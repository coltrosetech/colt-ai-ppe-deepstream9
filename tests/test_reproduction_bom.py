from __future__ import annotations

import copy
import errno
import hashlib
import os
import socket
import stat
import time
from pathlib import Path

import pytest

import validation.reproduction_bom as bom_module
from validation.reproduction_bom import (
    CANONICALIZATION,
    MAX_ENVIRONMENT_DEPTH,
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_PATH_DEPTH,
    SCHEMA_PATH,
    SCHEMA_SHA256,
    FileDeclaration,
    ReproductionBomError,
    build_reproduction_bom,
    canonical_sha256,
    validate_reproduction_bom_schema,
    verify_reproduction_bom,
)


def _file(
    path: str,
    *,
    roles: tuple[str, ...] = ("reproduction input",),
    read_by: tuple[str, ...] = ("validation runner",),
    provenance: tuple[str, ...] = ("caller declaration",),
) -> FileDeclaration:
    return FileDeclaration(path, roles, read_by, provenance)


def _resign(document: dict) -> dict:
    result = copy.deepcopy(document)
    result.pop("self_fingerprint", None)
    digest = canonical_sha256(result)
    result["self_fingerprint"] = {
        "algorithm": "sha256",
        "canonicalization": CANONICALIZATION,
        "value": digest,
    }
    return result


def test_builder_is_order_independent_normalized_and_schema_valid(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "z.txt").write_text("zeta", encoding="utf-8")
    (tmp_path / "inputs" / "a.txt").write_text("alpha", encoding="utf-8")

    first = build_reproduction_bom(
        tmp_path,
        files=[
            _file(
                "inputs/z.txt",
                roles=(" report input ", "source", "report   input"),
                read_by=(" generator ", "verifier", "generator"),
                provenance=(" local fixture ", "Local author"),
            ),
            _file("inputs/a.txt"),
        ],
        directory_snapshots=["inputs", "."],
        absences=["inputs/not-created.json"],
        environment_metadata={
            "versions": {"python": "3.12", "deepstream": "9.0"},
            "profile": 640,
        },
    )
    second = build_reproduction_bom(
        tmp_path,
        files=[
            _file("inputs/a.txt"),
            _file(
                "inputs/z.txt",
                roles=("source", "report   input", " report input "),
                read_by=("verifier", "generator", " generator "),
                provenance=("Local author", " local fixture "),
            ),
        ],
        directory_snapshots=[".", "inputs"],
        absences=["inputs/not-created.json"],
        environment_metadata={
            "profile": 640,
            "versions": {"deepstream": "9.0", "python": "3.12"},
        },
    )

    assert first == second
    assert [item["path"] for item in first["files"]] == [
        "inputs/a.txt",
        "inputs/z.txt",
    ]
    assert first["files"][1]["roles"] == ["report input", "source"]
    assert first["files"][1]["read_by"] == ["generator", "verifier"]
    assert first["scope"] == {
        "type": "declared_reproduction_closure",
        "covers": "only_files_directory_snapshots_absences_and_environment_listed_here",
        "runtime_complete": False,
    }
    assert first["environment_metadata"]["source"] == "caller_supplied"
    validate_reproduction_bom_schema(first)
    assert verify_reproduction_bom(tmp_path, first)["valid"] is True


def test_environment_is_only_caller_supplied_and_ambient_changes_do_not_enter_bom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "input.txt").write_text("stable", encoding="utf-8")
    declaration = [_file("input.txt")]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "first-ambient-value")
    first = build_reproduction_bom(
        tmp_path,
        files=declaration,
        environment_metadata={"gpu_uuid": "caller-fixed"},
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "different-ambient-value")
    monkeypatch.setenv("HOSTNAME", "also-not-discovered")
    second = build_reproduction_bom(
        tmp_path,
        files=declaration,
        environment_metadata={"gpu_uuid": "caller-fixed"},
    )

    assert first == second
    assert first["environment_metadata"] == {
        "source": "caller_supplied",
        "values": {"gpu_uuid": "caller-fixed"},
    }


def test_duplicate_file_path_is_a_declaration_merge_conflict(tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("same file", encoding="utf-8")
    with pytest.raises(ReproductionBomError, match="merge conflict"):
        build_reproduction_bom(
            tmp_path,
            files=[
                _file("input.txt", roles=("source",)),
                _file("input.txt", roles=("generated output",)),
            ],
            environment_metadata={},
        )


def test_same_length_rewrite_is_detected_by_live_hash_verification(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"AAAA-BBBB")
    document = build_reproduction_bom(
        tmp_path,
        files=[_file("input.bin")],
        environment_metadata={},
    )

    path.write_bytes(b"CCCC-DDDD")
    assert path.stat().st_size == document["files"][0]["size_bytes"]
    with pytest.raises(ReproductionBomError, match="live declared file mismatch.*sha256"):
        verify_reproduction_bom(tmp_path, document)


def test_same_length_rewrite_during_hash_is_rejected_as_unstable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "racing.bin"
    path.write_bytes(b"A" * (1024 * 1024 + 17))
    original_read = bom_module.os.read
    rewritten = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal rewritten
        chunk = original_read(descriptor, count)
        if chunk and not rewritten:
            rewritten = True
            path.write_bytes(b"B" * path.stat().st_size)
        return chunk

    monkeypatch.setattr(bom_module.os, "read", racing_read)
    with pytest.raises(ReproductionBomError, match="inode, size, or metadata changed"):
        build_reproduction_bom(
            tmp_path,
            files=[_file("racing.bin")],
            environment_metadata={},
        )
    assert rewritten is True


def test_self_fingerprint_and_resigned_scope_overclaim_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    document = build_reproduction_bom(
        tmp_path,
        files=[_file("input.txt")],
        environment_metadata={},
    )

    tampered = copy.deepcopy(document)
    tampered["files"][0]["roles"] = ["tampered"]
    with pytest.raises(ReproductionBomError, match="self-fingerprint mismatch"):
        verify_reproduction_bom(tmp_path, tampered)

    overclaim = copy.deepcopy(document)
    overclaim["scope"]["runtime_complete"] = True
    overclaim = _resign(overclaim)
    with pytest.raises(ReproductionBomError, match="schema validation failed"):
        validate_reproduction_bom_schema(overclaim)


@pytest.mark.parametrize("kind", ["leaf", "ancestor"])
def test_symlink_file_or_ancestor_is_rejected(
    tmp_path: Path, kind: str
) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "input.txt").write_text("input", encoding="utf-8")
    if kind == "leaf":
        (tmp_path / "linked.txt").symlink_to(tmp_path / "real" / "input.txt")
        declaration = _file("linked.txt")
    else:
        (tmp_path / "linked-dir").symlink_to(tmp_path / "real", target_is_directory=True)
        declaration = _file("linked-dir/input.txt")

    with pytest.raises(ReproductionBomError, match="securely open"):
        build_reproduction_bom(
            tmp_path,
            files=[declaration],
            environment_metadata={},
        )


def test_hardlinked_file_is_rejected_even_when_content_matches(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"content")
    os.link(source, tmp_path / "second-name.bin")
    assert source.stat().st_nlink == 2

    with pytest.raises(ReproductionBomError, match="exactly one hard link"):
        build_reproduction_bom(
            tmp_path,
            files=[_file("input.bin")],
            environment_metadata={},
        )


def test_platform_without_nofollow_support_cannot_silently_degrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bom_module, "_NO_FOLLOW", 0)
    with pytest.raises(ReproductionBomError, match="lacks required O_NOFOLLOW"):
        build_reproduction_bom(
            tmp_path,
            files=[],
            environment_metadata={},
        )


def test_directory_snapshot_is_sorted_and_enumerated_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "z").write_text("z", encoding="utf-8")
    (watched / "a").write_text("a", encoding="utf-8")

    original_open = bom_module._open_directory_at
    watched_open_count = 0

    def racing_open(root_fd: int, normalized_path: str) -> int:
        nonlocal watched_open_count
        if normalized_path == "watched":
            watched_open_count += 1
            if watched_open_count == 2:
                (watched / "appeared-between-enumerations").write_text(
                    "race", encoding="utf-8"
                )
        return original_open(root_fd, normalized_path)

    monkeypatch.setattr(bom_module, "_open_directory_at", racing_open)
    with pytest.raises(ReproductionBomError, match="entries changed between enumerations"):
        build_reproduction_bom(
            tmp_path,
            files=[],
            directory_snapshots=["watched"],
            environment_metadata={},
        )
    assert watched_open_count == 2

    monkeypatch.setattr(bom_module, "_open_directory_at", original_open)
    document = build_reproduction_bom(
        tmp_path,
        files=[],
        directory_snapshots=["watched"],
        environment_metadata={},
    )
    assert [item["name"] for item in document["directory_snapshots"][0]["entries"]] == [
        "a",
        "appeared-between-enumerations",
        "z",
    ]


def test_absence_is_bound_to_direct_parent_snapshot_and_live_addition_fails(
    tmp_path: Path,
) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "existing.json").write_text("{}", encoding="utf-8")
    document = build_reproduction_bom(
        tmp_path,
        files=[],
        absences=["reports/missing.json"],
        environment_metadata={},
    )

    snapshot = document["directory_snapshots"][0]
    absence = document["absence_commitments"][0]
    assert absence["witness_parent_directory"] == "reports"
    assert absence["first_missing_component"] == "missing.json"
    assert absence["remaining_components"] == ["missing.json"]
    assert absence["witness_parent_snapshot_sha256"] == snapshot[
        "snapshot_fingerprint_sha256"
    ]
    assert "missing.json" not in {item["name"] for item in snapshot["entries"]}

    (tmp_path / "reports" / "missing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReproductionBomError, match="live directory snapshot mismatch"):
        verify_reproduction_bom(tmp_path, document)


def test_nested_missing_parent_uses_closest_existing_ancestor_witness(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "keep.txt").write_text("keep", encoding="utf-8")
    document = build_reproduction_bom(
        tmp_path,
        files=[],
        absences=["reports/finalization/current/reproduction-bom.json"],
        environment_metadata={},
    )

    absence = document["absence_commitments"][0]
    assert absence == {
        "path": "reports/finalization/current/reproduction-bom.json",
        "witness_parent_directory": "reports",
        "first_missing_component": "finalization",
        "remaining_components": [
            "finalization",
            "current",
            "reproduction-bom.json",
        ],
        "commitment": "absent_via_first_missing_component_in_bound_ancestor_snapshot",
        "witness_parent_snapshot_sha256": document["directory_snapshots"][0][
            "snapshot_fingerprint_sha256"
        ],
    }
    assert verify_reproduction_bom(tmp_path, document)["valid"] is True

    (tmp_path / "reports" / "finalization").mkdir()
    with pytest.raises(ReproductionBomError, match="live directory snapshot mismatch"):
        verify_reproduction_bom(tmp_path, document)


@pytest.mark.parametrize("ancestor_kind", ["symlink", "regular_file"])
def test_absence_witness_never_traverses_symlink_or_non_directory(
    tmp_path: Path, ancestor_kind: str
) -> None:
    (tmp_path / "real").mkdir()
    if ancestor_kind == "symlink":
        (tmp_path / "blocked").symlink_to(tmp_path / "real", target_is_directory=True)
        expected = "traverses a symlink"
    else:
        (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")
        expected = "non-directory ancestor"
    with pytest.raises(ReproductionBomError, match=expected):
        build_reproduction_bom(
            tmp_path,
            files=[],
            absences=["blocked/deep/missing.json"],
            environment_metadata={},
        )


def test_resigned_absence_parent_substitution_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    document = build_reproduction_bom(
        tmp_path,
        files=[],
        directory_snapshots=["two"],
        absences=["one/missing"],
        environment_metadata={},
    )
    snapshots = {item["path"]: item for item in document["directory_snapshots"]}

    substituted = copy.deepcopy(document)
    absence = substituted["absence_commitments"][0]
    absence["witness_parent_directory"] = "two"
    absence["witness_parent_snapshot_sha256"] = snapshots["two"][
        "snapshot_fingerprint_sha256"
    ]
    substituted = _resign(substituted)
    with pytest.raises(ReproductionBomError, match="witness chain is inconsistent"):
        verify_reproduction_bom(tmp_path, substituted)


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/absolute",
        "double//separator",
        "dot/./segment",
        "back\\slash",
        "/".join(["deep"] * (MAX_PATH_DEPTH + 1)),
        "é" * 513,
    ],
)
def test_unbounded_or_non_project_paths_are_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(ReproductionBomError, match="path|depth|backslash"):
        build_reproduction_bom(
            tmp_path,
            files=[_file(path)],
            environment_metadata={},
        )


def test_environment_depth_float_and_cyclic_values_are_rejected(tmp_path: Path) -> None:
    nested: dict = {}
    cursor = nested
    for index in range(MAX_ENVIRONMENT_DEPTH + 1):
        child: dict = {}
        cursor[f"level_{index}"] = child
        cursor = child
    with pytest.raises(ReproductionBomError, match="exceeds depth"):
        build_reproduction_bom(
            tmp_path,
            files=[],
            environment_metadata=nested,
        )

    with pytest.raises(ReproductionBomError, match="contains a float"):
        build_reproduction_bom(
            tmp_path,
            files=[],
            environment_metadata={"driver": 590.48},
        )

    cyclic: dict = {}
    cyclic["self"] = cyclic
    with pytest.raises(ReproductionBomError, match="contains a cycle"):
        build_reproduction_bom(
            tmp_path,
            files=[],
            environment_metadata=cyclic,
        )


def test_file_count_environment_bytes_and_file_bytes_are_bounded(tmp_path: Path) -> None:
    too_many_files = (
        _file(f"file-{index:04d}.txt") for index in range(MAX_FILES + 1)
    )
    with pytest.raises(ReproductionBomError, match="files exceeds maximum item count"):
        build_reproduction_bom(
            tmp_path,
            files=too_many_files,
            environment_metadata={},
        )

    oversized_environment = {f"key-{index:03d}": "x" * 1000 for index in range(70)}
    with pytest.raises(ReproductionBomError, match="canonical bytes"):
        build_reproduction_bom(
            tmp_path,
            files=[],
            environment_metadata=oversized_environment,
        )

    oversized_file = tmp_path / "oversized-sparse.bin"
    try:
        with oversized_file.open("wb") as handle:
            handle.truncate(MAX_FILE_BYTES + 1)
    except OSError as exc:  # pragma: no cover - unusual filesystem file-size cap
        pytest.skip(f"filesystem cannot create a sparse bound-test file: {exc}")
    with pytest.raises(ReproductionBomError, match="exceeds maximum size"):
        build_reproduction_bom(
            tmp_path,
            files=[_file("oversized-sparse.bin")],
            environment_metadata={},
        )


def test_schema_rejects_unknown_fields_and_unsorted_sets_fail_internal_contract(
    tmp_path: Path,
) -> None:
    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    document = build_reproduction_bom(
        tmp_path,
        files=[_file("input.txt", roles=("a", "b"))],
        environment_metadata={},
    )

    unknown = copy.deepcopy(document)
    unknown["unexpected"] = True
    unknown = _resign(unknown)
    with pytest.raises(ReproductionBomError, match="schema validation failed"):
        validate_reproduction_bom_schema(unknown)

    unsorted = copy.deepcopy(document)
    unsorted["files"][0]["roles"] = ["b", "a"]
    unsorted = _resign(unsorted)
    with pytest.raises(ReproductionBomError, match="not normalized, sorted, and unique"):
        verify_reproduction_bom(tmp_path, unsorted)


def test_every_declared_and_schema_leaf_open_is_nonblocking_and_noctty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert bom_module._NONBLOCK != 0
    assert bom_module._NO_CTTY != 0
    assert bom_module._FILE_FLAGS & bom_module._NONBLOCK
    assert bom_module._FILE_FLAGS & bom_module._NO_CTTY

    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    original_open = bom_module.os.open
    observed: dict[str, list[int]] = {"input.txt": [], SCHEMA_PATH.name: []}

    def recording_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if path in observed:
            observed[str(path)].append(flags)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(bom_module.os, "open", recording_open)
    monkeypatch.setattr(
        bom_module.os,
        "supports_dir_fd",
        set(bom_module.os.supports_dir_fd).union({recording_open}),
    )
    build_reproduction_bom(
        tmp_path,
        files=[_file("input.txt")],
        environment_metadata={},
    )

    # Each accepted leaf is opened once for reading and once for path-identity
    # verification.  The safety flags must be on the initial open, not added
    # after fstat has already had a chance to block on a FIFO/device.
    assert len(observed["input.txt"]) == 2
    assert len(observed[SCHEMA_PATH.name]) == 2
    for flags in observed["input.txt"] + observed[SCHEMA_PATH.name]:
        assert flags & bom_module._NONBLOCK
        assert flags & bom_module._NO_CTTY
        assert flags & bom_module._NO_FOLLOW


@pytest.mark.parametrize("special_kind", ["fifo", "unix_socket", "device"])
def test_fifo_socket_and_device_leaf_probes_fail_quickly(
    tmp_path: Path, special_kind: str
) -> None:
    # This assertion intentionally precedes creation/open of the FIFO: if the
    # required flag is ever removed, the test fails rather than hanging.
    assert bom_module._FILE_FLAGS & os.O_NONBLOCK
    opened_socket: socket.socket | None = None
    if special_kind == "fifo":
        os.mkfifo(tmp_path / "special")
        root, relative = tmp_path, "special"
    elif special_kind == "unix_socket":
        opened_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        opened_socket.bind(str(tmp_path / "special"))
        root, relative = tmp_path, "special"
    else:
        if not Path("/dev/null").exists():  # pragma: no cover - non-POSIX host
            pytest.skip("/dev/null is unavailable")
        root, relative = Path("/"), "dev/null"

    started = time.monotonic()
    try:
        with pytest.raises(ReproductionBomError, match="regular file|securely open"):
            build_reproduction_bom(
                root,
                files=[_file(relative)],
                environment_metadata={},
            )
    finally:
        if opened_socket is not None:
            opened_socket.close()
    assert time.monotonic() - started < 2.0


def test_schema_content_is_exactly_digest_pinned() -> None:
    raw = SCHEMA_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SCHEMA_SHA256
    schema = bom_module._load_schema()
    assert schema["properties"]["schema_version"]["const"] == bom_module.SCHEMA_VERSION


def _install_private_schema_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, bytes]:
    root = tmp_path / "schema-root"
    schema_dir = root / "schemas"
    schema_dir.mkdir(parents=True)
    raw = SCHEMA_PATH.read_bytes()
    target = schema_dir / SCHEMA_PATH.name
    target.write_bytes(raw)
    monkeypatch.setattr(bom_module, "_SCHEMA_ROOT", root)
    monkeypatch.setattr(
        bom_module, "_SCHEMA_RELATIVE_PATH", f"schemas/{SCHEMA_PATH.name}"
    )
    monkeypatch.setattr(bom_module, "SCHEMA_PATH", target)
    monkeypatch.setattr(
        bom_module, "SCHEMA_SHA256", hashlib.sha256(raw).hexdigest()
    )
    return root, target, raw


def test_schema_digest_mismatch_fails_before_schema_is_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_private_schema_copy(tmp_path, monkeypatch)
    monkeypatch.setattr(bom_module, "SCHEMA_SHA256", "0" * 64)
    with pytest.raises(ReproductionBomError, match="pinned v1 SHA-256"):
        bom_module._load_schema()


def test_schema_ancestor_swap_is_detected_across_complete_reopen_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, raw = _install_private_schema_copy(tmp_path, monkeypatch)
    original_open_chain = bom_module._open_absolute_directory_chain
    open_count = 0

    def swapping_chain(path: Path):  # type: ignore[no-untyped-def]
        nonlocal open_count
        open_count += 1
        if open_count == 2:
            old_root = tmp_path / "schema-root-old"
            root.rename(old_root)
            (root / "schemas").mkdir(parents=True)
            (root / "schemas" / SCHEMA_PATH.name).write_bytes(raw)
        return original_open_chain(path)

    monkeypatch.setattr(bom_module, "_open_absolute_directory_chain", swapping_chain)
    with pytest.raises(ReproductionBomError, match="directory identity chain changed"):
        bom_module._load_schema()
    assert open_count == 2


def test_same_content_schema_leaf_path_replacement_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, target, raw = _install_private_schema_copy(tmp_path, monkeypatch)
    original_open_chain = bom_module._open_directory_chain_at
    parent_open_count = 0

    def replacing_parent_chain(root_fd: int, normalized_path: str):  # type: ignore[no-untyped-def]
        nonlocal parent_open_count
        result = original_open_chain(root_fd, normalized_path)
        parent_open_count += 1
        if parent_open_count == 2:
            # The second chain identities have already been captured and still
            # match the first.  Replace only the leaf before its second open so
            # the explicit leaf identity check, rather than a parent timestamp,
            # must catch the substitution.
            target.rename(target.with_suffix(".replaced"))
            target.write_bytes(raw)
        return result

    monkeypatch.setattr(bom_module, "_open_directory_chain_at", replacing_parent_chain)
    with pytest.raises(ReproductionBomError, match="schema path identity changed"):
        bom_module._load_schema()
    assert parent_open_count == 2


def test_schema_fifo_probe_is_nonblocking_before_type_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "schema-root"
    (root / "schemas").mkdir(parents=True)
    target = root / "schemas" / SCHEMA_PATH.name
    os.mkfifo(target)
    monkeypatch.setattr(bom_module, "_SCHEMA_ROOT", root)
    monkeypatch.setattr(
        bom_module, "_SCHEMA_RELATIVE_PATH", f"schemas/{target.name}"
    )
    monkeypatch.setattr(bom_module, "SCHEMA_PATH", target)
    assert bom_module._FILE_FLAGS & os.O_NONBLOCK
    started = time.monotonic()
    with pytest.raises(ReproductionBomError, match="single-link regular file"):
        bom_module._load_schema()
    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize(
    ("field_path", "replacement", "expected"),
    [
        (("schema_version",), "attacker/v9", "schema_version"),
        (("scope", "type"), "runtime_closure", "scope.type"),
        (("scope", "covers"), "everything", "scope.covers"),
        (("scope", "runtime_complete"), 0, "runtime_complete"),
        (("normalization", "paths"), "attacker/v9", "path normalization"),
        (("normalization", "text_sets"), "attacker/v9", "text-set normalization"),
        (("bounds", "max_files"), 4095, "hard limits"),
        (("environment_metadata", "source"), "ambient", "environment source"),
        (("self_fingerprint", "algorithm"), "sha1", "fingerprint algorithm"),
        (
            ("self_fingerprint", "canonicalization"),
            "implementation-defined",
            "fingerprint canonicalization",
        ),
        (
            ("absence_commitments", 0, "commitment"),
            "absence_claimed_without_bound_snapshot",
            "commitment does not match",
        ),
    ],
)
def test_critical_v1_semantics_are_enforced_without_schema_assistance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[object, ...],
    replacement: object,
    expected: str,
) -> None:
    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    document = build_reproduction_bom(
        tmp_path,
        files=[_file("input.txt")],
        absences=["not-created/deep/result.json"],
        environment_metadata={"gpu": "caller-fixed"},
    )
    tampered = copy.deepcopy(document)
    if field_path[0] != "self_fingerprint":
        cursor: object = tampered
        for component in field_path[:-1]:
            cursor = cursor[component]  # type: ignore[index]
        cursor[field_path[-1]] = replacement  # type: ignore[index]
        tampered = _resign(tampered)
    else:
        # self_fingerprint is excluded from its own digest, so mutate it after
        # constructing an otherwise valid signature-shaped object.
        tampered = _resign(tampered)
        tampered["self_fingerprint"][field_path[-1]] = replacement

    monkeypatch.setattr(bom_module, "validate_reproduction_bom_schema", lambda _: None)
    with pytest.raises(ReproductionBomError, match=expected):
        verify_reproduction_bom(tmp_path, tampered)


def test_internal_sha_check_rejects_trailing_newline_without_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    document = build_reproduction_bom(
        tmp_path,
        files=[_file("input.txt")],
        environment_metadata={},
    )
    document["files"][0]["sha256"] += "\n"
    document = _resign(document)
    monkeypatch.setattr(bom_module, "validate_reproduction_bom_schema", lambda _: None)
    with pytest.raises(ReproductionBomError, match="exact lowercase SHA-256"):
        verify_reproduction_bom(tmp_path, document)


def test_same_name_and_kind_inode_replacement_between_scans_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watched = tmp_path / "watched"
    watched.mkdir()
    target = watched / "same-name"
    target.write_text("same-content", encoding="utf-8")
    original_inode = target.stat().st_ino
    replacement = tmp_path / "replacement"
    replacement.write_text("same-content", encoding="utf-8")
    original_open = bom_module._open_directory_at
    watched_open_count = 0

    def replacing_open(root_fd: int, normalized_path: str) -> int:
        nonlocal watched_open_count
        if normalized_path == "watched":
            watched_open_count += 1
            if watched_open_count == 2:
                os.replace(replacement, target)
        return original_open(root_fd, normalized_path)

    monkeypatch.setattr(bom_module, "_open_directory_at", replacing_open)
    with pytest.raises(ReproductionBomError, match="entry identity changed"):
        build_reproduction_bom(
            tmp_path,
            files=[],
            directory_snapshots=["watched"],
            environment_metadata={},
        )
    assert target.stat().st_ino != original_inode


def test_full_directory_identity_is_compared_between_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "stable").write_text("stable", encoding="utf-8")
    original_open = bom_module._open_directory_at
    watched_open_count = 0

    def retiming_open(root_fd: int, normalized_path: str) -> int:
        nonlocal watched_open_count
        if normalized_path == "watched":
            watched_open_count += 1
            if watched_open_count == 2:
                metadata = watched.stat()
                os.utime(
                    watched,
                    ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000),
                )
        return original_open(root_fd, normalized_path)

    monkeypatch.setattr(bom_module, "_open_directory_at", retiming_open)
    with pytest.raises(ReproductionBomError, match="directory identity changed"):
        build_reproduction_bom(
            tmp_path,
            files=[],
            directory_snapshots=["watched"],
            environment_metadata={},
        )


def test_directory_limit_aborts_incremental_fd_scan_without_materializing_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "metadata-source").write_text("x", encoding="utf-8")
    metadata_result = (tmp_path / "metadata-source").stat()
    consumed = 0
    stat_calls = 0
    observed_fds: list[int] = []

    class FakeEntry:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeScandir:
        def __init__(self, total: int) -> None:
            self.total = total
            self.index = 0

        def __enter__(self) -> "FakeScandir":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __iter__(self) -> "FakeScandir":
            return self

        def __next__(self) -> FakeEntry:
            nonlocal consumed
            if self.index >= self.total:
                raise StopIteration
            name = f"entry-{self.index}"
            self.index += 1
            consumed += 1
            return FakeEntry(name)

    def fake_scandir(descriptor: int) -> FakeScandir:
        observed_fds.append(descriptor)
        return FakeScandir(total=100)

    def fake_stat(*args: object, **kwargs: object) -> os.stat_result:
        nonlocal stat_calls
        stat_calls += 1
        return metadata_result

    monkeypatch.setattr(bom_module, "MAX_ENTRIES_PER_DIRECTORY", 3)
    monkeypatch.setattr(bom_module.os, "scandir", fake_scandir)
    monkeypatch.setattr(bom_module.os, "stat", fake_stat)
    with pytest.raises(ReproductionBomError, match="exceeds 3 entries"):
        bom_module._enumerate_directory_once(9876, "synthetic")
    assert observed_fds == [9876]
    assert consumed == 4
    assert stat_calls == 3


def test_declared_file_same_content_path_replacement_during_reopen_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "input.txt"
    target.write_text("same-content", encoding="utf-8")
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("same-content", encoding="utf-8")
    original_inode = target.stat().st_ino
    original_open = bom_module._open_directory_at
    root_open_count = 0

    def replacing_parent_open(root_fd: int, normalized_path: str) -> int:
        nonlocal root_open_count
        if normalized_path == ".":
            root_open_count += 1
            if root_open_count == 2:
                os.replace(replacement, target)
        return original_open(root_fd, normalized_path)

    monkeypatch.setattr(bom_module, "_open_directory_at", replacing_parent_open)
    with pytest.raises(ReproductionBomError, match="file path identity changed"):
        build_reproduction_bom(
            tmp_path,
            files=[_file("input.txt")],
            environment_metadata={},
        )
    assert target.stat().st_ino != original_inode


def test_exact_string_schema_patterns_match_external_draft202012_on_newlines(
    tmp_path: Path
) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    document = build_reproduction_bom(
        tmp_path,
        files=[_file("input.txt")],
        directory_snapshots=["."],
        absences=["missing/deep.json"],
        environment_metadata={"fixed": "value"},
    )
    schema = bom_module._load_schema()
    external = jsonschema.Draft202012Validator(schema)

    def candidates() -> list[dict]:
        results: list[dict] = []
        for mutate in (
            lambda item: item["files"][0].__setitem__("path", "input.txt\n"),
            lambda item: item["files"][0].__setitem__(
                "sha256", item["files"][0]["sha256"] + "\n"
            ),
            lambda item: item["files"][0]["roles"].__setitem__(
                0, item["files"][0]["roles"][0] + "\n"
            ),
            lambda item: item["directory_snapshots"][0]["entries"][0].__setitem__(
                "name", item["directory_snapshots"][0]["entries"][0]["name"] + "\n"
            ),
            lambda item: item["absence_commitments"][0].__setitem__(
                "first_missing_component", "missing\n"
            ),
        ):
            candidate = copy.deepcopy(document)
            mutate(candidate)
            results.append(_resign(candidate))
        return results

    for candidate in candidates():
        external_errors = list(external.iter_errors(candidate))
        assert external_errors, "external Draft 2020-12 validator accepted newline suffix"
        with pytest.raises(ReproductionBomError, match="required pattern|longer than maxLength"):
            bom_module._validate_schema_subset(candidate, schema, schema, "<root>")
        with pytest.raises(ReproductionBomError, match="schema validation failed"):
            validate_reproduction_bom_schema(candidate)


def test_absolute_schema_chain_component_fstat_failure_is_wrapped_without_fd_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_fds = Path("/proc/self/fd")
    if not proc_fds.is_dir():  # pragma: no cover - Linux is the secure target
        pytest.skip("/proc/self/fd is unavailable")

    target = tmp_path / "schema-root"
    target.mkdir()
    original_fstat = bom_module.os.fstat

    def fd_count() -> int:
        return len(os.listdir(proc_fds))

    baseline = fd_count()
    with monkeypatch.context() as fault:
        calls = 0

        def failing_component_fstat(descriptor: int) -> os.stat_result:
            nonlocal calls
            calls += 1
            # First fstat owns the opened filesystem-root descriptor; the
            # second owns the first newly opened component descriptor.
            if calls == 2:
                raise OSError(errno.EIO, "injected component fstat failure")
            return original_fstat(descriptor)

        fault.setattr(bom_module.os, "fstat", failing_component_fstat)
        for _ in range(32):
            calls = 0
            with pytest.raises(
                ReproductionBomError,
                match="cannot securely inspect a BOM schema-root component",
            ) as captured:
                bom_module._open_absolute_directory_chain(target)
            assert isinstance(captured.value.__cause__, OSError)
            assert captured.value.__cause__.errno == errno.EIO
            assert fd_count() == baseline

    # Successful ownership transfer must also leave exactly one caller-owned
    # descriptor, which is returned and closed here on every repetition.
    for _ in range(32):
        descriptor, identities = bom_module._open_absolute_directory_chain(target)
        try:
            assert identities
            assert stat.S_ISDIR(os.fstat(descriptor).st_mode)
        finally:
            os.close(descriptor)
        assert fd_count() == baseline
