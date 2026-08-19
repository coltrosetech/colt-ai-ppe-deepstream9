from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable

import jsonschema
import pytest

from validation import person_rtdetrv4_tensorrt_r14j_v6 as lane


@pytest.fixture(scope="session")
def templates() -> dict[int, dict[str, Any]]:
    return {
        640: json.loads((lane.ROOT / lane.TEMPLATE_640_REL).read_text(encoding="utf-8")),
        960: json.loads((lane.ROOT / lane.TEMPLATE_960_REL).read_text(encoding="utf-8")),
    }


@pytest.fixture(scope="session")
def r14i_template() -> dict[str, Any]:
    path = lane.PINS["r14i_template"]["path"]
    return json.loads((lane.ROOT / path).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def audit() -> dict[str, Any]:
    return lane.audit(require_static_controller=True)


def nested_set(value: dict[str, Any], path: str, replacement: Any) -> None:
    parts = path.split(".")
    cursor: Any = value
    for component in parts[:-1]:
        cursor = cursor[int(component)] if isinstance(cursor, list) else cursor[component]
    if isinstance(cursor, list):
        cursor[int(parts[-1])] = replacement
    else:
        cursor[parts[-1]] = replacement


def verify_mutated(
    profile: int,
    templates: dict[int, dict[str, Any]],
    r14i_template: dict[str, Any],
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    value = copy.deepcopy(templates[profile])
    mutation(value)
    with lane.Reader() as reader:
        with pytest.raises(lane.PersonR14JV6Error):
            lane.verify_profile(profile, value, r14i_template, reader)


def test_strict_json_accepts_object() -> None:
    assert lane.strict_json(b'{"a":1}', "ok") == {"a": 1}


@pytest.mark.parametrize(
    "raw,match",
    [
        (b'{"a":1,"a":2}', "duplicate JSON key"),
        (b'{"a":NaN}', "non-finite"),
        (b'{"a":Infinity}', "non-finite"),
        (b'\xef\xbb\xbf{"a":1}', "BOM"),
        (b'{"a":"\xff"}', "invalid JSON"),
        (b'[]', "root differs"),
    ],
)
def test_strict_json_rejects_ambiguous_envelopes(raw: bytes, match: str) -> None:
    with pytest.raises(lane.PersonR14JV6Error, match=match):
        lane.strict_json(raw, "bad")


@pytest.mark.parametrize(
    "path", ["", "/absolute", "a/../b", "a/./b", "a//b", "a\\b", "trailing/"]
)
def test_reader_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(lane.PersonR14JV6Error, match="unsafe path"):
        lane.Reader.parts(path)


def test_reader_replays_exact_regular(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "subject"
    target.write_bytes(b"subject")
    target.chmod(0o440)
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    expected = lane.pin("subject", 7, hashlib.sha256(b"subject").hexdigest())
    with lane.Reader() as reader:
        assert reader.read_exact(expected, "subject") == b"subject"


def test_reader_rejects_final_symlink(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "target").write_bytes(b"x")
    (tmp_path / "link").symlink_to("target")
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    with lane.Reader() as reader:
        with pytest.raises(OSError):
            with reader.open_regular("link"):
                pass


def test_reader_rejects_parent_symlink(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "item").write_bytes(b"x")
    (tmp_path / "link").symlink_to("real", target_is_directory=True)
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    with lane.Reader() as reader:
        with pytest.raises(OSError):
            with reader.open_regular("link/item"):
                pass


def test_reader_rejects_hardlink(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "first").write_bytes(b"x")
    os.link(tmp_path / "first", tmp_path / "second")
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    with lane.Reader() as reader:
        with pytest.raises(lane.PersonR14JV6Error, match="regular identity differs"):
            with reader.open_regular("first"):
                pass


def test_reader_rejects_mode_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "subject"
    target.write_bytes(b"x")
    target.chmod(0o600)
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    expected = lane.pin("subject", 1, hashlib.sha256(b"x").hexdigest())
    with lane.Reader() as reader:
        with pytest.raises(lane.PersonR14JV6Error, match="pin metadata differs"):
            reader.read_exact(expected, "subject")


def test_author_audit_passes_without_execution(audit: dict[str, Any]) -> None:
    assert audit["status"] == "author_candidate_execution_closed"
    assert audit["resource_boundary"] == {
        "docker_called": False,
        "nvidia_smi_called": False,
        "gpu_called": False,
        "model_or_onnx_loaded": False,
        "tensorrt_called": False,
        "deepstream_called": False,
        "network_called": False,
    }


def test_static_controller_replays_exactly(audit: dict[str, Any]) -> None:
    static = lane.strict_json(
        (lane.ROOT / lane.STATIC_CONTROLLER_REL).read_bytes(), "static controller"
    )
    assert static == audit["controller"]
    assert static["fingerprint_sha256"] == lane.fingerprint(
        static, "fingerprint_sha256"
    )


@pytest.mark.parametrize(
    "schema_rel,value_rel",
    [
        (lane.TEMPLATE_SCHEMA_REL, lane.TEMPLATE_640_REL),
        (lane.TEMPLATE_SCHEMA_REL, lane.TEMPLATE_960_REL),
        (lane.CONTROLLER_SCHEMA_REL, lane.STATIC_CONTROLLER_REL),
    ],
)
def test_schemas_are_valid_closed_and_accept_exact_subject(
    schema_rel: str, value_rel: str
) -> None:
    schema = json.loads((lane.ROOT / schema_rel).read_text(encoding="utf-8"))
    value = json.loads((lane.ROOT / value_rel).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(value)
    mutated = copy.deepcopy(value)
    mutated["unexpected"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(mutated)


def test_all_subject_pins_are_exact_regular_single_link() -> None:
    with lane.Reader() as reader:
        for label, expected in lane.PINS.items():
            raw = reader.read_exact(expected, label)
            assert len(raw) == expected["bytes"]
            assert hashlib.sha256(raw).hexdigest() == expected["sha256"]


def test_frozen_r14i_acceptance_is_unmodified() -> None:
    expected = lane.PINS["r14i_independent_receipt"]
    path = lane.ROOT / expected["path"]
    assert path.stat().st_size == expected["bytes"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o440
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]


def test_foundations_remain_non_execution_only(audit: dict[str, Any]) -> None:
    foundations = audit["controller"]["foundations"]
    assert foundations == {
        "r14i_independent_acceptance": True,
        "gpu_lease_v6_static_bundle_acceptance": True,
        "gpu_lease_v6_api_ready": False,
        "current_boot": lane.BOOT_ID,
        "native_image_identity_exact": True,
        "gpu_runtime_qualified": False,
    }


@pytest.mark.parametrize("profile", [640, 960])
def test_profile_batch_shapes_and_precision_are_exact(
    templates: dict[int, dict[str, Any]], profile: int
) -> None:
    row = templates[profile]
    assert row["optimization_profile"] == {
        "images": {
            "min": [1, 3, profile, profile],
            "opt": [12, 3, profile, profile],
            "max": [12, 3, profile, profile],
        },
        "orig_target_sizes": {
            "min": [1, 2], "opt": [12, 2], "max": [12, 2]
        },
    }
    assert row["precision"] == {"fp16": True, "int8": False, "tf32": False}


def test_profiles_are_strictly_640_then_960(templates: dict[int, dict[str, Any]]) -> None:
    assert templates[640]["sequence"] == {
        "index": 1,
        "required_previous_profile": None,
        "parallel_with_other_profile": False,
        "order": "640_then_960",
    }
    assert templates[960]["sequence"] == {
        "index": 2,
        "required_previous_profile": 640,
        "parallel_with_other_profile": False,
        "order": "640_then_960",
    }


@pytest.mark.parametrize("profile", [640, 960])
def test_exact_foreground_argv_and_digest(
    templates: dict[int, dict[str, Any]], profile: int
) -> None:
    row = templates[profile]
    workload = row["future_v6_workload"]
    argv = workload["argv"]
    assert row["runtime"]["foreground"] is True
    assert argv[:2] == ["/usr/bin/docker", "run"]
    assert argv[16] == lane.IMAGE_REFERENCE
    assert workload["image_argv_index"] == 16
    assert workload["argv_sha256"] == lane.command_argv_sha256(argv)
    assert "--rm" not in argv and "--detach" not in argv and "-d" not in argv
    assert "--network=none" in argv and "--pull=never" in argv
    assert "--runtime=runc" in argv and "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges:true" in argv
    assert f"--gpus=device={lane.GPU_UUID}" in argv
    assert "--fp16" in argv and "--noTF32" in argv and "--skipInference" in argv


@pytest.mark.parametrize("profile", [640, 960])
def test_mounts_are_one_read_only_input_and_one_fresh_output(
    templates: dict[int, dict[str, Any]], profile: int
) -> None:
    argv = templates[profile]["future_v6_workload"]["argv"]
    mounts = [item for item in argv if item.startswith("--mount=")]
    assert len(mounts) == 2
    assert mounts[0].endswith(f"dst=/input/person-{profile}.onnx,readonly")
    assert mounts[1].endswith("dst=/output,rw")
    assert "/validation/work/person-r14j-v6/" in mounts[1]


@pytest.mark.parametrize("profile", [640, 960])
def test_templates_have_no_live_v6_authority(
    templates: dict[int, dict[str, Any]], profile: int
) -> None:
    row = templates[profile]
    assert row["authority"]["independent_template_acceptance_required"] is True
    assert all(
        value is False
        for key, value in row["authority"].items()
        if key != "independent_template_acceptance_required"
    )
    assert row["future_v6_workload"]["missing_authority"] == {
        "plan_id": None,
        "outer_acceptance": None,
        "user_notification": None,
        "activation_receipt": None,
    }
    assert row["future_v6_workload"]["caller_overrides"] is False


@pytest.mark.parametrize("profile", [640, 960])
def test_outputs_are_absent_fresh_and_non_overwriting(
    templates: dict[int, dict[str, Any]], profile: int
) -> None:
    outputs = templates[profile]["outputs"]
    assert outputs["output_directory"]["required_fresh_mode"] == "0700"
    assert outputs["output_directory"]["materialization_authorized"] is False
    for output in outputs.values():
        assert output["required_absent_now"] is True
        assert not (lane.ROOT / output["path"]).exists()
        if "published_pin" in output:
            assert output["published_pin"] is None
        if "overwrite_authorized" in output:
            assert output["overwrite_authorized"] is False


def test_profile_output_sets_are_disjoint(templates: dict[int, dict[str, Any]]) -> None:
    left = {item["path"] for item in templates[640]["outputs"].values()}
    right = {item["path"] for item in templates[960]["outputs"].values()}
    assert left.isdisjoint(right)


@pytest.mark.parametrize(
    "profile,path,replacement",
    [
        (640, "profile", 960),
        (640, "sequence.index", 2),
        (960, "sequence.required_previous_profile", None),
        (640, "runtime.image_reference", "mutable:latest"),
        (640, "gpu.uuid", "GPU-wrong"),
        (640, "precision.fp16", False),
        (960, "optimization_profile.images.opt.0", 11),
        (960, "onnx.mounted_read_only", False),
        (640, "future_v6_workload.image_argv_index", 15),
        (640, "future_v6_workload.argv_sha256", "0" * 64),
        (640, "future_v6_workload.missing_authority.plan_id", "live"),
        (960, "future_v6_workload.caller_overrides", True),
        (640, "outputs.output_directory.materialization_authorized", True),
        (640, "outputs.staging_engine.overwrite_authorized", True),
        (960, "outputs.final_engine.required_absent_now", False),
        (960, "outputs.engine_receipt.published_pin", {"sha256": "x"}),
        (640, "claim_boundary.gpu_executed", True),
        (960, "template_fingerprint_sha256", "0" * 64),
    ],
)
def test_semantic_mutations_fail_closed(
    templates: dict[int, dict[str, Any]],
    r14i_template: dict[str, Any],
    profile: int,
    path: str,
    replacement: Any,
) -> None:
    verify_mutated(
        profile,
        templates,
        r14i_template,
        lambda value: nested_set(value, path, replacement),
    )


@pytest.mark.parametrize("forbidden", ["--rm", "--detach", "--privileged", "--restart=always", "--label=x"])
def test_forbidden_argv_mutations_fail_closed(
    templates: dict[int, dict[str, Any]],
    r14i_template: dict[str, Any],
    forbidden: str,
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        argv = value["future_v6_workload"]["argv"]
        argv.insert(2, forbidden)
        value["future_v6_workload"]["argv_sha256"] = lane.command_argv_sha256(argv)

    verify_mutated(640, templates, r14i_template, mutate)


def test_writable_onnx_mount_mutation_fails_closed(
    templates: dict[int, dict[str, Any]], r14i_template: dict[str, Any]
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        argv = value["future_v6_workload"]["argv"]
        argv[12] = argv[12].removesuffix(",readonly") + ",rw"
        value["future_v6_workload"]["argv_sha256"] = lane.command_argv_sha256(argv)

    verify_mutated(640, templates, r14i_template, mutate)


def test_author_source_has_no_execution_or_mutation_surface() -> None:
    source = (lane.ROOT / lane.THIS_REL).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_imports = {
        "subprocess", "docker", "torch", "onnx", "onnxruntime", "tensorrt",
        "requests", "urllib", "socket", "gi", "pyds", "ctypes",
    }
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name.split(".")[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.add(node.func.id)
    assert imports.isdisjoint(forbidden_imports)
    assert calls.isdisjoint(
        {"system", "popen", "run", "Popen", "execv", "execve", "spawnv",
         "remove", "unlink", "rename", "replace", "mkdir", "makedirs",
         "rmdir", "chmod", "chown", "link", "symlink"}
    )


def test_all_compile_time_authority_constants_are_false() -> None:
    names = [
        "EXECUTION_AUTHORIZED", "USER_NOTIFICATION_ACCEPTED",
        "OUTER_ACCEPTANCE_PUBLISHED", "LIVE_V6_PLAN_PUBLISHED",
        "LAUNCH_AUTHORIZED", "ACTIVATION_RECEIPT_PUBLISHED",
        "PRODUCTION_AUTHORIZED",
    ]
    assert all(getattr(lane, name) is False for name in names)


def test_controller_gate_remains_closed(audit: dict[str, Any]) -> None:
    controller = audit["controller"]
    assert controller["execution_gate"]["authorized"] is False
    assert all(value is False for value in controller["authority"].values())
    assert controller["claim_boundary"]["engine_built"] is False
    assert controller["claim_boundary"]["production_ready"] is False


@pytest.fixture(scope="session")
def handoff() -> dict[str, Any]:
    return json.loads((lane.ROOT / lane.HANDOFF_REL).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def handoff_schema() -> dict[str, Any]:
    return json.loads(
        (lane.ROOT / lane.HANDOFF_SCHEMA_REL).read_text(encoding="utf-8")
    )


def test_handoff_schema_is_valid_and_recursively_closed(
    handoff_schema: dict[str, Any],
) -> None:
    jsonschema.Draft202012Validator.check_schema(handoff_schema)
    object_schemas = 0
    stack: list[Any] = [handoff_schema]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if item.get("type") == "object":
                object_schemas += 1
                assert item.get("additionalProperties") is False
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    assert object_schemas >= 15


def test_handoff_schema_accepts_exact_subject_only(
    handoff: dict[str, Any], handoff_schema: dict[str, Any]
) -> None:
    jsonschema.Draft202012Validator(handoff_schema).validate(handoff)
    mutated = copy.deepcopy(handoff)
    mutated["unexpected"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(handoff_schema).validate(mutated)


def test_handoff_fingerprint_replays(handoff: dict[str, Any]) -> None:
    assert handoff["self_fingerprint"]["canonical_sha256"] == lane.fingerprint(
        handoff, "self_fingerprint"
    )


def test_handoff_is_nonterminal_and_non_authorizing(handoff: dict[str, Any]) -> None:
    assert handoff["status"] == "author_handoff_pending_independent_review"
    assert handoff["decision"] is None
    assert handoff["terminal_accepted"] is False
    assert handoff["execution_authorized"] is False
    assert handoff["authority"]["independent_review_required"] is True
    assert all(value is False for value in handoff["permissions"].values())
    assert all(value is False for value in handoff["execution_outputs"].values())


def test_handoff_declares_exact_subject_paths_and_pins(
    handoff: dict[str, Any],
) -> None:
    declared = handoff["subject"]["artifacts"]
    assert set(declared) == set(lane.HANDOFF_ARTIFACT_PATHS)
    with lane.Reader() as reader:
        for label, path in lane.HANDOFF_ARTIFACT_PATHS.items():
            _, current = reader.read_current(path, label)
            assert declared[label] == current


def test_handoff_foundations_are_exact(handoff: dict[str, Any]) -> None:
    assert handoff["accepted_foundations"] == lane.expected_handoff_foundations()


@pytest.mark.parametrize(
    "path,replacement",
    [
        ("execution_authorized", True),
        ("decision", "ACCEPT"),
        ("authority.independent_review_required", False),
        ("permissions.accept", True),
        ("execution_outputs.engine_640_published", True),
        ("semantic_bindings.overwrite_authorized", True),
    ],
)
def test_handoff_authority_mutations_are_schema_rejected(
    handoff: dict[str, Any],
    handoff_schema: dict[str, Any],
    path: str,
    replacement: Any,
) -> None:
    mutated = copy.deepcopy(handoff)
    nested_set(mutated, path, replacement)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(handoff_schema).validate(mutated)


def test_verify_handoff_replays_all_files_without_execution() -> None:
    result = lane.verify_handoff()
    assert result["status"] == "PASS"
    assert set(result["artifacts_replayed"]) == set(lane.HANDOFF_ARTIFACT_PATHS)
    assert result["authority"] == {
        "independent_review_still_required": True,
        "execution_authorized": False,
        "live_v6_plan_published": False,
        "outer_acceptance_published": False,
        "launch_authorized": False,
    }
    assert all(value is False for value in result["resource_boundary"].values())


def test_author_test_command_is_gpu_hidden(handoff: dict[str, Any]) -> None:
    command = handoff["author_tests"]["command"]
    assert command.startswith(
        "CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=void "
        "PYTHONDONTWRITEBYTECODE=1 "
    )
    assert handoff["author_tests"]["required_replays"] == 2
    assert handoff["author_tests"]["author_reported_replays"] == 2
