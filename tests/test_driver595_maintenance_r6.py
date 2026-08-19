from __future__ import annotations

import copy
import hashlib
import io
import inspect
import json
import os
import stat
import struct
import subprocess
import tarfile
from pathlib import Path

import jsonschema
import pytest

from validation import driver595_maintenance_r5 as r5
from validation import driver595_maintenance_r6 as r6


def _candidate(package: str = "pkg", payload: bytes = b"payload") -> dict[str, object]:
    return {
        "package": package,
        "version": "1",
        "architecture": "amd64",
        "filename": f"pool/{package}.deb",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _record(candidate: dict[str, object], **changes: str) -> dict[str, str]:
    value = {
        "Package": str(candidate["package"]),
        "Version": str(candidate["version"]),
        "Architecture": str(candidate["architecture"]),
        "Filename": str(candidate["filename"]),
        "Size": str(candidate["size"]),
        "SHA256": str(candidate["sha256"]),
    }
    value.update(changes)
    return value


def _index(path: str, approved: bool, records: list[dict[str, str]]) -> dict[str, object]:
    return {"path": path, "approved": approved, "records": records}


def _valid_initramfs(kernel: str) -> str:
    root = f"usr/lib/modules/{kernel}"
    return "\n".join(
        [
            ".",
            "init",
            root,
            f"{root}/modules.dep",
            f"{root}/modules.alias",
            f"{root}/kernel/drivers/nvme/host/nvme.ko.zst",
            f"{root}/kernel/drivers/nvme/host/nvme-core.ko.zst",
        ]
    ) + "\n"


@pytest.fixture(scope="session")
def prepared_plan() -> dict[str, object]:
    plan = r6.prepare_plan()
    assert plan["status"] == "candidate_read_only_pending_independent_acceptance"
    return plan


def test_frozen_r5_dependency_pins_are_live_and_unchanged() -> None:
    report = r6.verify_frozen_r5()
    assert report["status"] == "pass"
    assert all(report["checks"].values())


def test_author_release_has_no_acceptance_or_operational_receipt_roots() -> None:
    assert r6.PINNED_ACCEPTED_REVIEW_RECEIPT_SHA256 is None
    assert r6.PINNED_ACCEPTED_CACHE_RECEIPT_SHA256 is None
    assert r6.PINNED_ACCEPTED_PRE_REBOOT_RECEIPT_SHA256 is None
    assert r6.EXPECTED_AUTHORITY["install_authorized"] is False
    assert r6.EXPECTED_AUTHORITY["reboot_authorized"] is False
    assert r6.EXPECTED_AUTHORITY["gpu_workload_authorized"] is False


def test_public_review_gate_cannot_be_enabled_by_caller_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        r6,
        "load_json_raw",
        lambda *args, **kwargs: pytest.fail("receipt must not be read without source root"),
    )
    result = r6.verify_accepted_review_receipt(
        Path("attacker.json"),
        plan_raw_sha256="a" * 64,
        plan_fingerprint_sha256="b" * 64,
    )
    assert result["verification_status"] == "blocked"
    assert result["pinned_receipt_root_present"] is False
    assert result["prepared_plan_accepted"] is False


def test_integrity_result_is_never_plain_pass_without_review_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(r6, "verify_candidate_static", lambda *a, **k: {"status": "pass"})
    result = r6.verify_plan_static(
        {"plan_fingerprint_sha256": "b" * 64},
        raw_sha256="a" * 64,
        recollect_apt=False,
    )
    assert result["status"] == "integrity_pass_non_authorizing"
    assert result["candidate_accepted"] is False
    assert result["install_authorized"] is False


@pytest.mark.parametrize(
    "argv",
    [
        ["/usr/bin/apt-get", "install", "nvidia-driver-595"],
        ["/usr/bin/apt-get", "download", "nvidia-driver-595"],
        ["/usr/bin/dpkg", "--install", "x.deb"],
        ["/usr/sbin/update-initramfs", "-u"],
        ["/usr/sbin/update-grub"],
        ["/usr/sbin/reboot"],
        ["/usr/bin/nvidia-smi", "--gpu-reset"],
    ],
)
def test_read_only_command_grammar_rejects_mutation(argv: list[str]) -> None:
    assert r6._allowed_command(argv) is False
    with pytest.raises(ValueError):
        r6._run(argv)


def test_all_r6_schemas_are_valid_draft_2020_12() -> None:
    for path in (
        r6.PLAN_SCHEMA,
        r6.REVIEW_RECEIPT_SCHEMA,
        r6.CACHE_RECEIPT_SCHEMA,
        r6.PRE_REBOOT_RECEIPT_SCHEMA,
    ):
        value, _, _ = r6.load_json_raw(path)
        jsonschema.Draft202012Validator.check_schema(value)


def test_receipt_fingerprint_changes_for_nested_authority_mutation() -> None:
    receipt = {"authority": {"install_authorized": False}, "decision": "PASS"}
    before = r6.receipt_fingerprint(receipt)
    receipt["authority"]["install_authorized"] = True
    assert r6.receipt_fingerprint(receipt) != before


def test_apt_mapping_accepts_exact_two_authenticated_records() -> None:
    candidate = _candidate()
    indexes = [
        _index("updates", True, [_record(candidate)]),
        _index("security", True, [_record(candidate)]),
        _index("unrelated", False, []),
    ]
    result = r6.map_candidates_to_indexes([candidate], indexes)
    assert result["status"] == "pass"
    assert result["mappings"][0]["approved_exact_count"] == 2


def test_apt_mapping_rejects_unapproved_same_version_record() -> None:
    candidate = _candidate()
    indexes = [
        _index("updates", True, [_record(candidate)]),
        _index("security", True, [_record(candidate)]),
        _index("ppa", False, [_record(candidate)]),
    ]
    result = r6.map_candidates_to_indexes([candidate], indexes)
    assert result["status"] == "blocked"
    assert result["mappings"][0]["unapproved_exact_count"] == 1


def test_apt_mapping_rejects_unapproved_same_tuple_conflicting_hash() -> None:
    candidate = _candidate()
    indexes = [
        _index("updates", True, [_record(candidate)]),
        _index("security", True, [_record(candidate)]),
        _index("cuda", False, [_record(candidate, SHA256="f" * 64)]),
    ]
    result = r6.map_candidates_to_indexes([candidate], indexes)
    assert result["status"] == "blocked"
    assert result["mappings"][0]["unapproved_conflict_count"] == 1


def test_apt_mapping_rejects_third_approved_duplicate() -> None:
    candidate = _candidate()
    indexes = [
        _index(name, True, [_record(candidate)])
        for name in ("updates", "security", "duplicate")
    ]
    result = r6.map_candidates_to_indexes([candidate], indexes)
    assert result["status"] == "blocked"
    assert result["mappings"][0]["approved_exact_count"] == 3


def test_cleartext_inrelease_parser_rejects_missing_signature() -> None:
    with pytest.raises(ValueError):
        r6._cleartext_release(b"Origin: Ubuntu\n")


def test_gpgv_status_requires_exact_good_and_valid_signature_projection() -> None:
    status = (
        b"[GNUPG:] NEWSIG\n"
        b"[GNUPG:] KEY_CONSIDERED F6ECB3762474EDA9D21B7022871920D1991BC93C 0\n"
        b"[GNUPG:] SIG_ID fixture 2026-07-17 1784320678\n"
        b"[GNUPG:] GOODSIG 871920D1991BC93C Ubuntu Archive Key\n"
        b"[GNUPG:] VALIDSIG F6ECB3762474EDA9D21B7022871920D1991BC93C "
        b"2026-07-17 1784320678 0 4 0 1 10 01 "
        b"F6ECB3762474EDA9D21B7022871920D1991BC93C\n"
    )
    report = r6._parse_gpgv_status(status)
    assert report["validsig_fingerprints"] == [
        r6.ACCEPTED_APT_SIGNING_FINGERPRINT
    ]
    assert report["goodsig_keyids"] == [
        r6.ACCEPTED_APT_SIGNING_FINGERPRINT[-16:]
    ]
    assert report["bad_status_tags"] == []


def test_gpgv_status_rejects_unknown_or_projects_bad_signature() -> None:
    with pytest.raises(ValueError):
        r6._parse_gpgv_status(b"[GNUPG:] ATTACKER ignored\n")
    report = r6._parse_gpgv_status(
        b"[GNUPG:] NEWSIG\n[GNUPG:] BADSIG 871920D1991BC93C bad\n"
    )
    assert report["bad_status_tags"] == ["BADSIG"]


def test_apt_indextarget_broken_symlink_is_present_and_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "fixture_Packages"
    link.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(r6, "APT_LISTS", tmp_path)
    monkeypatch.setattr(
        r6,
        "collect_inrelease",
        lambda suite, now=None: {"status": "pass", "suite": suite},
    )
    row = (
        "Packages\thttp://attacker.invalid/ubuntu\tnoble\tmain\tamd64\t"
        f"{link}\n"
    ).encode()
    monkeypatch.setattr(
        r6,
        "_run",
        lambda *args, **kwargs: r6.CommandResult(r6.APT_INDEXTARGETS_ARGV, 0, row),
    )
    report = r6.collect_apt_provenance([])
    assert report["indexes"] == [
        {
            "site": "http://attacker.invalid/ubuntu",
            "release": "noble",
            "component": "main",
            "architecture": "amd64",
            "path": str(link),
            "present": True,
            "is_symlink": True,
            "approved": False,
            "authenticated_by": None,
            "pin": None,
            "release_entry": None,
            "status": "blocked",
            "records": [],
        }
    ]
    assert report["checks"]["every_present_index_scanned_and_trusted"] is False


def test_real_apt_provenance_chains_22_candidates_to_four_indexes() -> None:
    candidates = r5.collect_candidate_records()
    report = r6.collect_apt_provenance(candidates)
    assert report["status"] == "pass"
    assert len(report["candidate_mapping"]["mappings"]) == 22
    assert all(
        item["approved_exact_count"] == 2
        for item in report["candidate_mapping"]["mappings"]
    )
    assert sum(
        1 for item in report["indexes"] if item["present"] and item["approved"]
    ) == 4


def _current_nvidia_crypto_inputs() -> tuple[Path, bytes, dict[str, str]]:
    fields, success = r5._module_fields(r6.FALLBACK_KERNEL, "nvidia")
    assert success
    return Path(fields["filename"]), r6._certificate_file_der(r6.LOCAL_MOK), fields


def test_current_module_detached_pkcs7_is_cryptographically_valid() -> None:
    path, certificate, fields = _current_nvidia_crypto_inputs()
    report = r6.verify_detached_module_signature(
        path,
        certificate,
        expected_signer=fields["signer"],
        expected_sig_key=fields["sig_key"],
    )
    assert report["status"] == "pass"
    assert report["checks"]["pkcs7_detached_signature_valid"] is True
    assert report["checks"]["pkcs7_has_no_embedded_certificates"] is True


def test_detached_module_signature_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, certificate, fields = _current_nvidia_crypto_inputs()
    original = r6._decompress_module

    def corrupt(module_path: Path, raw: bytes) -> bytes:
        value = bytearray(original(module_path, raw))
        value[0] ^= 1
        return bytes(value)

    monkeypatch.setattr(r6, "_decompress_module", corrupt)
    report = r6.verify_detached_module_signature(
        path,
        certificate,
        expected_signer=fields["signer"],
        expected_sig_key=fields["sig_key"],
    )
    assert report["status"] == "blocked"
    assert report["checks"]["pkcs7_detached_signature_valid"] is False


def test_module_signature_parser_rejects_duplicate_terminal_marker() -> None:
    with pytest.raises(ValueError):
        r6.split_module_signature(
            b"ELF" + r6.MODULE_SIGNATURE_MARKER + r6.MODULE_SIGNATURE_MARKER
        )


def test_mok_certificate_serial_and_fingerprint_are_distinct_and_exact() -> None:
    metadata = r6.certificate_metadata(r6._certificate_file_der(r6.LOCAL_MOK))
    assert r6._normalize_hex(metadata["serial"]) == (
        "147B5707B8B52394443EC50E244AA8416198533C"
    )
    assert metadata["sha1_fingerprint"] == r6.LOCAL_MOK_SHA1


def test_enrolled_authority_and_revocation_projection_is_live() -> None:
    report = r6.collect_enrolled_authorities()
    assert report["status"] == "pass"
    assert report["checks"]["canonical_ca_enrolled"] is True
    assert report["checks"]["local_mok_enrolled"] is True
    assert report["checks"]["accepted_anchors_not_revoked"] is True
    assert len(report["revocation_tokens"]) > 100


def test_target_crypto_gate_is_blocked_without_ten_595_modules(
    prepared_plan: dict[str, object],
) -> None:
    report = r6.collect_target_crypto_proof(prepared_plan)
    assert report["status"] == "blocked"
    assert report["checks"]["both_kernel_crypto_chains"] is False


@pytest.fixture(scope="session")
def cached_target_kernel_image() -> bytes:
    package = Path(
        "/var/cache/apt/archives/"
        "linux-image-7.0.0-28-generic_7.0.0-28.28~24.04.1_amd64.deb"
    )
    if not package.exists():
        pytest.skip("authenticated target kernel package is no longer cached")
    assert r6.sha256_file(package) == (
        "ed08599632c2822cf702d897f6567043e2f045ea78a3357363ee16cf6c27f5bc"
    )
    archive = subprocess.run(
        ["/usr/bin/dpkg-deb", "--fsys-tarfile", str(package)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:*") as handle:
        members = [
            member
            for member in handle.getmembers()
            if member.name.lstrip("./") == f"boot/vmlinuz-{r6.TARGET_KERNEL}"
        ]
        assert len(members) == 1 and members[0].isfile()
        stream = handle.extractfile(members[0])
        assert stream is not None
        return stream.read()


def test_cached_authenticated_target_pe_authenticode_projection(
    cached_target_kernel_image: bytes,
) -> None:
    report = r6.analyze_pe_authenticode(cached_target_kernel_image)
    assert report["status"] == "pass"
    assert report["authenticode_sha256"] == (
        "AD581C90DBBC3FCD0992F1F89C3E53BEA07ECCDB78CC713F3800FECC2C58FA1F"
    )
    assert report["leaf_certificate"]["sha1_fingerprint"] == (
        "5FB405E64BDBC403B572FF1E62735C12EDEBCA37"
    )
    assert report["leaf_certificate"]["der_sha256"] == (
        "996bccf3fcf8c289dc534d412d3f3de7c79af2f1a8fab8c2fb0d94ab4a237977"
    )


def test_pe_parser_rejects_non_pkcs7_win_certificate(
    cached_target_kernel_image: bytes,
) -> None:
    damaged = bytearray(cached_target_kernel_image)
    pe_offset = struct.unpack_from("<I", damaged, 0x3C)[0]
    optional = pe_offset + 24
    magic = struct.unpack_from("<H", damaged, optional)[0]
    directory = optional + (112 if magic == 0x20B else 96)
    certificate_offset = struct.unpack_from("<I", damaged, directory + 32)[0]
    struct.pack_into("<H", damaged, certificate_offset + 6, 1)
    with pytest.raises(ValueError, match="non-PKCS7"):
        r6.extract_pe_pkcs7(bytes(damaged))


def test_valid_canonical_initramfs_listing_passes() -> None:
    report = r6.validate_initramfs_listing(
        r6.TARGET_KERNEL, _valid_initramfs(r6.TARGET_KERNEL), ext4_builtin=True
    )
    assert report["status"] == "pass"
    assert report["required_counts"] == {"init": 1, "modules_dep": 1, "modules_alias": 1}


def test_initramfs_rejects_absolute_and_duplicate_required_members() -> None:
    listing = _valid_initramfs(r6.TARGET_KERNEL)
    absolute = "\n".join("/" + line for line in listing.splitlines()) + "\n"
    absolute_report = r6.validate_initramfs_listing(
        r6.TARGET_KERNEL, absolute, ext4_builtin=True
    )
    assert absolute_report["status"] == "blocked"
    assert absolute_report["checks"]["every_member_canonical_relative_or_root_dot"] is False
    root = f"usr/lib/modules/{r6.TARGET_KERNEL}"
    duplicate = listing + f"{root}/modules.dep\n{root}/modules.alias\n"
    duplicate_report = r6.validate_initramfs_listing(
        r6.TARGET_KERNEL, duplicate, ext4_builtin=True
    )
    assert duplicate_report["status"] == "blocked"
    assert duplicate_report["required_counts"]["modules_dep"] == 2
    assert duplicate_report["required_counts"]["modules_alias"] == 2


@pytest.mark.parametrize("member", ["init", "modules.dep", "modules.alias"])
def test_initramfs_required_multiplicity_is_exact_one(member: str) -> None:
    listing = _valid_initramfs(r6.FALLBACK_KERNEL)
    root = f"usr/lib/modules/{r6.FALLBACK_KERNEL}"
    value = "init" if member == "init" else f"{root}/{member}"
    report = r6.validate_initramfs_listing(
        r6.FALLBACK_KERNEL, listing + value + "\n", ext4_builtin=True
    )
    assert report["status"] == "blocked"


def test_initramfs_rejects_wrong_kernel_and_suffix_confusion() -> None:
    listing = _valid_initramfs(r6.TARGET_KERNEL)
    root = f"usr/lib/modules/{r6.TARGET_KERNEL}"
    listing = listing.replace(
        f"{root}/kernel/drivers/nvme/host/nvme.ko.zst",
        f"{root}/kernel/drivers/nvme/host/nvme.ko.attacker",
    )
    listing += f"usr/lib/modules/{r6.FALLBACK_KERNEL}/modules.dep\n"
    report = r6.validate_initramfs_listing(
        r6.TARGET_KERNEL, listing, ext4_builtin=True
    )
    assert report["status"] == "blocked"
    assert report["checks"]["one_nvme"] is False
    assert report["checks"]["no_wrong_kernel_roots"] is False


def test_real_fallback_initramfs_strict_listing_passes() -> None:
    report = r6.collect_initramfs_strict(r6.FALLBACK_KERNEL)
    assert report["status"] == "pass"
    assert report["listing"]["checks"]["one_modules_dep"] is True


def test_literal_grub_dropin_grammar_passes() -> None:
    report = r6.inspect_grub_defaults_text(
        'GRUB_DEFAULT="Advanced options>target"\nGRUB_TIMEOUT=5\n',
        pinned_base=False,
    )
    assert report == {
        "status": "pass",
        "assignments": ["Advanced options>target"],
        "unsupported_lines": [],
    }


@pytest.mark.parametrize(
    "text",
    [
        'X=GRUB_\nY=DEFAULT\neval "$X$Y=attacker"\n',
        ". /tmp/attacker\nGRUB_DEFAULT=0\n",
        "GRUB_DEFAULT=$(printf attacker)\n",
        "GRUB_DEFAULT=${ATTACKER}\n",
    ],
)
def test_grub_strict_parser_rejects_eval_indirection(text: str) -> None:
    report = r6.inspect_grub_defaults_text(text, pinned_base=False)
    assert report["status"] == "blocked"
    assert report["unsupported_lines"]


def test_real_grub_base_is_accepted_only_by_exact_pinned_hash() -> None:
    report = r6.collect_grub_defaults_r6()
    assert report["status"] == "pass"
    assert report["base"]["sha256"] == r6.APPROVED_GRUB_BASE_SHA256


def test_grub_cfg_gate_is_fail_closed_when_held_file_is_unreadable() -> None:
    report = r6.collect_grub_state_r6()
    assert report["status"] == "blocked"
    assert "PermissionError" in report["error"]


def _cache_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_kind: str | None = None,
    mount_transition: bool = False,
) -> dict[str, object]:
    cache = tmp_path / "archives"
    cache.mkdir()
    cache.chmod(0o750)
    candidates: list[dict[str, object]] = []
    for index in range(22):
        name = f"pkg{index}"
        payload = f"payload-{name}".encode()
        deb_path = cache / f"{name}.deb"
        deb_path.write_bytes(payload)
        deb_path.chmod(0o640)
        candidates.append(_candidate(name, payload))
    if extra_kind == "directory":
        (cache / "evil-extra.deb").mkdir()
    elif extra_kind == "symlink":
        (cache / "evil-link.deb").symlink_to(tmp_path / "missing")
    monkeypatch.setattr(r6, "_trusted_cache_path", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        r6,
        "_trusted_ancestry",
        lambda *args, **kwargs: {
            "status": "pass",
            "entries": [
                {
                    "path": "/var/cache/deepsafe-driver595-r6/state/archives",
                    "mount_id": 1,
                    "trusted": True,
                }
            ],
        },
    )

    def mount_id(fd: int) -> int:
        info = os.fstat(fd)
        return 2 if mount_transition and stat.S_ISREG(info.st_mode) else 1

    monkeypatch.setattr(r6, "_fd_mount_id", mount_id)

    def fake_run(argv: list[str], **kwargs: object) -> r6.CommandResult:
        path = Path(os.readlink(argv[2]))
        package = path.name.removesuffix(".deb")
        output = (
            f"Package: {package}\nVersion: 1\nArchitecture: amd64\n".encode()
        )
        return r6.CommandResult(tuple(argv), 0, output)

    monkeypatch.setattr(r6, "_run", fake_run)
    return r6.inspect_cache_fd(
        candidates,
        cache,
        expected_path="/var/cache/deepsafe-driver595-r6/" + "a" * 20 + "/archives",
        install_argv=["/usr/bin/apt-get", "--no-download", "install"],
        _expected_uid=os.getuid(),
    )


def test_fd_held_cache_fixture_passes_exact_22_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _cache_report(tmp_path, monkeypatch)
    assert report["status"] == "pass", json.dumps(report, indent=2, sort_keys=True)
    assert report["checks"]["exact_22_debs"] is True
    assert len(report["binding"]["deb_manifest"]) == 22


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_cache_classifier_rejects_nonregular_dot_deb_entries(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _cache_report(tmp_path, monkeypatch, extra_kind=kind)
    assert report["status"] == "blocked"
    assert report["unexpected_entries"]
    name = report["unexpected_entries"][0]
    assert report["entries"][name]["kind"] == kind


def test_cache_classifier_rejects_file_mount_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _cache_report(tmp_path, monkeypatch, mount_transition=True)
    assert report["status"] == "blocked"
    assert any(
        item.get("checks", {}).get("same_mount_as_directory") is False
        for item in report["entries"].values()
    )


def _gpu_xml(types: list[str]) -> bytes:
    processes = "".join(
        f"<process_info><pid>{100 + index}</pid><type>{kind}</type>"
        f"<process_name>/proc/{kind}</process_name></process_info>"
        for index, kind in enumerate(types)
    )
    return (
        "<nvidia_smi_log><driver_version>590.48.01</driver_version>"
        '<gpu id="00000000:01:00.0"><uuid>GPU-fixture</uuid>'
        f"<processes>{processes}</processes></gpu></nvidia_smi_log>"
    ).encode()


def test_gpu_xml_includes_graphics_and_all_required_process_types() -> None:
    report = r6.parse_nvidia_smi_processes(_gpu_xml(["C", "G", "C+G"]))
    assert [item["type"] for item in report["processes"]] == ["C", "G", "C+G"]


def test_gpu_xml_rejects_unknown_process_type_and_entity() -> None:
    with pytest.raises(ValueError):
        r6.parse_nvidia_smi_processes(_gpu_xml(["M"]))
    with pytest.raises(ValueError):
        r6.parse_nvidia_smi_processes(b"<!ENTITY x 'y'><nvidia_smi_log />")


def test_mountinfo_parser_binds_root_mount_id_and_instance() -> None:
    records = r6.parse_mountinfo(
        "32 2 259:6 / / rw,relatime shared:1 - ext4 /dev/nvme1n1p1 rw\n"
    )
    assert records == [
        {
            "mount_id": 32,
            "parent_id": 2,
            "major_minor": "259:6",
            "root": "/",
            "mount_point": "/",
            "mount_options": "rw,relatime",
            "optional_fields": ["shared:1"],
            "fs_type": "ext4",
            "mount_source": "/dev/nvme1n1p1",
            "super_options": "rw",
        }
    ]


def test_mountinfo_duplicate_mount_id_is_rejected() -> None:
    line = "32 2 259:6 / / rw - ext4 /dev/root rw\n"
    with pytest.raises(ValueError):
        r6.parse_mountinfo(line + line)


def test_real_live_guard_observes_graphics_process_and_root_fd_identity() -> None:
    report = r6.collect_live_guard_snapshot()
    assert report["status"] == "pass"
    assert any(item["type"] == "G" for item in report["gpu_processes"])
    assert report["root_mount"]["mount_id"] == report["root_mount"]["fd_mnt_id"]
    assert r6.validate_snapshot_self_digest(report) is True


def test_live_snapshot_digest_detects_mount_identity_tamper() -> None:
    report = r6.collect_live_guard_snapshot()
    report["root_mount"]["mount_id"] += 1
    assert r6.validate_snapshot_self_digest(report) is False


def test_prepared_plan_has_complete_finding_closure_matrix(
    prepared_plan: dict[str, object],
) -> None:
    assert prepared_plan["finding_closure"] == r6.FINDING_CLOSURE
    assert {item["finding"] for item in prepared_plan["finding_closure"]} == {
        "R5-P1-001",
        "R5-P1-002",
        "R5-P1-003",
        "R5-P1-004",
        "R5-P1-005",
        "R5-P1-006",
        "R5-P2-001",
        "R5-P2-002",
    }


def test_prepared_plan_passes_strict_r6_and_embedded_r5_schemas(
    prepared_plan: dict[str, object],
) -> None:
    report = r6.validate_plan_schema(prepared_plan)
    assert report["status"] == "pass"


def test_prepared_candidate_static_integrity_passes_without_recollection(
    prepared_plan: dict[str, object],
) -> None:
    report = r6.verify_candidate_static(prepared_plan, recollect_apt=False)
    assert report["status"] == "pass"
    assert report["candidate_accepted"] is False


def test_static_plan_result_remains_non_authorizing(
    prepared_plan: dict[str, object],
) -> None:
    raw_sha = r6.sha256_bytes(r6.canonical_json(prepared_plan))
    report = r6.verify_plan_static(
        prepared_plan, raw_sha256=raw_sha, recollect_apt=False
    )
    assert report["status"] == "integrity_pass_non_authorizing"
    assert report["candidate_accepted"] is False
    assert report["install_authorized"] is False


def test_plan_schema_rejects_unknown_root_and_nested_future_fields(
    prepared_plan: dict[str, object],
) -> None:
    value = copy.deepcopy(prepared_plan)
    value["unexpected"] = True
    value["future_commands"]["ignored"] = False
    report = r6.validate_plan_schema(value)
    assert report["status"] == "blocked"
    assert report["r6_schema"]["error_count"] >= 2


def test_plan_fingerprint_tamper_is_rejected(prepared_plan: dict[str, object]) -> None:
    value = copy.deepcopy(prepared_plan)
    value["authority"]["install_authorized"] = True
    value["plan_fingerprint_sha256"] = r6.plan_fingerprint(value)
    report = r6.verify_candidate_static(value, recollect_apt=False)
    assert report["status"] == "blocked"
    assert report["checks"]["authority_non_operational_exact"] is False


def test_preparation_check_tamper_is_rejected_even_with_new_fingerprint(
    prepared_plan: dict[str, object],
) -> None:
    value = copy.deepcopy(prepared_plan)
    value["preparation_checks"]["apt_signed_provenance"] = False
    value["plan_fingerprint_sha256"] = r6.plan_fingerprint(value)
    report = r6.verify_candidate_static(value, recollect_apt=False)
    assert report["status"] == "blocked"
    assert report["checks"]["preparation_checks_recomputed_exact"] is False


def test_apt_report_tamper_is_rejected_even_with_new_plan_fingerprint(
    prepared_plan: dict[str, object],
) -> None:
    value = copy.deepcopy(prepared_plan)
    value["apt_provenance"]["indexes"][0]["records"].append(
        _record(_candidate("attacker"))
    )
    value["plan_fingerprint_sha256"] = r6.plan_fingerprint(value)
    report = r6.verify_candidate_static(value, recollect_apt=False)
    assert report["status"] == "blocked"
    assert report["checks"]["apt_report_structural_proof"] is False


def test_review_receipt_schema_rejects_unknown_field() -> None:
    report = r6._schema_report(
        {"unexpected": True}, r6.REVIEW_RECEIPT_SCHEMA
    )
    assert report["status"] == "blocked"
    assert any("Additional properties" in error for error in report["errors"])


def test_cache_receipt_binding_rejects_changed_install_argv_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(r6, "_schema_report", lambda *a, **k: {"status": "pass"})
    cache_report = {
        "status": "pass",
        "binding": {
            "cache_path": "/var/cache/deepsafe-driver595-r6/" + "a" * 20 + "/archives",
            "install_argv_sha256": "1" * 64,
        },
        "binding_sha256": "2" * 64,
    }
    receipt = {
        "decision": "PASS",
        "authority": copy.deepcopy(r6.CACHE_AUTHORITY),
        "plan_raw_sha256": "3" * 64,
        "plan_fingerprint_sha256": "4" * 64,
        "review_receipt_raw_sha256": "5" * 64,
        "cache_path": cache_report["binding"]["cache_path"],
        "install_argv_sha256": "f" * 64,
        "cache_binding_sha256": "2" * 64,
        "deb_count": 22,
    }
    receipt["receipt_fingerprint_sha256"] = r6.receipt_fingerprint(receipt)
    result = r6._verify_cache_receipt_core(
        receipt,
        raw_sha256="6" * 64,
        expected_root_sha256="6" * 64,
        plan_raw_sha256="3" * 64,
        plan_fingerprint_sha256="4" * 64,
        review_receipt_sha256="5" * 64,
        cache_report=cache_report,
    )
    assert result["verification_status"] == "blocked"
    assert result["checks"]["install_argv_bound"] is False
    assert result["install_authorized"] is False


def test_pre_reboot_receipt_binding_rejects_listing_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(r6, "_schema_report", lambda *a, **k: {"status": "pass"})
    initramfs: dict[str, object] = {}
    receipt_listing: list[dict[str, object]] = []
    for kernel in (r6.FALLBACK_KERNEL, r6.TARGET_KERNEL):
        initramfs[kernel] = {
            "file": {"file": {"pin": {"sha256": "1" * 64}}},
            "listing": {
                "entry_count": 10,
                "raw_listing_sha256": "2" * 64,
                "canonical_listing_sha256": "3" * 64,
            },
        }
        receipt_listing.append(
            {
                "kernel": kernel,
                "file_sha256": "1" * 64,
                "entry_count": 10,
                "raw_listing_sha256": "2" * 64,
                "canonical_listing_sha256": "3" * 64,
            }
        )
    evidence = {
        "status": "pass",
        "evidence_sha256": "4" * 64,
        "initramfs": initramfs,
        "grub": {
            "grub_cfg": {"sha256": "5" * 64},
            "grubenv": {"sha256": "6" * 64},
            "evidence_sha256": "7" * 64,
        },
        "crypto": {"evidence_sha256": "8" * 64},
    }
    receipt = {
        "decision": "PASS",
        "authority": copy.deepcopy(r6.PRE_REBOOT_AUTHORITY),
        "plan_raw_sha256": "9" * 64,
        "plan_fingerprint_sha256": "a" * 64,
        "review_receipt_raw_sha256": "b" * 64,
        "pre_evidence_sha256": "4" * 64,
        "initramfs": copy.deepcopy(receipt_listing),
        "grub_cfg_sha256": "5" * 64,
        "grubenv_sha256": "6" * 64,
        "grub_evidence_sha256": "7" * 64,
        "crypto_evidence_sha256": "8" * 64,
    }
    receipt["initramfs"][1]["raw_listing_sha256"] = "f" * 64
    receipt["receipt_fingerprint_sha256"] = r6.receipt_fingerprint(receipt)
    result = r6._verify_pre_reboot_receipt_core(
        receipt,
        raw_sha256="c" * 64,
        expected_root_sha256="c" * 64,
        plan_raw_sha256="9" * 64,
        plan_fingerprint_sha256="a" * 64,
        review_receipt_sha256="b" * 64,
        evidence=evidence,
    )
    assert result["verification_status"] == "blocked"
    assert result["checks"]["both_complete_initramfs_listings_bound"] is False
    assert result["reboot_authorized"] is False


def test_real_preconditions_remain_fail_closed_and_non_authorizing(
    prepared_plan: dict[str, object],
) -> None:
    raw_sha = r6.sha256_bytes(r6.canonical_json(prepared_plan))
    report = r6.verify_pre_reboot(prepared_plan, raw_sha256=raw_sha)
    assert report["status"] == "blocked"
    assert report["pre_reboot_evidence_accepted"] is False
    assert report["reboot_authorized"] is False
    assert report["gpu_workload_authorized"] is False


def test_finding_closure_matrix_is_exact_and_complete() -> None:
    assert len(r6.FINDING_CLOSURE) == 8
    assert len({item["fixture"] for item in r6.FINDING_CLOSURE}) == 8
    source = inspect.getsource(r6)
    for token in (
        "VALIDSIG",
        "split_module_signature",
        "sbverify",
        "raw_listing_sha256",
        "grubenv",
        "inspect_cache_fd",
        "C+G",
        "mount_id",
    ):
        assert token in source
