from __future__ import annotations

import copy
import hashlib
import inspect
import subprocess
from pathlib import Path

import pytest

from validation import driver595_maintenance_r5 as r5
from validation import driver595_maintenance_r5_independent_review as review


def _load_review() -> dict[str, object]:
    return review.load_json(review.DEFAULT_REVIEW)


def _resign(value: dict[str, object]) -> dict[str, object]:
    value["review_fingerprint_sha256"] = review.review_fingerprint(value)
    return value


def test_real_r5_reject_record_verifies_but_never_accepts_or_authorizes() -> None:
    result = review.verify(_load_review())
    assert result["verification_status"] == "pass"
    assert result["review_decision"] == "REJECT"
    for field in (
        "subject_accepted",
        "prepared_plan_accepted",
        "cache_accepted",
        "download_authorized",
        "install_authorized",
        "reboot_authorized",
        "gpu_workload_authorized",
        "terminal_root_published",
    ):
        assert result[field] is False


def test_recomputed_fingerprint_cannot_turn_reject_into_accept() -> None:
    value = copy.deepcopy(_load_review())
    value["decision"] = "ACCEPT"
    value["authority"]["subject_accepted"] = True
    result = review.verify(_resign(value))
    assert result["verification_status"] == "blocked"
    assert any("REJECT" in failure for failure in result["failures"])


def test_missing_p1_or_wrong_severity_count_is_rejected() -> None:
    value = copy.deepcopy(_load_review())
    value["findings"].pop()
    value["severity_counts"]["P2"] = 1
    result = review.verify(_resign(value))
    assert result["verification_status"] == "blocked"
    assert any("finding ID/severity" in failure for failure in result["failures"])


def test_subject_pin_tamper_is_rejected_even_when_resigned() -> None:
    value = copy.deepcopy(_load_review())
    value["subject_pins"][0]["sha256"] = "0" * 64
    result = review.verify(_resign(value))
    assert result["verification_status"] == "blocked"
    assert any("subject pins" in failure for failure in result["failures"])


def test_schema_rejects_unknown_nested_or_root_review_fields() -> None:
    value = copy.deepcopy(_load_review())
    value["unexpected"] = True
    value["authority"]["ignored"] = False
    result = review.verify(_resign(value))
    assert result["verification_status"] == "blocked"
    assert result["schema_error_count"] >= 2


def test_failed_criterion_cannot_be_marked_pass_after_resign() -> None:
    value = copy.deepcopy(_load_review())
    value["criterion_assessment"][1]["result"] = "PASS"
    result = review.verify(_resign(value))
    assert result["verification_status"] == "blocked"
    assert any("criterion" in failure for failure in result["failures"])


def test_collected_node_id_pin_is_live_and_exact() -> None:
    value = _load_review()
    declared = value["test_replay"]["collected_node_ids_pin"]
    observed = review.pin(review.ROOT / review.NODE_IDS_PATH)
    assert declared == observed
    assert sum(1 for _ in (review.ROOT / review.NODE_IDS_PATH).open()) == 101


def test_integrity_strings_can_pass_without_independent_acceptance_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = {
        "status": "pass",
        "install_authorized": False,
        "reboot_authorized": False,
    }
    monkeypatch.setattr(r5, "verify_candidate_static", lambda plan: candidate)
    plan = {"plan_fingerprint_sha256": "b" * 64}
    result = r5.verify_plan_static(
        plan,
        raw_sha256="a" * 64,
        expected_raw_sha256="a" * 64,
        expected_fingerprint="b" * 64,
    )
    assert result["status"] == "pass"
    assert result["external_independent_acceptance_receipt_present"] is False
    assert result["candidate_accepted"] is False
    assert result["install_authorized"] is False


def test_apt_authority_has_no_inrelease_release_packages_signature_chain() -> None:
    source = inspect.getsource(r5.collect_apt_authority)
    whole = inspect.getsource(r5)
    assert "gpgv" not in r5.BIN
    assert "indextargets" not in whole
    assert "VALIDSIG" not in whole
    assert "InRelease" not in source
    report = r5.collect_apt_authority()
    assert report["status"] == "pass"
    assert set(report) == {
        "status",
        "checks",
        "contract",
        "source_policy",
        "source_file",
        "keyring",
        "keyring_package",
    }


def test_module_authority_has_no_cryptographic_pkcs7_or_pe_trust_verifier() -> None:
    whole = inspect.getsource(r5)
    assert "openssl" not in r5.BIN
    assert "sbverify" not in r5.BIN
    assert "--list-enrolled" not in whole
    assert "mokutil" in r5.BIN
    for kernel in (r5.FALLBACK_KERNEL, r5.TARGET_KERNEL):
        assert r5.collect_module_trust_reference(kernel)["status"] == "pass"


def _initramfs_fixture_report(
    monkeypatch: pytest.MonkeyPatch,
    lines: list[str],
) -> dict[str, object]:
    monkeypatch.setattr(r5, "_boot_file", lambda path: {"status": "pass"})
    monkeypatch.setattr(
        r5,
        "verify_kernel_boot_config",
        lambda kernel: {
            "status": "pass",
            "selected_values": {"CONFIG_EXT4_FS": "y"},
        },
    )
    output = "\n".join(lines)
    monkeypatch.setattr(
        r5,
        "_run",
        lambda argv, **kwargs: r5.CommandResult(tuple(argv), 0, output),
    )
    return r5.verify_initramfs(r5.TARGET_KERNEL)


def _valid_initramfs_lines() -> list[str]:
    root = f"usr/lib/modules/{r5.TARGET_KERNEL}"
    return [
        "init",
        root,
        f"{root}/modules.dep",
        f"{root}/modules.alias",
        f"{root}/kernel/drivers/nvme/host/nvme.ko.zst",
        f"{root}/kernel/drivers/nvme/host/nvme-core.ko.zst",
    ]


def test_absolute_initramfs_member_spellings_are_silently_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = ["/" + line for line in _valid_initramfs_lines()]
    report = _initramfs_fixture_report(monkeypatch, lines)
    assert report["status"] == "pass"
    assert report["checks"]["all_members_normalized"] is True


def test_duplicate_required_initramfs_metadata_is_collapsed_and_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _valid_initramfs_lines()
    root = f"usr/lib/modules/{r5.TARGET_KERNEL}"
    lines.extend([f"{root}/modules.dep", f"{root}/modules.alias"])
    report = _initramfs_fixture_report(monkeypatch, lines)
    assert report["status"] == "pass"
    assert report["checks"]["exact_kernel_modules_dep"] is True
    assert report["checks"]["exact_kernel_modules_alias"] is True
    assert report["duplicate_selected_modules"] == []


def test_base_grub_indirect_eval_changes_default_while_parser_passes() -> None:
    text = (
        "GRUB_DEFAULT=0\n"
        "X=GRUB_\n"
        "Y=DEFAULT\n"
        "eval \"$X$Y=attacker\"\n"
    )
    parsed = r5._inspect_grub_defaults_text(text, strict_dropin=False)
    shell = subprocess.run(
        ["/bin/sh", "-c", text + "printf '%s' \"$GRUB_DEFAULT\""],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        timeout=5,
    )
    assert parsed == {"status": "pass", "assignments": ["0"], "unsupported_lines": []}
    assert shell.returncode == 0
    assert shell.stdout == "attacker"


def test_extra_directory_and_symlink_named_deb_evade_cache_entry_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "archives"
    cache.mkdir()
    payload = b"reviewed bytes"
    good = cache / "good.deb"
    good.write_bytes(payload)
    (cache / "evil-extra.deb").mkdir()
    (cache / "evil-link.deb").symlink_to(tmp_path / "missing")
    candidate = {
        "package": "good",
        "version": "1",
        "architecture": "amd64",
        "filename": "pool/good.deb",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    plan = {
        "future_commands": {"cache_dir": str(cache)},
        "initial_state": {"snapshot": {"candidates": [candidate]}},
        "plan_fingerprint_sha256": "a" * 64,
    }
    monkeypatch.setattr(r5, "validate_plan_schema", lambda plan: {"status": "pass"})
    monkeypatch.setattr(r5, "verify_plan_static", lambda *args, **kwargs: {"status": "pass"})
    monkeypatch.setattr(r5, "verify_plan_current", lambda *args, **kwargs: {"status": "pass"})
    monkeypatch.setattr(
        r5,
        "_trusted_cache_permissions",
        lambda *args, **kwargs: {"status": "pass", "checks": {}},
    )
    monkeypatch.setattr(
        r5,
        "_run",
        lambda argv, **kwargs: r5.CommandResult(
            tuple(argv), 0, "Package: good\nVersion: 1\nArchitecture: amd64\n"
        ),
    )
    result = r5.verify_cache(plan, cache)
    assert result["actual_debs"] == ["good.deb"]
    assert result["unexpected_non_debs"] == []
    assert "evil-extra.deb" not in result["details"]
    assert "evil-link.deb" not in result["details"]


def test_live_snapshot_omits_graphics_processes_and_mount_instance_identity() -> None:
    source = inspect.getsource(r5.collect_live_baseline)
    assert "--query-compute-apps=pid,process_name" in source
    assert "--query-accounted-apps" not in source
    assert '"-q"' not in source
    assert "/proc/self/mountinfo" not in source
    assert "mount_id" not in source


def test_frozen_suite_count_is_101_but_new_bypass_classes_are_external() -> None:
    record = _load_review()
    assert record["test_replay"]["r5_collected"] == 101
    assert record["test_replay"]["r5_passed"] == 101
    finding = next(item for item in record["findings"] if item["id"] == "R5-P2-002")
    assert finding["severity"] == "P2"
    assert finding["blocks_acceptance"] is False
