from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import jsonschema
import pytest

from validation import driver595_maintenance_r4 as r4
from validation import driver595_maintenance_r4_independent_review as review


def _load_review() -> dict[str, object]:
    return review.load_json(review.DEFAULT_REVIEW)


def _resign(value: dict[str, object]) -> dict[str, object]:
    value["review_fingerprint_sha256"] = review.review_fingerprint(value)
    return value


def _apt_fixture(origin: str) -> str:
    lines = ["NOTE: This is only a simulation!"]
    for package, (version, _) in sorted(r4.EXPECTED_REMOVE_VERSIONS.items()):
        lines.append(f"Remv {package} [{version}]")
    for package, (version, architecture) in sorted(
        r4.EXPECTED_INSTALL_VERSIONS.items()
    ):
        lines.append(f"Inst {package} ({version} {origin} [{architecture}])")
    for package, (version, architecture) in sorted(
        r4.EXPECTED_CONFIGURE_VERSIONS.items()
    ):
        lines.append(f"Conf {package} ({version} {origin} [{architecture}])")
    lines.append(
        f"0 upgraded, {len(r4.EXPECTED_INSTALL_VERSIONS)} newly installed, "
        f"{len(r4.EXPECTED_REMOVE_VERSIONS)} to remove and 147 not upgraded."
    )
    return "\n".join(lines)


def _r4_static_plan(tmp_path: Path) -> dict[str, object]:
    pins: dict[str, object] = {}
    for name in ("validator", "schema", "documentation"):
        path = tmp_path / name
        path.write_bytes(name.encode("utf-8"))
        pins[name] = r4.file_pin(path)
    state_basis = {"fixture": "schema-bypass"}
    state_id = r4.sha256_bytes(r4.canonical_json(state_basis))[:20]
    cache_dir = f"/var/cache/deepsafe-driver595-r4/{state_id}/archives"
    plan: dict[str, object] = {
        "schema_version": r4.SCHEMA_VERSION,
        "plan_id": f"driver595-r4-{state_id}",
        "status": "prepared_read_only_pending_independent_review_cache_and_authorization",
        "authority": {
            "this_plan_executes_work": False,
            "downloads_performed": False,
            "install_enabled": False,
            "external_authorization_required": True,
            "external_authorization_receipt": None,
            "independent_review_required": True,
        },
        "target": {
            "driver_version": r4.TARGET_DRIVER,
            "driver_deb_version": r4.TARGET_DRIVER_DEB,
            "target_kernel": r4.TARGET_KERNEL,
            "target_kernel_deb_version": r4.TARGET_KERNEL_DEB,
            "fallback_kernel": r4.FALLBACK_KERNEL,
            "fallback_kernel_deb_version": r4.FALLBACK_KERNEL_DEB,
            "gpu_name": r4.TARGET_GPU,
            "architecture": r4.ARCH,
        },
        "source_pins": pins,
        "toolchain": r4.collect_toolchain(),
        "transaction_contract": {
            "install": [
                {"package": p, "version": v, "architecture": a}
                for p, (v, a) in sorted(r4.EXPECTED_INSTALL_VERSIONS.items())
            ],
            "remove": [
                {"package": p, "version": v, "architecture": a}
                for p, (v, a) in sorted(r4.EXPECTED_REMOVE_VERSIONS.items())
            ],
            "configure": [
                {"package": p, "version": v, "architecture": a}
                for p, (v, a) in sorted(r4.EXPECTED_CONFIGURE_VERSIONS.items())
            ],
            "forbidden_installed_packages": ["nvidia-dkms-595"],
            "autoremove_allowed": False,
            "package_download_allowed_before_external_authorization": False,
        },
        "future_commands": {
            "simulation_argv": r4.apt_simulation_argv(),
            "cache_dir": cache_dir,
            "cache_download_argv_after_authorization_only": r4.future_cache_download_argv(
                cache_dir
            ),
            "install_argv_after_authorization_only": r4.future_install_argv(cache_dir),
        },
        "initial_state": {"state_basis": state_basis},
    }
    plan["plan_fingerprint_sha256"] = r4.plan_fingerprint(plan)
    return plan


def test_real_reject_record_verifies_but_never_accepts_subject() -> None:
    result = review.verify(_load_review())
    assert result["verification_status"] == "pass"
    assert result["subject_decision"] == "REJECT"
    assert result["subject_accepted"] is False
    assert result["install_authorized"] is False
    assert result["reboot_authorized"] is False


def test_recomputed_fingerprint_cannot_turn_reject_into_accept() -> None:
    value = copy.deepcopy(_load_review())
    value["decision"] = "ACCEPT"
    value["authority"]["subject_accepted"] = True  # type: ignore[index]
    result = review.verify(_resign(value))
    assert result["verification_status"] == "blocked"
    assert any("REJECT" in item for item in result["failures"])


def test_missing_blocker_or_wrong_count_is_rejected() -> None:
    value = copy.deepcopy(_load_review())
    value["findings"].pop()  # type: ignore[union-attr]
    value["severity_counts"]["P2"] = 2  # type: ignore[index]
    result = review.verify(_resign(value))
    assert result["verification_status"] == "blocked"
    assert any("finding ID/severity" in item for item in result["failures"])


def test_subject_pin_tamper_is_rejected_even_when_resigned() -> None:
    value = copy.deepcopy(_load_review())
    value["subject_pins"][0]["sha256"] = "0" * 64  # type: ignore[index]
    result = review.verify(_resign(value))
    assert result["verification_status"] == "blocked"
    assert any("subject pins changed" in item for item in result["failures"])


def test_schema_is_strict_about_unknown_review_fields() -> None:
    value = copy.deepcopy(_load_review())
    value["unexpected"] = True
    result = review.verify(_resign(value))
    assert result["verification_status"] == "blocked"
    assert result["schema_error_count"] >= 1


def test_r4_static_verifier_accepts_a_schema_invalid_plan(tmp_path: Path) -> None:
    plan = _r4_static_plan(tmp_path)
    assert r4.verify_plan_static(plan)["status"] == "pass"
    schema = json.loads(r4.PLAN_SCHEMA.read_text(encoding="utf-8"))
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(plan))
    assert errors
    assert any(error.validator == "required" for error in errors)


def test_r4_origin_prefix_accepts_spoofed_suffix() -> None:
    origin = "Ubuntu:24.04/noble-evil-attacker"
    result = r4.parse_apt_actions(_apt_fixture(origin))
    assert result["status"] == "pass"
    assert result["checks"]["trusted_nonempty_origins"] is True


def test_r4_module_contract_omits_vermagic_and_trusted_key_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = inspect.getsource(r4.verify_module)
    assert '"vermagic"' not in source
    assert "trusted_sig_key" not in source

    def fake_module(kernel: str, module: str) -> dict[str, object]:
        return {
            "status": "pass",
            "fields": {"signer": "attacker-controlled", "sig_key": "DE:AD:BE:EF"},
        }

    monkeypatch.setattr(r4, "verify_module", fake_module)
    assert r4.verify_modules_for_kernel(r4.TARGET_KERNEL)["status"] == "pass"


def test_r4_initramfs_prefix_confusion_accepts_nonmodule_suffixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = r4.TARGET_KERNEL
    root = f"usr/lib/modules/{kernel}"
    output = "\n".join(
        [
            "init",
            f"{root}/modules.dep",
            f"{root}/modules.alias",
            f"{root}/kernel/drivers/nvme/host/nvme.ko.attacker",
            f"{root}/kernel/drivers/nvme/host/nvme-core.ko.not-a-module",
        ]
    )
    monkeypatch.setattr(r4, "_boot_file", lambda path: {"status": "pass"})
    monkeypatch.setattr(
        r4,
        "verify_kernel_boot_config",
        lambda kernel: {
            "status": "pass",
            "selected_values": {"CONFIG_EXT4_FS": "y"},
        },
    )
    monkeypatch.setattr(
        r4,
        "_run",
        lambda argv, **kwargs: r4.CommandResult(tuple(argv), 0, output),
    )
    assert r4.verify_initramfs(kernel)["status"] == "pass"


def test_r4_prepare_plan_does_not_measure_live_host_driver_baseline() -> None:
    source = inspect.getsource(r4.prepare_plan)
    for missing_probe in ("nvidia_smi", "mokutil", "uname", "dkms", "findmnt"):
        assert f'BIN["{missing_probe}"]' not in source


def test_r4_grub_verifier_ignores_effective_default_dropins() -> None:
    source = inspect.getsource(r4.verify_grub)
    assert 'Path("/etc/default/grub")' in source
    assert "/etc/default/grub.d" not in source


def test_r4_cache_verifier_does_not_bind_argument_to_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "wrong-state" / "archives"
    cache.mkdir(parents=True)
    payload = b"exact package bytes"
    deb = cache / "pkg_1_amd64.deb"
    deb.write_bytes(payload)
    plan = {
        "future_commands": {
            "cache_dir": "/var/cache/deepsafe-driver595-r4/"
            + "a" * 20
            + "/archives"
        },
        "initial_state": {
            "apt": {
                "candidates": [
                    {
                        "package": "pkg",
                        "version": "1",
                        "architecture": "amd64",
                        "filename": "pool/x/pkg_1_amd64.deb",
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            }
        },
    }
    monkeypatch.setattr(r4, "verify_plan_static", lambda plan: {"status": "pass"})
    monkeypatch.setattr(
        r4,
        "_trusted_cache_permissions",
        lambda cache_dir, paths: {"status": "pass", "checks": {}},
    )
    monkeypatch.setattr(
        r4,
        "_run",
        lambda argv, **kwargs: r4.CommandResult(
            tuple(argv), 0, "Package: pkg\nVersion: 1\nArchitecture: amd64\n"
        ),
    )
    assert str(cache) != plan["future_commands"]["cache_dir"]  # type: ignore[index]
    assert r4.verify_cache(plan, cache)["status"] == "pass"


def test_r4_cache_permission_gate_omits_hardlink_count() -> None:
    source = inspect.getsource(r4._trusted_cache_permissions)
    assert "st_nlink" not in source


def test_r4_command_guard_is_not_exact_for_every_allowlisted_tool() -> None:
    r4._validate_read_only_argv([r4.BIN["modinfo"], "--help"])
    r4._validate_read_only_argv(
        [r4.BIN["dpkg_query"], "-W", "--admindir=/attacker-selected"]
    )


def test_exact_subject_hashes_remain_frozen() -> None:
    for relative, expected in review.EXPECTED_SUBJECT_PINS.items():
        observed = review.pin(review.ROOT / relative)
        assert (observed["bytes"], observed["mode"], observed["sha256"]) == expected
