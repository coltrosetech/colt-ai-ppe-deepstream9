from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from validation import deepstream91_qualification as ds91


ROOT = Path(__file__).resolve().parents[1]


def contract() -> dict:
    return ds91.load_contract()


def test_locked_contract_and_exact_target_pins() -> None:
    value = contract()
    assert value["target"] == {
        "deepstream": "9.1",
        "cuda": "13.2",
        "tensorrt": "10.16.0.72",
        "image_ref": "nvcr.io/nvidia/deepstream:9.1-triton-multiarch",
        "image_index_digest": "sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b",
        "linux_amd64_manifest_digest": "sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994",
    }
    assert value["host_matrix"]["minimum_driver"] == "595.58.03"
    assert value["host_matrix"]["os_version"] == "24.04"
    assert set(value["artifacts"]) == {
        "implementation", "report_schema", "strict_json", "validation_package"
    }
    assert value["execution_policy"]["gpu_invocation"] is False


def test_current_host_shape_is_driver_blocked_without_live_actions(tmp_path: Path) -> None:
    driver = tmp_path / "version"
    driver.write_text(
        "NVRM version: NVIDIA UNIX x86_64 Kernel Module  590.48.01  Fri Jun 1\n",
        encoding="utf-8",
    )
    os_release = tmp_path / "os-release"
    os_release.write_text('ID=ubuntu\nVERSION_ID="24.04"\n', encoding="utf-8")
    report = ds91.build_report(
        contract(),
        driver_version=ds91.read_driver_version(driver),
        os_id=ds91.read_os_release(os_release)["ID"],
        os_version=ds91.read_os_release(os_release)["VERSION_ID"],
        machine="x86_64",
        generated_at_utc="2026-07-18T12:00:00Z",
    )
    assert report["status"] == "blocked_host_driver_upgrade_required"
    assert report["blockers"] == ["host_driver_upgrade_and_reboot_required"]
    assert set(report["actions_performed"].values()) == {False}
    assert report["live_qualification"] == {
        "authorized": False,
        "performed": False,
        "production_ready": False,
    }
    schema = json.loads(
        (ROOT / "validation/schemas/deepstream91-static-qualification-v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(report)
    unsigned = {key: value for key, value in report.items() if key != "report_fingerprint_sha256"}
    assert report["report_fingerprint_sha256"] == ds91.canonical_sha256(unsigned)


def test_supported_static_shape_still_is_not_live_approval() -> None:
    report = ds91.build_report(
        contract(),
        driver_version="595.58.03",
        os_id="ubuntu",
        os_version="24.04",
        machine="amd64",
        generated_at_utc="2026-07-18T12:00:00+00:00",
    )
    assert report["status"] == "static_prerequisites_met_live_probe_not_run"
    assert report["blockers"] == []
    assert report["live_qualification"]["production_ready"] is False


@pytest.mark.parametrize(
    ("driver", "minimum", "expected"),
    [
        ("595.58.03", "595.58.03", True),
        ("595.60.00", "595.58.03", True),
        ("590.48.01", "595.58.03", False),
        ("600.1.0", "595.58.03", True),
    ],
)
def test_numeric_driver_comparison(driver: str, minimum: str, expected: bool) -> None:
    assert ds91._version_at_least(driver, minimum) is expected


@pytest.mark.parametrize("os_version", ["24.040", "24.04-unsupported", "24.10"])
def test_lookalike_or_different_os_version_is_platform_blocked(
    os_version: str,
) -> None:
    report = ds91.build_report(
        contract(),
        driver_version="595.58.03",
        os_id="ubuntu",
        os_version=os_version,
        machine="amd64",
        generated_at_utc="2026-07-18T12:00:00Z",
    )
    assert report["status"] == "blocked_host_platform_mismatch"
    assert "host_operating_system_mismatch" in report["blockers"]


def test_contract_rejects_source_pin_or_fingerprint_tamper(tmp_path: Path) -> None:
    original = json.loads(ds91.DEFAULT_CONTRACT.read_text())
    original["artifacts"]["implementation"]["sha256"] = "f" * 64
    unsigned = {
        key: value
        for key, value in original.items()
        if key != "contract_fingerprint_sha256"
    }
    original["contract_fingerprint_sha256"] = ds91.canonical_sha256(unsigned)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ds91.QualificationError, match="implementation pin"):
        ds91.load_contract(path)


def test_source_contains_no_process_or_network_launcher() -> None:
    raw = (ROOT / "validation/deepstream91_qualification.py").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == contract()["artifacts"]["implementation"]["sha256"]
    text = raw.decode("utf-8")
    forbidden = ("subprocess", "os.system", "docker.from_env", "requests.", "urllib.request")
    assert all(token not in text for token in forbidden)


@pytest.mark.parametrize(
    "field",
    ["image_index_digest", "linux_amd64_manifest_digest"],
)
def test_report_schema_const_pins_both_oci_digests(field: str) -> None:
    report = ds91.build_report(
        contract(),
        driver_version="595.58.03",
        os_id="ubuntu",
        os_version="24.04",
        machine="amd64",
        generated_at_utc="2026-07-18T12:00:00Z",
    )
    schema = json.loads(
        (ROOT / "validation/schemas/deepstream91-static-qualification-v1.schema.json").read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(report)
    report["target"][field] = "sha256:" + "0" * 64
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(report)


def test_schema_and_semantic_replay_reject_forged_success_status() -> None:
    value = contract()
    report = ds91.build_report(
        value,
        driver_version="590.48.01",
        os_id="ubuntu",
        os_version="24.04",
        machine="amd64",
        generated_at_utc="2026-07-18T12:00:00Z",
    )
    report["status"] = "static_prerequisites_met_live_probe_not_run"
    unsigned = {
        key: item for key, item in report.items()
        if key != "report_fingerprint_sha256"
    }
    report["report_fingerprint_sha256"] = ds91.canonical_sha256(unsigned)
    schema = json.loads(ds91.DEFAULT_REPORT_SCHEMA.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(report)
    with pytest.raises(ds91.QualificationError):
        ds91.validate_report(report, value)


def test_schema_rejects_platform_blocker_that_contradicts_checks() -> None:
    report = ds91.build_report(
        contract(),
        driver_version="595.58.03",
        os_id="ubuntu",
        os_version="24.04",
        machine="amd64",
        generated_at_utc="2026-07-18T12:00:00Z",
    )
    report["status"] = "blocked_host_platform_mismatch"
    report["blockers"] = ["host_architecture_mismatch"]
    unsigned = {
        key: item for key, item in report.items()
        if key != "report_fingerprint_sha256"
    }
    report["report_fingerprint_sha256"] = ds91.canonical_sha256(unsigned)
    schema = json.loads(ds91.DEFAULT_REPORT_SCHEMA.read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(report)


@pytest.mark.parametrize(
    "field",
    ["contract_fingerprint_sha256", "report_fingerprint_sha256"],
)
def test_semantic_replay_rejects_forged_fingerprints(field: str) -> None:
    value = contract()
    report = ds91.build_report(
        value,
        driver_version="595.58.03",
        os_id="ubuntu",
        os_version="24.04",
        machine="amd64",
        generated_at_utc="2026-07-18T12:00:00Z",
    )
    report[field] = "0" * 64
    with pytest.raises(ds91.QualificationError):
        ds91.validate_report(report, value)


def test_driver_runbook_requires_complete_boot_and_mok_evidence() -> None:
    runbook = (ROOT / "docs/deepstream91-driver-maintenance.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "test -s /boot/initrd.img-7.0.0-28-generic",
        "readlink -f /boot/initrd.img",
        "audit=$(sudo dpkg --audit)",
        "test -z \"$audit\"",
        "mokutil --test-key /var/lib/shim-signed/mok/MOK.der",
        "openssl x509 -inform DER",
        "modinfo -k 7.0.0-28-generic -F signer nvidia",
        "modinfo -k 7.0.0-28-generic -F sig_key nvidia",
        "test \"$cert_serial\" = \"$module_sig_key\"",
        "test \"$(modinfo -k 7.0.0-28-generic -F version nvidia)\" = \"595.71.05\"",
        "dkms status | grep -Fx 'nvidia/595.71.05, 7.0.0-28-generic, x86_64: installed'",
    ):
        assert required in runbook
