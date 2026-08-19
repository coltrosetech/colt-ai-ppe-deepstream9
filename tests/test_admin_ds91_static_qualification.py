from __future__ import annotations

import json
from pathlib import Path

import pytest

from admin.ds91_status import load_deepstream91_static_status
from validation import deepstream91_qualification as ds91


ROOT = Path(__file__).resolve().parents[1]


def _host_files(tmp_path: Path, driver: str) -> tuple[Path, Path]:
    driver_file = tmp_path / "nvidia-version"
    driver_file.write_text(
        f"NVRM version: NVIDIA UNIX x86_64 Kernel Module  {driver}  Fri Jun 1\n",
        encoding="utf-8",
    )
    os_release = tmp_path / "os-release"
    os_release.write_text('ID=ubuntu\nVERSION_ID="24.04"\n', encoding="utf-8")
    return driver_file, os_release


def test_admin_projection_is_driver_blocked_and_actionless(tmp_path: Path) -> None:
    driver_file, os_release = _host_files(tmp_path, "590.48.01")
    card = load_deepstream91_static_status(
        driver_file=driver_file,
        os_release_path=os_release,
        machine="x86_64",
        generated_at_utc="2026-07-18T12:00:00Z",
    )
    assert card["state"] == "blocked_host_driver_upgrade_required"
    assert card["static_prerequisites_met"] is False
    assert card["live_probe_performed"] is False
    assert card["production_ready"] is False
    assert set(card["actions_performed"].values()) == {False}
    assert card["runtime"]["deepstream"] == "9.1"
    assert card["runtime"]["cuda"] == "13.2"
    assert card["runtime"]["tensorrt"] == "10.16.0.72"


def test_admin_projection_static_pass_never_claims_live_readiness(tmp_path: Path) -> None:
    driver_file, os_release = _host_files(tmp_path, "595.58.03")
    card = load_deepstream91_static_status(
        driver_file=driver_file,
        os_release_path=os_release,
        machine="amd64",
        generated_at_utc="2026-07-18T12:00:00Z",
    )
    assert card["state"] == "static_prerequisites_met_live_probe_not_run"
    assert card["static_prerequisites_met"] is True
    assert card["live_probe_performed"] is False
    assert card["production_ready"] is False


def test_admin_projection_fails_closed_when_host_evidence_is_missing(
    tmp_path: Path,
) -> None:
    card = load_deepstream91_static_status(
        driver_file=tmp_path / "missing-driver",
        os_release_path=tmp_path / "missing-os-release",
        generated_at_utc="2026-07-18T12:00:00Z",
    )
    assert card["state"] == "unavailable_integrity_error"
    assert card["available"] is False
    assert card["production_ready"] is False
    assert card["execution_actions_available"] is False


def test_admin_image_and_ui_include_static_projection_without_action() -> None:
    dockerfile = (ROOT / "admin/Dockerfile").read_text(encoding="utf-8")
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert (
        "COPY validation/deepstream91_qualification.py "
        "/app/validation/deepstream91_qualification.py"
    ) in dockerfile
    assert "deepstream91_static_qualification" in page
    assert "campaign.live_probe_performed" in page
    assert "fetch('/api/validation')" in page
    assert "fetch('/api/deepstream91/run')" not in page


def test_admin_projection_can_use_read_only_host_os_binding(
    tmp_path: Path, monkeypatch,
) -> None:
    driver_file, os_release = _host_files(tmp_path, "590.48.01")
    monkeypatch.setenv("DEEPSAFE_DS91_HOST_DRIVER_FILE", str(driver_file))
    monkeypatch.setenv("DEEPSAFE_DS91_HOST_OS_RELEASE", str(os_release))
    card = load_deepstream91_static_status(
        machine="amd64",
        generated_at_utc="2026-07-18T12:00:00Z",
    )
    assert card["state"] == "blocked_host_driver_upgrade_required"
    assert card["host"]["os_id"] == "ubuntu"
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "DEEPSAFE_DS91_HOST_OS_RELEASE: /host/etc/os-release" in compose
    assert "/etc/os-release:/host/etc/os-release:ro" in compose


def test_admin_projection_fails_closed_for_wrong_contract_container_type(
    tmp_path: Path,
) -> None:
    malformed = json.loads(ds91.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    malformed["artifacts"] = [
        "implementation",
        "report_schema",
        "strict_json",
        "validation_package",
    ]
    unsigned = {
        key: value
        for key, value in malformed.items()
        if key != "contract_fingerprint_sha256"
    }
    malformed["contract_fingerprint_sha256"] = ds91.canonical_sha256(unsigned)
    contract_path = tmp_path / "malformed-contract.json"
    contract_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(AttributeError):
        ds91.load_contract(contract_path)
    card = load_deepstream91_static_status(contract_path=contract_path)
    assert card["state"] == "unavailable_integrity_error"
    assert card["available"] is False
    assert card["production_ready"] is False
