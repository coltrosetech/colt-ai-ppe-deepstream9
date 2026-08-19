from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from admin import validation as admin_validation


ROOT = Path(__file__).resolve().parents[1]
CURRENT_TIME = datetime(2026, 7, 18, 2, 0, 0, tzinfo=timezone.utc)
EXPIRY_TIME = datetime(2026, 7, 19, 1, 21, 47, tzinfo=timezone.utc)


def _pin(path: str, raw: bytes) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_json(path: Path, value: dict) -> bytes:
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _copy_pin(workspace: Path, pin: dict) -> None:
    source = ROOT / pin["path"]
    destination = workspace / pin["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


@pytest.fixture
def ds9_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    for pin in admin_validation.DS9_RUNTIME_QUALIFICATION_ADMIN_PINS.values():
        _copy_pin(workspace, pin)
    return workspace


def _reader(workspace: Path) -> admin_validation.ArtifactReader:
    return admin_validation.ArtifactReader(
        workspace / "missing-results",
        workspace_root=workspace,
        schema_root=ROOT / "validation/schemas",
    )


def _project(workspace: Path, *, now: datetime = CURRENT_TIME) -> dict:
    return admin_validation._ds9_runtime_qualification(
        _reader(workspace), now=now
    )


def test_checked_in_ds9_qualification_replays_all_five_checks() -> None:
    projected = _project(ROOT)

    assert projected["available"] is True
    assert projected["current"] is True
    assert projected["state"] == "runtime_qualified_current"
    assert projected["runtime_qualification_passed"] is True
    assert projected["checks"]["passed"] == 5
    assert projected["checks"]["total"] == 5
    assert all(projected["integrity"].values())
    assert projected["runtime"] == {
        "deepstream": "9.0.0",
        "cuda": "13.1",
        "tensorrt": "10.14.1.48",
        "architecture": "sm_86",
    }
    assert projected["does_not_imply_product_readiness"] is True
    assert projected["product_production_ready"] is False
    assert projected["training_complete"] is False
    assert projected["final_test_complete"] is False


def test_separate_ds9_projection_does_not_revive_drifted_r2_chain() -> None:
    reader = _reader(ROOT)
    receipt_only = admin_validation._gpu_reentry_r2(reader)
    assert receipt_only is not None
    assert receipt_only["available"] is False
    assert receipt_only["reason"] == "current_receipt_schema_pin_mismatch"
    assert receipt_only["progress"]["completed"] == 0
    assert receipt_only["current_run"] == {}
    assert receipt_only["pending_gate_ids"] == [
        "deepstream9_engine_smoke",
        "seven_day_endurance",
    ]

    projected = admin_validation._gpu_reentry(reader, now=CURRENT_TIME)

    assert projected["progress"] == {
        "completed": 0,
        "total": 6,
        "remaining": 6,
        "fraction": 0.0,
    }
    assert projected["pending_gate_ids"] == [
        "deepstream9_engine_smoke",
        "seven_day_endurance",
    ]
    assert projected["current_run"] == {}
    assert projected["deepstream_runtime_qualification"]["current"] is True
    assert projected["state"] == "artifact_error"
    assert projected["reason"] == "current_receipt_schema_pin_mismatch"


def test_expiry_is_fail_closed_and_returns_top_gate_to_pending() -> None:
    at_expiry = _project(ROOT, now=EXPIRY_TIME)

    assert at_expiry["available"] is True
    assert at_expiry["current"] is False
    assert at_expiry["state"] == "stale_pending_requalification"
    assert at_expiry["reason"] == "qualification_expired"
    assert at_expiry["runtime_qualification_passed"] is False
    assert all(at_expiry["integrity"].values())

    merged = admin_validation._gpu_reentry(
        _reader(ROOT), now=EXPIRY_TIME
    )
    assert merged["progress"]["completed"] == 0
    assert merged["pending_gate_ids"] == [
        "deepstream9_engine_smoke",
        "seven_day_endurance",
    ]
    assert merged["deepstream_runtime_qualification"]["state"] == (
        "stale_pending_requalification"
    )


def test_missing_or_tampered_linked_evidence_fails_closed(
    ds9_workspace: Path,
) -> None:
    pin = admin_validation.DS9_RUNTIME_QUALIFICATION_ADMIN_PINS[
        "gpu_smoke_evidence"
    ]
    path = ds9_workspace / pin["path"]
    path.unlink()

    missing = _project(ds9_workspace)

    assert missing["available"] is False
    assert missing["current"] is False
    assert missing["reason"] == "gpu_smoke_evidence_missing"

    _copy_pin(ds9_workspace, pin)
    path.chmod(0o640)
    path.write_bytes(path.read_bytes() + b"tamper")

    tampered = _project(ds9_workspace)

    assert tampered["available"] is False
    assert tampered["reason"] == "gpu_smoke_evidence_pin_mismatch"
    assert tampered["integrity"][
        "gpu_smoke_evidence_exact_pin_verified"
    ] is False


def test_missing_runtime_control_reproduces_live_container_failure(
    ds9_workspace: Path,
) -> None:
    pin = admin_validation.DS9_RUNTIME_QUALIFICATION_ADMIN_PINS[
        "runtime_controls"
    ]
    (ds9_workspace / pin["path"]).unlink()

    projected = _project(ds9_workspace)

    assert projected["available"] is False
    assert projected["current"] is False
    assert projected["reason"] == "runtime_controls_missing"
    assert projected["checks"] == {"passed": 0, "total": 5}
    assert projected["integrity"][
        "runtime_controls_exact_pin_verified"
    ] is False


def test_exact_repinned_image_overclaim_fails_semantic_replay(
    ds9_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pins = copy.deepcopy(
        admin_validation.DS9_RUNTIME_QUALIFICATION_ADMIN_PINS
    )
    relative = pins["receipt"]["path"]
    path = ds9_workspace / relative
    path.chmod(0o640)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["requested_image"] = "unbound-deepstream-image:latest"
    receipt["image"]["requested_image"] = receipt["requested_image"]
    raw = _write_json(path, receipt)
    pins["receipt"] = _pin(relative, raw)
    monkeypatch.setattr(
        admin_validation, "DS9_RUNTIME_QUALIFICATION_ADMIN_PINS", pins
    )

    projected = _project(ds9_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "exact_pin_schema_or_semantic_replay_invalid"
    assert projected["integrity"]["receipt_exact_pin_verified"] is True
    assert projected["integrity"]["receipt_schema_replay_verified"] is True
    assert projected["integrity"]["receipt_semantics_replayed"] is False


def test_ds9_projection_redacts_paths_hashes_and_private_identity() -> None:
    serialized = json.dumps(_project(ROOT), ensure_ascii=False)

    assert re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", serialized) is None
    for fragment in (
        "/home/",
        "validation/results/",
        ".json",
        "sha256:",
        "GPU-",
        "ef7605fa",
        "ced1b591",
    ):
        assert fragment not in serialized


def test_admin_page_explains_separate_ds9_projection() -> None:
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")

    assert "Ayrı DS9 çalışma zamanı yeterliliği" in page
    assert "DS9 exact-pin / şema / replay" in page
    assert "fail-closed stale/pending" in page


def test_admin_image_packages_only_static_ds9_workspace_anchors() -> None:
    dockerfile = (ROOT / "admin/Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    pins = admin_validation.DS9_RUNTIME_QUALIFICATION_ADMIN_PINS

    for key in ("runtime_controls", "validator", "schema"):
        source = pins[key]["path"]
        destination = f"/workspace/{source}"
        assert f"COPY {source} {destination}" in dockerfile
        assert (
            f"!{source}" in dockerignore
            or (
                source.startswith("validation/schemas/")
                and "!validation/schemas/**" in dockerignore
            )
            or (
                source.startswith("deepstream/")
                and "!deepstream/**" in dockerignore
            )
        )
        raw = (ROOT / source).read_bytes()
        assert len(raw) == pins[key]["bytes"]
        assert hashlib.sha256(raw).hexdigest() == pins[key]["sha256"]

    # Generated receipts and GPU evidence remain read-only runtime mounts; a
    # stale image cannot silently substitute a baked result for live evidence.
    for key in ("receipt", "gpu_smoke_evidence", "parser_build_receipt"):
        assert pins[key]["path"] not in dockerfile
    assert "COPY validation /workspace/validation" not in dockerfile
    assert "COPY validation/results" not in dockerfile
