from __future__ import annotations

from pathlib import Path

from admin import ppe_construction_status as construction
from admin.validation import load_validation_status


ROOT = Path(__file__).resolve().parents[1]


def test_construction_ppe_card_replays_exact_diagnostic_receipt() -> None:
    card = construction.load_construction_ppe_status()
    assert card["state"] == "quarantined_diagnostic_only"
    assert card["available"] is True
    assert card["ready"] is False
    assert card["offline_diagnostic_model_evaluation_authorized"] is True
    assert card["training_authorized"] is False
    assert card["threshold_calibration_authorized"] is False
    assert card["independent_final_ground_truth_authorized"] is False
    assert card["production_ready"] is False
    assert card["execution_actions_available"] is False
    assert card["dataset"]["images"] == 1416
    assert card["dataset"]["paired_boxes"] == 11521
    assert card["dataset"]["target_class_mapping"] == {
        "helmet": 0,
        "no_helmet": 7,
        "hi_vis": 2,
        "no_hi_vis": None,
    }
    assert card["split_leakage"]["cross_split_pairs_by_max_hamming"] == {
        "0": 2,
        "2": 16,
        "4": 38,
        "6": 92,
        "8": 193,
    }
    assert all(value is False for value in card["model_readiness"].values())


def test_construction_ppe_card_fails_closed_on_pin_drift(monkeypatch) -> None:
    monkeypatch.setattr(
        construction,
        "MANIFEST_PIN",
        (construction.MANIFEST_PIN[0], "0" * 64),
    )
    card = construction.load_construction_ppe_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["available"] is False
    assert card["training_authorized"] is False
    assert card["production_ready"] is False


def test_validation_payload_and_ui_expose_read_only_construction_card() -> None:
    payload = load_validation_status()
    card = payload["campaigns"]["ppe_construction_ppe_quarantine"]
    assert card["state"] == "quarantined_diagnostic_only"
    assert payload["read_only"] is True
    assert payload["execution_actions_available"] is False
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert "ppe_construction_ppe_quarantine" in page
    assert "campaign.training_authorized" in page
    assert "fetch('/api/ppe/construction/train')" not in page


def test_admin_image_copies_projection_and_mounts_evidence_read_only() -> None:
    dockerfile = (ROOT / "admin/Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "COPY admin /app/admin" in dockerfile
    assert "./validation/results:/workspace/validation-results:ro" in compose
    assert "./validation:/workspace/validation:ro" in compose
    assert "./data:/workspace/data:ro" in compose
