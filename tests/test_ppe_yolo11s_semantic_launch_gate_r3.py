from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from validation import ppe_yolo11s_semantic_launch_gate_r3 as gate


ROOT = Path(__file__).resolve().parents[1]
GATE_FILE_SHA256 = (
    "a95aa81bf70bdfdf960c44e3cc65390876d1d736796421cfbdef00a1ed9b5c47"
)
GATE_FINGERPRINT = (
    "26680ed43b9ae6ffffa221a1e6bdf913347c45defdc220fe487a1555d7cce1c9"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def verified() -> tuple[dict, dict]:
    return gate.load_and_verify_gate(
        expected_gate_file_sha256=GATE_FILE_SHA256,
        expected_gate_fingerprint=GATE_FINGERPRINT,
    )


def test_gate_schema_file_pin_and_fingerprint_are_exact() -> None:
    value = json.loads(gate.GATE_PATH.read_text(encoding="utf-8"))
    schema = json.loads(gate.GATE_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert not list(
        Draft202012Validator(
            schema, format_checker=FormatChecker()
        ).iter_errors(value)
    )
    assert _sha256(gate.GATE_PATH) == GATE_FILE_SHA256
    assert value["fingerprint_sha256"] == GATE_FINGERPRINT
    assert gate.canonical_fingerprint(value) == GATE_FINGERPRINT


def test_historical_r2_plan_is_pinned_without_rewrite(
    verified: tuple[dict, dict],
) -> None:
    value, _ = verified
    pin = value["inputs"]["historical_r2_execution_plan"]
    assert pin == {
        "path": (
            "models/ppe/training-lanes/"
            "yolo11s-mendeley-five-class-internal-eval-r2/"
            "execution-plan-r2.json"
        ),
        "bytes": 12135,
        "sha256": (
            "d0d2a0b239c0575e8b7ff46b470b18c9fba5e568a85863a862ae399d45db7a27"
        ),
        "fingerprint_sha256": (
            "4d2c089624eaf53f8a8b33ef326b20e3a16dc8ee4af56d84fb12577c98a11118"
        ),
    }
    assert value["scope"]["historical_plan_immutable"] is True
    assert value["scope"]["historical_plan_authorization_not_rewritten"] is True


def test_r4_exact_evidence_closes_training_and_production(
    verified: tuple[dict, dict],
) -> None:
    value, result = verified
    assert value["semantic_evidence"] == {
        "sample_images": 20,
        "source_groups": 18,
        "bbox_rows_checked": 488,
        "questionable_needs_adjudication": 15,
        "rejected_development_candidates": 3,
        "accepted_with_guardrails": 2,
        "development_holdout_payload_files_opened": 0,
        "critical_findings": [
            "vest_to_hi_vis_harness_misclassification",
            "helmet_worn_vs_carried_ambiguous",
            "no_vest_to_no_hi_vis_unproven",
        ],
    }
    assert result["development_holdout_payload_files_opened"] == 0
    assert result["training_ready"] is False
    assert result["production_ready"] is False
    assert value["release_requirements"]["new_authorization_receipt"] is None
    assert value["release_requirements"]["all_satisfied"] is False


def test_only_image_build_preparation_is_allowed(
    verified: tuple[dict, dict],
) -> None:
    value, _ = verified
    allowed = gate.check_mode(value, "image_build")
    assert allowed == {
        "mode": "image_build",
        "allowed": True,
        "scope": "container_preparation_only_no_dataset_or_model_execution",
        "execution_performed": False,
    }
    for mode in gate.BLOCKED_MODES:
        with pytest.raises(gate.PpeSemanticLaunchGateError, match="launch blocked"):
            gate.check_mode(value, mode)


def test_resealed_training_overclaim_is_rejected(tmp_path: Path) -> None:
    value = json.loads(gate.GATE_PATH.read_text(encoding="utf-8"))
    value["launch_policy"]["full_train_150e"] = {
        "allowed": True,
        "scope": "container_preparation_only_no_dataset_or_model_execution",
    }
    value["fingerprint_sha256"] = gate.canonical_fingerprint(value)
    path = tmp_path / "overclaim.json"
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(raw)
    with pytest.raises(gate.PpeSemanticLaunchGateError, match="schema mismatch"):
        gate.load_and_verify_gate(
            gate_path=path,
            expected_gate_file_sha256=hashlib.sha256(raw).hexdigest(),
            expected_gate_fingerprint=value["fingerprint_sha256"],
        )


def test_module_declares_no_execution_history(verified: tuple[dict, dict]) -> None:
    value, _ = verified
    assert value["execution_history"] == {
        "docker": False,
        "gpu": False,
        "training": False,
        "evaluation": False,
        "export": False,
    }
    source = Path(gate.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "subprocess.run",
        "docker build",
        "docker run",
        "nvidia-smi",
        "urllib.request",
        "requests.",
    ):
        assert forbidden not in source
