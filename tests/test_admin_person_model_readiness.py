from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from admin import validation as admin_validation
from admin.app import app


ROOT = Path(__file__).resolve().parents[1]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: dict) -> bytes:
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _pin(path: str, raw: bytes) -> dict[str, object]:
    return {"path": path, "bytes": len(raw), "sha256": _sha256(raw)}


@pytest.fixture
def upgrade_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "workspace"
    source_plan = json.loads(
        (ROOT / "models/person/upgrade-provenance-plan.json").read_text(
            encoding="utf-8"
        )
    )

    for relative in (
        admin_validation.PERSON_UPGRADE_MANIFEST_PATH,
        admin_validation.PERSON_UPGRADE_TRAINING_PLAN_PATH,
        admin_validation.PERSON_UPGRADE_STRUCTURAL_SCHEMA_PIN["path"],
        admin_validation.PERSON_UPGRADE_STRUCTURAL_VALIDATOR_PIN["path"],
        admin_validation.PERSON_UPGRADE_FRAMEWORK_SCHEMA_PIN["path"],
        admin_validation.PERSON_UPGRADE_FRAMEWORK_VALIDATOR_PIN["path"],
        admin_validation.PERSON_UPGRADE_ONNX_EXPORTER_PIN["path"],
        admin_validation.PERSON_UPGRADE_ONNX_RECEIPT_SCHEMA_PIN["path"],
        admin_validation.PERSON_UPGRADE_REAL_IMAGE_PARITY_SCHEMA_PIN[
            "path"
        ],
        admin_validation.PERSON_UPGRADE_REAL_IMAGE_PARITY_VALIDATOR_PIN[
            "path"
        ],
        admin_validation.PERSON_UPGRADE_ONNX_BATCH12_SCHEMA_PIN["path"],
        admin_validation.PERSON_UPGRADE_ONNX_BATCH12_VALIDATOR_PIN["path"],
        *(
            pin["path"]
            for pin in admin_validation.PERSON_UPGRADE_DS9_PARSER_SOURCE_PINS.values()
        ),
    ):
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    checkpoint_raw = b"descriptor-bound-checkpoint-fixture\n"
    checkpoint_path = (
        workspace / admin_validation.PERSON_UPGRADE_RTDETR_CHECKPOINT_PATH
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(checkpoint_raw)
    checkpoint_pin = _pin(
        admin_validation.PERSON_UPGRADE_RTDETR_CHECKPOINT_PATH,
        checkpoint_raw,
    )

    structural_receipt = json.loads(
        (
            ROOT / admin_validation.PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN["path"]
        ).read_text(encoding="utf-8")
    )
    structural_receipt["inputs"]["checkpoint"] = checkpoint_pin
    structural_receipt.pop("receipt_sha256")
    structural_receipt["receipt_sha256"] = (
        admin_validation._canonical_sha256(structural_receipt)
    )
    structural_path = (
        workspace / admin_validation.PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN["path"]
    )
    structural_raw = _write_json(structural_path, structural_receipt)
    structural_pin = _pin(
        admin_validation.PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN["path"],
        structural_raw,
    )
    structural_pin["receipt_sha256"] = structural_receipt["receipt_sha256"]
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN",
        structural_pin,
    )

    framework_receipt = json.loads(
        (
            ROOT / admin_validation.PERSON_UPGRADE_FRAMEWORK_RECEIPT_PIN["path"]
        ).read_text(encoding="utf-8")
    )
    framework_receipt["inputs"]["checkpoint"] = checkpoint_pin
    framework_receipt["inputs"]["structural_receipt"] = structural_pin
    framework_receipt.pop("receipt_sha256")
    framework_receipt["receipt_sha256"] = admin_validation._canonical_sha256(
        framework_receipt
    )
    framework_path = (
        workspace / admin_validation.PERSON_UPGRADE_FRAMEWORK_RECEIPT_PIN["path"]
    )
    framework_raw = _write_json(framework_path, framework_receipt)
    framework_pin = _pin(
        admin_validation.PERSON_UPGRADE_FRAMEWORK_RECEIPT_PIN["path"],
        framework_raw,
    )
    framework_pin["receipt_sha256"] = framework_receipt["receipt_sha256"]
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_FRAMEWORK_RECEIPT_PIN",
        framework_pin,
    )

    export_plan = json.loads(
        (
            ROOT / admin_validation.PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN["path"]
        ).read_text(encoding="utf-8")
    )
    export_plan["inputs"]["checkpoint"] = checkpoint_pin
    export_plan["inputs"]["structural_receipt"] = _pin(
        structural_pin["path"], structural_raw
    )
    export_plan["inputs"]["framework_profiles_receipt"] = _pin(
        framework_pin["path"], framework_raw
    )
    export_plan["source_receipt_fingerprints"] = {
        "structural": structural_pin["receipt_sha256"],
        "framework_profiles": framework_pin["receipt_sha256"],
    }
    export_plan.pop("fingerprint_sha256")
    export_plan["fingerprint_sha256"] = admin_validation._canonical_sha256(
        export_plan
    )
    export_plan_path = (
        workspace / admin_validation.PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN["path"]
    )
    export_plan_raw = _write_json(export_plan_path, export_plan)
    export_plan_pin = _pin(
        admin_validation.PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN["path"],
        export_plan_raw,
    )
    export_plan_pin["fingerprint_sha256"] = export_plan[
        "fingerprint_sha256"
    ]
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN",
        export_plan_pin,
    )

    profile_pins: dict[int, dict[str, dict[str, object]]] = {}
    for profile in (640, 960):
        source_profile_pin = admin_validation.PERSON_UPGRADE_ONNX_PROFILE_PINS[
            profile
        ]
        onnx_raw = f"synthetic-onnx-fixture-{profile}\n".encode("ascii")
        onnx_path = workspace / source_profile_pin["onnx"]["path"]
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_path.write_bytes(onnx_raw)
        onnx_pin = _pin(source_profile_pin["onnx"]["path"], onnx_raw)

        receipt = json.loads(
            (ROOT / source_profile_pin["receipt"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        receipt["inputs"]["checkpoint"] = checkpoint_pin
        receipt["inputs"]["plan"] = _pin(
            export_plan_pin["path"], export_plan_raw
        )
        receipt["export"]["onnx"] = onnx_pin
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = admin_validation._canonical_sha256(receipt)
        receipt_path = workspace / source_profile_pin["receipt"]["path"]
        receipt_raw = _write_json(receipt_path, receipt)
        receipt_pin = _pin(source_profile_pin["receipt"]["path"], receipt_raw)
        receipt_pin["receipt_sha256"] = receipt["receipt_sha256"]
        profile_pins[profile] = {"receipt": receipt_pin, "onnx": onnx_pin}
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_ONNX_PROFILE_PINS",
        profile_pins,
    )

    real_image_plan = json.loads(
        (
            ROOT
            / admin_validation.PERSON_UPGRADE_REAL_IMAGE_PARITY_PLAN_PIN[
                "path"
            ]
        ).read_text(encoding="utf-8")
    )
    real_image_plan["inputs"]["checkpoint"] = checkpoint_pin
    real_image_plan["inputs"]["export_plan"] = _pin(
        export_plan_pin["path"], export_plan_raw
    )
    real_image_plan["inputs"]["onnx_profiles"] = {
        str(profile): {
            "export_receipt": deepcopy(profile_pins[profile]["receipt"]),
            "onnx": deepcopy(profile_pins[profile]["onnx"]),
        }
        for profile in (640, 960)
    }
    real_image_plan.pop("fingerprint_sha256")
    real_image_plan["fingerprint_sha256"] = (
        admin_validation._canonical_sha256(real_image_plan)
    )
    real_image_plan_path = (
        workspace
        / admin_validation.PERSON_UPGRADE_REAL_IMAGE_PARITY_PLAN_PIN["path"]
    )
    real_image_plan_raw = _write_json(
        real_image_plan_path, real_image_plan
    )
    real_image_plan_pin = _pin(
        admin_validation.PERSON_UPGRADE_REAL_IMAGE_PARITY_PLAN_PIN["path"],
        real_image_plan_raw,
    )
    real_image_plan_pin["fingerprint_sha256"] = real_image_plan[
        "fingerprint_sha256"
    ]
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_REAL_IMAGE_PARITY_PLAN_PIN",
        real_image_plan_pin,
    )

    real_image_receipt = json.loads(
        (
            ROOT
            / admin_validation.PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN[
                "path"
            ]
        ).read_text(encoding="utf-8")
    )
    real_image_receipt["inputs"]["plan"] = deepcopy(real_image_plan_pin)
    real_image_receipt["inputs"]["checkpoint"] = checkpoint_pin
    real_image_receipt["inputs"]["onnx_profiles"] = deepcopy(
        real_image_plan["inputs"]["onnx_profiles"]
    )
    for profile_row in real_image_receipt["profiles"]:
        profile = profile_row["profile"]
        profile_row["onnx"] = deepcopy(profile_pins[profile]["onnx"])
    real_image_receipt.pop("receipt_sha256")
    real_image_receipt["receipt_sha256"] = (
        admin_validation._canonical_sha256(real_image_receipt)
    )
    real_image_receipt_path = (
        workspace
        / admin_validation.PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN[
            "path"
        ]
    )
    real_image_receipt_raw = _write_json(
        real_image_receipt_path, real_image_receipt
    )
    real_image_receipt_pin = _pin(
        admin_validation.PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN["path"],
        real_image_receipt_raw,
    )
    real_image_receipt_pin["receipt_sha256"] = real_image_receipt[
        "receipt_sha256"
    ]
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN",
        real_image_receipt_pin,
    )

    batch12_receipt = json.loads(
        (
            ROOT
            / admin_validation.PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN["path"]
        ).read_text(encoding="utf-8")
    )
    batch12_receipt["inputs"] = {
        str(profile): {
            "export_receipt": deepcopy(profile_pins[profile]["receipt"]),
            "onnx": deepcopy(profile_pins[profile]["onnx"]),
        }
        for profile in (640, 960)
    }
    batch12_receipt.pop("receipt_sha256")
    batch12_receipt["receipt_sha256"] = admin_validation._canonical_sha256(
        batch12_receipt
    )
    batch12_receipt_path = (
        workspace
        / admin_validation.PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN["path"]
    )
    batch12_receipt_raw = _write_json(batch12_receipt_path, batch12_receipt)
    batch12_receipt_pin = _pin(
        admin_validation.PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN["path"],
        batch12_receipt_raw,
    )
    batch12_receipt_pin["receipt_sha256"] = batch12_receipt[
        "receipt_sha256"
    ]
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN",
        batch12_receipt_pin,
    )

    parser_artifact_raw = b"synthetic-parser-so-fixture\n"
    parser_artifact_path = (
        workspace / admin_validation.PERSON_UPGRADE_DS9_PARSER_ARTIFACT_PIN["path"]
    )
    parser_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    parser_artifact_path.write_bytes(parser_artifact_raw)
    parser_artifact_pin = _pin(
        admin_validation.PERSON_UPGRADE_DS9_PARSER_ARTIFACT_PIN["path"],
        parser_artifact_raw,
    )
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_DS9_PARSER_ARTIFACT_PIN",
        parser_artifact_pin,
    )

    parser_receipt = json.loads(
        (
            ROOT / admin_validation.PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN["path"]
        ).read_text(encoding="utf-8")
    )
    parser_receipt["inputs"] = {
        **deepcopy(admin_validation.PERSON_UPGRADE_DS9_PARSER_SOURCE_PINS),
        "export_plan": deepcopy(export_plan_pin),
        "onnx_640_receipt": deepcopy(profile_pins[640]["receipt"]),
        "onnx_960_receipt": deepcopy(profile_pins[960]["receipt"]),
    }
    parser_receipt["artifact"] = {
        **parser_artifact_pin,
        "mode": "0440",
        "elf": "ELF64_x86_64_shared_object",
    }
    parser_receipt.pop("receipt_sha256")
    parser_receipt["receipt_sha256"] = admin_validation._canonical_sha256(
        parser_receipt
    )
    parser_receipt_path = (
        workspace / admin_validation.PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN["path"]
    )
    parser_receipt_raw = _write_json(parser_receipt_path, parser_receipt)
    parser_receipt_pin = _pin(
        admin_validation.PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN["path"],
        parser_receipt_raw,
    )
    parser_receipt_pin["receipt_sha256"] = parser_receipt[
        "receipt_sha256"
    ]
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN",
        parser_receipt_pin,
    )

    provenance = json.loads(
        (
            ROOT / admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH
        ).read_text(encoding="utf-8")
    )
    provenance["checkpoint"].update(checkpoint_pin)
    provenance["structural_load_receipt"].update(structural_pin)
    provenance["framework_profiles"]["receipt"] = framework_pin
    provenance["onnx_export_evidence"]["export_plan"] = export_plan_pin
    provenance["onnx_export_evidence"]["profiles"] = {
        str(profile): deepcopy(profile_pins[profile]) for profile in (640, 960)
    }
    provenance["real_image_parity_evidence"]["plan"] = (
        real_image_plan_pin
    )
    provenance["real_image_parity_evidence"]["receipt"] = (
        real_image_receipt_pin
    )
    provenance["onnx_batch12_evidence"]["receipt"] = batch12_receipt_pin
    provenance["deepstream9_parser_evidence"][
        "build_receipt"
    ] = parser_receipt_pin
    provenance["deepstream9_parser_evidence"][
        "artifact"
    ] = parser_artifact_pin
    provenance_path = (
        workspace / admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH
    )
    provenance_raw = _write_json(provenance_path, provenance)
    provenance_pin = _pin(
        admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH,
        provenance_raw,
    )

    challenger = source_plan["upstream"]["rtdetrv4"]
    challenger["checkpoint"] = checkpoint_pin
    challenger["provenance"] = provenance_pin
    challenger["structural_load_verified"] = True
    challenger["structural_load_receipt"] = structural_pin
    challenger["framework_profiles_verified"] = True
    challenger["framework_profiles_receipt"] = framework_pin
    challenger["export_plan"] = export_plan_pin
    challenger["onnx_profiles_exported"] = [640, 960]
    challenger["onnx_profile_receipts"] = {
        str(profile): deepcopy(profile_pins[profile]["receipt"])
        for profile in (640, 960)
    }
    challenger["synthetic_onnx_parity_passed"] = True
    challenger["real_image_framework_parity_passed"] = False
    challenger["real_image_parity_evidence_verified"] = True
    challenger["real_image_parity_plan"] = real_image_plan_pin
    challenger["real_image_parity_receipt"] = real_image_receipt_pin
    challenger["real_image_parity_failure_count"] = 4
    challenger["onnx_batch12_shape_verified"] = True
    challenger["onnx_batch12_receipt"] = batch12_receipt_pin
    challenger["parser_cpu_contract_ready"] = True
    challenger["parser_build_receipt"] = parser_receipt_pin
    challenger["parser_artifact"] = parser_artifact_pin
    challenger["deepstream9_real_inference_validated"] = False
    plan_path = workspace / admin_validation.PERSON_UPGRADE_PLAN_PIN["path"]
    plan_raw = _write_json(plan_path, source_plan)
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_PLAN_PIN",
        _pin(admin_validation.PERSON_UPGRADE_PLAN_PIN["path"], plan_raw),
    )
    return workspace


def _project(workspace: Path) -> dict:
    reader = admin_validation.ArtifactReader(
        workspace / "missing-results",
        workspace_root=workspace,
        schema_root=ROOT / "validation/schemas",
    )
    return admin_validation._person_model_upgrade_readiness(reader)


def test_checked_in_upgrade_chain_matches_the_admin_trust_anchor() -> None:
    projected = _project(ROOT)

    assert projected["available"] is True
    assert projected["state"] == "prepared_not_evaluated"
    assert all(projected["integrity"].values())
    assert projected["license"]["decision"] is None
    assert all(value is False for value in projected["gates"].values())


def test_projection_is_preparation_only_and_pathless(
    upgrade_workspace: Path,
) -> None:
    projected = _project(upgrade_workspace)

    assert projected["available"] is True
    assert projected["state"] == "prepared_not_evaluated"
    assert projected["ready"] is False
    assert projected["final_claim_allowed"] is False
    assert projected["license"]["decision"] is None
    assert projected["license"]["download_and_training_authorized"] is False
    assert projected["preparation"] == {
        "training_data_prepared": True,
        "frozen_training_plan_verified": True,
        "permissive_checkpoint_acquired": True,
    }
    assert projected["dataset"]["train_frames"] == 1524
    assert projected["dataset"]["calibration_frames"] == 384
    assert projected["dataset"]["person_instances"] == 16652
    assert projected["dataset"]["official_test_output_frames"] == 0
    assert projected["training_plan"]["training_executed"] is False
    assert projected["training_plan"]["export_executed"] is False
    assert projected["permissive_challenger"]["checkpoint_integrity_verified"]
    assert projected["permissive_challenger"]["structural_load_verified"]
    assert projected["permissive_challenger"]["structural_receipt_verified"]
    assert projected["permissive_challenger"]["forward_pass_executed"] is False
    assert projected["permissive_challenger"]["framework_profiles_verified"]
    assert projected["permissive_challenger"]["onnx_profiles_exported"] == [
        640,
        960,
    ]
    assert projected["permissive_challenger"][
        "synthetic_onnx_parity_passed"
    ]
    assert projected["permissive_challenger"]["onnx_batch12_shape_verified"]
    assert projected["permissive_challenger"]["onnx_batch12_profiles"] == [
        640,
        960,
    ]
    assert projected["permissive_challenger"][
        "onnx_batch12_performance_claimed"
    ] is False
    assert projected["permissive_challenger"][
        "real_image_framework_onnx_evidence_verified"
    ] is True
    assert projected["permissive_challenger"][
        "real_image_inference_executed"
    ] is True
    assert projected["permissive_challenger"][
        "real_image_selected_frame_count"
    ] == 11
    assert projected["permissive_challenger"][
        "real_image_unique_video_type_count"
    ] == 11
    assert projected["permissive_challenger"]["real_image_profiles"] == {
        "640": {
            "batch1_passed": True,
            "batch2_passed": True,
            "passed": True,
        },
        "960": {
            "batch1_passed": False,
            "batch2_passed": True,
            "passed": False,
        },
    }
    assert projected["permissive_challenger"][
        "real_image_failure_count"
    ] == 4
    assert projected["permissive_challenger"][
        "real_image_tolerances_relaxed"
    ] is False
    assert projected["permissive_challenger"][
        "real_image_framework_onnx_parity_passed"
    ] is False
    assert projected["permissive_challenger"]["parser_cpu_contract_ready"]
    assert projected["permissive_challenger"]["parser_contract_test_passed"]
    assert projected["permissive_challenger"]["parser_max_batch_contract"] == 12
    assert all(value is False for value in projected["gates"].values())
    assert all(projected["integrity"].values())

    serialized = json.dumps(projected, ensure_ascii=False)
    for private_fragment in (
        "/workspace",
        "models/person/",
        "data/derived/",
        "docker.io/",
        "https://",
        "training_arguments",
        "run_yolo26_training",
    ):
        assert private_fragment not in serialized


def test_missing_checkpoint_fails_every_preparation_flag_closed(
    upgrade_workspace: Path,
) -> None:
    (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_RTDETR_CHECKPOINT_PATH
    ).unlink()

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert projected["state"] == "artifact_error"
    assert projected["ready"] is False
    assert projected["preparation"]["training_data_prepared"] is False
    assert projected["license"]["decision"] is None
    assert all(value is False for value in projected["gates"].values())
    assert projected["integrity"]["challenger_checkpoint_verified"] is False


def test_manifest_symlink_is_rejected_even_when_target_bytes_match(
    upgrade_workspace: Path,
) -> None:
    manifest = upgrade_workspace / admin_validation.PERSON_UPGRADE_MANIFEST_PATH
    target = upgrade_workspace / "same-manifest-outside-pinned-name.json"
    target.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(target)

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert projected["preparation"]["training_data_prepared"] is False
    assert projected["integrity"]["dataset_manifest_verified"] is False
    assert "unsafe_path" in projected["reason"]


def test_live_training_plan_pin_mismatch_is_rejected(
    upgrade_workspace: Path,
) -> None:
    training = (
        upgrade_workspace / admin_validation.PERSON_UPGRADE_TRAINING_PLAN_PATH
    )
    raw = training.read_bytes()
    assert b"yolo26s" in raw
    training.write_bytes(raw.replace(b"yolo26s", b"yolo26x", 1))

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert projected["preparation"]["frozen_training_plan_verified"] is False
    assert projected["integrity"]["training_plan_verified"] is False
    assert "pin_mismatch" in projected["reason"]


def test_trust_anchor_tamper_is_rejected_before_dependent_projection(
    upgrade_workspace: Path,
) -> None:
    plan = upgrade_workspace / admin_validation.PERSON_UPGRADE_PLAN_PIN["path"]
    plan.write_bytes(plan.read_bytes() + b" ")

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert projected["integrity"]["upgrade_plan_verified"] is False
    assert projected["preparation"]["training_data_prepared"] is False
    assert projected["license"]["decision"] is None


def test_missing_structural_receipt_fails_every_gate_closed(
    upgrade_workspace: Path,
) -> None:
    (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN["path"]
    ).unlink()

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert (
        projected["integrity"][
            "challenger_structural_receipt_file_verified"
        ]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_structural_receipt_hash_drift_fails_every_gate_closed(
    upgrade_workspace: Path,
) -> None:
    receipt = (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN["path"]
    )
    receipt.write_bytes(receipt.read_bytes() + b" ")

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert (
        projected["integrity"][
            "challenger_structural_receipt_file_verified"
        ]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_missing_onnx_receipt_fails_every_gate_closed(
    upgrade_workspace: Path,
) -> None:
    (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_ONNX_PROFILE_PINS[640]["receipt"][
            "path"
        ]
    ).unlink()

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert (
        projected["integrity"][
            "challenger_onnx_640_receipt_file_verified"
        ]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_missing_real_image_parity_receipt_fails_every_gate_closed(
    upgrade_workspace: Path,
) -> None:
    (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN[
            "path"
        ]
    ).unlink()

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert (
        projected["integrity"][
            "challenger_real_image_parity_receipt_file_verified"
        ]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_exact_repinned_real_image_parity_overclaim_is_rejected(
    upgrade_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN[
            "path"
        ]
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["acceptance"][
        "real_image_framework_onnx_parity_passed"
    ] = True
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = admin_validation._canonical_sha256(receipt)
    receipt_raw = _write_json(receipt_path, receipt)
    receipt_pin = _pin(
        admin_validation.PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN["path"],
        receipt_raw,
    )
    receipt_pin["receipt_sha256"] = receipt["receipt_sha256"]
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN",
        receipt_pin,
    )

    provenance_path = (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["real_image_parity_evidence"]["receipt"] = receipt_pin
    provenance_raw = _write_json(provenance_path, provenance)
    provenance_pin = _pin(
        admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH,
        provenance_raw,
    )

    plan_path = upgrade_workspace / admin_validation.PERSON_UPGRADE_PLAN_PIN[
        "path"
    ]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    challenger = plan["upstream"]["rtdetrv4"]
    challenger["provenance"] = provenance_pin
    challenger["real_image_parity_receipt"] = receipt_pin
    plan_raw = _write_json(plan_path, plan)
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_PLAN_PIN",
        _pin(admin_validation.PERSON_UPGRADE_PLAN_PIN["path"], plan_raw),
    )

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert projected["reason"] == (
        "challenger_real_image_parity_evidence_invalid"
    )
    assert projected["integrity"][
        "challenger_real_image_parity_self_hash_verified"
    ]
    assert (
        projected["integrity"][
            "challenger_real_image_parity_semantic_verified"
        ]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_onnx_artifact_hash_drift_fails_every_gate_closed(
    upgrade_workspace: Path,
) -> None:
    artifact = (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_ONNX_PROFILE_PINS[960]["onnx"][
            "path"
        ]
    )
    artifact.write_bytes(artifact.read_bytes() + b"tamper")

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert (
        projected["integrity"]["challenger_onnx_960_artifact_verified"]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_exact_repinned_onnx_overclaim_is_rejected_semantically(
    upgrade_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_pins = deepcopy(admin_validation.PERSON_UPGRADE_ONNX_PROFILE_PINS)
    receipt_path = upgrade_workspace / profile_pins[640]["receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["acceptance"]["production_ready"] = True
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = admin_validation._canonical_sha256(receipt)
    receipt_raw = _write_json(receipt_path, receipt)
    receipt_pin = _pin(profile_pins[640]["receipt"]["path"], receipt_raw)
    receipt_pin["receipt_sha256"] = receipt["receipt_sha256"]
    profile_pins[640]["receipt"] = receipt_pin
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_ONNX_PROFILE_PINS",
        profile_pins,
    )

    provenance_path = (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["onnx_export_evidence"]["profiles"]["640"] = profile_pins[640]
    provenance_raw = _write_json(provenance_path, provenance)
    provenance_pin = _pin(
        admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH,
        provenance_raw,
    )

    plan_path = upgrade_workspace / admin_validation.PERSON_UPGRADE_PLAN_PIN[
        "path"
    ]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    challenger = plan["upstream"]["rtdetrv4"]
    challenger["provenance"] = provenance_pin
    challenger["onnx_profile_receipts"]["640"] = receipt_pin
    plan_raw = _write_json(plan_path, plan)
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_PLAN_PIN",
        _pin(admin_validation.PERSON_UPGRADE_PLAN_PIN["path"], plan_raw),
    )

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "challenger_onnx_evidence_invalid"
    assert projected["integrity"]["challenger_onnx_640_self_hash_verified"]
    assert (
        projected["integrity"]["challenger_onnx_640_semantic_verified"]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_missing_ds9_parser_receipt_fails_every_gate_closed(
    upgrade_workspace: Path,
) -> None:
    (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN["path"]
    ).unlink()

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert (
        projected["integrity"][
            "challenger_ds9_parser_receipt_file_verified"
        ]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_missing_onnx_batch12_receipt_fails_every_gate_closed(
    upgrade_workspace: Path,
) -> None:
    (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN["path"]
    ).unlink()

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert (
        projected["integrity"][
            "challenger_onnx_batch12_receipt_file_verified"
        ]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_exact_repinned_onnx_batch12_capacity_overclaim_is_rejected(
    upgrade_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN["path"]
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["gates"]["capacity_passed"] = True
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = admin_validation._canonical_sha256(receipt)
    receipt_raw = _write_json(receipt_path, receipt)
    receipt_pin = _pin(
        admin_validation.PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN["path"],
        receipt_raw,
    )
    receipt_pin["receipt_sha256"] = receipt["receipt_sha256"]
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN",
        receipt_pin,
    )

    provenance_path = (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["onnx_batch12_evidence"]["receipt"] = receipt_pin
    provenance_raw = _write_json(provenance_path, provenance)
    provenance_pin = _pin(
        admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH,
        provenance_raw,
    )

    plan_path = upgrade_workspace / admin_validation.PERSON_UPGRADE_PLAN_PIN[
        "path"
    ]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    challenger = plan["upstream"]["rtdetrv4"]
    challenger["provenance"] = provenance_pin
    challenger["onnx_batch12_receipt"] = receipt_pin
    plan_raw = _write_json(plan_path, plan)
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_PLAN_PIN",
        _pin(admin_validation.PERSON_UPGRADE_PLAN_PIN["path"], plan_raw),
    )

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "challenger_onnx_batch12_evidence_invalid"
    assert projected["integrity"]["challenger_onnx_batch12_self_hash_verified"]
    assert (
        projected["integrity"]["challenger_onnx_batch12_semantic_verified"]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_ds9_parser_artifact_hash_drift_fails_every_gate_closed(
    upgrade_workspace: Path,
) -> None:
    artifact = (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_DS9_PARSER_ARTIFACT_PIN["path"]
    )
    artifact.write_bytes(artifact.read_bytes() + b"tamper")

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert (
        projected["integrity"]["challenger_ds9_parser_artifact_verified"]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_exact_repinned_ds9_parser_overclaim_is_rejected_semantically(
    upgrade_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN["path"]
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["readiness"]["deepstream9_real_inference_validated"] = True
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = admin_validation._canonical_sha256(receipt)
    receipt_raw = _write_json(receipt_path, receipt)
    receipt_pin = _pin(
        admin_validation.PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN["path"],
        receipt_raw,
    )
    receipt_pin["receipt_sha256"] = receipt["receipt_sha256"]
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN",
        receipt_pin,
    )

    provenance_path = (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["deepstream9_parser_evidence"]["build_receipt"] = receipt_pin
    provenance_raw = _write_json(provenance_path, provenance)
    provenance_pin = _pin(
        admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH,
        provenance_raw,
    )

    plan_path = upgrade_workspace / admin_validation.PERSON_UPGRADE_PLAN_PIN[
        "path"
    ]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    challenger = plan["upstream"]["rtdetrv4"]
    challenger["provenance"] = provenance_pin
    challenger["parser_build_receipt"] = receipt_pin
    plan_raw = _write_json(plan_path, plan)
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_PLAN_PIN",
        _pin(admin_validation.PERSON_UPGRADE_PLAN_PIN["path"], plan_raw),
    )

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "challenger_ds9_parser_evidence_invalid"
    assert projected["integrity"]["challenger_ds9_parser_self_hash_verified"]
    assert (
        projected["integrity"]["challenger_ds9_parser_semantic_verified"]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_exact_repinned_structural_overclaim_is_rejected_semantically(
    upgrade_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN["path"]
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["conclusions"]["production_ready"] = True
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = admin_validation._canonical_sha256(receipt)
    receipt_raw = _write_json(receipt_path, receipt)
    receipt_pin = _pin(
        admin_validation.PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN["path"],
        receipt_raw,
    )
    receipt_pin["receipt_sha256"] = receipt["receipt_sha256"]
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN",
        receipt_pin,
    )

    provenance_path = (
        upgrade_workspace
        / admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["structural_load_receipt"].update(receipt_pin)
    provenance_raw = _write_json(provenance_path, provenance)
    provenance_pin = _pin(
        admin_validation.PERSON_UPGRADE_RTDETR_PROVENANCE_PATH,
        provenance_raw,
    )

    plan_path = upgrade_workspace / admin_validation.PERSON_UPGRADE_PLAN_PIN[
        "path"
    ]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    challenger = plan["upstream"]["rtdetrv4"]
    challenger["provenance"] = provenance_pin
    challenger["structural_load_receipt"] = receipt_pin
    plan_raw = _write_json(plan_path, plan)
    monkeypatch.setattr(
        admin_validation,
        "PERSON_UPGRADE_PLAN_PIN",
        _pin(admin_validation.PERSON_UPGRADE_PLAN_PIN["path"], plan_raw),
    )

    projected = _project(upgrade_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "challenger_structural_receipt_invalid"
    assert (
        projected["integrity"]["challenger_structural_self_hash_verified"]
        is True
    )
    assert (
        projected["integrity"]["challenger_structural_semantic_verified"]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_validation_api_and_ui_expose_only_compact_read_only_projection(
    upgrade_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(results))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(upgrade_workspace)
    )
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT", str(ROOT / "validation/schemas")
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        page = client.get("/")

    assert response.status_code == 200
    projected = response.json()["campaigns"]["person_model_upgrade"]
    assert projected["state"] == "prepared_not_evaluated"
    assert projected["license"]["decision"] is None
    assert projected["gates"]["production_ready"] is False
    assert (
        projected["permissive_challenger"]["structural_load_verified"]
        is True
    )
    assert (
        projected["permissive_challenger"]["forward_pass_executed"]
        is False
    )
    assert (
        projected["permissive_challenger"]["framework_profiles_verified"]
        is True
    )
    assert projected["permissive_challenger"]["onnx_profiles_exported"] == [
        640,
        960,
    ]
    assert (
        projected["permissive_challenger"][
            "synthetic_onnx_parity_passed"
        ]
        is True
    )
    assert (
        projected["permissive_challenger"]["onnx_batch12_shape_verified"]
        is True
    )
    assert (
        projected["permissive_challenger"][
            "onnx_batch12_performance_claimed"
        ]
        is False
    )
    assert (
        projected["permissive_challenger"][
            "real_image_framework_onnx_parity_passed"
        ]
        is False
    )
    assert (
        projected["permissive_challenger"]["parser_cpu_contract_ready"]
        is True
    )
    assert (
        projected["permissive_challenger"]["parser_contract_test_passed"]
        is True
    )
    assert page.status_code == 200
    assert "Kişi modeli yükseltme hazırlığı" in page.text
    assert "Model/eğitim/export/parity/kabul" in page.text
    assert "Yapısal checkpoint yükleme" in page.text
    assert "CPU strict model + EMA doğrulandı" in page.text
    assert "Challenger framework profilleri" in page.text
    assert "Challenger ONNX profilleri" in page.text
    assert "ONNX batch-12 şekil/finite" in page.text
    assert "Gerçek-görüntü framework/ONNX parity" in page.text
    assert "DS9 parser CPU/ABI sözleşmesi" in page.text
    assert "indirme/eğitim kapalı" in page.text


def test_compose_mounts_upgrade_sources_read_only_and_docs_define_boundary() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    docs = (ROOT / "docs/admin-validation-dashboard.md").read_text(
        encoding="utf-8"
    )

    assert "./models:/workspace/models:ro" in compose
    assert "./data:/workspace/data:ro" in compose
    assert "component-by-component `O_NOFOLLOW`" in docs
    assert "license decision is deliberately projected as `null`" in docs
    assert "CPU-only strict model/EMA load" in docs
