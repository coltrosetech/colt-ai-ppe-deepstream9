from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "models/person/upgrade-provenance-plan.json"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def _assert_live_pin(pin: dict) -> None:
    path = ROOT / pin["path"]
    assert path.is_file()
    assert path.stat().st_size == pin["bytes"]
    assert _digest(path) == pin["sha256"]


def test_person_upgrade_plan_is_fail_closed_and_local_evidence_is_pinned() -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))

    assert value["schema_version"] == "deepsafe.person-upgrade-provenance-plan/v1"
    assert value["status"] == (
        "training_data_and_frozen_plan_prepared_license_and_training_not_started"
    )
    assert value["decision"]["control"] == "yolo11s"
    assert value["decision"]["primary_candidate"] == "yolo26s"
    assert value["decision"]["small_object_challenger"] == "yolo26s-p2"
    assert value["license_gate"]["decision"] is None
    assert value["license_gate"]["download_and_training_authorized"] is False
    generated = value["training_and_export_contract"]["generated_artifacts"]
    assert generated["dataset_manifest"] is not None
    assert generated["training_plan"] is not None
    assert all(
        generated[key] is None
        for key in generated
        if key not in {"dataset_manifest", "training_plan"}
    )
    assert value["readiness"]["training_data_prepared"] is True
    assert all(
        state is False
        for key, state in value["readiness"].items()
        if key != "training_data_prepared"
    )

    primary = value["training_data"]["primary"]
    _assert_live_pin(primary["source_contract"])
    _assert_live_pin(primary["archive"])
    _assert_live_pin(
        value["held_out_evidence"]["rlivit_test"]["threshold_sweep"]
    )
    _assert_live_pin(value["held_out_evidence"]["loaf_20_to_25m"]["aggregate"])
    _assert_live_pin(value["upstream"]["rtdetrv4"]["checkpoint"])
    _assert_live_pin(value["upstream"]["rtdetrv4"]["provenance"])
    _assert_live_pin(
        value["upstream"]["rtdetrv4"]["structural_load_receipt"]
    )
    _assert_live_pin(primary["prepared_dataset_manifest"])
    _assert_live_pin(generated["dataset_manifest"])
    _assert_live_pin(generated["training_plan"])


def test_prepared_training_manifest_keeps_official_test_out() -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    pin = value["training_data"]["primary"]["prepared_dataset_manifest"]
    manifest = json.loads((ROOT / pin["path"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "prepared_cpu_only"
    assert manifest["fingerprint_sha256"] == pin["fingerprint_sha256"]
    assert manifest["qa"]["output_frames"] == 1908
    assert manifest["qa"]["persons"] == 16652
    assert manifest["qa"]["train_calibration_sequence_overlap"] == 0
    excluded = manifest["splits"]["official_test_exclusion"]
    assert excluded["included_output_frames"] == 0
    assert excluded["included_output_labels"] == 0
    assert excluded["rgb_images_read"] == 0
    assert manifest["splits"]["quarantined_official_train"]["sequences"] == [
        "064"
    ]


def test_frozen_training_plan_binds_dataset_code_runtime_and_no_acceptance() -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    pin = value["training_and_export_contract"]["generated_artifacts"][
        "training_plan"
    ]
    training_plan = json.loads((ROOT / pin["path"]).read_text(encoding="utf-8"))

    assert training_plan["fingerprint_sha256"] == pin["fingerprint_sha256"]
    assert training_plan["status"] == "planned_license_required_not_executed"
    assert training_plan["runtime"]["container_image"].startswith(
        "docker.io/ultralytics/ultralytics@sha256:"
    )
    assert training_plan["runtime"]["implementation"]["harness"]["sha256"]
    assert training_plan["runtime"]["implementation"]["wrapper"]["sha256"]
    assert training_plan["dataset"]["official_test_output_frames"] == 0
    assert training_plan["license_gate"]["decision_recorded_in_plan"] is False
    assert training_plan["acceptance_effect"].startswith("none_until")


def test_rtdetrv4_challenger_is_acquired_but_cannot_claim_readiness() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    challenger = plan["upstream"]["rtdetrv4"]
    provenance_path = ROOT / challenger["provenance"]["path"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert challenger["declared_code_license"] == "Apache-2.0"
    assert challenger["status"] == (
        "official_checkpoint_acquired_onnx_profiles_exported_"
        "real_image_parity_failed_not_evaluated"
    )
    assert provenance["checkpoint"]["download_complete"] is True
    assert challenger["structural_load_verified"] is True
    assert provenance["checkpoint"]["structural_load_verified"] is True
    assert provenance["checkpoint"]["structural_load_blocker"] is None
    assert (
        provenance["structural_load_receipt"]["receipt_sha256"]
        == challenger["structural_load_receipt"]["receipt_sha256"]
    )
    assert challenger["framework_profiles_verified"] is True
    assert challenger["onnx_profiles_exported"] == [640, 960]
    assert challenger["synthetic_onnx_parity_passed"] is True
    assert challenger["real_image_framework_parity_passed"] is False
    assert challenger["real_image_parity_evidence_verified"] is True
    assert challenger["real_image_parity_failure_count"] == 4
    assert provenance["framework_profiles"]["profiles"] == [640, 960]
    assert provenance["framework_profiles"]["verified"] is True
    assert provenance["framework_profiles"][
        "real_image_inference_executed"
    ] is True
    assert provenance["onnx_export_evidence"][
        "synthetic_seeded_prng_parity_passed"
    ] is True
    assert provenance["onnx_export_evidence"][
        "real_image_framework_parity_passed"
    ] is False
    real_image = provenance["real_image_parity_evidence"]
    assert real_image["executed"] is True
    assert real_image["selected_frame_count"] == 11
    assert real_image["unique_primary_video_type_count"] == 11
    assert real_image["profiles"]["640"]["passed"] is True
    assert real_image["profiles"]["960"]["passed"] is False
    assert real_image["failure_count"] == 4
    assert real_image["tolerances_relaxed"] is False
    assert real_image["topk_tie_diagnostics_override_acceptance"] is False
    assert real_image["real_image_framework_onnx_parity_passed"] is False
    assert real_image["quality_passed"] is False
    assert real_image["latency_or_fps_passed"] is False
    assert real_image["production_ready"] is False
    assert challenger["onnx_batch12_shape_verified"] is True
    assert provenance["onnx_batch12_evidence"][
        "shape_and_finite_verified"
    ] is True
    assert provenance["onnx_batch12_evidence"]["profiles"] == [640, 960]
    assert provenance["onnx_batch12_evidence"][
        "latency_or_fps_claimed"
    ] is False
    assert provenance["onnx_batch12_evidence"]["capacity_passed"] is False
    assert challenger["parser_cpu_contract_ready"] is True
    assert challenger["deepstream9_real_inference_validated"] is False
    assert provenance["deepstream9_parser_evidence"][
        "cpu_contract_ready"
    ] is True
    assert provenance["deepstream9_parser_evidence"][
        "deepstream9_real_inference_validated"
    ] is False
    assert provenance["deployment_contract"]["onnx_status"] == (
        "profiles_640_960_exported_synthetic_parity_passed_"
        "real_image_parity_failed"
    )
    assert provenance["deployment_contract"]["deepstream9_status"] == "not_run"
    assert provenance["training_contract"]["official_r_livit_test_excluded"] is True
    assert provenance["training_contract"]["loaf_20_to_25m_excluded"] is True
    assert all(state is False for state in provenance["acceptance"].values())

    _assert_live_pin(challenger["checkpoint"])
    _assert_live_pin(challenger["provenance"])
    _assert_live_pin(challenger["structural_load_receipt"])
    _assert_live_pin(challenger["framework_profiles_receipt"])
    _assert_live_pin(challenger["export_plan"])
    _assert_live_pin(challenger["real_image_parity_plan"])
    _assert_live_pin(challenger["real_image_parity_receipt"])
    assert real_image["plan"] == challenger["real_image_parity_plan"]
    assert real_image["receipt"] == challenger["real_image_parity_receipt"]
    _assert_live_pin(real_image["schema"])
    _assert_live_pin(real_image["validator"])
    _assert_live_pin(challenger["onnx_batch12_receipt"])
    assert (
        provenance["onnx_batch12_evidence"]["receipt"]
        == challenger["onnx_batch12_receipt"]
    )
    _assert_live_pin(provenance["onnx_batch12_evidence"]["schema"])
    _assert_live_pin(provenance["onnx_batch12_evidence"]["validator"])
    _assert_live_pin(challenger["parser_build_receipt"])
    _assert_live_pin(challenger["parser_artifact"])
    assert (
        provenance["deepstream9_parser_evidence"]["build_receipt"]
        == challenger["parser_build_receipt"]
    )
    assert (
        provenance["deepstream9_parser_evidence"]["artifact"]
        == challenger["parser_artifact"]
    )
    _assert_live_pin(provenance["structural_load_receipt"]["schema"])
    _assert_live_pin(provenance["structural_load_receipt"]["validator"])
    _assert_live_pin(provenance["framework_profiles"]["schema"])
    _assert_live_pin(provenance["framework_profiles"]["validator"])
    _assert_live_pin(provenance["onnx_export_evidence"]["exporter"])
    _assert_live_pin(provenance["onnx_export_evidence"]["receipt_schema"])
    for profile in ("640", "960"):
        assert (
            challenger["onnx_profile_receipts"][profile]
            == provenance["onnx_export_evidence"]["profiles"][profile][
                "receipt"
            ]
        )
        _assert_live_pin(challenger["onnx_profile_receipts"][profile])
        _assert_live_pin(
            provenance["onnx_export_evidence"]["profiles"][profile]["onnx"]
        )
    for key in (
        "license_file",
        "readme",
        "requirements",
        "official_config",
        "official_onnx_exporter",
    ):
        pin = (
            provenance["upstream"]["license"][key]
            if key == "license_file"
            else provenance["upstream"][key]
        )
        _assert_live_pin(pin)


def test_official_test_is_never_a_training_or_calibration_source() -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    primary = value["training_data"]["primary"]
    held_out = value["held_out_evidence"]["rlivit_test"]

    assert primary["official_train_sequences"] == 160
    assert primary["official_test_sequences_excluded_from_training"] == 40
    assert held_out["role"] == "independent_test_never_training_or_threshold_calibration"
    assert "never tune" in value["training_and_export_contract"][
        "calibration_rule"
    ]
