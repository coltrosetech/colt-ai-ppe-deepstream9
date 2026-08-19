from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from validation import full_stack_benchmark_contract_v1 as contract


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "validation/plans/deepstream-full-stack-benchmark-v1.json"
PLAN_SCHEMA_PATH = (
    ROOT / "validation/schemas/deepstream-full-stack-benchmark-plan-v1.schema.json"
)
RECEIPT_SCHEMA_PATH = (
    ROOT / "validation/schemas/deepstream-full-stack-benchmark-receipt-v1.schema.json"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _pin(label: str) -> dict:
    return {
        "path": f"validation/results/full-stack-fixture/{label}.txt",
        "size_bytes": 10,
        "sha256": _sha(label),
    }


def _passing_receipt() -> dict:
    runs = []
    for profile in contract.PROFILES:
        sources = []
        for source_id in contract.SOURCE_IDS:
            sources.append(
                {
                    "source_id": source_id,
                    "media_sha256": _sha(f"media-{source_id}"),
                    "delivered_nonduplicated_frames": 9000,
                    "throughput": {"mean_fps": 30.0, "p05_fps": 26.0},
                    "latency": {
                        "coverage_fraction": 1.0,
                        "component_coverage": {
                            "person": 1.0,
                            "pose": 1.0,
                            "ppe": 1.0,
                            "fusion": 1.0,
                        },
                        "sample_count": 300,
                        "p50_ms": 18.0,
                        "p95_ms": 31.0,
                        "p99_ms": 44.0,
                    },
                    "metadata_coverage": {
                        "person": 1.0,
                        "pose": 1.0,
                        "ppe": 1.0,
                        "fusion": 1.0,
                    },
                    "module_observation_counts": {
                        "person_inference_batches": 9000,
                        "pose_inference_batches": 9000,
                        "ppe_inference_batches": 9000,
                        "fusion_frames": 9000,
                    },
                }
            )
        runtime_binding = {
            "image_id": "sha256:" + _sha("image"),
            "parser_sha256": _sha("parser"),
            "fusion_plugin_sha256": _sha("fusion"),
            "authorization_plan_sha256": _sha(f"authorization-{profile}"),
            "person_engine_sha256": _sha(f"person-engine-{profile}"),
            "person_infer_config_sha256": _sha(f"person-config-{profile}"),
            "pose_engine_sha256": _sha(f"pose-engine-{profile}"),
            "pose_infer_config_sha256": _sha(f"pose-config-{profile}"),
            "ppe_engine_sha256": _sha(f"ppe-engine-{profile}"),
            "ppe_infer_config_sha256": _sha(f"ppe-config-{profile}"),
        }
        runs.append(
            {
                "model_input": profile,
                "status": "measurement_complete",
                "deepstream_version": "9.0.0",
                "modules_enabled_together": ["person", "pose", "ppe"],
                "source_count": 12,
                "source_set_fingerprint_sha256": contract.source_set_fingerprint(sources),
                "started_at_utc": "2026-07-18T02:00:00Z",
                "finished_at_utc": "2026-07-18T02:05:15Z",
                "elapsed_ms": 315000,
                "measurement": {
                    "warmup_seconds": 15,
                    "steady_state_seconds": 300,
                    "foreground_gpu_lease_held_for_full_process": True,
                },
                "gpu_lease_receipt_pin": _pin(f"gpu-lease-{profile}"),
                "runtime_binding": runtime_binding,
                "aggregate_throughput": {
                    "mean_fps": 360.0,
                    "p05_fps": 312.0,
                    "per_stream_mean_fps": 30.0,
                    "per_stream_p05_fps": 26.0,
                    "sample_count": 60,
                },
                "per_source": sources,
                "telemetry": {
                    "gpu_sample_count": 300,
                    "gpu_sample_coverage": 1.0,
                    "thermal_sample_count": 300,
                    "thermal_sample_coverage": 1.0,
                    "max_gpu_temperature_c": 72.0,
                    "max_platform_temperature_c": 83.0,
                    "thermal_slowdown_sample_count": 0,
                    "power_brake_sample_count": 0,
                },
                "runtime_health": {
                    "xid_count": 0,
                    "oom_count": 0,
                    "fatal_count": 0,
                    "unexpected_restart_count": 0,
                    "engine_fallback_count": 0,
                    "module_disable_count": 0,
                    "metadata_schema_error_count": 0,
                },
                "aggregate_metadata": {
                    "person_object_count": 1200,
                    "pose_keypoint_count": 20400,
                    "ppe_observation_count": 900,
                    "fused_person_record_count": 1100,
                },
                "evidence_pins": [
                    {"role": role, "pin": _pin(f"{profile}-{role}")}
                    for role in contract.REQUIRED_EVIDENCE_ROLES
                ],
            }
        )
    receipt = {
        "schema_version": contract.RECEIPT_SCHEMA_VERSION,
        "state": "measurement_complete",
        "result_kind": "measured_full_stack",
        "execution_id": "fixture-three-module-001",
        "created_at_utc": "2026-07-18T02:10:00Z",
        "plan_pin": _pin("benchmark-plan"),
        "plan_fingerprint_sha256": _sha("plan-fingerprint"),
        "runs": runs,
        "acceptance": {
            "status": "pass",
            "minimum_output_fps_per_source": 25.0,
            "failure_reasons": [],
        },
        "fingerprint_sha256": "0" * 64,
    }
    receipt["fingerprint_sha256"] = contract.fingerprint(receipt)
    return receipt


def _refingerprint(document: dict) -> dict:
    document["fingerprint_sha256"] = contract.fingerprint(document)
    return document


def test_plan_and_receipt_schemas_are_valid_draft_2020_12() -> None:
    for path in (PLAN_SCHEMA_PATH, RECEIPT_SCHEMA_PATH):
        schema = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)


def test_repository_plan_is_schema_valid_semantically_blocked_and_replayable() -> None:
    plan = contract.load_json(PLAN_PATH)
    assert contract.validate_plan(plan) is plan
    replay = contract.replay_plan_files(plan, ROOT)
    assert replay["status"] == "pass"
    assert replay["execution_ready"] is False
    assert replay["replayed_unique_pins"] >= 26
    assert replay["safety"] == {
        "docker_called": False,
        "gpu_queried": False,
        "gpu_process_started": False,
        "inference_started": False,
    }


def test_plan_has_exact_12_distinct_sources_and_two_separate_300s_profiles() -> None:
    plan = contract.validate_plan(contract.load_json(PLAN_PATH))
    assert plan["execution"]["profile_order"] == [640, 960]
    assert plan["execution"]["separate_process_per_profile"] is True
    assert plan["execution"]["measurement_seconds"] == 300
    assert len(plan["sources"]) == 12
    assert len({item["video_type"] for item in plan["sources"]}) == 12
    assert len({item["media_pin"]["sha256"] for item in plan["sources"]}) == 12
    views = {view for item in plan["sources"] for view in item["view_types"]}
    assert {"medium_close", "overhead_security_camera"} <= views


def test_plan_pins_ds9_image_parser_fusion_and_current_person_artifacts() -> None:
    plan = contract.validate_plan(contract.load_json(PLAN_PATH))
    assert plan["runtime"]["deepstream_version"] == "9.0.0"
    assert plan["runtime"]["tensorrt_version"] == "10.14.1.48"
    assert plan["runtime"]["image"]["image_id"].startswith("sha256:ced1b591")
    assert plan["runtime"]["image"]["parser_sha256"] == (
        "2aa44a3395047ae371bee857476b1e78b438776c8a6b9643a055a16a0f15a7ae"
    )
    assert plan["runtime"]["fusion"]["publication_id"].endswith("-r3")
    for profile in plan["profiles"]:
        person, pose, ppe = profile["modules"]
        assert person["engine_pin"] and person["infer_config_pin"]
        assert pose["engine_pin"] is None and pose["infer_config_pin"] is None
        assert ppe["engine_pin"] is None and ppe["infer_config_pin"] is None


def test_person_only_baseline_and_full_stack_estimate_cannot_be_results() -> None:
    context = contract.validate_plan(contract.load_json(PLAN_PATH))["performance_context"]
    baseline = context["person_only_baseline"]
    assert baseline["classification"] == "person_only_measured_baseline"
    assert baseline["eligible_as_full_stack_result"] is False
    assert [item["aggregate_mean_fps"] for item in baseline["profiles"]] == [
        464.733,
        305.799,
    ]
    estimate = context["full_stack_estimate"]
    assert estimate["classification"] == "estimate_not_measured"
    assert estimate["eligible_as_result"] is False
    assert "Exact pose/PPE TensorRT engines" in estimate["rationale"]


def test_plan_rejects_readiness_overclaim_even_after_refingerprint() -> None:
    plan = contract.load_json(PLAN_PATH)
    plan["readiness"]["blockers"] = []
    plan["readiness"]["execution_ready"] = True
    plan["state"] = "ready"
    _refingerprint(plan)
    with pytest.raises(contract.BenchmarkContractError, match="blockers"):
        contract.validate_plan(plan)


def test_plan_rejects_duplicate_source_even_after_refingerprint() -> None:
    plan = contract.load_json(PLAN_PATH)
    plan["sources"][1]["media_pin"] = copy.deepcopy(plan["sources"][0]["media_pin"])
    plan["sources"][1]["plan_uri"] = plan["sources"][0]["plan_uri"]
    _refingerprint(plan)
    with pytest.raises(contract.BenchmarkContractError, match="distinct"):
        contract.validate_plan(plan)


def test_synthetic_measured_full_stack_receipt_passes_schema_and_semantics() -> None:
    receipt = _passing_receipt()
    assert contract.validate_receipt(receipt, require_pass=True) is receipt


def test_receipt_rejects_estimate_or_person_only_result_kind() -> None:
    for result_kind in ("estimate_not_measured", "person_only_measured_baseline"):
        receipt = _passing_receipt()
        receipt["result_kind"] = result_kind
        _refingerprint(receipt)
        with pytest.raises(contract.BenchmarkContractError, match="schema validation|result"):
            contract.validate_receipt(receipt)


def test_receipt_rejects_aggregate_arithmetic_overclaim() -> None:
    receipt = _passing_receipt()
    receipt["runs"][0]["aggregate_throughput"]["mean_fps"] = 999.0
    _refingerprint(receipt)
    with pytest.raises(contract.BenchmarkContractError, match="aggregate mean FPS"):
        contract.validate_receipt(receipt)


def test_receipt_rejects_delivered_frame_fps_arithmetic_overclaim() -> None:
    receipt = _passing_receipt()
    receipt["runs"][0]["per_source"][0]["delivered_nonduplicated_frames"] = 1
    _refingerprint(receipt)
    with pytest.raises(contract.BenchmarkContractError, match="delivered-frame FPS"):
        contract.validate_receipt(receipt)


def test_receipt_rejects_metadata_coverage_pass_overclaim() -> None:
    receipt = _passing_receipt()
    receipt["runs"][0]["per_source"][3]["metadata_coverage"]["pose"] = 0.8
    _refingerprint(receipt)
    with pytest.raises(contract.BenchmarkContractError, match="acceptance status"):
        contract.validate_receipt(receipt)


def test_receipt_can_honestly_record_failed_health_measurement() -> None:
    receipt = _passing_receipt()
    receipt["runs"][1]["runtime_health"]["xid_count"] = 1
    reasons = contract._receipt_failure_reasons(receipt)
    receipt["acceptance"] = {
        "status": "fail",
        "minimum_output_fps_per_source": 25.0,
        "failure_reasons": reasons,
    }
    _refingerprint(receipt)
    contract.validate_receipt(receipt)
    with pytest.raises(contract.BenchmarkContractError, match="did not pass"):
        contract.validate_receipt(receipt, require_pass=True)


def test_receipt_requires_exact_output_evidence_roles() -> None:
    receipt = _passing_receipt()
    receipt["runs"][0]["evidence_pins"].pop()
    _refingerprint(receipt)
    with pytest.raises(contract.BenchmarkContractError, match="schema validation"):
        contract.validate_receipt(receipt)


def test_validator_module_has_no_process_or_gpu_execution_imports() -> None:
    source = (ROOT / "validation/full_stack_benchmark_contract_v1.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "subprocess" not in imported
    assert "pynvml" not in imported
    assert "docker" not in imported


def test_cli_default_is_a_blocked_cpu_only_dry_run() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "validation.full_stack_benchmark_contract_v1"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "blocked_missing_runtime_artifacts"
    assert result["execution_ready"] is False
    assert result["sources"] == 12
    assert result["measurement_seconds_per_profile"] == 300
