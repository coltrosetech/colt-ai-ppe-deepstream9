from __future__ import annotations

import hashlib
import inspect
import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from validation import person_rtdetrv4_threshold_calibration_r13 as r13
from validation import person_rtdetrv4_threshold_calibration_r13b as lane


R13_IMMUTABLE_FILES = {
    "models/person/export-lanes/rtdetrv4-s-r-livit-person-r11/threshold-calibration-plan-r13.json": (
        6697,
        "1f591392f2ab89378d1e38bbc970a4f19049b81939e5d995fe792d232057474e",
    ),
    "validation/person_rtdetrv4_threshold_calibration_r13.py": (
        92334,
        "cfb813a60a95c0778fa714e1abc1eefafdd0a0235eb2c7d1f455d22194db7d26",
    ),
    "validation/schemas/person-rtdetrv4-threshold-sweep-r13.schema.json": (
        10212,
        "ea878566ad1b8d5b48f54ca0f04b4ffb683f0879ed3472400ef784ee9c62646f",
    ),
    "tests/test_person_rtdetrv4_threshold_calibration_r13.py": (
        18313,
        "dd885c48efd3da497da4b7a19ff15b68478f4c7c574dc8617d89e7d47892a4dc",
    ),
    "docs/person-rtdetrv4-threshold-calibration-r13.md": (
        9331,
        "0a5812ae30154213976bacd40472e42896471bd14f31c573d024c884c72f96cd",
    ),
}


def test_r13_artifacts_remain_byte_identical() -> None:
    for relative, (expected_bytes, expected_sha256) in R13_IMMUTABLE_FILES.items():
        raw = (lane.ROOT / relative).read_bytes()
        assert len(raw) == expected_bytes
        assert hashlib.sha256(raw).hexdigest() == expected_sha256


def test_exact_azure_plus_cpu_inventory_passes_without_session() -> None:
    lane.validate_provider_policy(
        ["AzureExecutionProvider", "CPUExecutionProvider"],
        [],
        session_required=False,
    )


def test_exact_azure_plus_cpu_inventory_with_cpu_only_session_passes() -> None:
    lane.validate_provider_policy(
        ["AzureExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
        session_required=True,
    )


@pytest.mark.parametrize(
    "provider",
    [
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "NvTensorRTRTXExecutionProvider",
        "ROCMExecutionProvider",
        "OpenVINOExecutionProvider",
        "UnknownExecutionProvider",
    ],
)
def test_acceleration_or_unknown_available_provider_fails_closed(
    provider: str,
) -> None:
    with pytest.raises(r13.ThresholdCalibrationR13Error):
        lane.validate_provider_policy(
            ["AzureExecutionProvider", "CPUExecutionProvider", provider],
            [],
            session_required=False,
        )


def test_missing_or_reordered_available_inventory_fails_closed() -> None:
    for providers in (
        ["CPUExecutionProvider"],
        ["CPUExecutionProvider", "AzureExecutionProvider"],
        ["AzureExecutionProvider"],
    ):
        with pytest.raises(
            r13.ThresholdCalibrationR13Error,
            match=r"exact Azure\+CPU pin",
        ):
            lane.validate_provider_policy(
                providers,
                [],
                session_required=False,
            )


@pytest.mark.parametrize(
    "active",
    [
        ["AzureExecutionProvider"],
        ["AzureExecutionProvider", "CPUExecutionProvider"],
        [],
    ],
)
def test_actual_session_must_be_cpu_only(active: list[str]) -> None:
    with pytest.raises(
        r13.ThresholdCalibrationR13Error,
        match="CPUExecutionProvider-only",
    ):
        lane.validate_provider_policy(
            ["AzureExecutionProvider", "CPUExecutionProvider"],
            active,
            session_required=True,
        )


def test_inherited_worker_constructs_and_asserts_cpu_only_session() -> None:
    source = inspect.getsource(lane._R13_INTERNAL_WORKER)
    assert 'providers=["CPUExecutionProvider"]' in source
    assert 'session.get_providers() == ["CPUExecutionProvider"]' in source
    network_boundary = source.index("_install_seccomp_denials(NETWORK_SYSCALLS)")
    ort_import = source.index("import onnxruntime as ort")
    assert network_boundary < ort_import


def test_r13b_preflight_installs_network_seccomp_before_ort_import() -> None:
    source = inspect.getsource(lane.internal_preflight)
    assert source.index("_install_seccomp_denials(base.NETWORK_SYSCALLS)") < source.index(
        "import onnxruntime as ort"
    )
    assert "session_constructed" in source
    assert "azure_provider_active" in source


def test_r13b_plan_is_canonical_and_supersedes_exact_r13() -> None:
    plan_path = lane.ROOT / lane.PLAN_RELATIVE
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    executor_raw = (lane.ROOT / lane.EXECUTOR_RELATIVE).read_bytes()
    executor_pin = {
        "path": lane.EXECUTOR_RELATIVE,
        "bytes": len(executor_raw),
        "sha256": hashlib.sha256(executor_raw).hexdigest(),
    }
    with lane._patched_base():
        lane._validate_plan_r13b(plan, executor_pin=executor_pin)
    assert plan["fingerprint_sha256"] == r13.fingerprint(plan)
    assert plan["supersedes"]["plan"] == lane.R13_PLAN_PIN
    assert plan["supersedes"]["executor"] == lane.R13_EXECUTOR_PIN
    assert plan["supersedes"]["preflight_failure"]["inference"] is False
    assert plan["supersedes"]["preflight_failure"]["gpu"] is False


def test_patched_base_context_restores_immutable_r13_module_globals() -> None:
    original = {
        "plan": r13.PLAN_RELATIVE,
        "executor": r13.EXECUTOR_RELATIVE,
        "schema": r13.SWEEP_SCHEMA_RELATIVE,
        "version": r13.SWEEP_SCHEMA_VERSION,
        "sweeps": r13.SWEEP_RECEIPT_RELATIVE,
    }
    with lane._patched_base():
        assert r13.PLAN_RELATIVE == lane.PLAN_RELATIVE
        assert r13.EXECUTOR_RELATIVE == lane.EXECUTOR_RELATIVE
        assert r13.SWEEP_SCHEMA_RELATIVE == lane.SWEEP_SCHEMA_RELATIVE
        assert r13.SWEEP_SCHEMA_VERSION == lane.SWEEP_SCHEMA_VERSION
        assert r13.SWEEP_RECEIPT_RELATIVE == lane.SWEEP_RECEIPT_RELATIVE
    assert r13.PLAN_RELATIVE == original["plan"]
    assert r13.EXECUTOR_RELATIVE == original["executor"]
    assert r13.SWEEP_SCHEMA_RELATIVE == original["schema"]
    assert r13.SWEEP_SCHEMA_VERSION == original["version"]
    assert r13.SWEEP_RECEIPT_RELATIVE == original["sweeps"]


def test_r13b_schema_separates_available_and_active_providers() -> None:
    schema = json.loads(
        (lane.ROOT / lane.SWEEP_SCHEMA_RELATIVE).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    execution = schema["properties"]["execution"]
    assert "provider" not in execution["properties"]
    assert execution["properties"]["available_providers"]["const"] == [
        "AzureExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert execution["properties"]["active_session_providers"]["const"] == [
        "CPUExecutionProvider"
    ]
    assert execution["properties"]["azure_provider_active"]["const"] is False


def test_r13b_outputs_are_new_but_final_r11_path_is_unchanged() -> None:
    assert lane.SWEEP_RECEIPT_RELATIVE == {
        640: "validation/results/person/export/rtdetrv4-s-r-livit-person-r11/threshold-calibration/full-sweep-640-r13b.json",
        960: "validation/results/person/export/rtdetrv4-s-r-livit-person-r11/threshold-calibration/full-sweep-960-r13b.json",
    }
    assert lane.FINAL_RECEIPT_RELATIVE == r13.FINAL_RECEIPT_RELATIVE


def _fake_accepted() -> SimpleNamespace:
    return SimpleNamespace(
        plan_pin={
            "path": lane.PLAN_RELATIVE,
            "bytes": 1,
            "sha256": "a" * 64,
            "fingerprint_sha256": "b" * 64,
        },
        r11_plan_pin={
            "path": r13.R11_PLAN_RELATIVE,
            "bytes": r13.R11_PLAN_BYTES,
            "sha256": r13.R11_PLAN_SHA256,
            "fingerprint_sha256": r13.R11_PLAN_FINGERPRINT,
        },
        sweep_schema=json.loads(
            (lane.ROOT / lane.SWEEP_SCHEMA_RELATIVE).read_text(encoding="utf-8")
        ),
        r11_schema=json.loads(
            (lane.ROOT / r13.R11_EVIDENCE_SCHEMA_RELATIVE).read_text(
                encoding="utf-8"
            )
        ),
    )


def _zero_detection_worker() -> dict:
    zero = r13.metrics_from_counts(r13.EXPECTED_ANNOTATIONS, 0, 0)
    points = [
        {"threshold": 0.0, "metrics": zero},
        {"threshold": 1.0, "metrics": zero},
    ]
    groups = []
    for index in range(r13.EXPECTED_CAPTURE_GROUPS):
        ground_truth = 126 if index < 6 else 125
        groups.append(
            {
                "capture_group_id": f"group-{index:02d}",
                "images": 15 if index < 20 else 14,
                "metrics": r13.metrics_from_counts(ground_truth, 0, 0),
            }
        )
    height_counts = [543, 543, 543, 543, 542, 542]
    bins = [
        {
            "id": name,
            "lower_exclusive": lower,
            "upper_inclusive": upper,
            "ground_truth": count,
            "tp": 0,
            "fn": count,
            "recall": 0.0,
        }
        for (name, lower, upper), count in zip(r13.HEIGHT_BINS, height_counts)
    ]
    return {
        "runtime": r13.EXPECTED_RUNTIME,
        "batches_executed": 32,
        "finite_output_score_count": r13.EXPECTED_OUTPUT_SCORES,
        "nonfinite_output_score_count": 0,
        "source_image_manifest_sha256": "c" * 64,
        "points": points,
        "points_sha256": hashlib.sha256(r13.canonical_bytes(points)).hexdigest(),
        "selected": {
            "threshold": 0.0,
            "objective": r13.OBJECTIVE,
            "tie_break": ["higher_exact_recall", "lower_threshold"],
            "metrics": zero,
            "threshold_finite": True,
        },
        "capture_group_report": {
            "threshold": 0.0,
            "groups": groups,
            "group_count": 26,
            "micro_totals_match_selected": True,
        },
        "apparent_scale_proxy": {
            "kind": "fine_tuned_person_original_gt_bbox_height_non_metric_distance_proxy",
            "metric_distance_claim": False,
            "threshold": 0.0,
            "height_source": "original_1280x720_ground_truth_bbox_height_pixels",
            "bins": bins,
            "unmatched_false_positives": 0,
        },
        "available_providers": list(lane.EXPECTED_AVAILABLE_PROVIDERS),
        "active_session_providers": list(
            lane.EXPECTED_ACTIVE_SESSION_PROVIDERS
        ),
        "azure_provider_active": False,
    }


def test_r13b_sweep_receipt_validates_and_reports_inventory_separately() -> None:
    accepted = _fake_accepted()
    with lane._patched_base():
        receipt = lane.build_sweep_receipt(
            accepted,
            profile=640,
            worker=_zero_detection_worker(),
            created_at_utc="2026-07-18T00:00:00+00:00",
        )
        lane._validate_sweep_receipt_semantics(
            receipt,
            accepted=accepted,
            profile=640,
        )
    assert receipt["schema_version"] == lane.SWEEP_SCHEMA_VERSION
    assert receipt["receipt_id"].endswith("-r13b")
    assert receipt["execution"]["available_providers"] == [
        "AzureExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert receipt["execution"]["active_session_providers"] == [
        "CPUExecutionProvider"
    ]
    assert receipt["execution"]["azure_provider_active"] is False
    assert "provider" not in receipt["execution"]


def test_r13b_final_receipt_keeps_exact_r11_schema_and_identity() -> None:
    accepted = _fake_accepted()
    worker = _zero_detection_worker()
    sweeps = {}
    with lane._patched_base():
        for profile in r13.PROFILES:
            receipt = lane.build_sweep_receipt(
                accepted,
                profile=profile,
                worker=worker,
                created_at_utc="2026-07-18T00:00:00+00:00",
            )
            sweeps[profile] = (
                receipt,
                {
                    "path": lane.SWEEP_RECEIPT_RELATIVE[profile],
                    "bytes": 123,
                    "sha256": ("d" if profile == 640 else "e") * 64,
                    "fingerprint_sha256": receipt["fingerprint_sha256"],
                },
            )
        final = lane.build_final_receipt(
            accepted,
            sweeps=sweeps,
            created_at_utc="2026-07-18T00:00:01+00:00",
        )
    r13._validate_schema(final, accepted.r11_schema)
    assert final["receipt_id"] == "rtdetrv4-s-r11-threshold-calibration-r13b"
    assert final["stage"] == "threshold_calibration"
    assert final["fingerprint_sha256"] == r13.fingerprint(final)
