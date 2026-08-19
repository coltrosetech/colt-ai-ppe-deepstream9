from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from admin import validation as admin_validation
from admin.app import app
from validation import product_readiness as product
from validation import product_readiness_replay as replay


ROOT = Path(__file__).resolve().parents[1]


def _copy(root: Path, relative: str) -> None:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, destination)


def _fixture_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    results = project / "validation/results"
    for relative in (
        "admin/validation.py",
        "admin/static/index.html",
        "admin/app.py",
        "deepstream/config.py",
        "validation/results/campaign-report/report.json",
        "validation/results/rlivit/current/status.json",
        "validation/results/rlivit/threshold-sweep/threshold-sweep.json",
        "validation/results/rlivit/threshold-sweep/receipt.json",
        "validation/results/ppe/qualitative-source-audit.json",
    ):
        _copy(project, relative)
    return project, results


def _schema() -> dict:
    return json.loads(
        (ROOT / "validation/schemas/product-readiness-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )


def _raw_replay_schema() -> dict:
    return json.loads(
        (
            ROOT
            / "validation/schemas/product-acceptance-raw-replay-v1.schema.json"
        ).read_text(encoding="utf-8")
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _resign(value: dict) -> None:
    value.pop("fingerprint_sha256", None)
    value["fingerprint_sha256"] = hashlib.sha256(
        product._canonical_bytes(value)
    ).hexdigest()


def _check(receipt: dict, check_id: str) -> dict:
    return next(item for item in receipt["checks"] if item["id"] == check_id)


def _rewrite_raw_replay(project: Path, gate_id: str, receipt: dict, raw: dict) -> None:
    contract = product.RECEIPT_CONTRACTS[gate_id]
    _resign(raw)
    raw_path = project / contract["raw_replay_path"]
    _write_json(raw_path, raw)
    _check(receipt, "semantic_raw_replay")["evidence_pins"] = [
        {
            "path": contract["raw_replay_path"].as_posix(),
            "size_bytes": raw_path.stat().st_size,
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        }
    ]
    _resign(receipt)


def _receipt(
    project: Path,
    gate_id: str,
    *,
    evidence_path: str = "evidence/proof.bin",
) -> dict:
    contract = product.RECEIPT_CONTRACTS[gate_id]
    evidence = project / evidence_path
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_bytes(b"hash-bound-acceptance-evidence\n")
    pin = {
        "path": evidence_path,
        "size_bytes": evidence.stat().st_size,
        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }
    kind = contract["kind"]
    measurements: dict = {
        "profiles": [
            {"model_input": 640, "aggregate_fps": 300.0, "per_stream_fps": 25.0},
            {"model_input": 960, "aggregate_fps": 240.0, "per_stream_fps": 20.0},
        ]
    }
    policy: dict = {
        "schema_version": f"deepsafe.{kind}-product-acceptance-policy/v1",
        "policy_id": f"owner-approved-{kind}-v1",
        "status": "approved",
        "approval_strictly_before_campaign": True,
        "profiles": [
            {
                "model_input": 640,
                "minimum_aggregate_fps": 240.0,
                "minimum_per_stream_fps": 20.0,
            },
            {
                "model_input": 960,
                "minimum_aggregate_fps": 180.0,
                "minimum_per_stream_fps": 15.0,
            },
        ],
    }
    if kind == "pose":
        policy["quality"] = {
            "metric_family": "OKS",
            "minimum_ground_truth_keypoints": 50,
            "minimum_metric_value": 0.70,
        }
        measurements.update(
            {
                "gt_metric_family": "OKS",
                "ground_truth_keypoints": 100,
                "ground_truth_metric_value": 0.75,
            }
        )
        area = 10_000.0
        sigma = replay.COCO_KEYPOINT_SIGMAS[0]
        # pycocotools uses vars=(2*sigma)^2 and divides by 2*area*vars.
        offset = math.sqrt(
            -2.0 * area * (2.0 * sigma) ** 2 * math.log(0.75)
        )
        raw_quality = {
            "metric_family": "OKS",
            "keypoints": [
                {
                    "sample_id": f"pose-sample-{index}",
                    "person_id": f"pose-person-{index}",
                    "keypoint_id": 0,
                    "visibility": 2,
                    "image_width_px": 640,
                    "image_height_px": 640,
                    "ground_truth_x_px": 100.0,
                    "ground_truth_y_px": 100.0,
                    "prediction_present": True,
                    "prediction_x_px": 100.0 + offset,
                    "prediction_y_px": 100.0,
                    "object_area_px2": area,
                }
                for index in range(100)
            ],
        }
    elif kind == "ppe":
        policy["quality"] = {
            "minimum_ground_truth_instances_by_class": {
                "helmet": 80,
                "hi_vis": 80,
            },
            "minimum_attribute_metrics": {
                "helmet": {"minimum_precision": 0.90, "minimum_recall": 0.95},
                "hi_vis": {"minimum_precision": 0.90, "minimum_recall": 0.95},
            },
            "minimum_ground_truth_events": 10,
            "minimum_event_precision": 0.90,
            "minimum_event_recall": 0.95,
            "maximum_event_latency_p95_ms": 500.0,
            "maximum_false_transition_rate": 0.05,
            "maximum_false_safe_rate": 0.01,
        }
        measurements.update(
            {
                "classes": ["helmet", "hi_vis"],
                "ground_truth_instances_by_class": {"helmet": 100, "hi_vis": 100},
                "attribute_metrics": {
                    "helmet": {
                        "precision": 0.96,
                        "recall": 0.96,
                        "f1": 0.96,
                    },
                    "hi_vis": {
                        "precision": 0.95,
                        "recall": 0.95,
                        "f1": 0.95,
                    },
                },
                "temporal": {
                    "ground_truth_events": 20,
                    "event_precision": 0.95,
                    "event_recall": 0.95,
                    "event_f1": 0.95,
                    "event_latency_p95_ms": 450.0,
                    "false_transition_rate": 0.02,
                    "false_safe_rate": 0.005,
                },
            }
        )
        attribute_observations = []
        for label, tp, fp, fn in (
            ("helmet", 96, 4, 4),
            ("hi_vis", 95, 5, 5),
        ):
            outcomes = (
                [(True, True)] * tp
                + [(False, True)] * fp
                + [(True, False)] * fn
            )
            attribute_observations.extend(
                {
                    "sample_id": f"ppe-{label}-sample-{index}",
                    "person_id": f"ppe-{label}-person-{index}",
                    "attribute": label,
                    "ground_truth_present": ground_truth_present,
                    "predicted_present": predicted_present,
                }
                for index, (ground_truth_present, predicted_present) in enumerate(
                    outcomes
                )
            )
        raw_quality = {
            "attribute_observations": attribute_observations,
            "temporal": {
                "events": [
                    {
                        "event_id": f"event-tp-{index}",
                        "outcome": "tp",
                        "latency_ms": 450.0,
                    }
                    for index in range(19)
                ]
                + [{"event_id": "event-fp-0", "outcome": "fp", "latency_ms": None}]
                + [{"event_id": "event-fn-0", "outcome": "fn", "latency_ms": None}],
                "transition_opportunities": [
                    {
                        "opportunity_id": f"transition-{index}",
                        "false_transition": index < 2,
                    }
                    for index in range(100)
                ],
                "safety_opportunities": [
                    {
                        "opportunity_id": f"safety-{index}",
                        "false_safe": index < 1,
                    }
                    for index in range(200)
                ],
            },
        }
    else:
        policy["quality"] = {
            "minimum_metadata_fusion_match_rate": 0.99,
            "maximum_unmatched_metadata_rate": 0.01,
            "maximum_dropped_frame_rate": 0.01,
            "maximum_fatal_error_count": 0,
        }
        measurements.update(
            {
                "modules": ["person", "pose", "ppe"],
                "metadata_fusion_verified": True,
                "metadata_fusion_match_rate": 0.995,
                "unmatched_metadata_rate": 0.005,
                "dropped_frame_rate": 1 / 162_001,
                "fatal_error_count": 0,
            }
        )
        raw_quality = {
            "modules_enabled": ["person", "pose", "ppe"],
            "fusion_observations": [
                {
                    "model_input": 640 if index % 2 == 0 else 960,
                    "source_id": (index // 2) % 12,
                    "frame_id": index,
                    "person_id": f"fusion-person-{index}",
                    "person_metadata_present": True,
                    "pose_metadata_present": index < 995,
                    "ppe_metadata_present": index < 995,
                }
                for index in range(1000)
            ],
        }
    policy["fingerprint_sha256"] = hashlib.sha256(
        product._canonical_bytes(policy)
    ).hexdigest()
    policy_path = project / contract["policy_path"]
    _write_json(policy_path, policy)
    policy_pin = {
        "path": contract["policy_path"].as_posix(),
        "size_bytes": policy_path.stat().st_size,
        "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }
    raw_replay = {
        "schema_version": replay.RAW_REPLAY_SCHEMA,
        "gate_id": gate_id,
        "kind": kind,
        "runtime": {
            "deepstream_version": "9.0",
            "gpu_inference_executed": True,
        },
        "profiles": [
            {
                "model_input": 640,
                "elapsed_ms": 300_000,
                "stream_frame_counts": [7_500] * 12,
                "expected_stream_frame_counts": [7_501] + [7_500] * 11,
                "fatal_error_count": 0,
            },
            {
                "model_input": 960,
                "elapsed_ms": 300_000,
                "stream_frame_counts": [6_000] * 12,
                "expected_stream_frame_counts": [6_000] * 12,
                "fatal_error_count": 0,
            },
        ],
        "quality": raw_quality,
    }
    _resign(raw_replay)
    raw_path = project / contract["raw_replay_path"]
    _write_json(raw_path, raw_replay)
    raw_pin = {
        "path": contract["raw_replay_path"].as_posix(),
        "size_bytes": raw_path.stat().st_size,
        "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    receipt = {
        "schema_version": contract["schema"],
        "gate_id": gate_id,
        "status": "complete",
        "acceptance_status": "pass",
        "scope": {
            "model_input_sizes": [640, 960],
            "simulated_streams": 12,
            "duration_seconds_per_profile": 300,
        },
        "runtime": {
            "deepstream_version": "9.0",
            "gpu_inference_executed": True,
        },
        "acceptance_policy": policy,
        "measurements": measurements,
        "checks": [
            {
                "id": check_id,
                "status": "pass",
                "evidence_pins": [
                    policy_pin
                    if check_id == "owner_approved_acceptance_policy"
                    else raw_pin
                    if check_id == "semantic_raw_replay"
                    else pin
                ],
            }
            for check_id in contract["checks"]
        ],
    }
    receipt["fingerprint_sha256"] = hashlib.sha256(
        product._canonical_bytes(receipt)
    ).hexdigest()
    return receipt


def test_report_is_deterministic_schema_valid_and_fail_closed(tmp_path):
    project, results = _fixture_project(tmp_path)

    first = product.build_product_readiness(
        project_root=project, results_root=results
    )
    second = product.build_product_readiness(
        project_root=project, results_root=results
    )

    assert first == second
    assert product._canonical_fingerprint_valid(first)
    schema = _schema()
    admin_validation._validate_schema_node(first, schema, schema)
    assert "generated_at" not in json.dumps(first)
    assert first["decision"]["status"] == "not_ready"
    assert first["decision"]["ready"] is False
    assert first["decision"]["final_claim_allowed"] is False
    assert first["person_validation_stage"]["product_acceptance_equivalent"] is False
    gates = {gate["id"]: gate for gate in first["required_gates"]}
    assert gates["pose_ds9_gt_and_capacity"]["state"] == "missing"
    assert gates["ppe_ds9_gt_temporal_and_capacity"]["state"] == "missing"
    assert gates["three_module_full_stack_capacity"]["state"] == "missing"
    assert gates["single_admin_visibility_and_readiness"]["state"] == "pass"
    diagnostic = next(
        item
        for item in first["optional_hardening"]
        if item["id"] == "rlivit_threshold_sweep_diagnostic"
    )
    assert diagnostic["state"] == "complete"
    assert diagnostic["required_for_product_readiness"] is False
    assert diagnostic["acceptance_effect"] == "none"
    ppe_diagnostic = next(
        item
        for item in first["optional_hardening"]
        if item["id"] == "ppe_qualitative_source_diagnostic"
    )
    assert ppe_diagnostic["state"] == "complete"
    assert ppe_diagnostic["required_for_product_readiness"] is False
    assert gates["ppe_ds9_gt_temporal_and_capacity"]["state"] == "missing"


@pytest.mark.parametrize(
    "gate_id",
    [
        "pose_ds9_gt_and_capacity",
        "ppe_ds9_gt_temporal_and_capacity",
        "three_module_full_stack_capacity",
    ],
)
def test_acceptance_receipt_requires_live_self_hash_and_pins(tmp_path, gate_id):
    project = tmp_path / "project"
    results = project / "validation/results"
    receipt = _receipt(project, gate_id)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    _write_json(results / contract["path"], receipt)

    reader = product.EvidenceReader(project, results)
    accepted = product._receipt_gate(reader, gate_id)
    assert accepted["passed"] is True
    assert accepted["state"] == "pass"

    (project / "evidence/proof.bin").write_bytes(b"tampered\n")
    rejected = product._receipt_gate(
        product.EvidenceReader(project, results), gate_id
    )
    assert rejected["passed"] is False
    assert rejected["state"] == "invalid"
    assert any(
        check["state"] == "invalid" for check in rejected["component_checks"]
    )


@pytest.mark.parametrize(
    "gate_id",
    [
        "pose_ds9_gt_and_capacity",
        "ppe_ds9_gt_temporal_and_capacity",
        "three_module_full_stack_capacity",
    ],
)
def test_normalized_raw_replay_fixture_matches_schema_and_receipt(
    tmp_path, gate_id
):
    project = tmp_path / "project"
    receipt = _receipt(project, gate_id)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    raw = json.loads(
        (project / contract["raw_replay_path"]).read_text(encoding="utf-8")
    )
    schema = _raw_replay_schema()

    admin_validation._validate_schema_node(raw, schema, schema)
    result = replay.replay_receipt(
        kind=contract["kind"],
        gate_id=gate_id,
        receipt=receipt,
        raw_evidence=raw,
    )

    assert result.valid is True
    assert result.errors == ()
    assert replay._equivalent(
        result.recomputed_measurements, receipt["measurements"]
    )


@pytest.mark.parametrize(
    ("gate_id", "mutate"),
    [
        (
            "pose_ds9_gt_and_capacity",
            lambda receipt: receipt["measurements"].__setitem__(
                "ground_truth_metric_value", 0.80
            ),
        ),
        (
            "ppe_ds9_gt_temporal_and_capacity",
            lambda receipt: receipt["measurements"]["temporal"].__setitem__(
                "false_safe_rate", 0.0
            ),
        ),
        (
            "three_module_full_stack_capacity",
            lambda receipt: (
                receipt["measurements"].__setitem__(
                    "metadata_fusion_match_rate", 0.996
                ),
                receipt["measurements"].__setitem__(
                    "unmatched_metadata_rate", 0.004
                ),
            ),
        ),
    ],
)
def test_semantic_replay_rejects_policy_passing_self_declared_measurements(
    tmp_path, gate_id, mutate
):
    project = tmp_path / "project"
    results = project / "validation/results"
    receipt = _receipt(project, gate_id)
    mutate(receipt)
    assert product._receipt_measurements_valid(
        product.RECEIPT_CONTRACTS[gate_id]["kind"], receipt
    )
    _resign(receipt)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    _write_json(results / contract["path"], receipt)

    gate = product._receipt_gate(product.EvidenceReader(project, results), gate_id)

    semantic = next(
        item
        for item in gate["component_checks"]
        if item["id"] == "semantic_raw_replay"
    )
    assert semantic["passed"] is False
    assert semantic["state"] == "invalid"
    assert gate["passed"] is False


def test_semantic_replay_rejects_arbitrary_but_live_proof_pin(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    gate_id = "pose_ds9_gt_and_capacity"
    receipt = _receipt(project, gate_id)
    _check(receipt, "semantic_raw_replay")["evidence_pins"] = copy.deepcopy(
        _check(receipt, "model_weights")["evidence_pins"]
    )
    _resign(receipt)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    _write_json(results / contract["path"], receipt)

    gate = product._receipt_gate(product.EvidenceReader(project, results), gate_id)

    semantic = next(
        item
        for item in gate["component_checks"]
        if item["id"] == "semantic_raw_replay"
    )
    assert semantic["passed"] is False
    assert gate["passed"] is False


def test_semantic_replay_rejects_rewritten_raw_counters_with_stale_claim(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    gate_id = "pose_ds9_gt_and_capacity"
    receipt = _receipt(project, gate_id)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    raw_path = project / contract["raw_replay_path"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    for keypoint in raw["quality"]["keypoints"]:
        keypoint["prediction_x_px"] = (
            keypoint["ground_truth_x_px"] + keypoint["prediction_x_px"]
        ) / 2.0
    _rewrite_raw_replay(project, gate_id, receipt, raw)
    _write_json(results / contract["path"], receipt)

    gate = product._receipt_gate(product.EvidenceReader(project, results), gate_id)

    semantic = next(
        item
        for item in gate["component_checks"]
        if item["id"] == "semantic_raw_replay"
    )
    assert semantic["passed"] is False
    assert gate["passed"] is False


def test_semantic_replay_rejects_boolean_profile_counter_type_confusion(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    gate_id = "three_module_full_stack_capacity"
    receipt = _receipt(project, gate_id)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    raw_path = project / contract["raw_replay_path"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["profiles"][0]["fatal_error_count"] = False
    _rewrite_raw_replay(project, gate_id, receipt, raw)
    _write_json(results / contract["path"], receipt)

    gate = product._receipt_gate(product.EvidenceReader(project, results), gate_id)

    semantic = next(
        item
        for item in gate["component_checks"]
        if item["id"] == "semantic_raw_replay"
    )
    assert semantic["passed"] is False
    assert gate["passed"] is False


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda raw: raw["profiles"][0].__setitem__(
                "elapsed_ms", 10**1_000
            ),
            "profile_counters",
        ),
        (
            lambda raw: raw["quality"]["keypoints"][0].__setitem__(
                "object_area_px2", 1.0e308
            ),
            "quality_counters",
        ),
    ],
)
def test_semantic_replay_fails_closed_on_numeric_overflow_inputs(
    tmp_path, mutate, expected_error
):
    project = tmp_path / "project"
    gate_id = "pose_ds9_gt_and_capacity"
    receipt = _receipt(project, gate_id)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    raw = json.loads(
        (project / contract["raw_replay_path"]).read_text(encoding="utf-8")
    )
    mutate(raw)
    _resign(raw)

    result = replay.replay_receipt(
        kind=contract["kind"],
        gate_id=gate_id,
        receipt=receipt,
        raw_evidence=raw,
    )

    assert result.valid is False
    assert result.errors == (expected_error,)


@pytest.mark.parametrize(
    ("target", "replacement"),
    [("attribute", []), ("outcome", [])],
)
def test_semantic_replay_rejects_unhashable_ppe_enum_values_without_crashing(
    tmp_path, target, replacement
):
    project = tmp_path / "project"
    gate_id = "ppe_ds9_gt_temporal_and_capacity"
    receipt = _receipt(project, gate_id)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    raw = json.loads(
        (project / contract["raw_replay_path"]).read_text(encoding="utf-8")
    )
    if target == "attribute":
        raw["quality"]["attribute_observations"][0][target] = replacement
    else:
        raw["quality"]["temporal"]["events"][0][target] = replacement
    _resign(raw)

    result = replay.replay_receipt(
        kind=contract["kind"],
        gate_id=gate_id,
        receipt=receipt,
        raw_evidence=raw,
    )

    assert result.valid is False
    assert result.errors == ("quality_counters",)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["profiles"][0].__setitem__("model_input", 640.0),
        lambda raw: raw["profiles"][0].__setitem__("fatal_error_count", 0.0),
    ],
)
def test_raw_replay_schema_rejects_float_for_integer_identity_fields(
    tmp_path, mutate
):
    project = tmp_path / "project"
    gate_id = "pose_ds9_gt_and_capacity"
    _receipt(project, gate_id)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    raw = json.loads(
        (project / contract["raw_replay_path"]).read_text(encoding="utf-8")
    )
    mutate(raw)
    _resign(raw)

    with pytest.raises(ValueError, match="type mismatch"):
        admin_validation._validate_schema_node(
            raw, _raw_replay_schema(), _raw_replay_schema()
        )


def test_huge_owner_policy_integer_fails_closed_without_overflow(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    gate_id = "pose_ds9_gt_and_capacity"
    receipt = _receipt(project, gate_id)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    policy = receipt["acceptance_policy"]
    policy["profiles"][0]["minimum_aggregate_fps"] = 10**1_000
    _resign(policy)
    policy_path = project / contract["policy_path"]
    _write_json(policy_path, policy)
    _check(receipt, "owner_approved_acceptance_policy")["evidence_pins"] = [
        {
            "path": contract["policy_path"].as_posix(),
            "size_bytes": policy_path.stat().st_size,
            "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        }
    ]
    _resign(receipt)
    _write_json(results / contract["path"], receipt)

    gate = product._receipt_gate(product.EvidenceReader(project, results), gate_id)

    assert gate["passed"] is False
    assert gate["state"] == "invalid"


def test_policy_is_evaluated_against_recomputed_not_tolerance_rounded_latency(
    tmp_path,
):
    project = tmp_path / "project"
    results = project / "validation/results"
    gate_id = "ppe_ds9_gt_temporal_and_capacity"
    receipt = _receipt(project, gate_id)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    policy = receipt["acceptance_policy"]
    policy["quality"]["maximum_event_latency_p95_ms"] = 450.0
    _resign(policy)
    policy_path = project / contract["policy_path"]
    _write_json(policy_path, policy)
    _check(receipt, "owner_approved_acceptance_policy")["evidence_pins"] = [
        {
            "path": contract["policy_path"].as_posix(),
            "size_bytes": policy_path.stat().st_size,
            "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        }
    ]
    raw_path = project / contract["raw_replay_path"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    for event in raw["quality"]["temporal"]["events"]:
        if event["outcome"] == "tp":
            event["latency_ms"] = 450.0000000005
    _rewrite_raw_replay(project, gate_id, receipt, raw)
    _write_json(results / contract["path"], receipt)

    replayed = replay.replay_receipt(
        kind=contract["kind"],
        gate_id=gate_id,
        receipt=receipt,
        raw_evidence=raw,
    )
    gate = product._receipt_gate(product.EvidenceReader(project, results), gate_id)

    assert replayed.valid is True
    assert gate["passed"] is False
    assert gate["state"] == "invalid"


def test_receipt_cannot_claim_pass_with_wrong_scope_or_fingerprint(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    gate_id = "pose_ds9_gt_and_capacity"
    receipt = _receipt(project, gate_id)
    receipt["scope"]["simulated_streams"] = 11
    contract = product.RECEIPT_CONTRACTS[gate_id]
    _write_json(results / contract["path"], receipt)

    gate = product._receipt_gate(product.EvidenceReader(project, results), gate_id)

    assert gate["passed"] is False
    assert gate["state"] == "invalid"
    envelope = gate["component_checks"][0]
    assert envelope["id"] == "receipt_envelope_and_measurements"
    assert envelope["passed"] is False


@pytest.mark.parametrize(
    ("gate_id", "mutate"),
    [
        (
            "pose_ds9_gt_and_capacity",
            lambda receipt: receipt["measurements"].__setitem__(
                "ground_truth_metric_value", 0.0
            ),
        ),
        (
            "ppe_ds9_gt_temporal_and_capacity",
            lambda receipt: receipt["measurements"]["temporal"].__setitem__(
                "false_safe_rate", 0.5
            ),
        ),
        (
            "three_module_full_stack_capacity",
            lambda receipt: receipt["measurements"].__setitem__(
                "metadata_fusion_match_rate", 0.1
            ),
        ),
    ],
)
def test_receipt_measurements_must_meet_preapproved_policy(
    tmp_path, gate_id, mutate
):
    project = tmp_path / "project"
    results = project / "validation/results"
    receipt = _receipt(project, gate_id)
    mutate(receipt)
    _resign(receipt)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    _write_json(results / contract["path"], receipt)

    gate = product._receipt_gate(product.EvidenceReader(project, results), gate_id)

    assert gate["passed"] is False
    assert gate["state"] == "invalid"
    assert gate["component_checks"][0]["id"] == "receipt_envelope_and_measurements"
    assert gate["component_checks"][0]["passed"] is False


def test_receipt_requires_live_pre_run_policy_fingerprint(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    gate_id = "pose_ds9_gt_and_capacity"
    receipt = _receipt(project, gate_id)
    receipt["acceptance_policy"]["quality"]["minimum_metric_value"] = 0.01
    _resign(receipt)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    _write_json(results / contract["path"], receipt)

    gate = product._receipt_gate(product.EvidenceReader(project, results), gate_id)

    assert gate["passed"] is False
    assert gate["state"] == "invalid"


def test_embedded_self_declared_policy_cannot_replace_fixed_live_policy(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    gate_id = "pose_ds9_gt_and_capacity"
    receipt = _receipt(project, gate_id)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    (project / contract["policy_path"]).unlink()
    _write_json(results / contract["path"], receipt)

    gate = product._receipt_gate(product.EvidenceReader(project, results), gate_id)

    assert gate["passed"] is False
    assert gate["state"] == "invalid"
    policy_check = next(
        item
        for item in gate["component_checks"]
        if item["id"] == "owner_approved_acceptance_policy"
    )
    assert policy_check["passed"] is False


def test_ppe_receipt_cannot_omit_accuracy_and_false_safe_metrics(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    gate_id = "ppe_ds9_gt_temporal_and_capacity"
    receipt = _receipt(project, gate_id)
    receipt["measurements"].pop("attribute_metrics")
    receipt["measurements"]["temporal"].pop("false_safe_rate")
    _resign(receipt)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    _write_json(results / contract["path"], receipt)

    gate = product._receipt_gate(product.EvidenceReader(project, results), gate_id)

    assert gate["passed"] is False
    assert gate["state"] == "invalid"


def test_full_stack_fatal_error_count_rejects_boolean_type_confusion(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    gate_id = "three_module_full_stack_capacity"
    receipt = _receipt(project, gate_id)
    receipt["measurements"]["fatal_error_count"] = False
    _resign(receipt)
    contract = product.RECEIPT_CONTRACTS[gate_id]
    _write_json(results / contract["path"], receipt)

    gate = product._receipt_gate(product.EvidenceReader(project, results), gate_id)

    assert gate["passed"] is False
    assert gate["state"] == "invalid"


def test_intermediate_symlink_is_rejected_for_receipt_and_pinned_evidence(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    target = project / "evidence-target"
    target.mkdir(parents=True)
    proof = target / "proof.bin"
    proof.write_bytes(b"proof\n")
    (project / "evidence-link").symlink_to(target, target_is_directory=True)
    reader = product.EvidenceReader(project, results)
    pin = {
        "path": "evidence-link/proof.bin",
        "size_bytes": proof.stat().st_size,
        "sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
    }
    assert reader.verify_pin(pin) is False

    real_receipts = results / "real-receipts"
    real_receipts.mkdir(parents=True)
    receipt = _receipt(project, "pose_ds9_gt_and_capacity")
    _write_json(real_receipts / "acceptance.json", receipt)
    product_validation = results / "product-validation"
    product_validation.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(product_validation / "pose")
    (product_validation / "pose").symlink_to(real_receipts, target_is_directory=True)

    gate = product._receipt_gate(
        product.EvidenceReader(project, results), "pose_ds9_gt_and_capacity"
    )
    assert gate["passed"] is False
    assert gate["state"] == "invalid"


@pytest.mark.parametrize(
    "raw_path",
    ["../outside.bin", "evidence/../../outside.bin", "/etc/hosts"],
)
def test_pinned_evidence_rejects_traversal_and_absolute_paths(tmp_path, raw_path):
    project = tmp_path / "project"
    project.mkdir()
    reader = product.EvidenceReader(project, project / "validation/results")

    assert reader.verify_pin(
        {
            "path": raw_path,
            "size_bytes": 1,
            "sha256": hashlib.sha256(b"x").hexdigest(),
        }
    ) is False


def test_oversize_receipt_fails_closed_without_parsing(tmp_path, monkeypatch):
    project = tmp_path / "project"
    results = project / "validation/results"
    path = results / product.POSE_RECEIPT
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{" + b" " * 128 + b"}")
    monkeypatch.setattr(product, "MAX_JSON_BYTES", 64)

    gate = product._receipt_gate(
        product.EvidenceReader(project, results), "pose_ds9_gt_and_capacity"
    )

    assert gate["passed"] is False
    assert gate["state"] == "invalid"
    reader = product.EvidenceReader(project, results)
    assert reader.result_json("pose", product.POSE_RECEIPT).state == "too_large"


def test_deeply_nested_invalid_receipt_does_not_crash_reporter(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    path = results / product.POSE_RECEIPT
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"nested":' + "[" * 2_000 + "0" + "]" * 2_000 + "}",
        encoding="utf-8",
    )

    gate = product._receipt_gate(
        product.EvidenceReader(project, results), "pose_ds9_gt_and_capacity"
    )

    assert gate["passed"] is False
    assert gate["state"] == "invalid"


def test_inconsistent_person_campaign_is_not_promoted(tmp_path):
    project, results = _fixture_project(tmp_path)
    path = results / "campaign-report/report.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    requirement = next(
        item
        for item in campaign["requirements"]
        if item["id"] == "person_detection_quality"
    )
    requirement["state"] = "pass"
    _write_json(path, campaign)

    report = product.build_product_readiness(
        project_root=project, results_root=results
    )
    person = report["required_gates"][0]
    contract = next(
        item
        for item in person["component_checks"]
        if item["id"] == "campaign_report_contract"
    )
    assert contract["passed"] is False
    assert contract["state"] == "invalid"
    assert person["passed"] is False
    assert report["decision"]["ready"] is False


def test_admin_projects_product_readiness_as_read_only_card(tmp_path, monkeypatch):
    project, results = _fixture_project(tmp_path)
    report = product.build_product_readiness(
        project_root=project, results_root=results
    )
    product.write_product_readiness(report, results / "product-readiness/current")
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(results))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(project))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT", str(ROOT / "validation/schemas")
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        raw = client.get(
            "/api/validation", params={"artifact": "product_readiness_json"}
        )
        page = client.get("/")

    assert response.status_code == 200
    projected = response.json()["campaigns"]["product_readiness"]
    assert projected["state"] == "not_ready"
    assert projected["ready"] is False
    assert projected["read_only"] is True
    assert projected["execution_actions_available"] is False
    assert projected["progress"]["total"] == 6
    assert len(projected["required_gates"]) == 6
    assert raw.status_code == 404
    assert page.status_code == 200
    assert 'id="productReadinessCard"' in page.text
    assert "renderProductReadiness(payload.campaigns?.product_readiness)" in page.text
    assert "salt okunur" in page.text


def test_admin_gate_rejects_markers_in_comments_and_strings(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    sources = {
        "admin/validation.py": '''\n# PRODUCT_READINESS_SCHEMA\n# "product_readiness_json": ArtifactSpec\n# def _product_readiness(reader): pass\n# "product_readiness": _product_readiness(reader)\nMARKERS = "product_readiness_json load_validation_status"\n''',
        "admin/app.py": '''\n# @app.get("/api/validation")\nTEXT = "load_validation_status load_validation_artifact"\n''',
        "deepstream/config.py": '''\n# MODULE_ARTIFACTS_CONFIGURED = {"person": True, "pose": False, "ppe": False}\nTEXT = "validate_module_selection module_readiness"\n''',
        "admin/static/index.html": '''\n<!-- id="productReadinessCard" salt okunur -->\n<script>\nconst fake = "function renderProductReadiness() {}";\n// renderProductReadiness(payload.campaigns?.product_readiness)\n</script>\n''',
    }
    for relative, content in sources.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    gate = product._admin_gate(product.EvidenceReader(project, results))

    assert gate["passed"] is False
    assert gate["state"] == "invalid"
    assert all(check["passed"] is False for check in gate["component_checks"])


def test_admin_projection_rejects_product_report_schema_mismatch(tmp_path):
    project, results = _fixture_project(tmp_path)
    report = product.build_product_readiness(
        project_root=project, results_root=results
    )
    report.pop("contract_id")
    _write_json(results / "product-readiness/current/report.json", report)
    reader = admin_validation.ArtifactReader(
        results,
        workspace_root=project,
        schema_root=ROOT / "validation/schemas",
    )

    projected = admin_validation._product_readiness(reader)

    assert projected["state"] == "artifact_error"
    assert projected["ready"] is False
    assert projected["final_claim_allowed"] is False
    assert projected["required_gates"] == []


def test_admin_cannot_promote_gate_when_a_component_check_failed(tmp_path):
    project, results = _fixture_project(tmp_path)
    report = product.build_product_readiness(
        project_root=project, results_root=results
    )
    for gate in report["required_gates"]:
        gate["passed"] = True
        gate["state"] = "pass"
    report["decision"].update(
        status="ready",
        ready=True,
        final_claim_allowed=True,
        failed_required_gate_ids=[],
    )
    report["summary"].update(
        passed_required_gate_count=6,
        remaining_required_gate_count=0,
        state_counts={"pass": 6},
    )
    _resign(report)
    _write_json(results / "product-readiness/current/report.json", report)
    schema = _schema()
    with pytest.raises(ValueError):
        admin_validation._validate_schema_node(report, schema, schema)
    reader = admin_validation.ArtifactReader(
        results,
        workspace_root=project,
        schema_root=ROOT / "validation/schemas",
    )

    projected = admin_validation._product_readiness(reader)

    assert projected["state"] == "artifact_error"
    assert projected["ready"] is False
    assert projected["final_claim_allowed"] is False


def test_admin_projection_rejects_tampered_product_report_fingerprint(tmp_path):
    project, results = _fixture_project(tmp_path)
    report = product.build_product_readiness(
        project_root=project, results_root=results
    )
    report["required_gates"][0]["title"] = "tampered but schema-valid"
    _write_json(results / "product-readiness/current/report.json", report)
    reader = admin_validation.ArtifactReader(
        results,
        workspace_root=project,
        schema_root=ROOT / "validation/schemas",
    )

    projected = admin_validation._product_readiness(reader)

    assert reader.validates_schema(report, admin_validation.PRODUCT_READINESS_SCHEMA)
    assert projected["state"] == "artifact_error"
    assert projected["ready"] is False


def test_admin_projection_rejects_symlinked_product_report(tmp_path):
    project, results = _fixture_project(tmp_path)
    report = product.build_product_readiness(
        project_root=project, results_root=results
    )
    real = results / "product-readiness/real-report.json"
    _write_json(real, report)
    link = results / "product-readiness/current/report.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)
    reader = admin_validation.ArtifactReader(
        results,
        workspace_root=project,
        schema_root=ROOT / "validation/schemas",
    )

    projected = admin_validation._product_readiness(reader)

    assert reader.read("product_readiness_json").state == "unsafe_path"
    assert projected["state"] == "artifact_error"
    assert projected["ready"] is False


def test_admin_projection_rejects_duplicate_json_keys(tmp_path):
    project = tmp_path / "project"
    results = project / "validation/results"
    path = results / "product-readiness/current/report.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"schema_version":"deepsafe.product-readiness/v1",'
        '"schema_version":"deepsafe.product-readiness/v1"}',
        encoding="utf-8",
    )
    reader = admin_validation.ArtifactReader(
        results,
        workspace_root=project,
        schema_root=ROOT / "validation/schemas",
    )

    projected = admin_validation._product_readiness(reader)

    assert reader.read("product_readiness_json").state == "invalid_json"
    assert projected["state"] == "artifact_error"
    assert projected["ready"] is False


def test_writer_produces_byte_identical_outputs(tmp_path):
    project, results = _fixture_project(tmp_path)
    report = product.build_product_readiness(
        project_root=project, results_root=results
    )
    output = tmp_path / "out"

    json_path, markdown_path = product.write_product_readiness(report, output)
    first = (json_path.read_bytes(), markdown_path.read_bytes())
    json_path, markdown_path = product.write_product_readiness(report, output)
    second = (json_path.read_bytes(), markdown_path.read_bytes())

    assert first == second
    assert json.loads(first[0]) == report
    assert b"person_validation_stage" in first[0]
    assert "CAVIAR" in first[1].decode("utf-8")


def test_writer_refuses_symlink_output_without_touching_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    link = tmp_path / "current"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot be a symlink"):
        product.write_product_readiness({"decision": {}}, link)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert link.is_symlink()
