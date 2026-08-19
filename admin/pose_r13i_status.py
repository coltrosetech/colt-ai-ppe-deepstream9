"""Fail-closed projection of the independent Pose R13I static acceptance."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .model_gate_status import (
    ModelGateStatusError,
    _expect,
    _expected_projection,
    _pin_projection,
    _read_exact,
    _strict_json,
    _validate_fingerprint,
    _validate_pin_rows,
)


REVIEW_PATH = (
    "validation/results/pose/models/mmpose-yoloxpose-s-fixed-k100-r13i/"
    "r13i-independent-review-r1/receipt.json"
)
REVIEW_FINGERPRINT = (
    "518aa982c1ac849a279e101552f46dc265f52caef19653940c14aefdadb94b7e"
)
HANDOFF_PATH = (
    "validation/results/pose/models/mmpose-yoloxpose-s-fixed-k100-r13i/"
    "author-handoff-r1/handoff.json"
)
PINS: dict[str, tuple[int, str, str]] = {
    REVIEW_PATH: (
        17350,
        "1c650119b1e1748679ff6bedf0666b25d562d29482c99b5bf755ebc7941158f4",
        "0440",
    ),
    "validation/pose_mmpose_yoloxpose_tensorrt_ds91_r13i_independent_review.py": (
        68567,
        "b84c1487c1c823a6898660816749df62e83b509f844e07fb4f1820fd1ed0067b",
        "0555",
    ),
    "validation/schemas/pose-mmpose-yoloxpose-tensorrt-ds91-r13i-independent-review-v1.schema.json": (
        11071,
        "e0371f89d0202b53b6a360e925ee03564b3944132e2578718ed0879839419ff2",
        "0440",
    ),
    "tests/test_pose_mmpose_yoloxpose_tensorrt_ds91_r13i_independent_review.py": (
        18722,
        "743ccdef4042b690748d0b923ff97e489d87f605654d35cc21343b126c76cbfd",
        "0440",
    ),
    "docs/pose-mmpose-yoloxpose-tensorrt-ds91-r13i-independent-review.md": (
        3959,
        "e142a504a23e56552470ce16922d87dbabf057b39a1f6305636515df2aab9df8",
        "0440",
    ),
    "validation/pose_mmpose_yoloxpose_tensorrt_ds91_r13i.py": (
        62867,
        "74d5e0e0a8b97ba5efaf29bcf3791f840b638cc5336b5b4848871b27a66549bf",
        "0440",
    ),
    "validation/schemas/pose-mmpose-yoloxpose-tensorrt-ds91-controller-r13i.schema.json": (
        92525,
        "f3b1290099b4b06e383debbdac8c6e6c06a2f1b7a17de688bae345f44533c3dd",
        "0440",
    ),
    "tests/test_pose_mmpose_yoloxpose_tensorrt_ds91_r13i.py": (
        27024,
        "582d7462152c8106e2f79dca7a49d5a95d9ecede217df4cf6da8c74b390e2a9e",
        "0440",
    ),
    "docs/pose-mmpose-yoloxpose-tensorrt-ds91-r13i.md": (
        5637,
        "5ea8cca5503f2269921925ed01bf2de1b0153572c623e4a5862a2b515e1a46aa",
        "0440",
    ),
    "models/pose/challengers/mmpose-yoloxpose-s/tensorrt-ds91-controller-r13i.json": (
        24080,
        "9df085389deca034bac018eb8dc754a9d9a32a833fc4535c3bf70537637ed847",
        "0440",
    ),
    "validation/plans/pose-mmpose-yoloxpose-tensorrt-ds91-workload-template-r13i.json": (
        21057,
        "eeef25da428ec0c5894b04d3331559d46a947c84c5a0c273590297cea6ed0cce",
        "0440",
    ),
    "validation/schemas/pose-mmpose-yoloxpose-tensorrt-ds91-workload-template-r13i.schema.json": (
        22954,
        "9c1092f9f2d2c5d39314576db67dde8b7ae1f54bb0034dc5ed457d0fd03a584b",
        "0440",
    ),
    HANDOFF_PATH: (
        7577,
        "f409e871f31576c862e45ec530912f4751960350c331e77aec1eb5f4dfad325c",
        "0440",
    ),
}


def _unavailable() -> dict[str, Any]:
    return {
        "label": "Pose YOLOXPose/MMPose TensorRT R13I kapısı",
        "state": "unavailable_integrity_error",
        "reason": "pose_r13i_independent_acceptance_integrity_failed",
        "available": False,
        "decision": None,
        "independent_static_acceptance": False,
        "v5_host_realization_eligible": False,
        "engine_build_authorized": False,
        "gpu_workload_authorized": False,
        "deepstream_runtime_authorized": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "evidence": [],
        "caveats": ["Pose R13I exact-pin zinciri doğrulanamadı; bütün yürütme kapıları kapalıdır."],
    }


def _validate_external_blocker(review: dict[str, Any]) -> None:
    observations = review.get("external_gate_observations")
    _expect(isinstance(observations, list) and len(observations) == 1, "Pose external gate differs")
    blocker = observations[0]
    _expect(
        blocker.get("id") == "GPU_LEASE_V5_HOST_TOOL_PIN_DRIFT"
        and blocker.get("classification")
        == "execution_gate_blocker_outside_static_subject_acceptance"
        and blocker.get("subject_severity") is None
        and blocker.get("would_be_execution_readiness_severity") == "P1",
        "Pose external blocker classification differs",
    )
    details = blocker.get("details")
    _expect(isinstance(details, list) and len(details) == 2, "Pose host tool details differ")
    rows = {row.get("tool"): row for row in details if isinstance(row, dict)}
    expected = {
        "docker_cli": {
            "old_bytes": 45410374,
            "old_sha": "aa5d6553089f1139456466c754d1f45d064dc2762c77c403d3bcc0e399eb4e1d",
            "new_bytes": 45356984,
            "new_sha": "628af575ee8499596e7d266c221ee6bb74fbc20de3a81dd6fe5106f81e652db4",
        },
        "nvidia_smi": {
            "old_bytes": 1260192,
            "old_sha": "78c2d203c12619673ddd5ae9c487a79689c9461de40a9b948f3f4615f26dba29",
            "new_bytes": 1259616,
            "new_sha": "7896b7cdd9cb84b1e0fbc4baf91dfa3039b44ab6641cd08d7cc1dd23f81d6deb",
        },
    }
    _expect(set(rows) == set(expected), "Pose host tool set differs")
    for key, pins in expected.items():
        old = rows[key].get("expected")
        new = rows[key].get("observed")
        _expect(
            isinstance(old, dict)
            and old.get("bytes") == pins["old_bytes"]
            and old.get("sha256") == pins["old_sha"]
            and old.get("executed") is False,
            f"Pose expected tool pin differs: {key}",
        )
        _expect(
            isinstance(new, dict)
            and new.get("bytes") == pins["new_bytes"]
            and new.get("sha256") == pins["new_sha"]
            and new.get("executed") is False,
            f"Pose observed tool pin differs: {key}",
        )


def _validate_review(review: dict[str, Any]) -> None:
    _expect(
        review.get("schema_version")
        == "deepsafe.pose-mmpose-yoloxpose-tensorrt-ds91-r13i-independent-review/v1",
        "Pose schema differs",
    )
    _expect(
        review.get("status")
        == "accepted_static_execution_closed_controller_definition_with_external_v5_host_gate_blocked",
        "Pose status differs",
    )
    _expect(review.get("decision") == "ACCEPT", "Pose decision differs")
    _expect(review.get("severity_counts") == {"P0": 0, "P1": 0, "P2": 0}, "Pose severity differs")
    _expect(review.get("findings") == [], "Pose findings differ")
    _validate_fingerprint(review, REVIEW_FINGERPRINT)

    review_paths = {
        "validation/pose_mmpose_yoloxpose_tensorrt_ds91_r13i_independent_review.py",
        "validation/schemas/pose-mmpose-yoloxpose-tensorrt-ds91-r13i-independent-review-v1.schema.json",
        "tests/test_pose_mmpose_yoloxpose_tensorrt_ds91_r13i_independent_review.py",
        "docs/pose-mmpose-yoloxpose-tensorrt-ds91-r13i-independent-review.md",
    }
    _validate_pin_rows(review.get("review_control_pins"), review_paths, PINS, "Pose review")
    subject = review.get("subject")
    _expect(isinstance(subject, dict), "Pose subject differs")
    author_paths = {
        "validation/pose_mmpose_yoloxpose_tensorrt_ds91_r13i.py",
        "validation/schemas/pose-mmpose-yoloxpose-tensorrt-ds91-controller-r13i.schema.json",
        "tests/test_pose_mmpose_yoloxpose_tensorrt_ds91_r13i.py",
        "docs/pose-mmpose-yoloxpose-tensorrt-ds91-r13i.md",
        "models/pose/challengers/mmpose-yoloxpose-s/tensorrt-ds91-controller-r13i.json",
        "validation/plans/pose-mmpose-yoloxpose-tensorrt-ds91-workload-template-r13i.json",
        "validation/schemas/pose-mmpose-yoloxpose-tensorrt-ds91-workload-template-r13i.schema.json",
    }
    author_artifacts = subject.get("author_artifacts")
    _expect(isinstance(author_artifacts, dict), "Pose author artifact map differs")
    _validate_pin_rows(
        list(author_artifacts.values()), author_paths, PINS, "Pose author"
    )
    handoff = subject.get("handoff")
    _expect(isinstance(handoff, dict), "Pose handoff differs")
    _expect(
        _pin_projection(handoff.get("pin")) == _expected_projection(HANDOFF_PATH, PINS),
        "Pose handoff pin differs",
    )
    _expect(
        handoff.get("fingerprint_sha256")
        == "60be82356eeb251a73bb0e7ad9ba4d5f04f0bda8bd69113c4df75a5d907aaf54",
        "Pose handoff fingerprint differs",
    )
    _expect(
        handoff.get("status") == "author_handoff_pending_independent_review"
        and handoff.get("author_tests_claimed_passed_twice") is True
        and handoff.get("execution_authorized") is False,
        "Pose handoff projection differs",
    )

    tests = review.get("independent_test_replay")
    _expect(
        isinstance(tests, dict)
        and tests.get("collected") == 85
        and tests.get("passed") == 85
        and tests.get("failed") == 0
        and tests.get("repeat_runs_required") == 2,
        "Pose independent tests differ",
    )
    scope = review.get("accepted_scope")
    _expect(isinstance(scope, dict), "Pose scope differs")
    _expect(scope.get("profiles") == [640, 960], "Pose profiles differ")
    _expect(scope.get("batch_min_opt_max") == [1, 12, 12], "Pose batch differs")
    _expect(scope.get("precision") == "fp16_tf32_off_int8_off", "Pose precision differs")
    _expect(scope.get("same_index_coco17_contract") is True, "Pose keypoint contract differs")
    _expect(scope.get("static_r13i_controller_definition") is True, "Pose static scope differs")
    _expect(scope.get("gpu_lease_v5_current_host_realization_eligible") is False, "Pose V5 gate overclaim")
    for key in (
        "runtime_execution_authorized",
        "engine_build_authorized",
        "config_publication_authorized",
        "parity_authorized",
        "gpu_lease_activation_authorized",
        "production_ready",
        "deepstream91_gpu_runtime_qualified",
    ):
        _expect(scope.get(key) is False, f"Pose scope overclaim: {key}")
    permissions = review.get("permissions")
    _expect(isinstance(permissions, dict) and permissions and all(value is False for value in permissions.values()), "Pose permissions overclaim")
    boundary = review.get("authority_boundary")
    _expect(
        isinstance(boundary, dict)
        and boundary.get("runtime_or_build_authority_granted") is False
        and boundary.get("v5_host_realization_accepted") is False,
        "Pose authority boundary differs",
    )
    resource = review.get("resource_boundary")
    _expect(isinstance(resource, dict) and resource.get("host_tool_bytes_hashed_only") is True, "Pose resource observation differs")
    for key, value in resource.items():
        if key != "host_tool_bytes_hashed_only":
            _expect(value is False, f"Pose resource overclaim: {key}")
    profiles = review.get("verification", {}).get("profiles")
    _expect(isinstance(profiles, dict) and set(profiles) == {"640", "960"}, "Pose profile rows differ")
    for size in (640, 960):
        row = profiles[str(size)]
        shape = [12, 3, size, size]
        _expect(
            row.get("shape_min_opt_max")
            == {"min": [1, 3, size, size], "opt": shape, "max": shape},
            f"Pose shape differs: {size}",
        )
        _expect(row.get("precision") == "fp16_tf32_off_int8_off", f"Pose precision differs: {size}")
        _expect(isinstance(row.get("outputs_absent"), list) and len(row["outputs_absent"]) == 10, f"Pose outputs differ: {size}")
    _validate_external_blocker(review)


def _validate_author_handoff(handoff: dict[str, Any]) -> None:
    _expect(
        handoff.get("schema_version")
        == "deepsafe.pose-mmpose-yoloxpose-tensorrt-ds91-r13i-author-handoff/v1",
        "Pose author handoff schema differs",
    )
    _expect(
        handoff.get("status") == "author_handoff_pending_independent_review"
        and handoff.get("decision") is None
        and handoff.get("terminal_accepted") is False
        and handoff.get("execution_authorized") is False,
        "Pose author handoff authority differs",
    )
    fingerprint = handoff.get("self_fingerprint")
    _expect(
        isinstance(fingerprint, dict)
        and fingerprint.get("algorithm")
        == "sha256-canonical-json-without-self_fingerprint"
        and fingerprint.get("canonical_sha256")
        == "60be82356eeb251a73bb0e7ad9ba4d5f04f0bda8bd69113c4df75a5d907aaf54",
        "Pose author handoff fingerprint declaration differs",
    )
    unsigned = {key: value for key, value in handoff.items() if key != "self_fingerprint"}
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    _expect(
        hashlib.sha256(canonical).hexdigest() == fingerprint["canonical_sha256"],
        "Pose author handoff fingerprint replay differs",
    )
    permissions = handoff.get("permissions")
    resources = handoff.get("resource_boundary")
    _expect(
        isinstance(permissions, dict)
        and permissions
        and all(value is False for value in permissions.values()),
        "Pose author handoff permissions overclaim",
    )
    _expect(
        isinstance(resources, dict)
        and resources
        and all(value is False for value in resources.values()),
        "Pose author handoff resource boundary overclaim",
    )


def load_pose_r13i_status() -> dict[str, Any]:
    try:
        raw = {relative: _read_exact(relative, PINS) for relative in PINS}
        review = _strict_json(raw[REVIEW_PATH])
        _validate_review(review)
        handoff = _strict_json(raw[HANDOFF_PATH])
        _validate_author_handoff(handoff)
        author_tests = handoff.get("author_tests")
        _expect(
            isinstance(author_tests, dict)
            and author_tests.get("collected") == 83
            and author_tests.get("passed") == 83
            and author_tests.get("failed") == 0
            and author_tests.get("repeat_runs_required") == 2,
            "Pose author tests differ",
        )
    except (OSError, KeyError, TypeError, ModelGateStatusError):
        return _unavailable()

    return {
        "label": "Pose YOLOXPose/MMPose TensorRT R13I kapısı",
        "state": "independent_static_controller_accepted_v5_host_gate_blocked",
        "reason": "gpu_lease_v6_successor_and_runtime_engine_gates_pending",
        "available": True,
        "decision": "ACCEPT",
        "severity": {"p0": 0, "p1": 0, "p2": 0},
        "external_execution_gate": {"severity": "P1", "id": "GPU_LEASE_V5_HOST_TOOL_PIN_DRIFT"},
        "independent_static_acceptance": True,
        "v5_host_realization_eligible": False,
        "profiles": [640, 960],
        "batch": {"min": 1, "opt": 12, "max": 12},
        "precision": "FP16 (TF32 kapalı, INT8 kapalı)",
        "keypoints": {"layout": "COCO", "count": 17, "same_index_contract": True},
        "tests": {"author_passed": 83, "independent_passed": 85, "combined_passed": 168, "failed": 0},
        "planned_outputs": 20,
        "outputs_present": 0,
        "next_gate": "GPU Lease V6 successor kabulü; ardından DS9.1 smoke ve ayrı 640/960 engine planları",
        "engine_build_authorized": False,
        "gpu_workload_authorized": False,
        "deepstream_runtime_authorized": False,
        "parity_authorized": False,
        "config_publication_authorized": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "does_not_imply_product_readiness": True,
        "evidence": [],
        "caveats": [
            "ACCEPT yalnız 640/960 Pose TensorRT controller tanımına aittir; engine veya inference üretilmedi.",
            "Current V5 host-tool realization P1 blocker'dır; GPU, DeepStream, parity ve production kapalıdır.",
        ],
    }
