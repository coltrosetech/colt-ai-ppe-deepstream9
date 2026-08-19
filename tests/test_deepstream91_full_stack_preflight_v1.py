from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from validation import deepstream91_full_stack_preflight_v1 as preflight


ROOT = Path(__file__).resolve().parents[1]


def plan() -> dict:
    return preflight.load_plan()


def resign(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("self_fingerprint", None)
    value["self_fingerprint"] = hashlib.sha256(preflight.canonical_bytes(value)).hexdigest()
    return value


def test_checked_in_preflight_replays_12_exact_media_and_stays_blocked() -> None:
    report = preflight.validate_plan(plan())
    assert report["status"] == "pass_blocked_as_designed"
    assert report["execution_ready"] is False
    assert report["source_count"] == 12
    assert report["distinct_video_types"] == 12
    assert report["profiles"] == [640, 960]
    assert report["measurement_seconds_per_profile"] == 300
    assert report["legacy_ds90_runtime_reused"] is False
    assert report["gpu_or_docker_called"] is False


def test_semantically_resigned_ds90_runtime_reuse_is_rejected() -> None:
    value = plan()
    value["source_matrix"]["legacy_runtime_projection_allowed"] = True
    with pytest.raises(preflight.DS91PreflightError, match="legacy runtime reuse"):
        preflight.validate_plan(resign(value), replay_media=False)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("runtime_target", "deepstream", "9.0.0", "runtime target"),
        ("runtime_target", "minimum_driver", "590.48.01", "runtime target"),
        ("execution", "measurement_seconds", 30, "measurement window"),
        ("execution", "simulated_streams", 11, "stream count"),
        ("gpu_lease", "execution_authorized", True, "lease overclaim"),
    ],
)
def test_resigned_runtime_execution_and_lease_overclaims_fail_closed(
    section: str, key: str, value: object, match: str
) -> None:
    document = plan()
    document[section][key] = value
    with pytest.raises(preflight.DS91PreflightError, match=match):
        preflight.validate_plan(resign(document), replay_media=False)


def test_resigned_module_artifact_without_successor_contract_is_rejected() -> None:
    value = plan()
    value["module_artifacts"]["person"]["engine_640"] = {
        "path": "models/person/640/legacy.engine",
        "bytes": 1,
        "sha256": "0" * 64,
    }
    with pytest.raises(preflight.DS91PreflightError, match="unaccepted person artifact"):
        preflight.validate_plan(resign(value), replay_media=False)


def test_resigned_blocker_removal_is_rejected() -> None:
    value = plan()
    value["blockers"].pop()
    with pytest.raises(preflight.DS91PreflightError, match="blocker derivation"):
        preflight.validate_plan(resign(value), replay_media=False)


def test_resigned_extra_nested_field_is_rejected() -> None:
    value = plan()
    value["authorization"]["hidden_gpu_override"] = False
    with pytest.raises(preflight.DS91PreflightError, match="authorization shape"):
        preflight.validate_plan(resign(value), replay_media=False)


def test_resigned_qualitative_matrix_cannot_claim_ground_truth() -> None:
    value = plan()
    value["source_matrix"]["ground_truth_policy"] = "ground_truth_available"
    with pytest.raises(preflight.DS91PreflightError, match="ground-truth policy"):
        preflight.validate_plan(resign(value), replay_media=False)


def test_duplicate_json_key_is_rejected() -> None:
    with pytest.raises(preflight.DS91PreflightError, match="duplicate JSON key"):
        preflight.strict_object(b'{"x":1,"x":2}', "fixture")


def test_corrupt_control_pin_is_rejected_before_any_execution() -> None:
    value = plan()
    value["runtime_target"]["prepared_closed_engine_builder_root"]["sha256"] = "0" * 64
    with pytest.raises(preflight.DS91PreflightError, match="pin SHA mismatch"):
        preflight.validate_plan(resign(value), replay_media=False)
