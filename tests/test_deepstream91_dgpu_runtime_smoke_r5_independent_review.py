from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from validation import deepstream91_dgpu_runtime_smoke_r5_independent_review as review


@pytest.fixture(scope="session")
def report() -> dict[str, Any]:
    return review.audit()


@pytest.fixture(scope="session")
def result() -> dict[str, Any]:
    return json.loads((review.ROOT / review.PINS["result_r5"]["path"]).read_text())


def test_audit_accepts(report: dict[str, Any]) -> None:
    assert report["decision"] == "ACCEPT"
    assert report["severity_counts"] == {"P0": 0, "P1": 0, "P2": 0}


@pytest.mark.parametrize("name", sorted(review.PINS))
def test_every_subject_pin_replays(name: str) -> None:
    expected = review.PINS[name]
    with review.Reader() as reader:
        raw = reader.read(expected, name)
    assert len(raw) == expected["bytes"]
    assert hashlib.sha256(raw).hexdigest() == expected["sha256"]


@pytest.mark.parametrize("path", ["", "/x", "a/../b", "a/./b", "a//b", "a\\b"])
def test_unsafe_paths_rejected(path: str) -> None:
    with pytest.raises(review.ReviewError):
        review._parts(path)


def test_reader_rejects_hardlink(tmp_path: Path) -> None:
    first = tmp_path / "a"; first.write_bytes(b"x"); first.chmod(0o440); os.link(first, tmp_path / "b")
    with review.Reader(tmp_path) as reader:
        with pytest.raises(review.ReviewError):
            reader.read(review.pin("a", 1, hashlib.sha256(b"x").hexdigest(), "0440"), "a")


def test_result_fingerprint(report: dict[str, Any]) -> None:
    assert report["verification"]["result"]["result_fingerprint_sha256"] == review.RESULT_FP


@pytest.mark.parametrize("probe_id", list(review.EXPECTED_COMMANDS))
def test_each_probe_passes(result: dict[str, Any], probe_id: str) -> None:
    row = {item["probe_id"]: item for item in result["observations"]}[probe_id]
    assert row["argv"] == review.EXPECTED_COMMANDS[probe_id]
    assert row["returncode"] == 0
    assert row["passed"] is True
    assert row["required_tokens_present"] is True
    assert row["timed_out"] is False


def test_probes_unique(result: dict[str, Any]) -> None:
    ids = [item["probe_id"] for item in result["observations"]]
    assert len(ids) == len(set(ids)) == 7


def test_runtime_tokens(report: dict[str, Any]) -> None:
    row = report["verification"]["result"]
    assert row["deepstream"] == "9.1.0"
    assert row["tensorrt"] == "10.16"
    assert row["trtexec_banner"] == "TensorRT v101600 [b72]"
    assert row["plugins"] == ["nvinfer", "nvstreammux", "nvvideoconvert"]


def test_gpu_driver_tokens(report: dict[str, Any]) -> None:
    row = report["verification"]["result"]
    assert row["gpu_uuid"] == review.GPU_UUID
    assert row["driver_version"] == review.DRIVER


def test_model_free_pipeline_passed(report: dict[str, Any]) -> None:
    assert report["verification"]["result"]["model_free_nvvideoconvert_pipeline_passed"] is True


def test_network_none(report: dict[str, Any]) -> None:
    assert report["verification"]["plan"]["network"] == "none"
    assert report["verification"]["result"]["network"] == "none"


def test_v6_activation_terminal_rc0(report: dict[str, Any]) -> None:
    row = report["verification"]["activation"]
    assert row["returncode"] == 0
    assert row["receipt_fingerprint_sha256"] == review.ACTIVATION_FP


@pytest.mark.parametrize("field", ["terminal_release", "container_verified_absent", "host_scope_verified_absent", "active_state_absent"])
def test_cleanup_closed(report: dict[str, Any], field: str) -> None:
    assert report["verification"]["activation"][field] is True


def test_plan_outer_binding(report: dict[str, Any]) -> None:
    row = report["verification"]["plan"]
    assert row["plan_fingerprint_sha256"] == review.PLAN_FP
    assert row["outer_fingerprint_sha256"] == review.OUTER_FP
    assert row["outer_accepted"] is True and row["user_notified"] is True


def test_r4_failed_only_trtexec_version(report: dict[str, Any]) -> None:
    row = report["verification"]["r4_failure_lineage"]
    assert row["immutable_mode"] == "0440"
    assert row["passed_probes"] == 6
    assert row["only_failed_probe"] == "trtexec_version"
    assert row["argv"] == ["/usr/src/tensorrt/bin/trtexec", "--version"]
    assert row["returncode"] == 1


def test_worker_delta_only_help_and_tokens(report: dict[str, Any]) -> None:
    assert report["verification"]["worker_delta"]["only_delta"] == "trtexec_--version_to_--help_and_required_tokens"


def test_build_and_verify_receipt(report: dict[str, Any]) -> None:
    value = review.build_receipt(report, 50, 2)
    assert review.verify_receipt(value)["decision"] == "ACCEPT"
    assert value["review_fingerprint_sha256"] == review.fingerprint(value, "review_fingerprint_sha256")


@pytest.mark.parametrize("section,key", [("permissions", "call_docker"), ("permissions", "call_gpu"), ("resource_boundary", "network_called"), ("authority_boundary", "production_accepted")])
def test_authority_mutations_rejected(report: dict[str, Any], section: str, key: str) -> None:
    value = review.build_receipt(report, 50, 2)
    value[section][key] = True
    value["review_fingerprint_sha256"] = review.fingerprint(value, "review_fingerprint_sha256")
    with pytest.raises(review.ReviewError):
        review.verify_receipt(value)
