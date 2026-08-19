from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ppe_dataset.five_class_materialization_r3 import (
    CANONICAL_CLASSES,
    FiveClassMaterializationError,
    _parse_and_remap_label,
    materialize,
    verify_materialization_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "data/manifests/"
    "ppe-mendeley-five-class-v1-development-materialization-r3.plan.json"
)
PLAN_SCHEMA = (
    ROOT
    / "ppe_dataset/schemas/"
    "ppe-five-class-materialization-plan-v1.schema.json"
)
RECEIPT = (
    ROOT
    / "validation/results/ppe/materialization/"
    "mendeley-ppe-five-class-v1-development-r3.json"
)
RECEIPT_SCHEMA = (
    ROOT
    / "validation/schemas/"
    "ppe-five-class-materialization-receipt-v1.schema.json"
)
DATASET = (
    ROOT / "data/derived/ppe/mendeley-ppe-five-class-v1-development-r3"
)
PLAN_SHA256 = "9ef2fb03b05a9d3a2bc1e3d8e53e6ff699181ab88df042aa8a349f7d96cb61d3"
RECEIPT_SHA256 = "bf51c0da5db4392723c91f7be72f74ff5c946d8a54c0bb391d3650080e6a2529"
RECEIPT_FILE_SHA256 = "453227cd1fabc4d835142389d42075d572ca209abcb8deb91794e30a87df4d0e"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def verified() -> dict:
    return verify_materialization_receipt(
        RECEIPT,
        expected_receipt_sha256=RECEIPT_SHA256,
        root=ROOT,
        schema_path=RECEIPT_SCHEMA,
    )


def test_materialization_plan_is_strict_schema_valid_and_exact_pinned() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    schema = json.loads(PLAN_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(plan)
    assert _sha256(PLAN) == PLAN_SHA256
    assert plan["execution_constraints"] == {
        "cpu_only": True,
        "network_allowed": False,
        "gpu_query_allowed": False,
        "docker_allowed": False,
        "training_allowed": False,
        "export_allowed": False,
        "admin_rebuild_allowed": False,
    }


def test_materialization_receipt_and_complete_tree_replay(verified: dict) -> None:
    assert RECEIPT.stat().st_size == 7770
    assert _sha256(RECEIPT) == RECEIPT_FILE_SHA256
    assert verified == {
        "valid": True,
        "path": str(RECEIPT),
        "file_sha256": RECEIPT_FILE_SHA256,
        "receipt_sha256": RECEIPT_SHA256,
        "images": 2586,
        "groups": 292,
        "bbox_rows": 17827,
        "leakage_zero": True,
        "classes": list(CANONICAL_CLASSES),
        "roles": {
            "train": 2068,
            "calibration": 259,
            "development_holdout": 259,
        },
        "external_pin_verified": True,
        "dataset_tree_rehashed": True,
        "training_authorized": False,
        "production_eligible": False,
    }


def test_roles_counts_class_order_and_four_leakage_dimensions_are_exact() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["dataset"]["classes"] == [
        "person",
        "helmet",
        "no_helmet",
        "hi_vis",
        "no_hi_vis",
    ]
    assert receipt["dataset"]["roles"] == {
        "train": {
            "groups": 145,
            "images": 2068,
            "bbox_rows": 14262,
            "canonical_class_bbox_counts": {
                "person": 4758,
                "helmet": 4032,
                "no_helmet": 820,
                "hi_vis": 2156,
                "no_hi_vis": 2496,
            },
        },
        "calibration": {
            "groups": 75,
            "images": 259,
            "bbox_rows": 1796,
            "canonical_class_bbox_counts": {
                "person": 608,
                "helmet": 502,
                "no_helmet": 105,
                "hi_vis": 271,
                "no_hi_vis": 310,
            },
        },
        "development_holdout": {
            "groups": 72,
            "images": 259,
            "bbox_rows": 1769,
            "canonical_class_bbox_counts": {
                "person": 589,
                "helmet": 502,
                "no_helmet": 101,
                "hi_vis": 267,
                "no_hi_vis": 310,
            },
        },
    }
    assert receipt["leakage_audit"] == {
        "image_path_role_leakage_count": 0,
        "capture_key_role_leakage_count": 0,
        "exact_duplicate_role_leakage_count": 0,
        "group_role_leakage_count": 0,
        "leakage_zero": True,
    }


def test_dataset_has_no_test_directory_or_yaml_test_claim() -> None:
    assert not (DATASET / "images/test").exists()
    assert not (DATASET / "labels/test").exists()
    yaml_text = (DATASET / "dataset.yaml").read_text(encoding="utf-8")
    assert "\ntest:" not in "\n" + yaml_text
    assert "official_test" not in yaml_text
    role_contract = json.loads(
        (DATASET / "metadata/role-contract.json").read_text(encoding="utf-8")
    )
    assert role_contract["final_test_role_present"] is False
    assert role_contract["official_test_claimed"] is False
    assert role_contract["roles"]["development_holdout"] == (
        "development_only_not_final_test"
    )


def test_materialized_tree_is_read_only_and_single_link() -> None:
    assert stat.S_IMODE(DATASET.stat().st_mode) == 0o550
    for path in (
        DATASET / "dataset.yaml",
        DATASET / "metadata/file-ledger.jsonl",
        DATASET / "metadata/group-ledger.json",
        DATASET / "metadata/role-contract.json",
        RECEIPT,
    ):
        info = path.lstat()
        assert stat.S_ISREG(info.st_mode)
        assert stat.S_IMODE(info.st_mode) == 0o440
        assert info.st_nlink == 1


def test_bbox_parser_is_fail_closed_and_canonical_remap_is_exact() -> None:
    payload = b"3 0.5 0.5 0.25 0.5\n0 0.25 0.25 0.1 0.1\n"
    remapped, source, canonical, rows = _parse_and_remap_label("fixture.txt", payload)
    assert remapped == b"0 0.5 0.5 0.25 0.5\n1 0.25 0.25 0.1 0.1\n"
    assert dict(source) == {3: 1, 0: 1}
    assert dict(canonical) == {0: 1, 1: 1}
    assert rows == 2
    for invalid in (
        b"0 nan 0.5 0.1 0.1\n",
        b"0 0.99 0.5 0.1 0.1\n",
        b"9 0.5 0.5 0.1 0.1\n",
        b"0 0.5 0.5 0 0.1\n",
        b"0 0.5 0.5 0.1\n",
    ):
        with pytest.raises(FiveClassMaterializationError):
            _parse_and_remap_label("invalid.txt", invalid)


def test_wrong_receipt_pin_and_overwrite_attempt_fail_closed() -> None:
    with pytest.raises(FiveClassMaterializationError, match="external receipt pin"):
        verify_materialization_receipt(
            RECEIPT,
            expected_receipt_sha256="0" * 64,
            root=ROOT,
            schema_path=RECEIPT_SCHEMA,
        )
    with pytest.raises(FiveClassMaterializationError, match="already exists"):
        materialize(
            PLAN,
            expected_plan_sha256=PLAN_SHA256,
            output_receipt=RECEIPT,
            root=ROOT,
        )


def test_rights_and_production_blockers_are_separate() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    blockers = receipt["blocker_partitions"]
    assert "source_provenance_review_required" in blockers[
        "development_training_authorization"
    ]
    assert "embedded_third_party_rights_review_required" in blockers[
        "development_training_authorization"
    ]
    assert "independent_final_test_dataset_required" in blockers[
        "production_acceptance"
    ]
    assert "deepstream_9_tensorrt_runtime_qualification_required" in blockers[
        "production_acceptance"
    ]
    assert receipt["readiness"]["mechanical_materialization_complete"] is True
    assert receipt["readiness"]["development_training_launch_authorized"] is False
    assert receipt["readiness"]["production_eligible"] is False
