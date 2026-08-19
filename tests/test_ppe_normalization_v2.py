from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from ppe_dataset.normalization_v2 import (
    NormalizationContractError,
    _scan_overlay,
    assign_deterministic_group_splits,
    audit_deterministic_group_splits,
    verify_assessment_receipt,
)


PLAN_PATH = Path("data/manifests/ppe-normalization-mendeley-v6.plan.json")
PLAN_SHA256 = "41c9943dc0487a75f859a825f77da0e281b8699cb95b40604ac8c293e71df463"
RECEIPT_PATH = Path(
    "validation/results/ppe/normalization/mendeley-ppe-v6-normalization-assessment-r1.json"
)
RECEIPT_SHA256 = "ae5b5473edbb3b8b76ffb934b09679d30d52760977982af8b74ad44ca56f3c1f"


def _grouped_dataset() -> dict:
    videos = [
        {
            "id": "v1",
            "site_id": "site-a",
            "camera_group_id": "cam-a",
            "session_id": "shift-1",
            "provenance_group_id": "capture-a",
        },
        {
            "id": "v2",
            "site_id": "site-a",
            "camera_group_id": "cam-b",
            "session_id": "shift-1",
            "provenance_group_id": "capture-b",
        },
        {
            "id": "v3",
            "site_id": "site-b",
            "camera_group_id": "cam-c",
            "session_id": "shift-2",
            "provenance_group_id": "capture-c",
        },
        {
            "id": "v4",
            "site_id": "site-c",
            "camera_group_id": "cam-d",
            "session_id": "shift-3",
            "provenance_group_id": "capture-d",
        },
    ]
    images = [
        {
            "id": index,
            "video_id": video_id,
            "sha256": f"{index:064x}",
            "split": "unassigned",
        }
        for index, video_id in enumerate(("v1", "v2", "v3", "v4"), start=1)
    ]
    return {"videos": videos, "images": images}


def test_frozen_plan_and_assessment_are_exact_pinned_and_fail_closed() -> None:
    assert hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() == PLAN_SHA256
    verification = verify_assessment_receipt(
        RECEIPT_PATH,
        expected_receipt_sha256=RECEIPT_SHA256,
    )
    assert verification["valid"] is True
    assert verification["normalization_ready"] is False
    assert verification["source_archive_training_eligible"] is False
    assert verification["normalized_dataset_training_eligible"] is False

    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    audit = receipt["observations"]["offline_overlay_label_audit"]
    assert audit["image_count"] == audit["label_file_count"] == 2286
    assert audit["label_row_count"] == 6038
    assert audit["class_id_histogram"] == {
        "0": 2421,
        "1": 1174,
        "2": 1058,
        "3": 1385,
    }
    assert audit["issue_counts"] == {"bbox_out_of_range": 52}
    assert audit["silent_bbox_clipping_applied"] is False
    assert audit["automatic_repairs_applied"] == []
    assert audit["canonical_labels_emitted"] is False


def test_assessment_rejects_wrong_external_trust_anchor() -> None:
    with pytest.raises(NormalizationContractError, match="trust-anchor"):
        verify_assessment_receipt(
            RECEIPT_PATH,
            expected_receipt_sha256="0" * 64,
        )


def test_overlay_audit_is_non_mutating_and_never_clips_bbox(tmp_path: Path) -> None:
    archive_path = tmp_path / "seed.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("root/train/images/a.jpg", b"not-decoded-by-overlay")
        archive.writestr("root/train/labels/a.txt", "0 0.5 0.5 1.00002 0.5\n")
        archive.writestr("root/valid/images/b.jpg", b"not-decoded-by-overlay")
        archive.writestr("root/valid/labels/b.txt", "3 0.5 0.5 0.2 0.2\n")
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    audit = _scan_overlay(
        archive_path,
        digest,
        {
            "classes": ["Helmet", "NoHelmet", "NoVest", "Vest"],
            "splits": {
                "train": {
                    "images": "root/train/images/",
                    "labels": "root/train/labels/",
                },
                "validation": {
                    "images": "root/valid/images/",
                    "labels": "root/valid/labels/",
                },
            },
            "max_label_bytes": 1024,
        },
    )
    assert audit["label_row_count"] == 2
    assert audit["issue_counts"] == {"bbox_out_of_range": 1}
    assert audit["bbox_overflow"]["maximum"] == pytest.approx(2e-5)
    assert audit["automatic_repairs_applied"] == []
    assert audit["silent_bbox_clipping_applied"] is False


def test_group_split_is_deterministic_and_keeps_shared_session_together() -> None:
    source = _grouped_dataset()
    first = assign_deterministic_group_splits(source, seed="stable")
    reordered = {
        "videos": list(reversed(copy.deepcopy(source["videos"]))),
        "images": list(reversed(copy.deepcopy(source["images"]))),
    }
    second = assign_deterministic_group_splits(reordered, seed="stable")
    first_by_id = {image["id"]: image["split"] for image in first["images"]}
    second_by_id = {image["id"]: image["split"] for image in second["images"]}
    assert first_by_id == second_by_id
    assert first_by_id[1] == first_by_id[2]
    assert audit_deterministic_group_splits(first)["valid"] is True
    assert all(image["split"] == "unassigned" for image in source["images"])


def test_static_seed_cannot_reuse_upstream_split_without_group_metadata() -> None:
    with pytest.raises(NormalizationContractError) as error:
        assign_deterministic_group_splits(
            {"videos": [], "images": [{"id": 1, "sha256": "a" * 64, "split": "train"}]}
        )
    assert error.value.code == "camera_group_metadata_required"


def test_duplicate_copy_is_connected_but_still_blocks_final_split_audit() -> None:
    source = _grouped_dataset()
    source["images"][2]["sha256"] = source["images"][0]["sha256"]
    assigned = assign_deterministic_group_splits(source, seed="stable")
    by_id = {image["id"]: image["split"] for image in assigned["images"]}
    assert by_id[1] == by_id[3]
    audit = audit_deterministic_group_splits(assigned)
    assert audit["valid"] is False
    assert {issue["code"] for issue in audit["issues"]} == {
        "exact_duplicate_not_removed"
    }


def test_group_split_audit_detects_manual_leakage() -> None:
    document = _grouped_dataset()
    document["images"][0]["split"] = "train"
    document["images"][1]["split"] = "test"
    document["images"][2]["split"] = "val"
    document["images"][3]["split"] = "test"
    audit = audit_deterministic_group_splits(document)
    assert audit["valid"] is False
    assert "group_split_leakage" in {issue["code"] for issue in audit["issues"]}
