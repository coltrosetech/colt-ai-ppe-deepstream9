from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from ppe_dataset.provenance import (
    ProvenanceContractError,
    _canonical_sha256,
    audit_seed_provenance,
    main,
    verify_provenance_receipt_file,
)


def _jpeg_fixture(quality: int) -> bytes:
    y, x = np.mgrid[0:96, 0:128]
    image = np.dstack(
        (
            (x * 2 + y) % 256,
            (x + y * 2) % 256,
            ((x // 8 + y // 8) % 2) * 180 + 30,
        )
    ).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return encoded.tobytes()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "root"
    root.mkdir()
    embedded = {
        "seed/README.dataset.txt": b"fixture aggregate readme\n",
        "seed/README.roboflow.txt": b"fixture roboflow readme\n",
        "seed/data.yaml": b"train: train/images\nval: valid/images\n",
        "seed/split.py": b"import random\nrandom.sample(['a', 'b'], 1)\n",
    }
    first = _jpeg_fixture(95)
    second = _jpeg_fixture(90)
    assert first != second
    archive_path = root / "raw" / "seed.zip"
    archive_path.parent.mkdir()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in embedded.items():
            archive.writestr(name, content)
        archive.writestr(
            "seed/train/images/scene_001_jpg.rf.11111111111111111111111111111111.jpg",
            first,
        )
        archive.writestr(
            "seed/valid/images/scene_002_jpg.rf.22222222222222222222222222222222.jpg",
            second,
        )
    archive_bytes = archive_path.read_bytes()

    quarantine = {
        "schema_version": "deepsafe.ppe-seed-quarantine-receipt/v1",
        "archive": {"bytes": len(archive_bytes), "sha256": _sha256(archive_bytes)},
        "structural_pass": False,
        "accepted_to_quarantine": False,
        "training_eligible": False,
    }
    quarantine["receipt_sha256"] = _canonical_sha256(quarantine, omit="receipt_sha256")
    quarantine_path = root / "receipts" / "quarantine.json"
    quarantine_path.parent.mkdir()
    quarantine_path.write_text(json.dumps(quarantine), encoding="utf-8")

    role_for_path = {
        "seed/README.dataset.txt": "aggregate_dataset_readme",
        "seed/README.roboflow.txt": "roboflow_export_readme",
        "seed/data.yaml": "upstream_yolo_config",
        "seed/split.py": "published_random_split_script",
    }
    plan = {
        "schema_version": "deepsafe.ppe-seed-provenance-review-plan/v1",
        "plan_id": "fixture-provenance-r1",
        "status": "planned_fail_closed_training_blocked",
        "source_id": "fixture-seed",
        "research_cutoff": "2026-07-17",
        "inputs": {
            "archive": {
                "path": "raw/seed.zip",
                "bytes": len(archive_bytes),
                "sha256": _sha256(archive_bytes),
            },
            "quarantine_receipt": {
                "path": "receipts/quarantine.json",
                "file_sha256": _sha256(quarantine_path.read_bytes()),
                "receipt_sha256": quarantine["receipt_sha256"],
                "structural_pass": False,
            },
            "embedded_members": [
                {
                    "path": name,
                    "role": role_for_path[name],
                    "bytes": len(content),
                    "sha256": _sha256(content),
                }
                for name, content in embedded.items()
            ],
        },
        "research_evidence": [
            {
                "id": "fixture-primary",
                "kind": "dataset_doi",
                "url": "https://example.invalid/fixture",
                "scope": "Fixture-only aggregate metadata; no item rights.",
                "retrieved_sha256": None,
            }
        ],
        "component_sources": [
            {
                "id": "fixture-component",
                "declared_images": 2,
                "declared_license": "fixture aggregate declaration",
                "item_level_member_mapping_complete": False,
                "embedded_media_chain_of_title_verified": False,
                "depicted_person_rights_verified": False,
                "location_capture_rights_verified": False,
                "evidence_ids": ["fixture-primary"],
            }
        ],
        "rights_policy": {
            "aggregate_license_is_not_item_level_clearance": True,
            "publication_license_is_not_dataset_license": True,
            "commercial_training_requires_item_level_chain_of_title": True,
            "person_and_location_rights_require_separate_review": True,
            "embedded_third_party_rights_review_complete": False,
            "review_note": "Fixture rights deliberately remain incomplete.",
        },
        "filename_family_policy": {
            "export_suffix_regex": "_jpg\\.rf\\.[0-9a-f]{32}$",
            "family_method": "strip_roboflow_suffix_then_trailing_numeric_segments_then_ascii_alpha_normalize_v1",
            "split_roots": {
                "train": "seed/train/images/",
                "validation": "seed/valid/images/",
            },
            "expected_families": [
                {
                    "family": "scene",
                    "images": 2,
                    "train": 1,
                    "validation": 1,
                    "source_mapping_status": "ambiguous_aggregate_not_item_mapped",
                    "component_source_id": None,
                }
            ],
        },
        "near_duplicate_policy": {
            "method": "opencv_grayscale_dct_phash63_plus_horizontal_dhash64_cross_split_v1",
            "strict_thresholds": {"phash_hamming_max": 4, "dhash_hamming_max": 4},
            "high_confidence_thresholds": {
                "phash_hamming_max": 1,
                "dhash_hamming_max": 1,
                "resized_rgb_mae_max": 2.0,
                "grayscale_correlation_min": 0.999,
            },
            "max_candidate_pairs": 10,
            "human_confirmation_required": True,
        },
        "expected_observations": {
            "images": 2,
            "train_images": 1,
            "validation_images": 1,
            "filename_families": 1,
            "cross_split_filename_families": 1,
            "exact_content_duplicate_groups": 0,
            "exact_original_key_duplicate_groups": 0,
            "strict_cross_split_candidate_pairs": 1,
            "strict_cross_split_validation_members": 1,
            "high_confidence_automated_candidate_pairs": 1,
        },
        "required_blockers": [
            "embedded_third_party_rights_review_incomplete",
            "camera_site_session_metadata_absent",
            "cross_split_near_duplicate_candidates_detected",
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return root, plan_path, plan


def test_fixture_audit_is_mechanically_complete_but_training_remains_blocked(
    tmp_path: Path,
) -> None:
    pytest.importorskip("jsonschema")
    root, plan_path, _ = _write_fixture(tmp_path)

    receipt = audit_seed_provenance(plan_path, root=root)

    assert receipt["mechanical_audit_complete"] is True
    assert receipt["training_eligible"] is False
    assert receipt["embedded_third_party_rights_review_complete"] is False
    assert receipt["camera_group_split_review_complete"] is False
    assert receipt["observations"]["strict_cross_split_candidate_pairs"] == 1
    assert receipt["duplicate_review"]["high_confidence_automated_candidate_pairs"] == 1
    assert receipt["duplicate_review"]["cross_split_candidates"][0]["human_confirmed"] is False
    assert receipt["split_review"]["all_images_in_cross_split_families"] == 2
    assert receipt["duplicate_review"]["exact_content_duplicate_groups"] == []


def test_plan_cannot_promote_item_rights_or_training_by_boolean_edit(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    _, _, plan = _write_fixture(tmp_path)
    schema = json.loads(
        Path("ppe_dataset/schemas/ppe-seed-provenance-review-plan-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    plan["component_sources"][0]["depicted_person_rights_verified"] = True
    plan["rights_policy"]["embedded_third_party_rights_review_complete"] = True
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(plan))
    assert len(errors) >= 2


@pytest.mark.parametrize("symlink_level", ["final", "intermediate"])
def test_archive_symlink_is_rejected_even_when_bytes_match(
    tmp_path: Path, symlink_level: str
) -> None:
    pytest.importorskip("jsonschema")
    root, plan_path, plan = _write_fixture(tmp_path)
    original = root / "raw" / "seed.zip"
    if symlink_level == "final":
        target = root / "raw" / "target.zip"
        original.rename(target)
        original.symlink_to(target.name)
    else:
        target_directory = root / "raw-target"
        original.parent.rename(target_directory)
        (root / "raw").symlink_to(target_directory.name)
    plan["inputs"]["archive"]["path"] = "raw/seed.zip"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ProvenanceContractError) as error:
        audit_seed_provenance(plan_path, root=root)
    assert error.value.code == "unsafe_file_path"


def test_receipt_schema_self_hash_external_pin_and_tamper_detection(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    root, plan_path, _ = _write_fixture(tmp_path)
    receipt = audit_seed_provenance(plan_path, root=root)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    verified = verify_provenance_receipt_file(
        receipt_path, expected_receipt_sha256=receipt["receipt_sha256"]
    )
    assert verified["valid"] is True
    assert verified["external_pin_verified"] is True
    assert verified["training_eligible"] is False

    receipt["blockers"].pop()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ProvenanceContractError) as error:
        verify_provenance_receipt_file(receipt_path)
    assert error.value.code == "receipt_self_hash_mismatch"


def test_cli_publishes_immutable_blocked_receipt_without_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    root, plan_path, _ = _write_fixture(tmp_path)
    receipt_path = tmp_path / "out" / "receipt.json"
    arguments = [
        "audit",
        "--plan",
        str(plan_path),
        "--root",
        str(root),
        "--receipt",
        str(receipt_path),
    ]

    assert main(arguments) == 2
    first = receipt_path.read_bytes()
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o440
    assert main(arguments) == 2
    assert receipt_path.read_bytes() == first


def test_frozen_real_plan_and_schemas_are_valid() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    for schema_path in (
        Path("ppe_dataset/schemas/ppe-seed-provenance-review-plan-v1.schema.json"),
        Path("ppe_dataset/schemas/ppe-seed-provenance-review-receipt-v1.schema.json"),
    ):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
    plan_schema = json.loads(
        Path("ppe_dataset/schemas/ppe-seed-provenance-review-plan-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    plans = []
    for path in (
        Path("data/manifests/ppe-mendeley-v6-provenance-review.plan.json"),
        Path("data/manifests/ppe-mendeley-v6-provenance-review-r2.plan.json"),
    ):
        plan = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(plan_schema).validate(plan)
        assert plan["expected_observations"]["strict_cross_split_candidate_pairs"] == 7
        assert sum(item["declared_images"] for item in plan["component_sources"]) == 2286
        plans.append(plan)
    assert plans[-1]["plan_id"].endswith("-r2")
    assert hashlib.sha256(
        Path("data/manifests/ppe-mendeley-v6-provenance-review-r2.plan.json").read_bytes()
    ).hexdigest() == "e7a9afbf4c5c9b78c0dcfdb4548d1ff2dccfabf0b36cda0de119eaa03817982c"

    verification = verify_provenance_receipt_file(
        Path("validation/results/ppe/provenance/mendeley-ppe-v6-provenance-review-r2.json"),
        expected_receipt_sha256="9358c9d6d67b302c0161c4de6d50389b8c683f8c005f8b0042757dec0ae195fa",
    )
    assert verification["valid"] is True
    assert verification["training_eligible"] is False
