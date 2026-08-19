from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from jsonschema import Draft202012Validator

from ppe_dataset.acquisition import verify_seed_receipt_file


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/ppe-mendeley-five-class-v1-acquisition-r2.json"
MANIFEST_SCHEMA = ROOT / "ppe_dataset/schemas/ppe-training-seed-sources-v1.schema.json"
ARCHIVE = ROOT / "data/raw/ppe/mendeley-ppe-five-class-v1/8vf7z6v5sb-1.zip"
RECEIPT = (
    ROOT
    / "validation/results/ppe/quarantine/mendeley-ppe-five-class-v1-quarantine-r2.json"
)

ARCHIVE_BYTES = 208_799_718
ARCHIVE_SHA256 = "bf9af5cefc9a35e5fa6158b0d72789c13c1b4fcb564e223d3ced02f8f41f6e26"
RECEIPT_FILE_SHA256 = "c06e735749accdea2cae3cd08e9942816918f7f2418b8fa8c6898d5dfab71323"
RECEIPT_FINGERPRINT = "2c845f047bc7983adb0f1f7f7a67831f973052e56c291ea1c543a582af326c9c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_r2_manifest_is_schema_valid_and_keeps_training_blocked() -> None:
    schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)

    source = manifest["sources"][0]
    assert source["archive"]["bytes"] == ARCHIVE_BYTES
    assert source["archive"]["sha256"] == ARCHIVE_SHA256
    assert source["declared_content"]["train_images"] == 2069
    assert source["declared_content"]["validation_images"] == 517
    assert source["eligibility"]["download"] is True
    assert source["eligibility"]["quarantine_inspection"] is True
    assert source["eligibility"]["training"] is False
    assert source["eligibility"]["final_validation_or_test"] is False


def test_pinned_archive_and_immutable_r2_receipt_replay() -> None:
    archive_info = ARCHIVE.stat()
    receipt_info = RECEIPT.stat()
    assert stat.S_ISREG(archive_info.st_mode)
    assert stat.S_ISREG(receipt_info.st_mode)
    assert stat.S_IMODE(archive_info.st_mode) == 0o440
    assert stat.S_IMODE(receipt_info.st_mode) == 0o440
    assert archive_info.st_size == ARCHIVE_BYTES
    assert _sha256(ARCHIVE) == ARCHIVE_SHA256
    assert _sha256(RECEIPT) == RECEIPT_FILE_SHA256

    verified = verify_seed_receipt_file(
        RECEIPT, expected_receipt_sha256=RECEIPT_FINGERPRINT
    )
    assert verified["valid"] is True
    assert verified["external_pin_verified"] is True


def test_quarantine_passes_structure_without_overclaiming_training() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["failure"] is None
    assert receipt["accepted_to_quarantine"] is True
    assert receipt["structural_pass"] is True
    assert receipt["training_eligible"] is False
    assert receipt["archive"]["bytes"] == ARCHIVE_BYTES
    assert receipt["archive"]["sha256"] == ARCHIVE_SHA256
    assert all(gate["passed"] is True for gate in receipt["gates"])

    yolo = receipt["yolo"]
    assert yolo["image_count"] == 2586
    assert yolo["decoded_image_count"] == 2586
    assert yolo["label_file_count"] == 2586
    assert yolo["label_row_count"] == 17_827
    assert yolo["split_image_counts"] == {"train": 2069, "validation": 517}
    assert yolo["class_names"] == [
        "helmet",
        "no_helmet",
        "no_vest",
        "person",
        "vest",
    ]
    assert len(yolo["exact_duplicate_image_groups"]) == 31
    cross_split_exact = [
        group
        for group in yolo["exact_duplicate_image_groups"]
        if any("/train/" in path for path in group["paths"])
        and any("/valid/" in path for path in group["paths"])
    ]
    assert len(cross_split_exact) == 10
    assert "exact_duplicate_image_review_required" in receipt["eligibility_blockers"]
    assert "camera_group_split_review_required" in receipt["eligibility_blockers"]
    assert "embedded_third_party_rights_review_required" in receipt[
        "eligibility_blockers"
    ]
