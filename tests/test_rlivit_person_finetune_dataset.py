from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from pathlib import Path

import pytest

from validation import rlivit_person_finetune_dataset as prep
from validation.rlivit_contract import RLiViTError, canonical_sha256


def _png(width: int = 8, height: int = 6) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + b"\x20\x40\x60" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _annotation(
    sequence: str,
    *,
    timestamp_ns: int,
    person: bool = True,
    invalid_bbox: bool = False,
) -> bytes:
    xmax = 9 if invalid_bbox else 7
    objects = (
        f"""
  <object><name>person</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>{xmax}</xmax><ymax>5</ymax></bndbox></object>"""
        if person
        else ""
    )
    return f"""<annotation verified="yes">
  <folder>{sequence}</folder><filename>02.xml</filename>
  <source><database>R-LiViT</database></source>
  <size><width>8</width><height>6</height><depth>3</depth></size>
  <sequence>{sequence}</sequence><positionInSequence>02</positionInSequence>
  <timestampRGB>{timestamp_ns}</timestampRGB><timestampThermal>{timestamp_ns - 10}</timestampThermal>{objects}
  <object><name>car</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object>
</annotation>
""".encode()


def _pin(path: Path, logical: str) -> dict:
    raw = path.read_bytes()
    return {"path": logical, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _sign(value: dict, field: str = "fingerprint_sha256") -> dict:
    value[field] = canonical_sha256(value)
    return value


def _source_contract(archive: bytes) -> dict:
    return {
        "schema_version": "deepsafe.rlivit-source-contract/v1",
        "dataset": {
            "dataset_id": "R-LiViT_RGB-T_fixture",
            "title": "R-LiViT fixture",
            "version": "1.0",
            "doi": "10.5281/zenodo.16356714",
            "zenodo_record_id": 16356714,
            "publication_date": "2025-07-23",
        },
        "archive": {
            "filename": "R-LiViT_RGB-T.zip",
            "size_bytes": len(archive),
            "checksum": {
                "algorithm": "md5",
                "value": hashlib.md5(archive, usedforsecurity=False).hexdigest(),
            },
            "content_url": "https://zenodo.org/fixture",
        },
        "license": {
            "id": "cc-by-4.0",
            "name": "Creative Commons Attribution 4.0 International",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "R-LiViT fixture",
        },
        "official_references": {
            "github_repository": "https://github.com/XITASO/r-livit",
            "github_commit": "1" * 40,
            "files": [{"path": "README.md", "size_bytes": 1, "sha256": "2" * 64}],
        },
        "archive_dataset_layout": {
            "root_directory": "R-LiViT_RGB-T",
            "rgb_directory": "rgb",
            "thermal_directory": "thermal",
            "annotations_directory": "annotations",
            "sequence_metadata_file": "sequences.xml",
            "sequence_id_xml_element": "rgbt_seq_id",
            "train_split_file": "train.txt",
            "test_split_file": "test.txt",
            "readme_file": "README.md",
        },
        "dataset_expectations": {
            "sequence_count": 10,
            "train_sequence_count": 8,
            "test_sequence_count": 2,
            "annotated_frame_indices": [2],
            "verified_frame_count": 10,
            "total_object_count": 19,
            "source_capture_fps": 5.0,
            "annotation_fps": 1.25,
            "rgb_dimensions": [8, 6],
            "thermal_dimensions": [8, 6],
            "supported_classes": [
                "person", "bicycle", "car", "motorcycle", "bus", "tramway", "truck", "escooter"
            ],
            "required_daytime_values": ["day", "night"],
            "minimum_distinct_locations": 2,
            "campaign_split": "test",
            "model_input_profiles": [640, 960],
        },
        "extraction_limits": {
            "max_entries": 100,
            "max_path_bytes": 256,
            "max_file_uncompressed_bytes": 1024 * 1024,
            "max_total_uncompressed_bytes": 16 * 1024 * 1024,
            "max_overall_expansion_ratio": 100.0,
            "allowed_compression_methods": [0, 8],
            "normalized_mtime_epoch": 0,
        },
        "provenance_notes": ["Synthetic test fixture."],
    }


def _fixture(
    tmp_path: Path,
    *,
    invalid_sequence: str | None = None,
    malformed_sequence: str | None = None,
    split_overlap: bool = False,
) -> tuple[Path, Path, Path]:
    materialized = tmp_path / "materialized"
    source = materialized / "archive/R-LiViT_RGB-T"
    train = tuple(f"{index:03d}" for index in range(8))
    test = ("000", "009") if split_overlap else ("008", "009")
    all_ids = tuple(sorted(set(train) | set(test)))
    metadata = {
        **{f"{index:03d}": ("day", "0") for index in range(4)},
        **{f"{index:03d}": ("night", "1") for index in range(4, 8)},
        "008": ("day", "0"),
        "009": ("night", "1"),
    }
    timestamps = {
        "000": 100_000_000_000,
        "001": 200_000_000_000,
        "002": 300_000_000_000,
        "003": 400_000_000_000,
        "004": 500_000_000_000,
        "005": 600_000_000_000,
        "006": 700_000_000_000,
        "007": 800_000_000_000,
        "008": 900_000_000_000,
        "009": 1_000_000_000_000,
    }
    source.mkdir(parents=True)
    (source / "train.txt").write_text("\n".join(train) + "\n")
    (source / "test.txt").write_text("\n".join(test) + "\n")
    (source / "README.md").write_text("fixture\n")
    sequences_xml = ["<sequences>"]
    for sequence in all_ids:
        daytime, location = metadata[sequence]
        sequences_xml.append(
            f"<sequence><rgbt_seq_id>{sequence}</rgbt_seq_id><daytime>{daytime}</daytime><location>{location}</location></sequence>"
        )
        rgb = source / "rgb" / sequence / "02.png"
        annotation = source / "annotations" / sequence / "02.xml"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        annotation.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(_png())
        annotation.write_bytes(
            b"<annotation>"
            if sequence == malformed_sequence
            else _annotation(
                sequence,
                timestamp_ns=timestamps[sequence],
                person=sequence != "007",
                invalid_bbox=sequence == invalid_sequence,
            )
        )
    sequences_xml.append("</sequences>")
    (source / "sequences.xml").write_text("\n".join(sequences_xml) + "\n")

    archive = b"synthetic archive bytes"
    archive_sha = hashlib.sha256(archive).hexdigest()
    archive_md5 = hashlib.md5(archive, usedforsecurity=False).hexdigest()
    contract = _source_contract(archive)
    contract_path = tmp_path / "source-contract.json"
    _write_json(contract_path, contract)
    contract_pin = _pin(contract_path, "validation/rlivit-source-contract.json")

    extraction_files = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source).as_posix()
        extraction_files.append(
            _pin(path, f"R-LiViT_RGB-T/{relative}")
        )
    archive_value = {
        "filename": "R-LiViT_RGB-T.zip",
        "size_bytes": len(archive),
        "md5": archive_md5,
        "sha256": archive_sha,
    }
    extraction = {
        "schema_version": "deepsafe.rlivit-extraction-manifest/v1",
        "status": "extracted_exactly",
        "archive": archive_value,
        "file_count": len(extraction_files),
        "files": extraction_files,
        "files_fingerprint_sha256": canonical_sha256(extraction_files),
    }
    extraction_path = materialized / "evidence/extraction-manifest.json"
    _write_json(extraction_path, extraction)

    sequence_summary = [
        {
            "sequence_id": sequence,
            "split": "train" if sequence in train else "test",
            "daytime": metadata[sequence][0],
            "location": metadata[sequence][1],
        }
        for sequence in all_ids
    ]
    dataset = _sign(
        {
            "schema_version": "deepsafe.rlivit-dataset-validation/v1",
            "status": "validated",
            "dataset_id": "R-LiViT_RGB-T_fixture",
            "dataset_root": "archive/R-LiViT_RGB-T",
            "gpu_docker_ffmpeg_executed": False,
            "counts": {
                "sequences": len(all_ids),
                "train_sequences": len(train),
                "test_sequences": len(test),
            },
            "sequence_summary": sequence_summary,
        }
    )
    dataset_path = materialized / "derived/dataset-validation-receipt.json"
    _write_json(dataset_path, dataset)

    source_license = {
        "schema_version": "deepsafe.rlivit-source-license-receipt/v1",
        "status": "source_and_license_verified",
        "source_contract": {
            "size_bytes": contract_pin["size_bytes"],
            "sha256": contract_pin["sha256"],
        },
        "archive": archive_value,
        "license": {"id": "cc-by-4.0"},
    }
    source_license_path = materialized / "evidence/source-license-receipt.json"
    _write_json(source_license_path, source_license)

    materialization = _sign(
        {
            "schema_version": "deepsafe.rlivit-materialization-receipt/v1",
            "status": "materialized_cpu_only",
            "dataset_id": "R-LiViT_RGB-T_fixture",
            "gpu_docker_ffmpeg_executed": False,
            "archive": archive_value,
            "receipts": {
                "dataset_validation": _pin(dataset_path, "derived/dataset-validation-receipt.json"),
                "extraction": _pin(extraction_path, "evidence/extraction-manifest.json"),
                "source_license": _pin(source_license_path, "evidence/source-license-receipt.json"),
            },
        }
    )
    _write_json(materialized / "evidence/materialization-receipt.json", materialization)
    return materialized, source, contract_path


def _plan(materialized: Path, source: Path, contract: Path) -> prep.DatasetPlan:
    return prep.build_plan(
        materialized_root=materialized,
        source_root=source,
        source_contract_path=contract,
        expected_train_sequences=8,
        expected_test_sequences=2,
    )


def test_mini_fixture_is_deterministic_stratified_and_test_excluded(tmp_path: Path) -> None:
    first_inputs = _fixture(tmp_path / "first")
    second_inputs = _fixture(tmp_path / "second")
    first = _plan(*first_inputs)
    second = _plan(*second_inputs)

    assert first.manifest_base["plan_fingerprint_sha256"] == second.manifest_base["plan_fingerprint_sha256"]
    assert set(first.train_sequences).isdisjoint(first.calibration_sequences)
    assert (set(first.train_sequences) | set(first.calibration_sequences)) == set(first.official_train_sequences)
    assert (set(first.train_sequences) | set(first.calibration_sequences)).isdisjoint(first.official_test_sequences)
    assert not first.quarantined_train_sequences
    assert len(first.calibration_sequences) == 2
    assert {first.manifest_base["capture_groups"][index]["daytime"] for index in range(len(first.capture_groups))} == {"day", "night"}
    assert all(frame.sequence_id not in first.official_test_sequences for frame in first.frames)


def test_official_test_rgb_is_never_opened_for_plan(tmp_path: Path, monkeypatch) -> None:
    inputs = _fixture(tmp_path)
    original = prep._verify_source_member

    def guarded(**kwargs):
        relative = kwargs["relative"]
        parts = relative.parts
        if len(parts) >= 3 and parts[0] == "rgb" and parts[1] in {"008", "009"}:
            raise AssertionError("official test RGB must not be opened")
        return original(**kwargs)

    monkeypatch.setattr(prep, "_verify_source_member", guarded)
    plan = _plan(*inputs)
    assert not ({"008", "009"} & {frame.sequence_id for frame in plan.frames})


def test_execute_creates_relative_symlinks_yolo_labels_and_audited_empty_label(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path / "inputs")
    plan = _plan(*inputs)
    output = tmp_path / "dataset"
    manifest = prep.execute_plan(plan, output_root=output)

    assert manifest["status"] == "prepared_cpu_only"
    assert manifest["execution"] == {
        "gpu_executed": False,
        "docker_executed": False,
        "model_training_executed": False,
        "model_inference_executed": False,
    }
    assert manifest["qa"]["regular_image_copies"] == 0
    assert manifest["qa"]["hardlinked_images"] == 0
    links = sorted((output / "images").rglob("*.png"))
    assert len(links) == len(plan.frames)
    assert all(path.is_symlink() and not os.path.isabs(os.readlink(path)) for path in links)
    assert (output / "labels/train/007/02.txt").read_bytes() == b""
    nonempty = next(path for path in (output / "labels").rglob("*.txt") if path.stat().st_size)
    fields = nonempty.read_text().strip().split()
    assert fields[0] == "0" and len(fields) == 5
    listed = (output / "train.txt").read_text() + (output / "val.txt").read_text()
    assert "/008/" not in listed and "/009/" not in listed
    dataset_yaml = (output / "dataset.yaml").read_text()
    assert "path:" not in dataset_yaml
    assert dataset_yaml.endswith("  0: person\n")
    assert all(
        line.startswith("./images/train/")
        for line in (output / "train.txt").read_text().splitlines()
    )
    assert all(
        line.startswith("./images/val/")
        for line in (output / "val.txt").read_text().splitlines()
    )


def test_default_plan_is_read_only_and_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path / "inputs")
    plan = _plan(*inputs)
    output = tmp_path / "dataset"
    public = prep.public_plan(plan, output_root=output)
    assert public["status"] == "planned_not_executed"
    assert not output.exists()
    output.mkdir()
    marker = output / "owner.txt"
    marker.write_text("keep\n")
    with pytest.raises(RLiViTError, match="overwrite forbidden"):
        prep.execute_plan(plan, output_root=output)
    assert marker.read_text() == "keep\n"


def test_invalid_person_bbox_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path, invalid_sequence="000")
    with pytest.raises(RLiViTError, match="invalid or out-of-bounds bbox"):
        _plan(*inputs)


def test_malformed_xml_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path, malformed_sequence="000")
    with pytest.raises(RLiViTError, match="malformed"):
        _plan(*inputs)


def test_official_split_overlap_fails_closed(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path, split_overlap=True)
    with pytest.raises(RLiViTError, match="split counts|overlap"):
        _plan(*inputs)


def test_symlinked_source_image_is_rejected(tmp_path: Path) -> None:
    materialized, source, contract = _fixture(tmp_path)
    image = source / "rgb/000/02.png"
    outside = tmp_path / "outside.png"
    outside.write_bytes(image.read_bytes())
    image.unlink()
    image.symlink_to(outside)
    with pytest.raises(RLiViTError, match="non-file source member|safely open"):
        _plan(materialized, source, contract)


def test_symlinked_contracted_source_root_is_rejected(tmp_path: Path) -> None:
    materialized, source, contract = _fixture(tmp_path)
    outside = tmp_path / "moved-source"
    source.rename(outside)
    source.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RLiViTError, match="symlink or non-directory"):
        _plan(materialized, source, contract)


def test_capture_group_boundary_is_integer_inclusive_and_uses_component_max_end() -> None:
    items = (
        prep.SequenceInfo("a", "day", "0", 0, 2_000_000_000),
        prep.SequenceInfo("b", "day", "0", 3_250_000_000, 3_300_000_000),
        prep.SequenceInfo("c", "day", "0", 4_550_000_001, 4_600_000_000),
        prep.SequenceInfo("d", "day", "0", 1_000_000_000, 10_000_000_000),
        prep.SequenceInfo("e", "day", "0", 11_250_000_000, 11_300_000_000),
    )
    groups = prep.build_capture_groups(items)
    # a overlaps d; component max-end is d.end, so e joins at exactly 1.25 s.
    # c is chronological before e and also joins the same expanded component.
    assert len(groups) == 1
    split = prep.build_capture_groups(
        (
            prep.SequenceInfo("a", "day", "0", 0, 1),
            prep.SequenceInfo("b", "day", "0", 1 + prep.CAPTURE_GAP_NS + 1, 10_000_000_000),
        )
    )
    assert len(split) == 2


def test_checked_in_corpus_has_frozen_golden_split() -> None:
    plan = prep.build_plan()
    assert len(plan.capture_groups) == 150
    assert plan.quarantined_train_sequences == ("064",)
    assert plan.calibration_sequences == prep.EXPECTED_CALIBRATION_SEQUENCE_IDS
    assert len(plan.train_sequences) == 127
    assert len(plan.frames) == 159 * 12
    assert plan.manifest_base["splits"]["official_test_exclusion"]["included_output_frames"] == 0
    train = set(plan.train_sequences)
    calibration = set(plan.calibration_sequences)
    official_test = set(plan.official_test_sequences)
    for group in plan.capture_groups:
        members = set(group.sequence_ids)
        assert not (members & train and members & calibration)
        assert not (members & calibration and members & official_test)
    expected_calibration_by_stratum = {
        "day/location-0": 4,
        "day/location-1": 2,
        "day/location-2": 1,
        "day/location-3": 3,
        "day/location-4": 3,
        "day/location-5": 1,
        "day/location-6": 3,
        "day/location-7": 4,
        "night/location-1": 3,
        "night/location-3": 3,
        "night/location-7": 5,
    }
    assert {
        key: row["sequences"]
        for key, row in plan.manifest_base["splits"]["calibration"]["strata"].items()
    } == expected_calibration_by_stratum
