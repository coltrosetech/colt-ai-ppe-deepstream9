import copy
import hashlib
import json
import os
import stat
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from validation import rlivit_contract as contract_module
from validation import rlivit_materializer as materializer
from validation.rlivit_contract import RLiViTError


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
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def _annotation(*, verified: str = "yes", xmax: int = 6) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<annotation verified="{verified}">
  <size><width>8</width><height>6</height><depth>3</depth></size>
  <object><name>person</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>{xmax}</xmax><ymax>5</ymax></bndbox></object>
</annotation>
"""


def _source_contract(archive: Path) -> dict:
    raw = archive.read_bytes()
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
            "size_bytes": len(raw),
            "checksum": {
                "algorithm": "md5",
                "value": hashlib.md5(raw, usedforsecurity=False).hexdigest(),
            },
            "content_url": "https://zenodo.org/api/records/16356714/files/R-LiViT_RGB-T.zip/content",
        },
        "license": {
            "id": "cc-by-4.0",
            "name": "Creative Commons Attribution 4.0 International",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "R-LiViT fixture attribution",
        },
        "official_references": {
            "github_repository": "https://github.com/XITASO/r-livit",
            "github_commit": "1" * 40,
            "files": [
                {"path": "README.md", "size_bytes": 1, "sha256": "2" * 64}
            ],
        },
        "archive_dataset_layout": {
            "root_directory": "RLiViT",
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
            "sequence_count": 2,
            "train_sequence_count": 1,
            "test_sequence_count": 1,
            "annotated_frame_indices": [2, 6],
            "verified_frame_count": 4,
            "total_object_count": 4,
            "source_capture_fps": 5.0,
            "annotation_fps": 1.25,
            "rgb_dimensions": [8, 6],
            "thermal_dimensions": [8, 6],
            "supported_classes": [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "bus",
                "tramway",
                "truck",
                "escooter",
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
        "provenance_notes": ["Synthetic fixture only."],
    }


def _build_archive(
    tmp_path: Path,
    *,
    verified: str = "yes",
    xmax: int = 6,
    extra: list[tuple[zipfile.ZipInfo | str, bytes | str]] | None = None,
    split_overlap: bool = False,
) -> tuple[Path, Path, dict]:
    archive = tmp_path / "R-LiViT_RGB-T.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("RLiViT/train.txt", "S001\n")
        zf.writestr("RLiViT/test.txt", "S001\n" if split_overlap else "S002\n")
        zf.writestr("RLiViT/README.md", "Synthetic R-LiViT fixture.\n")
        zf.writestr(
            "RLiViT/sequences.xml",
            """<sequences>
<sequence><rgbt_seq_id>S001</rgbt_seq_id><daytime>day</daytime><location>L1</location></sequence>
<sequence><rgbt_seq_id>S002</rgbt_seq_id><daytime>night</daytime><location>L2</location></sequence>
</sequences>""",
        )
        for sequence in ("S001", "S002"):
            for frame in (2, 6):
                zf.writestr(f"RLiViT/rgb/{sequence}/{frame:02d}.png", _png())
                zf.writestr(f"RLiViT/thermal/{sequence}/{frame:02d}.png", _png())
                zf.writestr(
                    f"RLiViT/annotations/{sequence}/{frame:02d}.xml",
                    _annotation(
                        verified=verified if (sequence, frame) == ("S002", 6) else "yes",
                        xmax=xmax if (sequence, frame) == ("S002", 6) else 6,
                    ),
                )
        for name, content in extra or []:
            zf.writestr(name, content)
    source_contract = _source_contract(archive)
    contract_path = tmp_path / "source-contract.json"
    contract_path.write_text(
        json.dumps(source_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archive, contract_path, source_contract


def test_status_is_stat_only_and_does_not_open_archive(tmp_path: Path, monkeypatch) -> None:
    archive, _contract_path, source_contract = _build_archive(tmp_path)

    def forbidden_open(*_args, **_kwargs):
        raise AssertionError("status must not call os.open")

    monkeypatch.setattr(contract_module.os, "open", forbidden_open)
    result = contract_module.archive_status(archive, source_contract)
    assert result["status"] == "expected_size_reached_confirmation_required"
    assert result["bytes_read_from_archive"] == 0
    assert result["checksum_computed"] is False


def test_checked_in_source_contract_pins_official_zenodo_object() -> None:
    value, pin = contract_module.load_source_contract()
    assert value["dataset"]["zenodo_record_id"] == 16356714
    assert value["dataset"]["doi"] == "10.5281/zenodo.16356714"
    assert value["archive"] == {
        "checksum": {
            "algorithm": "md5",
            "value": "88e3db20698705017e4f06e4bbb2dde6",
        },
        "content_url": "https://zenodo.org/api/records/16356714/files/R-LiViT_RGB-T.zip/content",
        "filename": "R-LiViT_RGB-T.zip",
        "size_bytes": 3798751628,
    }
    assert value["license"]["id"] == "cc-by-4.0"
    assert pin["size_bytes"] > 0
    assert len(pin["sha256"]) == 64


def test_archive_read_requires_explicit_completion_confirmation(tmp_path: Path) -> None:
    archive, _contract_path, source_contract = _build_archive(tmp_path)
    with pytest.raises(RLiViTError, match="explicit download-complete"):
        contract_module.hash_verified_archive(
            archive,
            source_contract,
            download_complete_confirmed=False,
        )


def test_read_only_inspection_validates_central_directory_without_extraction(
    tmp_path: Path,
) -> None:
    archive, _contract_path, source_contract = _build_archive(tmp_path)
    result = materializer.inspect_verified_archive(
        archive_path=archive,
        contract=source_contract,
        download_complete_confirmed=True,
    )
    assert result["status"] == "archive_checksum_and_zip_metadata_validated"
    assert result["file_count"] == 16
    assert result["directory_entry_count"] == 0
    assert result["archive_members_read"] is False
    assert result["extraction_executed"] is False
    assert result["dataset_semantics_validated"] is False
    assert result["source_contract_canonical_sha256"] == materializer.canonical_sha256(
        source_contract
    )
    assert not (tmp_path / "RLiViT").exists()


def test_materializer_passes_checksum_verified_handle_to_zipfile(
    tmp_path: Path, monkeypatch
) -> None:
    archive, contract_path, _source_contract_value = _build_archive(tmp_path)
    original = materializer.zipfile.ZipFile
    opened_with: list[object] = []

    def checking_zipfile(file, *args, **kwargs):
        opened_with.append(file)
        assert not isinstance(file, (str, os.PathLike))
        assert hasattr(file, "fileno")
        return original(file, *args, **kwargs)

    monkeypatch.setattr(materializer.zipfile, "ZipFile", checking_zipfile)
    materializer.materialize(
        archive_path=archive,
        output_root=tmp_path / "materialized",
        contract_path=contract_path,
        download_complete_confirmed=True,
    )
    assert len(opened_with) == 1


def test_materializes_valid_fixture_atomically_and_builds_cpu_only_plans(tmp_path: Path) -> None:
    archive, contract_path, _source_contract_value = _build_archive(tmp_path)
    output = tmp_path / "materialized"
    receipt = materializer.materialize(
        archive_path=archive,
        output_root=output,
        contract_path=contract_path,
        download_complete_confirmed=True,
    )
    assert receipt["status"] == "materialized_cpu_only"
    assert receipt["gpu_docker_ffmpeg_executed"] is False
    assert archive.is_file()
    assert output.is_dir()
    source = json.loads((output / "evidence/source-license-receipt.json").read_text())
    assert source["license"]["id"] == "cc-by-4.0"
    assert source["attribution_required"] is True
    dataset = json.loads((output / "derived/dataset-validation-receipt.json").read_text())
    assert dataset["counts"] == {
        "campaign_jobs": 2,
        "campaign_sequences": 1,
        "objects": 4,
        "persons": 4,
        "sequences": 2,
        "test_sequences": 1,
        "train_sequences": 1,
        "verified_frames": 4,
    }
    gt_lines = (output / "derived/gt/person-visible-all.jsonl").read_text().splitlines()
    assert len(gt_lines) == 4
    assert all(json.loads(line)["person_count"] == 1 for line in gt_lines)
    campaign = json.loads((output / "derived/plans/person-campaign-plan.json").read_text())
    assert campaign["profiles"] == [640, 960]
    assert campaign["sequence_count"] == 1
    assert campaign["expected_jobs"] == 2
    assert campaign["gpu_execution_requested"] is False
    mp4 = json.loads((output / "derived/plans/mp4-plan.json").read_text())
    assert mp4["expected_video_count"] == 2
    assert mp4["ffmpeg_executed"] is False
    assert list((output / "derived/videos").iterdir()) == []
    assert all(path.lstat().st_mtime_ns == 0 for path in output.rglob("*") if path.is_file())


def test_materialization_is_byte_deterministic_for_same_archive(tmp_path: Path) -> None:
    archive, contract_path, _source_contract_value = _build_archive(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    materializer.materialize(
        archive_path=archive,
        output_root=first,
        contract_path=contract_path,
        download_complete_confirmed=True,
    )
    materializer.materialize(
        archive_path=archive,
        output_root=second,
        contract_path=contract_path,
        download_complete_confirmed=True,
    )
    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


def test_wrong_zenodo_checksum_fails_before_zip_inspection(tmp_path: Path, monkeypatch) -> None:
    archive, _contract_path, source_contract = _build_archive(tmp_path)
    source_contract["archive"]["checksum"]["value"] = "0" * 32

    def forbidden_zip(*_args, **_kwargs):
        raise AssertionError("ZIP must not open before checksum succeeds")

    monkeypatch.setattr(materializer.zipfile, "ZipFile", forbidden_zip)
    with pytest.raises(RLiViTError, match="MD5 differs"):
        contract_module.hash_verified_archive(
            archive,
            source_contract,
            download_complete_confirmed=True,
        )


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("zip_slip", "escapes extraction root"),
        ("symlink", "symlink entry"),
        ("casefold", "case-folding ZIP collision"),
    ],
)
def test_rejects_unsafe_zip_entries_without_partial_publication(
    tmp_path: Path, mode: str, match: str
) -> None:
    if mode == "zip_slip":
        extra = [("../escape.txt", b"escape")]
    elif mode == "symlink":
        link = zipfile.ZipInfo("RLiViT/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        extra = [(link, b"../../outside")]
    else:
        extra = [("RLiViT/README", b"one"), ("rlivit/readme", b"two")]
    archive, contract_path, _source_contract_value = _build_archive(
        tmp_path, extra=extra
    )
    output = tmp_path / "materialized"
    with pytest.raises(RLiViTError, match=match):
        materializer.materialize(
            archive_path=archive,
            output_root=output,
            contract_path=contract_path,
            download_complete_confirmed=True,
        )
    assert not output.exists()
    assert not (tmp_path / "escape.txt").exists()
    assert not list(tmp_path.glob(".materialized.staging-*"))
    assert archive.is_file()


def test_rejects_zip_bomb_by_declared_total_before_extraction(tmp_path: Path) -> None:
    archive, contract_path, source_contract = _build_archive(tmp_path)
    source_contract["extraction_limits"]["max_total_uncompressed_bytes"] = 100
    contract_path.write_text(json.dumps(source_contract), encoding="utf-8")
    with pytest.raises(RLiViTError, match="total uncompressed"):
        materializer.materialize(
            archive_path=archive,
            output_root=tmp_path / "materialized",
            contract_path=contract_path,
            download_complete_confirmed=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"verified": "no"}, "not verified=yes"),
        ({"xmax": 99}, "x bounds are invalid"),
        ({"split_overlap": True}, "train/test sequence leakage"),
    ],
)
def test_dataset_semantic_validation_fails_closed(
    tmp_path: Path, kwargs: dict, match: str
) -> None:
    archive, contract_path, _source_contract_value = _build_archive(tmp_path, **kwargs)
    output = tmp_path / "materialized"
    with pytest.raises(RLiViTError, match=match):
        materializer.materialize(
            archive_path=archive,
            output_root=output,
            contract_path=contract_path,
            download_complete_confirmed=True,
        )
    assert not output.exists()
    assert archive.is_file()


def test_refuses_archive_symlink_and_existing_output(tmp_path: Path) -> None:
    archive, contract_path, _source_contract_value = _build_archive(tmp_path)
    link = tmp_path / "linked" / archive.name
    link.parent.mkdir()
    link.symlink_to(archive)
    with pytest.raises(RLiViTError, match="symlink"):
        materializer.materialize(
            archive_path=link,
            output_root=tmp_path / "materialized",
            contract_path=contract_path,
            download_complete_confirmed=True,
        )
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(RLiViTError, match="overwrite"):
        materializer.materialize(
            archive_path=archive,
            output_root=output,
            contract_path=contract_path,
            download_complete_confirmed=True,
        )


def test_contract_rejects_incoherent_frame_matrix(tmp_path: Path) -> None:
    archive, _contract_path, source_contract = _build_archive(tmp_path)
    broken = copy.deepcopy(source_contract)
    broken["dataset_expectations"]["verified_frame_count"] = 3
    with pytest.raises(RLiViTError, match="frame matrix"):
        contract_module.validate_source_contract(broken)
