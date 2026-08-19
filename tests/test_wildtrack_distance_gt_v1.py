from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
import struct
import zipfile
import zlib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from validation import wildtrack_distance_gt_v1 as lane


def test_position_id_mapping_uses_x_first_480_grid() -> None:
    assert lane.position_id_to_ground_m(0) == (-3.0, -9.0)
    assert lane.position_id_to_ground_m(479) == pytest.approx((8.975, -9.0))
    assert lane.position_id_to_ground_m(480) == pytest.approx((-3.0, -8.975))
    assert lane.position_id_to_ground_m(480 * 1440 - 1) == pytest.approx(
        (8.975, 26.975)
    )
    for invalid in (-1, 480 * 1440, True, 1.5):
        with pytest.raises(lane.WildtrackDistanceError):
            lane.position_id_to_ground_m(invalid)  # type: ignore[arg-type]


def test_rodrigues_is_orthonormal_and_matches_quarter_turn() -> None:
    assert np.array_equal(lane.rodrigues([0.0, 0.0, 0.0]), np.eye(3))
    observed = lane.rodrigues([0.0, 0.0, math.pi / 2.0])
    expected = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert np.allclose(observed, expected, atol=1e-12)
    assert np.allclose(observed.T @ observed, np.eye(3), atol=1e-12)
    assert np.linalg.det(observed) == pytest.approx(1.0)


def test_distance_bins_have_exact_20_to_25_boundaries() -> None:
    assert lane._distance_bin(19.999999) == "15-20m"
    assert lane._distance_bin(20.0) == "20-21m"
    assert lane._distance_bin(20.999999) == "20-21m"
    assert lane._distance_bin(21.0) == "21-22m"
    assert lane._distance_bin(24.999999) == "24-25m"
    assert lane._distance_bin(25.0) == "25-30m"


def test_view_contract_accepts_intersecting_truncation_but_not_outside() -> None:
    sentinel = {
        "viewNum": 0,
        "xmin": -1,
        "ymin": -1,
        "xmax": -1,
        "ymax": -1,
    }
    assert lane._validate_view(sentinel, person_id=1) == sentinel
    truncated = {
        "viewNum": 1,
        "xmin": -50,
        "ymin": 100,
        "xmax": 50,
        "ymax": 400,
    }
    assert lane._validate_view(truncated, person_id=1) == truncated
    outside = {
        "viewNum": 1,
        "xmin": 2000,
        "ymin": 100,
        "xmax": 2100,
        "ymax": 400,
    }
    with pytest.raises(lane.WildtrackDistanceError):
        lane._validate_view(outside, person_id=1)


def _extrinsic_xml(camera_x_cm: float, camera_y_cm: float) -> str:
    # pi around X looks toward z=0 from a camera at positive z.
    return f"""<?xml version="1.0"?>
<opencv_storage>
  <rvec>{math.pi} 0 0</rvec>
  <tvec>{-camera_x_cm} {camera_y_cm} 200</tvec>
</opencv_storage>
"""


def _intrinsic_xml() -> str:
    return """<?xml version="1.0"?>
<opencv_storage>
  <camera_matrix type_id="opencv-matrix">
    <rows>3</rows><cols>3</cols><dt>d</dt>
    <data>100 0 960 0 100 540 0 0 1</data>
  </camera_matrix>
  <distortion_coefficients type_id="opencv-matrix">
    <rows>5</rows><cols>1</cols><dt>d</dt><data>0 0 0 0 0</data>
  </distortion_coefficients>
</opencv_storage>
"""


def _make_metadata_fixture(root: Path) -> None:
    annotation_dir = root / "annotations_positions"
    extrinsic_dir = root / "calibrations" / "extrinsic"
    intrinsic_dir = root / "calibrations" / "intrinsic_zero"
    annotation_dir.mkdir(parents=True)
    extrinsic_dir.mkdir(parents=True)
    intrinsic_dir.mkdir(parents=True)
    camera_ground_cm = (-300.0, -900.0)
    for index, name in enumerate(lane.CAMERA_NAMES):
        camera_x = camera_ground_cm[0] + index * 10.0
        camera_y = camera_ground_cm[1]
        (extrinsic_dir / f"extr_{name}.xml").write_text(
            _extrinsic_xml(camera_x, camera_y), encoding="utf-8"
        )
        (intrinsic_dir / f"intr_{name}.xml").write_text(
            _intrinsic_xml(), encoding="utf-8"
        )
    for frame in (0, 5):
        views = []
        for index in range(7):
            # Ground point (-300,-900) projects near the image centre; the
            # camera's x offset shifts it by five pixels per ten centimetres.
            centre_x = 960 - index * 5
            views.append(
                {
                    "viewNum": index,
                    "xmin": centre_x - 20,
                    "ymin": 440,
                    "xmax": centre_x + 20,
                    "ymax": 540,
                }
            )
        payload = [{"personID": 7, "positionID": 0, "views": views}]
        (annotation_dir / f"{frame:08d}.json").write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )


def test_metadata_scan_builds_seven_camera_frame_rows(tmp_path: Path) -> None:
    root = tmp_path / "Wildtrack_dataset"
    _make_metadata_fixture(root)
    result = lane.scan_dataset(
        root,
        require_images=False,
        hash_images=False,
        frame_numbers=(0, 5),
    )
    assert len(result.calibrations) == 7
    assert len(result.rows) == 14
    assert len(result.manifest_rows) == 16
    assert result.summary["annotation_frames"] == 2
    assert result.summary["camera_frames"] == 14
    assert result.summary["source_files"] == 16
    assert result.summary["visible_person_instances"] == 14
    assert result.summary["distance_evaluation_eligible_instances"] == 14
    assert result.summary["truncated_instances"] == 0
    assert all(len(row["persons"]) == 1 for row in result.rows)
    assert all(row["image"]["present"] is False for row in result.rows)
    assert result.summary["position_to_bbox_reprojection_error_px"]["maximum"] < 1e-6
    assert result.summary["bbox_bottom_to_position_ground_error_m"]["maximum"] < 1e-6
    lane._validate_result_schemas(result)


def test_metadata_scan_rejects_missing_frame_and_duplicate_json_key(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Wildtrack_dataset"
    _make_metadata_fixture(root)
    (root / "annotations_positions" / "00000005.json").unlink()
    with pytest.raises(lane.WildtrackDistanceError, match="frame set differs"):
        lane.scan_dataset(
            root,
            require_images=False,
            hash_images=False,
            frame_numbers=(0, 5),
        )
    (root / "annotations_positions" / "00000005.json").write_text(
        '[{"personID":7,"personID":8,"positionID":0,"views":[]}]',
        encoding="utf-8",
    )
    with pytest.raises(lane.WildtrackDistanceError, match="invalid strict JSON"):
        lane.scan_dataset(
            root,
            require_images=False,
            hash_images=False,
            frame_numbers=(0, 5),
        )


def test_xml_entities_are_rejected_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "bad.xml"
    path.write_text(
        '<!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]>'
        "<opencv_storage><rvec>&e;</rvec></opencv_storage>",
        encoding="utf-8",
    )
    with pytest.raises(lane.WildtrackDistanceError, match="unsafe XML"):
        lane._parse_xml(path)


def test_atomic_publication_is_read_only_and_exact(tmp_path: Path) -> None:
    pin = lane._atomic_write(tmp_path / "artifact.json", b'{"ok":true}\n')
    assert pin == {
        "path": "artifact.json",
        "bytes": 12,
        "sha256": "e5f1eb4d806641698a35efe20e098efd20d7d57a9b90ee69079d5bb650920726",
    }
    assert (tmp_path / "artifact.json").stat().st_mode & 0o777 == 0o440
    with pytest.raises(FileExistsError):
        lane._atomic_write(tmp_path / "artifact.json", b'{"changed":true}\n')
    assert (tmp_path / "artifact.json").read_bytes() == b'{"ok":true}\n'


def test_row_and_manifest_schemas_reject_semantic_shape_drift(tmp_path: Path) -> None:
    root = tmp_path / "Wildtrack_dataset"
    _make_metadata_fixture(root)
    result = lane.scan_dataset(
        root,
        require_images=False,
        hash_images=False,
        frame_numbers=(0, 5),
    )
    wrong_camera = copy.deepcopy(result.rows[0])
    wrong_camera["camera_id"] = "IDIAP3"
    with pytest.raises(lane.WildtrackDistanceError, match="violates"):
        lane._schema_validate(
            wrong_camera, lane.ROW_SCHEMA_NAME, context="adversarial row"
        )
    wrong_policy = copy.deepcopy(result.rows[0])
    wrong_policy["persons"][0]["distance_evaluation_eligible"] = False
    with pytest.raises(lane.WildtrackDistanceError, match="violates"):
        lane._schema_validate(
            wrong_policy, lane.ROW_SCHEMA_NAME, context="adversarial row"
        )
    extra_manifest_field = {**result.manifest_rows[0], "unbound": True}
    with pytest.raises(lane.WildtrackDistanceError, match="violates"):
        lane._schema_validate(
            extra_manifest_field,
            lane.MANIFEST_SCHEMA_NAME,
            context="adversarial manifest row",
        )


def test_scan_rejects_extra_entry_and_symlinked_source_leaf(tmp_path: Path) -> None:
    root = tmp_path / "Wildtrack_dataset"
    _make_metadata_fixture(root)
    (root / "annotations_positions" / "ignored-directory").mkdir()
    with pytest.raises(lane.WildtrackDistanceError, match="frame set differs"):
        lane.scan_dataset(
            root,
            require_images=False,
            hash_images=False,
            frame_numbers=(0, 5),
        )

    root = tmp_path / "Wildtrack_dataset_symlink"
    _make_metadata_fixture(root)
    target = root / "calibrations" / "extrinsic" / "extr_CVLab2.xml"
    link = root / "calibrations" / "extrinsic" / "extr_CVLab1.xml"
    link.unlink()
    link.symlink_to(target.name)
    with pytest.raises(lane.WildtrackDistanceError, match="non-regular"):
        lane.scan_dataset(
            root,
            require_images=False,
            hash_images=False,
            frame_numbers=(0, 5),
        )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _minimal_png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1920, 1080, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(b""))
        + _png_chunk(b"IEND", b"")
    )


def test_png_pin_uses_full_payload_and_rejects_crc_drift(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    payload = _minimal_png()
    path.write_bytes(payload)
    assert lane._read_png_pin(path) == {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    changed = bytearray(payload)
    changed[-5] ^= 1
    path.write_bytes(changed)
    with pytest.raises(lane.WildtrackDistanceError, match="CRC differs"):
        lane._read_png_pin(path)


def _manifest_pin(kind: str, path: str, payload: bytes) -> dict:
    return {
        "schema_version": lane.MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_zip(path: Path, members: list[tuple[zipfile.ZipInfo | str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members:
            archive.writestr(name, payload)


def test_archive_member_binding_matches_unique_safe_member_bytes(tmp_path: Path) -> None:
    annotation = b'[{"personID":1}]\n'
    calibration = b"<opencv_storage/>\n"
    rows = [
        _manifest_pin("annotation", "annotations_positions/00000000.json", annotation),
        _manifest_pin(
            "calibration_extrinsic",
            "calibrations/extrinsic/extr_CVLab1.xml",
            calibration,
        ),
    ]
    archive_path = tmp_path / "wildtrack.zip"
    _write_zip(
        archive_path,
        [
            ("Wildtrack_dataset/annotations_positions/00000000.json", annotation),
            (
                "Wildtrack_dataset/calibrations/extrinsic/extr_CVLab1.xml",
                calibration,
            ),
        ],
    )
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    observed = lane._verify_archive_members(
        archive_path,
        accept_archive_sha256=digest,
        manifest_rows=rows,
        expected_archive_bytes=archive_path.stat().st_size,
    )
    assert observed == {
        "bytes": archive_path.stat().st_size,
        "sha256": digest,
        "matched_source_files": 2,
    }

    tampered = copy.deepcopy(rows)
    tampered[0]["sha256"] = "0" * 64
    with pytest.raises(lane.WildtrackDistanceError, match="content differs"):
        lane._verify_archive_members(
            archive_path,
            accept_archive_sha256=digest,
            manifest_rows=tampered,
            expected_archive_bytes=archive_path.stat().st_size,
        )


@pytest.mark.parametrize(
    "attack", ["duplicate", "traversal", "symlink", "directory_type"]
)
def test_archive_member_binding_rejects_unsafe_central_directory(
    tmp_path: Path, attack: str
) -> None:
    payload = b"annotation"
    relative = "annotations_positions/00000000.json"
    member = f"Wildtrack_dataset/{relative}"
    members: list[tuple[zipfile.ZipInfo | str, bytes]] = [(member, payload)]
    if attack == "duplicate":
        members.append((member, payload))
    elif attack == "traversal":
        members.append(("../escape", b"bad"))
    elif attack == "symlink":
        link = zipfile.ZipInfo("Wildtrack_dataset/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        members.append((link, b"target"))
    else:
        wrong_directory = zipfile.ZipInfo("Wildtrack_dataset/wrong-directory/")
        wrong_directory.create_system = 3
        wrong_directory.external_attr = (stat.S_IFREG | 0o755) << 16
        members.append((wrong_directory, b""))
    archive_path = tmp_path / f"{attack}.zip"
    with (pytest.warns(UserWarning) if attack == "duplicate" else nullcontext()):
        _write_zip(archive_path, members)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with pytest.raises(lane.WildtrackDistanceError, match="archive"):
        lane._verify_archive_members(
            archive_path,
            accept_archive_sha256=digest,
            manifest_rows=[_manifest_pin("annotation", relative, payload)],
            expected_archive_bytes=archive_path.stat().st_size,
        )


def test_archive_member_binding_rejects_nested_spoof_root(tmp_path: Path) -> None:
    payload = b"annotation"
    relative = "annotations_positions/00000000.json"
    archive_path = tmp_path / "nested-root.zip"
    _write_zip(
        archive_path,
        [(f"spoof/Wildtrack_dataset/{relative}", payload)],
    )
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with pytest.raises(lane.WildtrackDistanceError, match="does not contain"):
        lane._verify_archive_members(
            archive_path,
            accept_archive_sha256=digest,
            manifest_rows=[_manifest_pin("annotation", relative, payload)],
            expected_archive_bytes=archive_path.stat().st_size,
        )


def test_local_header_resolver_repairs_unique_32bit_wrap_candidate(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "wrapped-offset.zip"
    _write_zip(archive_path, [("dataset/file.txt", b"payload")])
    descriptor = os.open(archive_path, os.O_RDONLY)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo("dataset/file.txt")
            actual_offset = info.header_offset
            info.header_offset += 1 << 32
            local = lane._resolve_zip_local_member(
                descriptor,
                info,
                central_start=archive.start_dir,
            )
            assert local.header_offset == actual_offset
            size, digest, crc = lane._hash_raw_zip_member(
                descriptor,
                info,
                local,
                expected_bytes=len(b"payload"),
            )
            assert (size, digest, crc) == (
                7,
                hashlib.sha256(b"payload").hexdigest(),
                zlib.crc32(b"payload"),
            )
    finally:
        os.close(descriptor)


def test_local_header_resolver_rejects_wrong_local_name(tmp_path: Path) -> None:
    archive_path = tmp_path / "wrong-local-name.zip"
    _write_zip(archive_path, [("dataset/file.txt", b"payload")])
    with zipfile.ZipFile(archive_path) as archive:
        offset = archive.getinfo("dataset/file.txt").header_offset
    with archive_path.open("r+b") as stream:
        stream.seek(offset + 30)
        stream.write(b"X")
    descriptor = os.open(archive_path, os.O_RDONLY)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            with pytest.raises(lane.WildtrackDistanceError, match="invalid or ambiguous"):
                lane._resolve_zip_local_member(
                    descriptor,
                    archive.getinfo("dataset/file.txt"),
                    central_start=archive.start_dir,
                )
    finally:
        os.close(descriptor)


def test_local_header_resolver_rejects_malformed_offset(tmp_path: Path) -> None:
    archive_path = tmp_path / "malformed-offset.zip"
    _write_zip(archive_path, [("dataset/file.txt", b"payload")])
    descriptor = os.open(archive_path, os.O_RDONLY)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo("dataset/file.txt")
            info.header_offset += 123
            with pytest.raises(lane.WildtrackDistanceError, match="invalid or ambiguous"):
                lane._resolve_zip_local_member(
                    descriptor,
                    info,
                    central_start=archive.start_dir,
                )
    finally:
        os.close(descriptor)


def test_local_header_resolver_rejects_ambiguous_wrap_candidates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ambiguous-local-headers.bin"
    name = b"file"
    header = struct.pack(
        "<IHHHHHIIIHH",
        0x04034B50,
        20,
        0,
        zipfile.ZIP_STORED,
        0,
        0,
        0,
        0,
        0,
        len(name),
        0,
    ) + name
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.pwrite(descriptor, header, 0)
        os.pwrite(descriptor, header, 1 << 32)
        info = zipfile.ZipInfo("file")
        info.header_offset = 1 << 32
        info.flag_bits = 0
        info.compress_type = zipfile.ZIP_STORED
        info.compress_size = 0
        info.file_size = 0
        info.CRC = 0
        with pytest.raises(lane.WildtrackDistanceError, match="invalid or ambiguous"):
            lane._resolve_zip_local_member(
                descriptor,
                info,
                central_start=(1 << 32) + len(header) + 1,
            )
    finally:
        os.close(descriptor)


def _receipt_ready_result(root: Path) -> lane.ScanResult:
    _make_metadata_fixture(root)
    result = lane.scan_dataset(
        root,
        require_images=False,
        hash_images=False,
        frame_numbers=(0, 5),
    )
    result.summary["annotation_frames"] = 400
    result.summary["camera_frames"] = 2800
    result.summary["source_files"] = 3214
    result.summary["twenty_to_twenty_five_m_visible_instances"] = 1000
    result.summary["twenty_to_twenty_five_m_eligible_instances"] = 1000
    result.manifest_rows.extend(
        [copy.deepcopy(result.manifest_rows[-1]) for _ in range(3214 - 16)]
    )
    return result


def _valid_receipt(root: Path) -> tuple[dict, lane.ScanResult, dict, dict, dict]:
    result = _receipt_ready_result(root)
    archive_pin = {
        "bytes": lane.ARCHIVE_EXPECTED_BYTES,
        "sha256": "a" * 64,
        "matched_source_files": 3214,
    }
    artifacts = {
        "ground_truth": {"path": "ground-truth.jsonl", "bytes": 1, "sha256": "b" * 64},
        "source_manifest": {"path": "source-manifest.jsonl", "bytes": 1, "sha256": "c" * 64},
    }
    implementation = lane._implementation_pins()
    receipt = lane._build_receipt(
        result=result,
        archive_pin=archive_pin,
        artifacts=artifacts,
        implementation=implementation,
        created_at_utc="2026-07-18T12:34:56Z",
    )
    return receipt, result, archive_pin, artifacts, implementation


def test_receipt_schema_is_closed_and_resealed_tamper_fails_replay(
    tmp_path: Path,
) -> None:
    receipt, result, archive_pin, artifacts, implementation = _valid_receipt(
        tmp_path / "Wildtrack_dataset"
    )
    lane._schema_validate(receipt, lane.RECEIPT_SCHEMA_NAME, context="test receipt")
    extra = {**receipt, "unbound": True}
    with pytest.raises(lane.WildtrackDistanceError, match="violates"):
        lane._schema_validate(extra, lane.RECEIPT_SCHEMA_NAME, context="test receipt")

    tampered = copy.deepcopy(receipt)
    tampered["summary"]["twenty_to_twenty_five_m_visible_instances"] += 1
    tampered["fingerprint_sha256"] = lane._fingerprint(tampered)
    lane._schema_validate(tampered, lane.RECEIPT_SCHEMA_NAME, context="tampered receipt")
    with pytest.raises(lane.WildtrackDistanceError, match="semantic replay differs"):
        lane._verify_receipt_replay(
            tampered,
            result=result,
            archive_pin=archive_pin,
            artifacts=artifacts,
            implementation=implementation,
        )


def test_published_receipt_requires_external_hash_and_canonical_bytes(
    tmp_path: Path,
) -> None:
    receipt, *_ = _valid_receipt(tmp_path / "Wildtrack_dataset")
    output = tmp_path / "output"
    output.mkdir()
    payload = lane._canonical_bytes(receipt) + b"\n"
    (output / "receipt.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    observed, pin = lane._load_published_receipt(
        output, expected_receipt_sha256=digest
    )
    assert observed == receipt
    assert pin["sha256"] == digest
    with pytest.raises(lane.WildtrackDistanceError, match="external SHA-256"):
        lane._load_published_receipt(output, expected_receipt_sha256="0" * 64)

    noncanonical = lane._canonical_bytes(receipt) + b" \n"
    (output / "receipt.json").write_bytes(noncanonical)
    with pytest.raises(lane.WildtrackDistanceError, match="not canonical"):
        lane._load_published_receipt(
            output,
            expected_receipt_sha256=hashlib.sha256(noncanonical).hexdigest(),
        )


def test_artifact_semantic_replay_rejects_resealed_distance_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Wildtrack_dataset"
    _make_metadata_fixture(root)
    result = lane.scan_dataset(
        root,
        require_images=False,
        hash_images=False,
        frame_numbers=(0, 5),
    )
    output = tmp_path / "published"
    output.mkdir()
    rows = copy.deepcopy(result.rows)
    rows[0]["persons"][0]["horizontal_distance_m"] = 99.0
    row_pin = lane._atomic_write(output / "ground-truth.jsonl", lane._jsonl(rows))
    manifest_pin = lane._atomic_write(
        output / "source-manifest.jsonl", lane._jsonl(result.manifest_rows)
    )
    receipt_projection = {
        "artifacts": {
            "ground_truth": row_pin,
            "source_manifest": manifest_pin,
        }
    }
    with pytest.raises(lane.WildtrackDistanceError, match="semantic replay differs"):
        lane._load_and_verify_artifacts(
            output, receipt_projection, expected_result=result
        )
