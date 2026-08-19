import json
from pathlib import Path

import pytest

from evaluation.readers import load_predictions_jsonl
from validation.kitti_to_jsonl import convert_kitti_directory
from validation.run_caviar import calculate_streammux_dimensions, parse_deepstream_fps


def _kitti_row(label, left, top, right, bottom, confidence):
    return (
        f"{label} 0.0 0 0.0 {left} {top} {right} {bottom} "
        f"0.0 0.0 0.0 0.0 0.0 0.0 0.0 {confidence}\n"
    )


def test_kitti_conversion_keeps_empty_gt_frames_and_parses_multiword_labels(tmp_path):
    kitti = tmp_path / "kitti"
    kitti.mkdir()
    (kitti / "00_000_000000.txt").write_text(
        _kitti_row("person", -1, 10, 101, 60, 0.9)
        + _kitti_row("traffic light", 1, 2, 3, 4, 0.8),
        encoding="utf-8",
    )
    (kitti / "00_000_000001.txt").write_text("", encoding="utf-8")
    (kitti / "00_000_000002.txt").write_text(
        _kitti_row("person", 20, 20, 30, 40, 0.7), encoding="utf-8"
    )
    labels = tmp_path / "labels.txt"
    labels.write_text("person\ntraffic light\n", encoding="utf-8")
    output = tmp_path / "predictions.jsonl"

    stats = convert_kitti_directory(
        kitti,
        output,
        sequence_id="clip",
        image_width=100,
        image_height=80,
        labels_path=labels,
        expected_frames=3,
        include_frames={0, 1},
        fps=25,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["frame_index"] for row in rows] == [0, 1]
    assert rows[1]["detections"] == []
    assert rows[0]["detections"][0]["bbox_norm_xywh"] == [0.0, 0.125, 1.0, 0.625]
    assert rows[0]["timestamp_ns"] == 0
    assert stats["decoded_frame_files"] == 3
    assert stats["exported_frame_records"] == 2
    assert stats["skipped_unannotated_frames"] == 1
    assert stats["person_detections"] == 1
    assert stats["ignored_non_person_detections"] == 1
    assert stats["clipped_person_boxes"] == 1

    loaded = load_predictions_jsonl(output)
    assert loaded.metadata["frame_records"] == 2
    assert loaded.metadata["person_detections"] == 1


def test_kitti_conversion_rejects_incomplete_decode_sequence(tmp_path):
    kitti = tmp_path / "kitti"
    kitti.mkdir()
    (kitti / "00_000_000000.txt").write_text("", encoding="utf-8")
    (kitti / "00_000_000002.txt").write_text("", encoding="utf-8")
    labels = tmp_path / "labels.txt"
    labels.write_text("person\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        convert_kitti_directory(
            kitti,
            tmp_path / "predictions.jsonl",
            sequence_id="clip",
            image_width=100,
            image_height=80,
            labels_path=labels,
            expected_frames=3,
        )


def test_deepstream_fps_parser_ignores_header(tmp_path):
    log = tmp_path / "deepstream.log"
    log.write_text(
        "**PERF:  FPS 0 (Avg)\n"
        "**PERF:  220.00 (357.71)\n"
        "**PERF:  366.00 (362.75)\n",
        encoding="utf-8",
    )
    parsed = parse_deepstream_fps(log)
    assert parsed["last_reported_average"] == 362.75
    assert parsed["samples"][0] == {"current": 220.0, "average": 357.71}


def test_kitti_mux_coordinates_are_normalized_to_original_image(tmp_path):
    kitti = tmp_path / "kitti"
    kitti.mkdir()
    (kitti / "00_000_000000.txt").write_text(
        _kitti_row("person", 100, 80, 300, 240, 0.8), encoding="utf-8"
    )
    labels = tmp_path / "labels.txt"
    labels.write_text("person\n", encoding="utf-8")
    output = tmp_path / "predictions.jsonl"

    stats = convert_kitti_directory(
        kitti,
        output,
        sequence_id="clip",
        image_width=200,
        image_height=160,
        coordinate_width=400,
        coordinate_height=320,
        labels_path=labels,
        expected_frames=1,
    )

    detection = json.loads(output.read_text())["detections"][0]
    assert detection["bbox_norm_xywh"] == [0.25, 0.25, 0.5, 0.5]
    assert stats["json_image_dimensions"] == [200, 160]
    assert stats["kitti_coordinate_dimensions"] == [400, 320]


def test_streammux_model_active_area_scales_both_low_and_high_resolution_sources():
    assert calculate_streammux_dimensions(
        384, 288, 640, policy="model-active-area"
    ) == (640, 480, pytest.approx(640 / 384))
    assert calculate_streammux_dimensions(
        1920, 1080, 960, policy="model-active-area"
    ) == (960, 540, 0.5)
    assert calculate_streammux_dimensions(
        2720, 1530, 960, policy="model-active-area"
    ) == (960, 540, pytest.approx(960 / 2720))
    # Existing/default behavior remains backward compatible for CAVIAR runs.
    assert calculate_streammux_dimensions(1920, 1080, 960) == (1920, 1080, 1.0)
