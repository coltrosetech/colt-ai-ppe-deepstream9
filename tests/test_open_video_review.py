import json
from pathlib import Path

import pytest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from validation import open_video_review as review


def _record(sequence, frame, width, height, detections):
    return {
        "schema_version": "deepsafe.person-detections/v1",
        "sequence_id": sequence,
        "frame_index": frame,
        "image_width": width,
        "image_height": height,
        "timestamp_ns": frame * 200_000_000,
        "source_uri": "file:///fixture.avi",
        "model_id": "fixture-model",
        "detections": detections,
    }


def _detection(confidence):
    return {
        "class_id": 0,
        "class_name": "person",
        "confidence": confidence,
        "bbox_norm_xywh": [0.1, 0.1, 0.3, 0.6],
    }


def _write_predictions(path: Path, sequence="clip"):
    rows = [
        _record(sequence, 0, 100, 80, [_detection(0.9)]),
        _record(sequence, 1, 100, 80, []),
        _record(sequence, 2, 100, 80, [_detection(0.1)]),
        _record(sequence, 3, 100, 80, [_detection(0.8)]),
        _record(sequence, 4, 100, 80, []),
    ]
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _scene(tmp_path: Path, *, sensitive=False):
    segments = (
        [
            {
                "label": "closed",
                "role": "closed_review",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "is_ground_truth": False,
            }
        ]
        if sensitive
        else [
            {
                "label": "people",
                "role": "person_visible",
                "start_seconds": 0.0,
                "end_seconds": 0.6,
                "is_ground_truth": False,
            },
            {
                "label": "empty_hint",
                "role": "likely_empty",
                "start_seconds": 0.6,
                "end_seconds": 1.0,
                "is_ground_truth": False,
            },
        ]
    )
    return {
        "id": "clip",
        "sensitive": sensitive,
        "video": {"path": str(tmp_path / "fixture.avi"), "sha256": "fixture"},
        "original_source": {"path": "fixture-source", "sha256": "fixture"},
        "video_metadata": {
            "width": 100,
            "height": 80,
            "fps": 5.0,
            "exact_decoded_frame_count": 5,
        },
        "normalization": {"frame_count_preserved": True},
        "source_catalog": {
            "ground_truth": {"available": False},
            "license": {"spdx": "CC0-1.0", "attribution": "fixture"},
        },
        "segments": segments,
    }


def test_model_active_area_mux_contract():
    assert review.expected_mux_dimensions(384, 288, 640) == (
        640,
        480,
        pytest.approx(640 / 384),
    )
    assert review.expected_mux_dimensions(1920, 1080, 960) == (960, 540, 0.5)


def test_gt_free_candidate_categories_respect_threshold_and_segment_hints(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions)
    categories, summary = review.build_review_candidates(
        _scene(tmp_path), predictions, confidence=0.25
    )

    assert [item.frame_index for item in categories["top_detections"]] == [0, 3]
    assert [item.frame_index for item in categories["presence_zero_candidates"]] == [1, 2]
    assert [item.frame_index for item in categories["negative_detection_candidates"]] == [3]
    assert len(categories["diverse_overview"]) == 5
    assert summary["raw_exported_detections"] == 3
    assert summary["accepted_detections"] == 2
    assert summary["segments"]["empty_hint"]["frames_with_detections"] == 1


def test_alignment_inspection_rejects_missing_frame_and_accepts_original_coordinates(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_predictions(run_root / "predictions.jsonl")
    (run_root / "deepstream.log").write_text("fixture\n", encoding="utf-8")
    (run_root / "run-manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "sequence_id": "clip",
                "model_input": 640,
                "video": str(tmp_path / "fixture.avi"),
                "streammux": {
                    "width": 640,
                    "height": 512,
                    "max_nvinfer_upscale": 1.0,
                    "policy": "model-active-area",
                },
                "gpu_safety": {
                    "status": "complete",
                    "execution_boundary": "validation.gpu_guarded_process/v1",
                },
            }
        ),
        encoding="utf-8",
    )
    conversion = {
        "decoded_frame_files": 5,
        "exported_frame_records": 5,
        "skipped_unannotated_frames": 0,
        "kitti_coordinate_dimensions": [640, 512],
        "json_image_dimensions": [100, 80],
    }
    (run_root / "conversion.json").write_text(json.dumps(conversion), encoding="utf-8")
    safety = run_root / "safety/gpu-guard-report.json"
    safety.parent.mkdir(parents=True)
    safety.write_text(
        json.dumps(
            {
                "schema_version": "deepsafe.gpu-guarded-process/v1",
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )
    scene = _scene(tmp_path)
    job = {
        "run_root": str(run_root),
        "model_input": 640,
        "expected_frame_count": 5,
        "coordinate_contract": {
            "source_dimensions": [100, 80],
            "streammux_dimensions": [640, 512],
        },
    }

    result = review.inspect_job(job, scene)
    assert result["state"] == "complete"
    assert result["checks"]["contiguous_zero_based"] is True

    rows = (run_root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    (run_root / "predictions.jsonl").write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    result = review.inspect_job(job, scene)
    assert result["state"] == "incomplete"
    assert any("frame sequence mismatch" in reason for reason in result["reasons"])


def test_sensitive_scene_writes_report_but_withholds_images_and_contact_sheets(
    tmp_path, monkeypatch
):
    if cv2 is None or np is None:
        pytest.skip("OpenCV visual extras are not installed")
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_predictions(run_root / "predictions.jsonl")
    scene = _scene(tmp_path, sensitive=True)
    job = {"run_root": str(run_root), "model_input": 640}
    monkeypatch.setattr(
        review,
        "inspect_job",
        lambda *_: {
            "state": "complete",
            "reasons": [],
            "checks": {"contiguous_zero_based": True},
        },
    )

    report = review.render_review(
        scene,
        job,
        confidence=0.25,
        max_frames=3,
        render_sensitive=False,
    )

    assert report["sensitive_media"]["rendering"] == "withheld"
    assert report["sensitive_media"]["included_in_default_campaign_contact_sheet"] is False
    assert all(
        category["contact_sheet"] is None
        for category in report["categories"].values()
    )
    assert all(
        frame["frame_image"] is None
        for category in report["categories"].values()
        for frame in category["ranked_frames"]
    )
    assert (run_root / "qualitative-review/review.json").is_file()
