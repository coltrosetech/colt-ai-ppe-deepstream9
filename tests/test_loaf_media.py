import json
from pathlib import Path

import pytest

from validation.loaf_distance_band import METRIC_GEOMETRY, SCHEMA_VERSION
from validation import loaf_media
from validation.loaf_media import build_media_plan


def _write_gt(path: Path, images):
    path.write_text(
        json.dumps(
            {
                "info": {
                    "deepsafe_distance_band": {
                        "schema_version": SCHEMA_VERSION,
                        "distance_band_m": {
                            "minimum_inclusive": 20,
                            "maximum_exclusive": 25,
                        },
                        "metric_geometry": METRIC_GEOMETRY,
                    }
                },
                "images": images,
                "annotations": [],
                "categories": [{"id": 1, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )


def test_build_media_plan_preserves_frame_order_and_is_dry(tmp_path):
    images_root = tmp_path / "images"
    images_root.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (images_root / name).write_bytes(b"not-decoded-in-plan")
    gt = tmp_path / "gt.json"
    _write_gt(
        gt,
        [
            {"id": 2, "sequence_id": "scene/one", "frame_id": 1, "file_name": "b.jpg", "width": 512, "height": 512},
            {"id": 1, "sequence_id": "scene/one", "frame_id": 0, "file_name": "a.jpg", "width": 512, "height": 512},
        ],
    )

    plan = build_media_plan(gt, images_root, tmp_path / "out")

    assert plan["status"] == "planned_not_encoded"
    assert plan["sequence_count"] == 1
    assert plan["frame_count"] == 2
    assert plan["execution_policy"] == {
        "process": "local ffmpeg only",
        "decoder_hardware_acceleration": "disabled (-hwaccel none)",
        "encoder": "libx264",
        "gpu_used": False,
        "docker_used": False,
        "network_used": False,
        "local_files_only": True,
    }
    assert plan["evaluation_policy"]["inference_performed"] is False
    assert plan["evaluation_policy"]["metrics_computed"] is False
    assert plan["evaluation_policy"]["threshold_or_model_tuning_performed"] is False
    sequence = plan["sequences"][0]
    assert [frame["image_id"] for frame in sequence["frames"]] == [1, 2]
    assert all(len(frame["sha256"]) == 64 for frame in sequence["frames"])
    assert sequence["frame_count"] == 2
    assert "-hwaccel" in sequence["ffmpeg"]
    assert sequence["ffmpeg"][sequence["ffmpeg"].index("-hwaccel") + 1] == "none"
    assert sequence["ffmpeg"][sequence["ffmpeg"].index("-c:v") + 1] == "libx264"
    assert not (tmp_path / "out").exists()


def test_build_media_plan_marks_test_unseen_as_media_only(tmp_path):
    images_root = tmp_path / "images"
    images_root.mkdir()
    (images_root / "a.jpg").write_bytes(b"local")
    gt = tmp_path / "gt.json"
    _write_gt(
        gt,
        [
            {
                "id": 1,
                "sequence_id": "scene",
                "frame_id": 0,
                "file_name": "a.jpg",
                "width": 512,
                "height": 512,
            }
        ],
    )
    payload = json.loads(gt.read_text(encoding="utf-8"))
    payload["info"]["deepsafe_distance_band"].update(
        {
            "source": {"path": "annotations/instances_test-unseen.json"},
            "leakage_policy": "never tune on test-unseen",
        }
    )
    gt.write_text(json.dumps(payload), encoding="utf-8")

    plan = build_media_plan(gt, images_root, tmp_path / "out")

    assert plan["source_split"] == "test-unseen"
    assert plan["evaluation_policy"]["test_unseen_tuning_allowed"] is False
    assert plan["evaluation_policy"]["source_leakage_policy"] == "never tune on test-unseen"
    assert "media-only" in plan["evaluation_policy"]["note"]


def test_build_media_plan_rejects_path_escape(tmp_path):
    images_root = tmp_path / "images"
    images_root.mkdir()
    (tmp_path / "outside.jpg").write_bytes(b"x")
    gt = tmp_path / "gt.json"
    _write_gt(
        gt,
        [{"id": 1, "sequence_id": "scene", "frame_id": 0, "file_name": "../outside.jpg", "width": 512, "height": 512}],
    )
    with pytest.raises(ValueError, match="leaves images root"):
        build_media_plan(gt, images_root, tmp_path / "out")


def test_build_media_plan_requires_contiguous_frame_ids(tmp_path):
    images_root = tmp_path / "images"
    images_root.mkdir()
    (images_root / "a.jpg").write_bytes(b"x")
    gt = tmp_path / "gt.json"
    _write_gt(
        gt,
        [{"id": 1, "sequence_id": "scene", "frame_id": 1, "file_name": "a.jpg", "width": 512, "height": 512}],
    )
    with pytest.raises(ValueError, match="contiguous from zero"):
        build_media_plan(gt, images_root, tmp_path / "out")


def test_build_media_plan_rejects_wrong_geometry_contract(tmp_path):
    images_root = tmp_path / "images"
    images_root.mkdir()
    (images_root / "a.jpg").write_bytes(b"x")
    gt = tmp_path / "gt.json"
    _write_gt(
        gt,
        [
            {
                "id": 1,
                "sequence_id": "scene",
                "frame_id": 0,
                "file_name": "a.jpg",
                "width": 512,
                "height": 512,
            }
        ],
    )
    payload = json.loads(gt.read_text(encoding="utf-8"))
    payload["info"]["deepsafe_distance_band"]["metric_geometry"] = "raw_loaf_bbox"
    gt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="required rotated-box AABB envelope"):
        build_media_plan(gt, images_root, tmp_path / "out")


def test_execute_media_plan_rejects_wrong_encoded_frame_rate(tmp_path, monkeypatch):
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    output = tmp_path / "video.mp4"
    output.write_bytes(b"encoded")
    plan = {
        "sequences": [
            {
                "sequence_id": "loaf-0001",
                "frame_count": 1,
                "fps": 1,
                "width": 512,
                "height": 512,
                "frames": [
                    {
                        "frame_id": 0,
                        "source_path": str(source),
                        "size_bytes": source.stat().st_size,
                        "sha256": loaf_media._sha256(source),
                    }
                ],
                "staging_pattern": str(tmp_path / "staging" / "%06d.jpg"),
                "output_video": str(output),
                "ffmpeg": loaf_media._cpu_ffmpeg_command(
                    fps=1,
                    frame_count=1,
                    staging_pattern=tmp_path / "staging" / "%06d.jpg",
                    output_path=output,
                ),
            }
        ]
    }
    monkeypatch.setattr(loaf_media.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        loaf_media,
        "_probe_exact",
        lambda path: {
            "codec": "h264",
            "width": 512,
            "height": 512,
            "fps_fraction": "2/1",
            "exact_decoded_frame_count": 1,
        },
    )

    with pytest.raises(ValueError, match="frame rate"):
        loaf_media.execute_media_plan(plan)


def test_execute_media_plan_rejects_modified_or_gpu_encode_command(tmp_path):
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    output = tmp_path / "video.mp4"
    staging_pattern = tmp_path / "staging" / "%06d.jpg"
    command = loaf_media._cpu_ffmpeg_command(
        fps=1,
        frame_count=1,
        staging_pattern=staging_pattern,
        output_path=output,
    )
    command[command.index("libx264")] = "h264_nvenc"
    plan = {
        "sequences": [
            {
                "sequence_id": "loaf-0001",
                "frame_count": 1,
                "fps": 1,
                "width": 512,
                "height": 512,
                "frames": [
                    {
                        "frame_id": 0,
                        "source_path": str(source),
                        "size_bytes": source.stat().st_size,
                        "sha256": loaf_media._sha256(source),
                    }
                ],
                "staging_pattern": str(staging_pattern),
                "output_video": str(output),
                "ffmpeg": command,
            }
        ]
    }

    with pytest.raises(ValueError, match="only local ffmpeg"):
        loaf_media.execute_media_plan(plan)


def test_execute_media_plan_rejects_source_changed_after_planning(tmp_path):
    images_root = tmp_path / "images"
    images_root.mkdir()
    image = images_root / "a.jpg"
    image.write_bytes(b"before")
    gt = tmp_path / "gt.json"
    _write_gt(
        gt,
        [
            {
                "id": 1,
                "sequence_id": "scene",
                "frame_id": 0,
                "file_name": "a.jpg",
                "width": 512,
                "height": 512,
            }
        ],
    )
    plan = build_media_plan(gt, images_root, tmp_path / "out")
    image.write_bytes(b"aft3r!")

    with pytest.raises(ValueError, match="changed after plan creation"):
        loaf_media.execute_media_plan(plan)
