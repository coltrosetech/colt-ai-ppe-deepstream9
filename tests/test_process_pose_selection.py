import json
from pathlib import Path

import pytest

from video_selector.process_pose_selection import (
    IMAGE,
    PoseSelectionError,
    build_plan,
)


SELECTION_ID = "d58c660f38d947b9b4db3583c5981889"


def _fixture(root: Path) -> None:
    paths = [
        root / "content/pose_video.py",
        root
        / "third_party/mmpose/configs/body_2d_keypoint/yoloxpose/coco/"
        "yoloxpose_s_8xb32-300e_coco-640.py",
        root
        / "models/pose/candidates/mmpose-yoloxpose-s/"
        "yoloxpose_s_8xb32-300e_coco-640-56c79c1f_20230829.pth",
        root / "content/raw/office.mp4",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    queue = {
        "selection_id": SELECTION_ID,
        "items": [
            {
                "video_id": "04",
                "source_path": "content/raw/office.mp4",
                "clip": {"start_seconds": 1.0, "end_seconds": 3.0},
                "source_video": {
                    "fps": 25.0,
                    "width": 1920,
                    "height": 1080,
                },
            }
        ],
    }
    queue_path = (
        root / "content/video-selector/state/queues" / f"{SELECTION_ID}.json"
    )
    queue_path.parent.mkdir(parents=True)
    queue_path.write_text(json.dumps(queue), encoding="utf-8")


def test_plan_is_separate_gpu_pose_delivery_with_shared_theme(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)

    plan = build_plan(SELECTION_ID, repository_root=tmp_path)

    assert plan["runtime"]["container_image"] == IMAGE
    assert plan["runtime"]["device"] == "cuda:0"
    assert plan["runtime"]["checkpoint_license"] == "Apache-2.0"
    assert plan["visual_contract"]["brand_name"] == "COLT AI - COLLBRAI"
    assert plan["visual_contract"]["model_name_visible"] is False
    assert plan["jobs"][0]["expected"]["frames"] == 50
    assert plan["jobs"][0]["paths"]["final"].endswith(
        "COLT-AI-COLLBRAI-CAM-04-POSE-d58c660f.mp4"
    )


def test_video_filter_and_smoke_frame_cap(tmp_path: Path) -> None:
    _fixture(tmp_path)

    plan = build_plan(
        SELECTION_ID,
        repository_root=tmp_path,
        video_ids={"04"},
        max_frames=12,
    )

    assert plan["max_frames"] == 12
    assert plan["jobs"][0]["expected"]["frames"] == 12
    with pytest.raises(PoseSelectionError, match="not in selection"):
        build_plan(
            SELECTION_ID,
            repository_root=tmp_path,
            video_ids={"99"},
        )
