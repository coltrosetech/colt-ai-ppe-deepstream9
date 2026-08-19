import numpy as np
import pytest

from content.pose_video import (
    PoseVideoError,
    SAFE,
    draw_pose_frame,
    prediction_row,
)


def _pose() -> tuple[np.ndarray, ...]:
    keypoints = np.zeros((1, 17, 2), dtype=np.float32)
    keypoints[0, :, 0] = np.linspace(20, 140, 17)
    keypoints[0, :, 1] = np.linspace(30, 180, 17)
    scores = np.full((1, 17), 0.9, dtype=np.float32)
    boxes = np.asarray([[10, 20, 160, 190]], dtype=np.float32)
    box_scores = np.asarray([0.95], dtype=np.float32)
    return keypoints, scores, boxes, box_scores


def test_draw_pose_frame_adds_shared_brand_panel_and_skeleton() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    keypoints, scores, boxes, box_scores = _pose()

    rendered, people = draw_pose_frame(
        frame,
        keypoints=keypoints,
        keypoint_scores=scores,
        bboxes=boxes,
        bbox_scores=box_scores,
        keypoint_threshold=0.3,
        camera_label="CAM-04",
    )

    assert people == 1
    assert rendered.shape == (240, 320, 3)
    assert np.count_nonzero(rendered) > 0
    assert np.any(np.all(rendered == np.asarray(SAFE), axis=2))


def test_draw_pose_frame_rejects_wrong_keypoint_contract() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    keypoints, scores, boxes, box_scores = _pose()

    with pytest.raises(PoseVideoError, match="keypoints"):
        draw_pose_frame(
            frame,
            keypoints=keypoints[:, :16],
            keypoint_scores=scores[:, :16],
            bboxes=boxes,
            bbox_scores=box_scores,
            keypoint_threshold=0.3,
            camera_label="CAM-04",
        )


def test_prediction_row_uses_normalized_coco17_contract() -> None:
    keypoints, scores, boxes, box_scores = _pose()

    row = prediction_row(
        frame_index=25,
        fps=25.0,
        width=320,
        height=240,
        keypoints=keypoints,
        keypoint_scores=scores,
        bboxes=boxes,
        bbox_scores=box_scores,
        keypoint_threshold=0.3,
        camera_label="CAM-04",
    )

    assert row["schema_version"] == "colt-ai.pose-prediction-frame/v1"
    assert row["timestamp_seconds"] == 1.0
    assert len(row["poses"]) == 1
    assert len(row["poses"][0]["keypoints"]) == 17
    assert row["poses"][0]["keypoints"][0]["name"] == "nose"
    assert row["poses"][0]["keypoints"][0]["visible"] is True
