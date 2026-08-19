import copy

import pytest

from validation.loaf_distance_band import SCHEMA_VERSION
from validation.loaf_visual_audit import (
    EXPECTED_METRIC_GEOMETRY,
    VisualAuditError,
    analyze_ground_truth,
)


def _annotation(
    annotation_id,
    image_id,
    bbox,
    *,
    role="target",
    distance_cm=2200,
):
    x, y, width, height = bbox
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": list(bbox),
        "deepsafe_loaf_bbox_raw": list(bbox),
        "area": width * height,
        "rotated_box": [x + width / 2, y + height / 2, width, height, 0],
        "world_location": [distance_cm, 0],
        "deepsafe_distance_band_role": role,
    }


def _payload():
    images = []
    annotations = []
    annotation_id = 0
    for sequence_index, sequence_id in enumerate(("loaf-a", "loaf-b")):
        image_offset = sequence_index * 10
        for frame_id in range(4):
            image_id = image_offset + frame_id
            images.append(
                {
                    "id": image_id,
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "file_name": f"{sequence_id}-{frame_id}.jpg",
                    "width": 100,
                    "height": 100,
                }
            )
        # Each criterion has an unambiguous, distinct winner.
        annotations.append(_annotation(annotation_id, image_offset, [48, 48, 2, 2]))
        annotation_id += 1
        annotations.append(_annotation(annotation_id, image_offset + 1, [90, 48, 4, 8]))
        annotation_id += 1
        annotations.append(_annotation(annotation_id, image_offset + 2, [45, 45, 8, 8]))
        annotation_id += 1
        annotations.append(
            _annotation(
                annotation_id,
                image_offset + 2,
                [5, 5, 8, 8],
                role="ignore_outside_band",
                distance_cm=1000,
            )
        )
        annotation_id += 1
        annotations.append(
            _annotation(
                annotation_id,
                image_offset + 2,
                [15, 15, 8, 8],
                role="ignore_outside_band",
                distance_cm=3000,
            )
        )
        annotation_id += 1
        annotations.append(_annotation(annotation_id, image_offset + 3, [40, 40, 10, 10]))
        annotation_id += 1
        annotations.append(
            _annotation(
                annotation_id,
                image_offset + 3,
                [40, 40, 10, 10],
                role="ignore_outside_band",
                distance_cm=1500,
            )
        )
        annotation_id += 1
    return {
        "info": {
            "deepsafe_distance_band": {
                "schema_version": SCHEMA_VERSION,
                "metric_geometry": EXPECTED_METRIC_GEOMETRY,
                "distance_band_m": {
                    "minimum_inclusive": 20,
                    "maximum_exclusive": 25,
                },
            }
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }


def test_selects_four_distinct_stress_frames_per_sequence_and_never_claims_accuracy():
    report = analyze_ground_truth(_payload())

    assert report["status"] == "ground_truth_visual_audit_only"
    assert report["detector_predictions_used"] is False
    assert report["detector_accuracy_claim"] == "none"
    assert report["summary"]["sequence_count"] == 2
    assert report["summary"]["selected_frame_count"] == 8
    assert report["summary"]["target_count"] == 8
    for sequence in report["sequences"]:
        assert [item["selection_criterion"] for item in sequence["selected_frames"]] == [
            "tiny_target",
            "edge_target",
            "highest_density",
            "highest_overlap",
        ]
        assert [item["frame_id"] for item in sequence["selected_frames"]] == [0, 1, 2, 3]
        assert sequence["selected_frames"][-1]["maximum_target_overlap_ioa"] == 1.0


def test_rejects_old_raw_bbox_proxy_geometry():
    payload = _payload()
    payload["info"]["deepsafe_distance_band"]["metric_geometry"] = "axis_aligned_bbox_iou"

    with pytest.raises(VisualAuditError, match="unsafe LOAF bbox geometry"):
        analyze_ground_truth(payload)


def test_rejects_active_target_outside_declared_distance_band():
    payload = copy.deepcopy(_payload())
    payload["annotations"][0]["world_location"][0] = 2500

    with pytest.raises(VisualAuditError, match="outside active band"):
        analyze_ground_truth(payload)
