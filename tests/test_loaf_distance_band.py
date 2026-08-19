import json
import math

import pytest

from evaluation.readers import load_coco
from validation.loaf_distance_band import main, prepare_distance_band


def _source():
    return {
        "info": {"name": "synthetic LOAF"},
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "chair"}],
        "images": [
            {"id": 10, "file_name": "scene-a/0002.jpg", "width": 512, "height": 512},
            {"id": 11, "file_name": "scene-a/0001.jpg", "width": 512, "height": 512},
            {"id": 20, "file_name": "scene-b/0001.jpg", "width": 512, "height": 512},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 10,
                "category_id": 1,
                "bbox": [10, 10, 20, 30],
                "rotated_box": [20, 25, 20, 30, 5],
                "world_location": [2000, 0],
                "area": 600,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 10,
                "category_id": 1,
                "bbox": [40, 40, 20, 30],
                "rotated_box": [50, 55, 20, 30, 5],
                "world_location": [2500, 0],
                "area": 600,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 11,
                "category_id": 1,
                "bbox": [70, 70, 20, 30],
                "rotated_box": [80, 85, 20, 30, 5],
                "world_location": [2499, 0],
                "area": 600,
                "iscrowd": 0,
            },
            {
                "id": 4,
                "image_id": 11,
                "category_id": 1,
                "bbox": [100, 100, 20, 30],
                "rotated_box": [110, 115, 20, 30, 5],
                "world_location": [2200, 0],
                "area": 600,
                "difficult": 1,
                "iscrowd": 0,
            },
            {
                "id": 5,
                "image_id": 11,
                "category_id": 2,
                "bbox": [0, 0, 10, 10],
                "area": 100,
            },
            {
                "id": 6,
                "image_id": 20,
                "category_id": 1,
                "bbox": [10, 10, 20, 30],
                "rotated_box": [20, 25, 20, 30, 5],
                "world_location": [1000, 0],
                "area": 600,
                "iscrowd": 0,
            },
        ],
    }


def test_distance_band_is_half_open_and_outside_people_become_ignore(tmp_path):
    filtered, report = prepare_distance_band(
        _source(), minimum_m=20, maximum_m=25, exclude_difficult=True
    )

    assert [image["id"] for image in filtered["images"]] == [11, 10]
    assert [image["frame_id"] for image in filtered["images"]] == [0, 1]
    assert {image["sequence_id"] for image in filtered["images"]} == {"scene-a"}
    roles = {
        annotation["id"]: annotation.get("deepsafe_distance_band_role")
        for annotation in filtered["annotations"]
    }
    assert roles == {1: "target", 2: "ignore_outside_band", 3: "target", 4: "ignore_difficult", 5: None}
    ignored = {
        annotation["id"]
        for annotation in filtered["annotations"]
        if annotation.get("ignore")
    }
    assert ignored == {2, 4}
    assert report["selected_counts"]["roles"] == {
        "ignore_difficult": 1,
        "ignore_outside_band": 1,
        "target": 2,
    }
    assert report["selected_counts"]["target_people_by_1m_bin"] == {
        "20-21": 1,
        "24-25": 1,
    }
    first = next(annotation for annotation in filtered["annotations"] if annotation["id"] == 1)
    assert first["deepsafe_loaf_bbox_raw"] == [10.0, 10.0, 20.0, 30.0]
    assert first["bbox"] != first["deepsafe_loaf_bbox_raw"]
    assert first["area"] == pytest.approx(first["bbox"][2] * first["bbox"][3])
    assert report["metric_geometry"] == "axis_aligned_envelope_of_rotated_box"

    path = tmp_path / "filtered.json"
    path.write_text(json.dumps(filtered), encoding="utf-8")
    dataset = load_coco(path)
    assert sum(
        not annotation.ignored
        for frame in dataset.frames.values()
        for annotation in frame.annotations
    ) == 2
    assert sum(
        annotation.ignored
        for frame in dataset.frames.values()
        for annotation in frame.annotations
    ) == 2


def test_distance_band_fails_when_no_eligible_target():
    with pytest.raises(ValueError, match="contains no eligible"):
        prepare_distance_band(_source(), minimum_m=30, maximum_m=31)


def test_difficult_is_active_by_default_for_loaf_far_range():
    _, report = prepare_distance_band(_source(), minimum_m=20, maximum_m=25)
    assert report["selected_counts"]["roles"]["target"] == 3
    assert "ignore_difficult" not in report["selected_counts"]["roles"]


def test_native_ignore_is_suppressed_and_policy_discloses_official_loader_divergence():
    source = _source()
    source["annotations"][3]["ignore"] = 1
    filtered, report = prepare_distance_band(source, minimum_m=20, maximum_m=25)

    ignored = next(annotation for annotation in filtered["annotations"] if annotation["id"] == 4)
    assert ignored["deepsafe_distance_band_role"] == "ignore_native"
    assert ignored["ignore"] == 1
    assert ignored["iscrowd"] == 1
    assert report["selected_counts"]["roles"]["ignore_native"] == 1
    assert "does not consume ignore" in report["native_ignore_policy"]


def test_flat_official_filename_preserves_loaf_sequence_boundary():
    source = _source()
    source["images"][0]["file_name"] = "0064_00001.jpg"
    source["images"][1]["file_name"] = "0064_00000.jpg"
    filtered, report = prepare_distance_band(source, minimum_m=20, maximum_m=25)
    assert {image["sequence_id"] for image in filtered["images"]} == {"loaf-0064"}
    assert report["sequence_frame_counts"] == {"loaf-0064": 2}


def test_rotated_box_envelope_is_exact_at_90_degrees_and_preserves_raw_bbox():
    source = _source()
    source["images"] = [
        {"id": 10, "file_name": "0064_00000.jpg", "width": 100, "height": 100}
    ]
    source["annotations"] = [
        {
            "id": 1,
            "image_id": 10,
            "category_id": 1,
            "bbox": [40, 35, 20, 30],
            "rotated_box": [50, 50, 20, 30, 90],
            "world_location": [2000, 0],
            "area": 600,
            "iscrowd": 0,
        }
    ]

    filtered, _ = prepare_distance_band(source, minimum_m=20, maximum_m=25)
    annotation = filtered["annotations"][0]
    assert annotation["deepsafe_loaf_bbox_raw"] == [40.0, 35.0, 20.0, 30.0]
    assert annotation["bbox"] == [35.0, 40.0, 30.0, 20.0]
    assert annotation["area"] == 600.0


def test_rotated_box_envelope_is_clipped_to_image_and_area_is_recomputed():
    source = _source()
    source["images"] = [
        {"id": 10, "file_name": "0064_00000.jpg", "width": 100, "height": 100}
    ]
    source["annotations"] = [
        {
            "id": 1,
            "image_id": 10,
            "category_id": 1,
            "bbox": [-5, 0, 20, 10],
            "rotated_box": [5, 5, 20, 10, 0],
            "world_location": [2000, 0],
            "area": 200,
            "iscrowd": 0,
        }
    ]

    filtered, report = prepare_distance_band(source, minimum_m=20, maximum_m=25)
    annotation = filtered["annotations"][0]
    assert annotation["deepsafe_loaf_bbox_raw"] == [-5.0, 0.0, 20.0, 10.0]
    assert annotation["bbox"] == [0.0, 0.0, 15.0, 10.0]
    assert annotation["area"] == 150.0
    assert report["selected_counts"]["boxes_clipped_to_image"] == 1


@pytest.mark.parametrize(
    "rotated_box",
    (
        None,
        [50, 50, 20, 30],
        [50, 50, 0, 30, 0],
        [50, 50, 20, 30, math.inf],
    ),
)
def test_malformed_rotated_box_fails_closed(rotated_box):
    source = _source()
    source["annotations"][0]["rotated_box"] = rotated_box
    with pytest.raises(ValueError, match="rotated_box"):
        prepare_distance_band(source, minimum_m=20, maximum_m=25)


def test_cli_pins_source_and_output_hashes(tmp_path, capsys):
    source = tmp_path / "instances.json"
    output = tmp_path / "band.json"
    report = tmp_path / "report.json"
    source.write_text(json.dumps(_source()), encoding="utf-8")

    assert main(["--input", str(source), "--output", str(output), "--report", str(report)]) == 0
    rendered = json.loads(capsys.readouterr().out)
    stored = json.loads(report.read_text(encoding="utf-8"))
    assert rendered == stored
    assert len(stored["source"]["sha256"]) == 64
    assert len(stored["output"]["sha256"]) == 64
    assert stored["status"] == "prepared_not_evaluated"
