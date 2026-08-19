import json
from pathlib import Path

import pytest

try:
    import cv2
    import numpy as np
except ImportError:  # The matching test remains runnable without visual extras.
    cv2 = None
    np = None

from evaluation.readers import load_caviar, load_predictions_jsonl
from evaluation.visualize import (
    COLORS_BGR,
    build_frame_reviews,
    main,
    rank_frame_reviews,
)


FIXTURES = Path(__file__).parent / "fixtures" / "evaluation"


def _write_fixture_video(path: Path) -> None:
    assert cv2 is not None and np is not None
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (100, 100)
    )
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((100, 100, 3), 35 + index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _contains_color(image, color: tuple[int, int, int]) -> bool:
    assert np is not None
    return bool(np.any(np.all(image == np.asarray(color, dtype=np.uint8), axis=2)))


def test_frame_review_ranking_and_size_filter_match_evaluator_policy():
    ground_truth = load_caviar(
        FIXTURES / "caviar.xml", image_width=100, image_height=100
    )
    predictions = load_predictions_jsonl(FIXTURES / "caviar-predictions.jsonl")

    reviews = build_frame_reviews(
        ground_truth, predictions, sequence_id="caviar-fixture"
    )
    assert sum(review.tp for review in reviews) == 1
    assert sum(review.fp for review in reviews) == 1
    assert sum(review.fn for review in reviews) == 1
    assert [item.key.frame_index for item in rank_frame_reviews(reviews, limit=2)] == [
        2,
        1,
    ]
    assert [
        item.key.frame_index
        for item in rank_frame_reviews(reviews, rank_by="fp", limit=2)
    ] == [1]

    medium_reviews = build_frame_reviews(
        ground_truth,
        predictions,
        sequence_id="caviar-fixture",
        bucket="medium",
    )
    assert sum(review.fp for review in medium_reviews) == 0
    assert sum(review.fn for review in medium_reviews) == 1
    assert sum(len(review.ignored_predictions) for review in medium_reviews) == 2


def test_cli_renders_lossless_color_layers_contact_sheet_and_json(tmp_path, capsys):
    if cv2 is None or np is None:
        pytest.skip("optional visual-review dependencies are not installed")
    video = tmp_path / "fixture.avi"
    _write_fixture_video(video)
    output = tmp_path / "review"

    assert (
        main(
            [
                "--predictions",
                str(FIXTURES / "caviar-predictions.jsonl"),
                "--ground-truth",
                str(FIXTURES / "caviar.xml"),
                "--ground-truth-format",
                "caviar",
                "--video",
                str(video),
                "--output-dir",
                str(output),
                "--image-width",
                "100",
                "--image-height",
                "100",
                "--max-frames",
                "2",
                "--columns",
                "2",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    report = json.loads((output / "visual-review.json").read_text(encoding="utf-8"))
    assert printed["summary"] == report["summary"]
    assert report["summary"] == {
        "frames_considered": 3,
        "frames_with_errors": 2,
        "selected_frames": 2,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "ignored_predictions": 0,
    }
    assert [item["frame_index"] for item in report["ranked_frames"]] == [2, 1]
    assert (output / "contact-sheet.png").is_file()

    fn_image = cv2.imread(
        str(output / report["ranked_frames"][0]["frame_image"]), cv2.IMREAD_COLOR
    )
    fp_image = cv2.imread(
        str(output / report["ranked_frames"][1]["frame_image"]), cv2.IMREAD_COLOR
    )
    assert _contains_color(fn_image, COLORS_BGR["ground_truth"])
    assert _contains_color(fn_image, COLORS_BGR["false_negative"])
    assert _contains_color(fp_image, COLORS_BGR["ground_truth"])
    assert _contains_color(fp_image, COLORS_BGR["true_positive"])
    assert _contains_color(fp_image, COLORS_BGR["false_positive"])
