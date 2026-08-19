from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from validation import run_person_deepstream_direct as runner


def _fixture(tmp_path: Path) -> Path:
    source = tmp_path / "content/clips/ppe.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    for profile in (640, 960):
        model = tmp_path / f"models/person/{profile}"
        model.mkdir(parents=True)
        (model / "yolo11s_b12_gpu0_fp16.engine").write_bytes(b"engine")
        (model / "yolo11s.onnx").write_bytes(b"onnx")
        (model / "labels.txt").write_text("person\n", encoding="utf-8")
    return source


def _execute_fixture(
    tmp_path: Path, monkeypatch
) -> tuple[dict, dict]:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = _fixture(tmp_path)
    plan = runner.build_plan(
        video=source,
        run_root=tmp_path / "results/person/640",
        profile=640,
        gpu=0,
        sequence_id="ppe-P01-person-640",
        video_probe=lambda _path: {
            "width": 320,
            "height": 180,
            "fps": 25.0,
            "frames": 1,
        },
    )

    def fake_command(_command, log_path: Path) -> None:
        kitti = tmp_path / plan["paths"]["kitti"]
        kitti.mkdir(parents=True, exist_ok=True)
        tracker_kitti = tmp_path / plan["paths"]["tracker_kitti"]
        tracker_kitti.mkdir(parents=True, exist_ok=True)
        row = (
            "person 42 0.0 0 0.0 10 20 100 170 "
            "0 0 0 0 0 0 0 0.91\n"
        )
        (kitti / "00_000_000000.txt").write_text(row, encoding="utf-8")
        (tracker_kitti / "00_000_000000.txt").write_text(
            row, encoding="utf-8"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake deepstream log\n", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "attest_engine_load",
        lambda _log, profile: {"status": "pass", "model_input": profile},
    )
    terminal = runner.execute_plan(plan, command_runner=fake_command)
    return plan, terminal


def test_plan_is_direct_no_guard_and_profile_bound(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = _fixture(tmp_path)

    plan = runner.build_plan(
        video=source,
        run_root=tmp_path / "results/person/640",
        profile=640,
        gpu=0,
        sequence_id="ppe-P01-person-640",
        video_probe=lambda _path: {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "frames": 2,
        },
    )

    assert plan["execution_mode"] == "direct_no_guard"
    assert plan["model"]["id"] == "yolo11s-person-640"
    assert plan["streammux"]["width"] == 640
    assert plan["streammux"]["height"] == 360
    assert plan["paths"]["predictions"].endswith("/predictions.jsonl")
    assert plan["paths"]["tracker_kitti"].endswith("/tracker-kitti")
    assert plan["tracker"]["backend"] == "NvDCF-perf"
    assert plan["tracker"]["native_object_id_output"] is True
    assert plan["tracker"]["width"] == 640
    assert plan["tracker"]["height"] == 384
    command_text = " ".join(plan["docker_command"]).casefold()
    assert "gpu_guard" not in command_text
    assert "token" not in command_text
    assert "--gpus device=0" in command_text


def test_execute_converts_complete_kitti_and_resumes_without_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = _fixture(tmp_path)
    plan = runner.build_plan(
        video=source,
        run_root=tmp_path / "results/person/640",
        profile=640,
        gpu=0,
        sequence_id="ppe-P01-person-640",
        video_probe=lambda _path: {
            "width": 320,
            "height": 180,
            "fps": 25.0,
            "frames": 2,
        },
    )
    calls: list[list[str]] = []

    def fake_command(command, log_path: Path) -> None:
        calls.append(list(command))
        kitti = tmp_path / plan["paths"]["kitti"]
        kitti.mkdir(parents=True, exist_ok=True)
        tracker_kitti = tmp_path / plan["paths"]["tracker_kitti"]
        tracker_kitti.mkdir(parents=True, exist_ok=True)
        row = (
            "person 42 0.0 0 0.0 10 20 100 170 "
            "0 0 0 0 0 0 0 0.91\n"
        )
        (kitti / "00_000_000000.txt").write_text(row, encoding="utf-8")
        (kitti / "00_000_000001.txt").write_text("", encoding="utf-8")
        (tracker_kitti / "00_000_000000.txt").write_text(
            row, encoding="utf-8"
        )
        (tracker_kitti / "00_000_000001.txt").write_text(
            "", encoding="utf-8"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake deepstream log\n", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "attest_engine_load",
        lambda _log, profile: {"status": "pass", "model_input": profile},
    )
    terminal = runner.execute_plan(plan, command_runner=fake_command)

    assert terminal["status"] == "complete"
    assert terminal["conversion"]["exported_frame_records"] == 2
    assert terminal["conversion"]["person_detections"] == 1
    predictions = tmp_path / plan["paths"]["predictions"]
    rows = [
        json.loads(raw)
        for raw in predictions.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["frame_index"] for row in rows] == [0, 1]
    assert len(rows[0]["detections"]) == 1
    assert rows[0]["detections"][0]["track_id"] == 42
    assert rows[0]["detections"][0]["tracker_confidence"] == 0.91
    assert rows[1]["detections"] == []
    app_config = (
        tmp_path / plan["paths"]["deepstream_config"]
    ).read_text(encoding="utf-8")
    assert "kitti-track-output-dir=/workspace/" in app_config
    assert "[tracker]\nenable=1" in app_config
    assert "config_tracker_NvDCF_perf.yml" in app_config

    resumed = runner.execute_plan(
        plan,
        command_runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("resume unexpectedly launched a command")
        ),
    )
    assert resumed == terminal
    assert len(calls) == 1


def test_tracked_kitti_keeps_native_id_and_clamps_nvdcf_response(
    tmp_path: Path,
) -> None:
    tracker_kitti = tmp_path / "tracker-kitti"
    tracker_kitti.mkdir()
    (tracker_kitti / "00_000_000000.txt").write_text(
        "person 7 0.0 0 0.0 10 20 100 170 "
        "0 0 0 0 0 0 0 1.016471\n",
        encoding="utf-8",
    )
    output = tmp_path / "predictions.jsonl"

    stats = runner._convert_tracked_kitti_directory(
        tracker_kitti,
        output,
        sequence_id="clip",
        image_width=320,
        image_height=180,
        coordinate_width=320,
        coordinate_height=180,
        expected_frames=1,
        fps=25.0,
        source_uri="clip.mp4",
        model_id="person",
    )

    detection = json.loads(output.read_text(encoding="utf-8"))["detections"][0]
    assert detection["track_id"] == 7
    assert detection["confidence"] == 1.0
    assert detection["tracker_confidence"] == 1.0
    assert detection["tracker_confidence_raw"] == 1.016471
    assert stats["native_track_ids"] is True
    assert stats["unique_track_ids"] == [7]


def test_person_vehicle_mode_exports_tracked_truck_as_forklift_candidate(
    tmp_path: Path,
) -> None:
    tracker_kitti = tmp_path / "tracker-kitti"
    tracker_kitti.mkdir()
    (tracker_kitti / "00_000_000000.txt").write_text(
        (
            "Person 7 0.0 0 0.0 10 20 100 170 0 0 0 0 0 0 0 0.91\n"
            "Truck 8 0.0 0 0.0 4 5 180 175 0 0 0 0 0 0 0 0.72\n"
        ),
        encoding="utf-8",
    )
    output = tmp_path / "predictions.jsonl"

    stats = runner._convert_tracked_kitti_directory(
        tracker_kitti,
        output,
        sequence_id="clip",
        image_width=320,
        image_height=180,
        coordinate_width=320,
        coordinate_height=180,
        expected_frames=1,
        fps=25.0,
        source_uri="clip.mp4",
        model_id="person-vehicle",
        included_class_ids=(0, 7),
    )

    detections = json.loads(output.read_text(encoding="utf-8"))["detections"]
    assert [(item["class_id"], item["class_name"]) for item in detections] == [
        (0, "person"),
        (7, "forklift_candidate"),
    ]
    assert detections[1]["detector_class_name"] == "truck"
    assert stats["person_detections"] == 1
    assert stats["detections_by_class"] == {
        "person": 1,
        "forklift_candidate": 1,
    }


def test_forklift_candidate_plan_keeps_one_engine_and_opens_only_classes_0_7(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = _fixture(tmp_path)
    plan = runner.build_plan(
        video=source,
        run_root=tmp_path / "results/person-vehicle/960",
        profile=960,
        gpu=0,
        sequence_id="ppe-S04-person-vehicle-960",
        include_forklift_candidates=True,
        video_probe=lambda _path: {
            "width": 1920,
            "height": 1080,
            "fps": 24.0,
            "frames": 2,
        },
    )

    assert plan["model"]["included_class_ids"] == [0, 7]
    assert plan["model"]["forklift_evidence"]["semantic_class"] == (
        "forklift_candidate"
    )
    infer = runner.render_infer_config(
        960,
        0.05,
        parser="cuda",
        included_class_ids=(0, 7),
    )
    filtered = infer.split("filter-out-class-ids=", 1)[1].splitlines()[0]
    assert "0" not in filtered.split(";")
    assert "7" not in filtered.split(";")
    assert "1" in filtered.split(";")


@pytest.mark.parametrize(
    "artifact_name",
    [
        "predictions",
        "conversion",
        "deepstream_log",
        "deepstream_config",
        "infer_config",
    ],
)
def test_resume_rejects_tamper_of_every_pinned_artifact(
    tmp_path: Path, monkeypatch, artifact_name: str
) -> None:
    plan, terminal = _execute_fixture(tmp_path, monkeypatch)
    artifact_path = tmp_path / terminal["artifacts"][artifact_name]["path"]
    with artifact_path.open("ab") as stream:
        stream.write(b" ")

    with pytest.raises(
        runner.PersonDeepStreamError,
        match=f"resume artifact integrity differs: {artifact_name}",
    ):
        runner.execute_plan(
            plan,
            command_runner=lambda *_args: (_ for _ in ()).throw(
                AssertionError("tampered resume unexpectedly launched a command")
            ),
        )


def test_resume_rejects_same_size_prediction_sha256_tamper(
    tmp_path: Path, monkeypatch
) -> None:
    plan, terminal = _execute_fixture(tmp_path, monkeypatch)
    predictions = tmp_path / terminal["artifacts"]["predictions"]["path"]
    original = predictions.read_bytes()
    tampered = original.replace(b'"track_id":42', b'"track_id":43', 1)
    assert tampered != original
    assert len(tampered) == len(original)
    predictions.write_bytes(tampered)
    assert runner._predictions_complete(plan, predictions) is True

    with pytest.raises(
        runner.PersonDeepStreamError,
        match="resume artifact integrity differs: predictions",
    ):
        runner.execute_plan(
            plan,
            command_runner=lambda *_args: (_ for _ in ()).throw(
                AssertionError("tampered resume unexpectedly launched a command")
            ),
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("track_id", True),
        ("confidence", True),
        ("tracker_confidence", float("nan")),
        ("tracker_confidence", float("inf")),
        ("tracker_confidence_raw", float("nan")),
        ("tracker_confidence_raw", float("-inf")),
        ("bbox_norm_xywh", [0.1, 0.1, float("nan"), 0.4]),
        ("bbox_norm_xywh", [False, 0.1, 0.2, 0.4]),
        ("bbox_norm_xywh", [0.9, 0.1, 0.2, 0.4]),
        ("bbox_norm_xywh", [0.1, 0.1, 0.0, 0.4]),
    ],
)
def test_prediction_completeness_rejects_invalid_tracker_metadata(
    tmp_path: Path, monkeypatch, field: str, invalid_value: object
) -> None:
    plan, terminal = _execute_fixture(tmp_path, monkeypatch)
    predictions = tmp_path / terminal["artifacts"]["predictions"]["path"]
    row = json.loads(predictions.read_text(encoding="utf-8"))
    corrupted = copy.deepcopy(row)
    corrupted["detections"][0][field] = invalid_value
    predictions.write_text(
        json.dumps(corrupted, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    assert runner._predictions_complete(plan, predictions) is False
