import json
from pathlib import Path

import pytest

from validation import run_ppe_deepstream as ppe


def test_infer_config_selects_person_helmet_and_hi_vis_classes():
    config = ppe.render_infer_config(960, threshold=0.2)
    assert "num-detected-classes=13" in config
    assert "parse-bbox-func-name=NvDsInferParseYoloCuda" in config
    assert "engine-create-func-name=" not in config
    assert "onnx-file=" not in config
    assert (
        "filter-out-class-ids=0;1;2;4;5;6;8;10" in config
    )
    assert "pre-cluster-threshold=0.2" in config
    assert "safetyvision_yolov8s_v2_ds9raw6_b12_gpu0_fp16.engine" in config


def test_infer_config_rejects_unknown_profile_or_threshold():
    with pytest.raises(ppe.PpeDeepStreamError, match="unsupported"):
        ppe.render_infer_config(896)
    with pytest.raises(ppe.PpeDeepStreamError, match="threshold"):
        ppe.render_infer_config(640, threshold=0)


def test_docker_command_uses_writable_profile_directory(tmp_path):
    run_root = ppe.ROOT / "validation/results/ppe/unit-command"
    app_config = run_root / "640/generated/deepstream-app.txt"
    command = ppe.build_docker_command(
        app_config=app_config,
        run_root=run_root,
        profile=640,
        gpu=0,
        container_name="fixture",
    )
    workdir = command[command.index("-w") + 1]
    assert workdir == (
        "/workspace/models/ppe/safetyvision-yolov8s-v2/640"
    )
    assert (
        f"{ppe.ROOT / ppe.MODEL_ROOT}:"
        "/workspace/models/ppe/safetyvision-yolov8s-v2:rw"
    ) in command


def test_engine_builder_uses_dynamic_batch12_profile():
    command = ppe.build_engine_command(profile=640, gpu=0)
    assert "--fp16" in command
    assert "--noTF32" in command
    assert "--minShapes=images:1x3x640x640" in command
    assert "--optShapes=images:12x3x640x640" in command
    assert "--maxShapes=images:12x3x640x640" in command
    assert any(
        value.endswith(
            "/safetyvision_yolov8s_v2_ds9raw6_b12_gpu0_fp16.engine"
        )
        for value in command
        if value.startswith("--saveEngine=")
    )


def test_engine_load_attestation_rejects_fallback(tmp_path):
    engine = (
        ppe.ROOT
        / ppe.MODEL_ROOT
        / "640/safetyvision_yolov8s_v2_ds9raw6_b12_gpu0_fp16.engine"
    )
    expected = "/workspace/" + engine.relative_to(ppe.ROOT).as_posix()
    log = tmp_path / "deepstream.log"
    log.write_text(
        f"deserialized trt engine from :{expected}\n"
        f"Use deserialized engine model: {expected}\n",
        encoding="utf-8",
    )
    assert ppe.attest_engine_load(log, engine)["status"] == "pass"
    log.write_text(
        f"deserialized trt engine from :{expected}\n"
        "Trying to create engine from model files\n",
        encoding="utf-8",
    )
    with pytest.raises(ppe.PpeDeepStreamError, match="fallback"):
        ppe.attest_engine_load(log, engine)


def test_plan_keeps_license_and_nonproduction_state(monkeypatch):
    monkeypatch.setattr(
        ppe,
        "probe_video",
        lambda _: {"width": 1920, "height": 1080, "fps": 30.0, "frames": None},
    )
    plan = ppe.build_plan(
        # The GitHub handoff deliberately contains no redistributable sample
        # video. build_plan only needs an in-repository regular file here; media
        # decoding belongs to the execute-path tests.
        video=Path("README.md"),
        run_root=Path("validation/results/ppe/unit-plan"),
        profiles=(640, 960),
        gpu=0,
        start_seconds=80.0,
        duration_seconds=12.0,
        threshold=0.1,
    )
    assert plan["purpose"] == "diagnostic_content_evaluation"
    assert plan["production_accepted"] is False
    assert plan["commercially_cleared"] is False
    assert plan["model"]["accepted_model"] is False
    assert plan["model"]["license_id"] == "AGPL-3.0"
    assert plan["model"]["class_mapping"] == {
        "helmet": 3,
        "no_helmet": 7,
        "no_hi_vis": 9,
        "person": 11,
        "hi_vis": 12,
    }
    assert [item["profile"] for item in plan["profiles"]] == [640, 960]
    assert plan["adapter_parity"]["path"].endswith(
        "ds9-raw6-real-frame-parity.json"
    )
    assert len(plan["contract_sha256"]) == 64


def _kitti_line(label: str, confidence: float, left: float = 10.0) -> str:
    values = [
        0,
        0,
        0,
        left,
        20,
        left + 30,
        60,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        confidence,
    ]
    return label + " " + " ".join(str(value) for value in values) + "\n"


def test_kitti_conversion_preserves_four_ppe_semantics(tmp_path):
    kitti = tmp_path / "kitti"
    kitti.mkdir()
    (kitti / "00_000_000000.txt").write_text(
        _kitti_line("Hardhat", 0.9)
        + _kitti_line("NO-Safety Vest", 0.8, left=50),
        encoding="utf-8",
    )
    (kitti / "00_000_000001.txt").write_text(
        _kitti_line("Safety Vest", 0.7)
        + _kitti_line("NO-Hardhat", 0.05, left=80),
        encoding="utf-8",
    )
    output = tmp_path / "predictions.jsonl"
    stats = ppe.convert_kitti(
        kitti,
        output,
        sequence_id="fixture",
        source_width=1920,
        source_height=1080,
        coordinate_width=640,
        coordinate_height=360,
        expected_frames=2,
        threshold=0.1,
    )
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["production_accepted"] is False
    assert {
        detection["canonical_class"]
        for row in rows
        for detection in row["detections"]
    } == {"helmet", "no_hi_vis", "hi_vis"}
    assert rows[0]["detections"][0]["bbox_xywh"] == [
        30.0,
        60.0,
        90.0,
        120.0,
    ]
    assert stats["detections"] == 3
    assert stats["ppe_detections"] == 3
    assert stats["person_detections"] == 0
    assert stats["dropped_below_threshold"] == 1


def test_kitti_conversion_keeps_person_for_person_centric_fusion(tmp_path):
    kitti = tmp_path / "kitti"
    kitti.mkdir()
    (kitti / "00_000_000000.txt").write_text(
        _kitti_line("Person", 0.92)
        + _kitti_line("Hardhat", 0.85, left=15),
        encoding="utf-8",
    )
    output = tmp_path / "predictions.jsonl"
    stats = ppe.convert_kitti(
        kitti,
        output,
        sequence_id="person-ppe-fixture",
        source_width=640,
        source_height=360,
        coordinate_width=640,
        coordinate_height=360,
        expected_frames=1,
        threshold=0.1,
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert [
        (item["canonical_class"], item["compliance"])
        for item in row["detections"]
    ] == [("person", "neutral"), ("helmet", "compliant")]
    assert stats["class_counts"]["person"] == 1
    assert stats["person_detections"] == 1
    assert stats["ppe_detections"] == 1


def test_person_only_kitti_output_does_not_count_as_ppe(tmp_path):
    kitti = tmp_path / "kitti"
    kitti.mkdir()
    (kitti / "00_000_000000.txt").write_text(
        _kitti_line("Person", 0.92),
        encoding="utf-8",
    )
    stats = ppe.convert_kitti(
        kitti,
        tmp_path / "predictions.jsonl",
        sequence_id="person-only",
        source_width=640,
        source_height=360,
        coordinate_width=640,
        coordinate_height=360,
        expected_frames=1,
        threshold=0.1,
    )

    assert stats["detections"] == 1
    assert stats["person_detections"] == 1
    assert stats["ppe_detections"] == 0
    with pytest.raises(
        ppe.PpeDeepStreamError,
        match="no selected PPE detections",
    ):
        ppe._require_selected_ppe(stats, profile=640)
    with pytest.raises(
        ppe.PpeDeepStreamError,
        match="no selected PPE detections",
    ):
        ppe._require_selected_ppe(
            {
                "detections": 9,
                "class_counts": {
                    "helmet": 0,
                    "no_helmet": 0,
                    "hi_vis": 0,
                    "no_hi_vis": 0,
                    "person": 9,
                },
            },
            profile=640,
        )
    ppe._require_selected_ppe(
        {
            "class_counts": {
                "helmet": 1,
                "no_helmet": 0,
                "hi_vis": 0,
                "no_hi_vis": 0,
                "person": 9,
            },
        },
        profile=640,
    )


def test_kitti_conversion_requires_complete_frame_sequence(tmp_path):
    kitti = tmp_path / "kitti"
    kitti.mkdir()
    (kitti / "00_000_000001.txt").write_text("", encoding="utf-8")
    with pytest.raises(ppe.PpeDeepStreamError, match="sequence mismatch"):
        ppe.convert_kitti(
            kitti,
            tmp_path / "out.jsonl",
            sequence_id="fixture",
            source_width=640,
            source_height=360,
            coordinate_width=640,
            coordinate_height=360,
            expected_frames=2,
            threshold=0.1,
        )
