import copy
import json
import os
from pathlib import Path

import pytest

from content import roi_demo
import validation.run_content_person_batch as batch


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_plan(tmp_path: Path, monkeypatch) -> tuple[dict, dict]:
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path.resolve())
    schema = tmp_path / "validation/schemas/content-roi-demo-v1.schema.json"
    schema.parent.mkdir(parents=True)
    schema.write_bytes(
        (
            ROOT / "validation/schemas/content-roi-demo-v1.schema.json"
        ).read_bytes()
    )
    monkeypatch.setattr(roi_demo, "SCHEMA_PATH", schema)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic video")
    engine = tmp_path / "engine.plan"
    engine.write_bytes(b"synthetic engine")
    reentry = tmp_path / "reentry.json"
    receipt = tmp_path / "ds9-receipt.json"
    _write_json(reentry, {"status": "pass"})
    _write_json(receipt, {"status": "production_ready"})

    config = json.loads(
        (
            ROOT / "content/configs/colt-collbrai-person-full-01.json"
        ).read_text(encoding="utf-8")
    )
    config["source"]["video_path"] = "source.mp4"
    config["detections"]["path"] = "inference/01/960/predictions.jsonl"
    config["detections"]["sequence_id"] = "colt-candidate-01"
    config["clip"] = {"start_seconds": 0.0, "end_seconds": 0.3}
    config["output"]["directory"] = "deliveries/01"
    config["output"]["width"] = 640
    config["output"]["height"] = 360
    config_path = (
        tmp_path / "content/configs/colt-collbrai-person-full-01.json"
    )
    _write_json(config_path, config)
    monkeypatch.setattr(
        batch,
        "probe_video",
        lambda _: {"width": 640, "height": 360, "fps": 10.0, "frames": 3},
    )
    plan = batch.build_plan(
        [config_path],
        expected_ids=("01",),
        control_root=tmp_path / "control",
        engine_path=engine,
        control_paths=(),
        reentry_evidence=reentry,
        ds9_compatibility_receipt=receipt,
    )
    return plan, plan["jobs"][0]


def _write_complete_inference(tmp_path: Path, job: dict) -> Path:
    root = tmp_path / job["inference_root"]
    root.mkdir(parents=True)
    manifest = {
        "status": "complete",
        "sequence_id": job["sequence_id"],
        "video": job["source"]["path"],
        "ground_truth": None,
        "model": job["model_id"],
        "model_input": 960,
        "bbox_parser": "cuda",
        "export_threshold": 0.001,
        "evaluation_confidence": 0.25,
        "streammux": {"policy": "model-active-area"},
        "gpu_safety": {"status": "complete"},
        "ds9_runtime_compatibility": {"status": "production_ready"},
        "engine_load_attestation": {"status": "pass"},
    }
    conversion = {
        "schema_version": "deepsafe.person-detections/v1",
        "sequence_id": job["sequence_id"],
        "decoded_frame_files": 3,
        "exported_frame_records": 3,
        "skipped_unannotated_frames": 0,
        "json_image_dimensions": [640, 360],
    }
    rows = [
        {
            "schema_version": "deepsafe.person-detections/v1",
            "sequence_id": job["sequence_id"],
            "frame_index": index,
            "image_width": 640,
            "image_height": 360,
            "model_id": job["model_id"],
            "detections": [],
        }
        for index in range(3)
    ]
    _write_json(root / "run-manifest.json", manifest)
    _write_json(root / "conversion.json", conversion)
    (root / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    for relative in (
        "deepstream.log",
        "generated/config-infer-primary.txt",
        "generated/deepstream-app.txt",
        "safety/gpu-guard-report.json",
        "safety/gpu-guard-artifact-receipt.json",
        "safety/gpu-preflight.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("complete\n", encoding="utf-8")
    return root


def _write_complete_delivery(tmp_path: Path, job: dict) -> Path:
    root = tmp_path / job["delivery_root"]
    root.mkdir(parents=True)
    events = [
        {
            "schema_version": roi_demo.EVENT_SCHEMA_VERSION,
            "demo_id": job["demo_id"],
            "event_type": "zone_enter",
        }
    ]
    (root / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    (root / "demo.mp4").write_bytes(b"encoded video")
    (root / "preview-safe.png").write_bytes(b"safe")
    (root / "preview-alert.png").write_bytes(b"alert")
    artifacts = {
        name: {
            "bytes": (root / name).stat().st_size,
            "sha256": batch._sha256_file(root / name),
        }
        for name in (
            "demo.mp4",
            "events.jsonl",
            "preview-safe.png",
            "preview-alert.png",
        )
    }
    predictions = tmp_path / job["predictions"]
    manifest = {
        "schema_version": roi_demo.MANIFEST_SCHEMA_VERSION,
        "demo_id": job["demo_id"],
        "status": "rendered",
        "theme": roi_demo.THEME.as_dict(),
        "plan": {
            "config_path": job["config"]["path"],
            "theme": roi_demo.THEME.as_dict(),
            "video": {"sha256": job["source"]["sha256"]},
            "detections": {
                "path": job["predictions"],
                "sha256": batch._sha256_file(predictions),
            },
            "clip": {
                "start_frame": 0,
                "end_frame_exclusive": 3,
                "output_frame_count": 3,
            },
        },
        "statistics": {
            "rendered_frames": 3,
            "maximum_roi_person_count": 1,
            "event_count": 1,
        },
        "artifacts": artifacts,
    }
    _write_json(root / "manifest.json", manifest)
    return root


def test_build_plan_pins_full_clip_and_requires_explicit_gpu_execution(
    tmp_path, monkeypatch
):
    plan, job = _fixture_plan(tmp_path, monkeypatch)
    assert plan["execution_policy"]["gpu_requires_explicit_execute"] is True
    assert plan["execution_policy"]["render_starts_after_all_inference_complete"] is True
    assert plan["contract_sha256"] == batch._contract_hash(plan)
    assert job["job_id"] == "01"
    assert job["video_metadata"]["frames"] == 3
    assert job["inference_command"][:3] == [
        os.sys.executable,
        "-m",
        "validation.run_caviar",
    ]
    assert "--dry-run" not in job["inference_command"]
    assert job["render_command"][2] == "content.roi_demo"

    args = batch.build_parser().parse_args([])
    assert args.execute is False
    assert args.render_only is False
    assert batch.build_parser().parse_args(["--execute"]).execute is True


def test_inference_resume_inspection_rejects_frame_drift(tmp_path, monkeypatch):
    _, job = _fixture_plan(tmp_path, monkeypatch)
    root = _write_complete_inference(tmp_path, job)
    inspection = batch.inspect_inference(job)
    assert inspection == {
        "state": "complete",
        "reasons": [],
        "metrics": {"frame_records": 3, "person_detections": 0},
    }

    rows = (root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    bad = json.loads(rows[1])
    bad["frame_index"] = 9
    rows[1] = json.dumps(bad)
    (root / "predictions.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    inspection = batch.inspect_inference(job)
    assert inspection["state"] == "conflict"
    assert any("expected 1" in reason for reason in inspection["reasons"])


def test_delivery_qc_verifies_full_video_and_manifest_hashes(
    tmp_path, monkeypatch
):
    _, job = _fixture_plan(tmp_path, monkeypatch)
    _write_complete_inference(tmp_path, job)
    root = _write_complete_delivery(tmp_path, job)
    inspection = batch.inspect_delivery(job)
    assert inspection["state"] == "complete", inspection
    assert inspection["metrics"]["frames"] == 3

    (root / "preview-alert.png").write_bytes(b"tampered")
    inspection = batch.inspect_delivery(job)
    assert inspection["state"] == "conflict"
    assert any("preview-alert.png" in reason for reason in inspection["reasons"])


def test_event_ledger_is_append_only_and_hash_chained(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path.resolve())
    plan = {
        "batch_id": batch.BATCH_ID,
        "contract_sha256": "a" * 64,
    }
    path = tmp_path / "control/events.jsonl"
    first = batch.append_event(
        path,
        plan=plan,
        event_type="stage_started",
        job_id="01",
        stage="inference",
    )
    second = batch.append_event(
        path,
        plan=plan,
        event_type="stage_completed",
        job_id="01",
        stage="inference",
    )
    assert second["previous_event_sha256"] == first["event_sha256"]
    assert len(batch._read_events(path)) == 2

    records = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(records[0])
    tampered["event_type"] = "tampered"
    records[0] = json.dumps(tampered)
    path.write_text("\n".join(records) + "\n", encoding="utf-8")
    with pytest.raises(batch.BatchError, match="hash differs"):
        batch._read_events(path)


def test_logged_process_output_is_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path.resolve())
    log = tmp_path / "control/logs/test.log"
    returncode, _ = batch._run_logged(
        ["/usr/bin/printf", "hello\n"],
        log_path=log,
    )
    assert returncode == 0
    assert "hello" in log.read_text(encoding="utf-8")
    assert log.stat().st_mode & 0o222 == 0
