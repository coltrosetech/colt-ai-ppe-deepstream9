import json
from pathlib import Path

import pytest

import validation.run_caviar_batch as batch


ROOT = Path(__file__).resolve().parents[1]


def _write_xml(path: Path, frames: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f'<frame number="{index}" />' for index in frames)
    path.write_text(f"<dataset>{body}</dataset>", encoding="utf-8")


def test_default_catalog_discovers_all_eight_pairs_and_explicit_xml_tail_policy():
    sequences = batch.discover_sequences()
    assert len(sequences) == 8
    assert len({item["sequence_id"] for item in sequences}) == 8
    assert all((ROOT / item["video"]).is_file() for item in sequences)
    assert all((ROOT / item["ground_truth"]).is_file() for item in sequences)
    assert all(item["frame_mapping"]["xml_frames_outside_video"] == [] for item in sequences)

    data1 = [item for item in sequences if item["collection"] == "CAVIARDATA1"]
    data2 = [item for item in sequences if item["collection"] == "CAVIARDATA2"]
    assert len(data1) == 4 and len(data2) == 4
    assert all(item["frame_mapping"]["unannotated_decoded_frame_count"] == 1 for item in data1)
    assert all(item["frame_mapping"]["unannotated_decoded_frame_count"] == 0 for item in data2)


def test_discovery_rejects_xml_frame_outside_video(tmp_path, monkeypatch):
    gt_root = tmp_path / "gt"
    video_root = tmp_path / "video"
    xml = gt_root / "CAVIARDATA1" / "Clip" / "clip.xml"
    video = video_root / "CAVIARDATA1" / "Clip" / "Clip.mp4"
    _write_xml(xml, [0, 1, 3])
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not decoded because probe is mocked")

    monkeypatch.setattr(batch, "_inside_repo", lambda path: Path(path).resolve())
    monkeypatch.setattr(batch, "_repo_relative", lambda path: str(Path(path).resolve()))
    monkeypatch.setattr(
        batch,
        "probe_video",
        lambda path: {"width": 384, "height": 288, "fps": 25.0, "frames": 3},
    )
    with pytest.raises(ValueError, match="outside decoded video"):
        batch.discover_sequences(gt_root, video_root)


def _synthetic_plan(tmp_path: Path):
    sequence = {
        "sequence_id": "Clip",
        "collection": "CAVIARDATA1",
        "video": "video.mp4",
        "ground_truth": "gt.xml",
        "video_metadata": {"width": 384, "height": 288, "fps": 25.0, "frames": 3},
        "frame_mapping": {
            "decoded_frame_count": 3,
            "annotated_frame_count": 2,
            "unannotated_decoded_frame_count": 1,
        },
    }
    campaign = {
        "model_input_sizes": [640],
        "bbox_parser": "cuda",
        "export_threshold": 0.001,
        "evaluation_confidence": 0.25,
        "iou": 0.5,
        "max_nvinfer_upscale": 1.0,
    }
    job = {
        "job_id": "Clip:640",
        "sequence_id": "Clip",
        "model_input": 640,
        "run_root": "results/Clip/640",
    }
    return {
        "schema_version": batch.PLAN_SCHEMA_VERSION,
        "campaign": campaign,
        "sequences": [sequence],
        "jobs": [job],
    }, sequence, job


def _write_complete_run(root: Path) -> None:
    root.mkdir(parents=True)
    manifest = {
        "status": "complete",
        "sequence_id": "Clip",
        "model_input": 640,
        "video": "video.mp4",
        "ground_truth": "gt.xml",
        "bbox_parser": "cuda",
        "export_threshold": 0.001,
        "evaluation_confidence": 0.25,
        "iou": 0.5,
        "streammux": {"max_nvinfer_upscale": 1.0},
        "deepstream_fps": {"last_reported_average": 123.0},
        "gpu_safety": {
            "status": "complete",
            "execution_boundary": "validation.gpu_guarded_process/v1",
        },
    }
    conversion = {
        "sequence_id": "Clip",
        "decoded_frame_files": 3,
        "exported_frame_records": 2,
        "skipped_unannotated_frames": 1,
    }
    metric = {
        "ground_truth": 10,
        "tp": 4,
        "fp": 2,
        "fn": 6,
        "precision": 0.666667,
        "recall": 0.4,
        "f1": 0.5,
        "ap_101_point": 0.35,
        "ap_serialized_predictions": 20,
    }
    evaluation = {
        "schema_version": "deepsafe.person-evaluation/v1",
        "diagnostics": {
            "prediction_only_frames": [],
            "ground_truth_only_frame_count": 0,
        },
        "overall": metric,
        "size_buckets": {
            "small": {"recall": 0.1},
            "medium": {"recall": 0.5},
            "large": {"recall": None},
        },
    }
    (root / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "conversion.json").write_text(json.dumps(conversion), encoding="utf-8")
    (root / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    prediction_rows = [
        {
            "schema_version": "deepsafe.person-detections/v1",
            "sequence_id": "Clip",
            "frame_index": index,
            "image_width": 384,
            "image_height": 288,
            "detections": [],
        }
        for index in (0, 1)
    ]
    (root / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    (root / "deepstream.log").write_text("done\n", encoding="utf-8")
    safety = root / "safety/gpu-guard-report.json"
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
    (root / "safety/gpu-preflight.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (root / "safety/gpu-guard-artifact-receipt.json").write_text(
        json.dumps({"guard_status": "complete"}), encoding="utf-8"
    )
    (root / "job-receipt.json").write_text(
        json.dumps({"job_id": "Clip:640"}), encoding="utf-8"
    )


def test_resume_inspection_and_aggregate_require_alignment_safe_complete_run(
    tmp_path, monkeypatch
):
    plan, sequence, job = _synthetic_plan(tmp_path)
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path)
    run_root = tmp_path / job["run_root"]
    _write_complete_run(run_root)

    assert batch.inspect_job(job, sequence, plan["campaign"])["state"] == "complete"
    aggregate = batch.aggregate_plan(plan)
    assert aggregate["completeness"]["complete_jobs"] == 1
    assert aggregate["by_model_input"]["640"]["micro"] == {
        "ground_truth": 10,
        "tp": 4,
        "fp": 2,
        "fn": 6,
        "precision": 0.666667,
        "recall": 0.4,
        "f1": 0.5,
    }
    assert aggregate["visual_audit_priority"]["worst_fn"][0]["job_id"] == "Clip:640"
    assert "evaluation.visualize" in aggregate["visual_audit_priority"]["worst_fn"][0][
        "visualize_command"
    ]

    conversion_path = run_root / "conversion.json"
    conversion = json.loads(conversion_path.read_text())
    conversion["exported_frame_records"] = 3
    conversion_path.write_text(json.dumps(conversion), encoding="utf-8")
    inspection = batch.inspect_job(job, sequence, plan["campaign"])
    assert inspection["state"] == "conflict"
    assert any("exported_frame_records" in reason for reason in inspection["reasons"])


def test_cli_is_gpu_free_unless_execute_is_explicit():
    args = batch.build_parser().parse_args([])
    assert args.execute is False
    assert args.dry_run is False


def _authorized_claim_plan(nonce: str) -> dict:
    results_root = f"validation/results/caviar/sessions/{nonce}"
    authorization = {
        "campaign_nonce": nonce,
        "expires_at_utc": "2099-07-16T12:00:00Z",
        "authorized_results_root": results_root,
        "single_use": True,
    }
    return {
        "campaign": {
            "results_root": results_root,
            "campaign_nonce": nonce,
            "session_claim": results_root + "/session-claim.json",
            "batch_receipt": results_root + "/batch-receipt.json",
            "session_claim_artifact": None,
            "quality_policy": {
                "artifact": {"path": "policy.json", "bytes": 1, "sha256": "0" * 64},
                "policy_id": "fixture",
                "policy_contract_sha256": "1" * 64,
                "status": "approved",
                "campaign_authorization": authorization,
            },
        }
    }


def test_campaign_nonce_claim_is_single_use(tmp_path, monkeypatch):
    nonce = "a" * 64
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path.resolve())
    plan = _authorized_claim_plan(nonce)
    first = batch.claim_execution_session(plan)
    assert first == plan["campaign"]["session_claim_artifact"]
    claim = tmp_path / first["path"]
    assert claim.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        batch.claim_execution_session(plan)


def test_expired_campaign_nonce_is_rejected_before_claim(tmp_path, monkeypatch):
    nonce = "b" * 64
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path.resolve())
    plan = _authorized_claim_plan(nonce)
    plan["campaign"]["quality_policy"]["campaign_authorization"][
        "expires_at_utc"
    ] = "2000-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="expired"):
        batch.claim_execution_session(plan)
    assert not (tmp_path / plan["campaign"]["session_claim"]).exists()
