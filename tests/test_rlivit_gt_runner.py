from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evaluation.rlivit import (
    RLiViTEvaluationError,
    canonical_sha256,
    evaluate_sequences,
    load_sequence_ground_truth,
    load_sequence_predictions,
    stable_file_pin,
    validate_test_split_ground_truth,
)
from validation import run_rlivit as single
from validation import run_rlivit_batch as batch


def _jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in records),
        encoding="utf-8",
    )


def _source_pin(root: Path, relative: str, content: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return stable_file_pin(path, root=root, display_path=relative)


def _sequence_fixture(root: Path, sequence_id: str = "002") -> tuple[Path, Path, Path]:
    gt_records = []
    map_records = []
    prediction_records = []
    source_indices = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46]
    for video_index, source_index in enumerate(source_indices):
        stem = f"{source_index:02d}"
        rgb = _source_pin(root, f"archive/R-LiViT_RGB-T/rgb/{sequence_id}/{stem}.png", b"rgb" + bytes([video_index]))
        thermal = _source_pin(root, f"archive/R-LiViT_RGB-T/thermal/{sequence_id}/{stem}.png", b"thermal" + bytes([video_index]))
        annotation = _source_pin(root, f"archive/R-LiViT_RGB-T/annotations/{sequence_id}/{stem}.xml", b"xml" + bytes([video_index]))
        # The three dimensions exercise all COCO area and height strata.
        if video_index % 3 == 0:
            box = [10, 10, 20, 20]
        elif video_index % 3 == 1:
            box = [20, 20, 60, 70]
        else:
            box = [30, 30, 150, 180]
        gt_records.append(
            {
                "schema_version": "deepsafe.rlivit-visible-person-gt/v1",
                "dataset_id": "R-LiViT_RGB-T_v1.0",
                "split": "test",
                "sequence_id": sequence_id,
                "daytime": "day",
                "location": "0",
                "source_frame_index": source_index,
                "video_frame_index": video_index,
                "image": {"width": 1280, "height": 720, "modality": "rgb_visible_full_frame", **rgb},
                "paired_thermal": {"width": 1280, "height": 720, **thermal},
                "annotation": {
                    "all_object_count": 1,
                    "coordinate_convention": "source_integer_xyxy_values_unmodified",
                    "format": "pascal_voc_xml",
                    "verified": True,
                    **annotation,
                },
                "person_count": 1,
                "persons": [{"class_name": "person", "bbox_xyxy_px": box}],
            }
        )
        map_records.append(
            {
                "schema_version": "deepsafe.rlivit-visible-frame-map/v1",
                "dataset_id": "R-LiViT_RGB-T_v1.0",
                "split": "test",
                "sequence_id": sequence_id,
                "daytime": "day",
                "location": "0",
                "source_frame_index": source_index,
                "video_frame_index": video_index,
                "rgb_image": rgb,
                "annotation": annotation,
            }
        )
        x1, y1, x2, y2 = box
        prediction_records.append(
            {
                "schema_version": "deepsafe.person-detections/v1",
                "sequence_id": sequence_id,
                "frame_index": video_index,
                "image_width": 1280,
                "image_height": 720,
                "model_id": "synthetic",
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.9 - video_index / 100,
                        "bbox_norm_xywh": [x1 / 1280, y1 / 720, (x2 - x1) / 1280, (y2 - y1) / 720],
                    }
                ],
            }
        )
    gt = root / f"derived/gt/sequences/{sequence_id}.jsonl"
    frame_map = root / f"derived/frame-maps/sequences/{sequence_id}.jsonl"
    predictions = root / f"predictions/{sequence_id}.jsonl"
    _jsonl(gt, gt_records)
    _jsonl(frame_map, map_records)
    _jsonl(predictions, prediction_records)
    return gt.relative_to(root), frame_map.relative_to(root), predictions.relative_to(root)


def test_strict_reader_and_raw_replay_perfect_metrics(tmp_path: Path) -> None:
    gt_path, map_path, prediction_path = _sequence_fixture(tmp_path)
    gt = load_sequence_ground_truth(gt_path, map_path, materialized_root=tmp_path)
    predictions = load_sequence_predictions(prediction_path, root=tmp_path, sequence_id="002")
    result = evaluate_sequences([gt], [predictions], iou_threshold=0.5, confidence_threshold=0.25)

    assert result["inputs"]["frame_count"] == 12
    assert result["inputs"]["ground_truth_persons"] == 12
    assert result["overall"]["tp"] == 12
    assert result["overall"]["fp"] == 0
    assert result["overall"]["fn"] == 0
    assert result["overall"]["precision"] == 1.0
    assert result["overall"]["recall"] == 1.0
    assert result["overall"]["f1"] == 1.0
    assert result["overall"]["ap_101_point"] == 1.0
    assert {key: value["ground_truth"] for key, value in result["coco_area"].items()} == {
        "small": 4,
        "medium": 4,
        "large": 4,
    }
    assert {key: value["ground_truth"] for key, value in result["height_bands"].items()} == {
        "lt32": 4,
        "32to95": 4,
        "gte96": 4,
    }


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda records: records.pop(), "exactly 12"),
        (lambda records: records[1].__setitem__("video_frame_index", 0), "video index differs"),
        (lambda records: records[0]["persons"].append(copy.deepcopy(records[0]["persons"][0])), "person count differs"),
        (lambda records: records[0]["persons"][0].__setitem__("bbox_xyxy_px", [0, 0, 1281, 20]), "outside image"),
        (lambda records: records[0].__setitem__("split", "train"), "not test split"),
    ],
)
def test_ground_truth_adversarial_records_fail_closed(tmp_path: Path, mutation, match: str) -> None:
    gt_path, map_path, _predictions = _sequence_fixture(tmp_path)
    path = tmp_path / gt_path
    records = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(records)
    _jsonl(path, records)
    with pytest.raises(RLiViTEvaluationError, match=match):
        load_sequence_ground_truth(gt_path, map_path, materialized_root=tmp_path)


def test_duplicate_ground_truth_box_is_rejected(tmp_path: Path) -> None:
    gt_path, map_path, _predictions = _sequence_fixture(tmp_path)
    path = tmp_path / gt_path
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["persons"].append(copy.deepcopy(records[0]["persons"][0]))
    records[0]["person_count"] = 2
    records[0]["annotation"]["all_object_count"] = 2
    _jsonl(path, records)
    with pytest.raises(RLiViTEvaluationError, match="duplicate bbox"):
        load_sequence_ground_truth(gt_path, map_path, materialized_root=tmp_path)


def test_live_source_pin_tamper_is_rejected(tmp_path: Path) -> None:
    gt_path, map_path, _predictions = _sequence_fixture(tmp_path)
    record = json.loads((tmp_path / gt_path).read_text().splitlines()[0])
    (tmp_path / record["image"]["path"]).write_bytes(b"changed")
    with pytest.raises(RLiViTEvaluationError, match="live size/SHA-256 differs"):
        load_sequence_ground_truth(gt_path, map_path, materialized_root=tmp_path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda records: records.pop(), "exactly 12"),
        (lambda records: records[0].__setitem__("image_width", 1279), "dimensions differ"),
        (lambda records: records[1].__setitem__("frame_index", 0), "frame index differs"),
        (lambda records: records[0]["detections"][0].__setitem__("class_name", "car"), "non-person"),
        (lambda records: records[0]["detections"][0].__setitem__("bbox_norm_xywh", [0.9, 0.0, 0.2, 0.2]), "outside normalized"),
    ],
)
def test_prediction_adversarial_records_fail_closed(tmp_path: Path, mutation, match: str) -> None:
    _gt, _map, prediction_path = _sequence_fixture(tmp_path)
    path = tmp_path / prediction_path
    records = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(records)
    _jsonl(path, records)
    with pytest.raises(RLiViTEvaluationError, match=match):
        load_sequence_predictions(prediction_path, root=tmp_path, sequence_id="002")


def test_duplicate_prediction_is_rejected(tmp_path: Path) -> None:
    _gt, _map, prediction_path = _sequence_fixture(tmp_path)
    path = tmp_path / prediction_path
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["detections"].append(copy.deepcopy(records[0]["detections"][0]))
    _jsonl(path, records)
    with pytest.raises(RLiViTEvaluationError, match="duplicate detection"):
        load_sequence_predictions(prediction_path, root=tmp_path, sequence_id="002")


def test_official_test_split_contract_is_exact_without_rehashing_sources() -> None:
    root = Path("data/derived/r-livit/materialized-v1").resolve()
    plan = json.loads((root / "derived/plans/person-campaign-plan.json").read_text())
    jobs = {job["sequence_id"]: job for job in plan["jobs"] if job["model_input"] == 640}
    sequences, summary = validate_test_split_ground_truth(
        [(Path(job["ground_truth"]["path"]), Path(job["frame_map"]["path"])) for job in jobs.values()],
        materialized_root=root,
        expected_sequences=40,
        expected_persons=4318,
        verify_live_sources=False,
    )
    assert len(sequences) == 40
    assert summary["frame_count"] == 480
    assert summary["person_count"] == 4318
    assert summary["locations"] == [str(item) for item in range(8)]
    assert summary["daytime"] == ["day", "night"]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _root_pin(root: Path, relative: str) -> dict:
    return stable_file_pin(root / relative, root=root, display_path=relative)


def _synthetic_mp4_chain(root: Path, *, bad_mapping: bool = False) -> tuple[list[dict], Path]:
    materialized = root / "materialized"
    mp4 = root / "mp4"
    sequences = []
    campaign_jobs = []
    job_pins = []
    outputs = []
    for number in range(40):
        sequence_id = f"{number:03d}"
        gt_rel = f"derived/gt/sequences/{sequence_id}.jsonl"
        map_rel = f"derived/frame-maps/sequences/{sequence_id}.jsonl"
        (materialized / gt_rel).parent.mkdir(parents=True, exist_ok=True)
        (materialized / gt_rel).write_text("{}\n")
        (materialized / map_rel).parent.mkdir(parents=True, exist_ok=True)
        (materialized / map_rel).write_text("{}\n")
        gt_pin = _root_pin(materialized, gt_rel)
        map_pin = _root_pin(materialized, map_rel)
        project_gt = {**gt_pin, "path": f"materialized/{gt_rel}"}
        project_map = {**map_pin, "path": f"materialized/{map_rel}"}
        sequences.append(
            {
                "sequence_id": sequence_id,
                "daytime": "day" if number < 20 else "night",
                "location": str(number % 8),
                "ground_truth": project_gt,
                "frame_map": project_map,
                "ground_truth_materialized_relative": gt_rel,
                "frame_map_materialized_relative": map_rel,
                "gpu_job_ids": [f"rlivit:{sequence_id}:640", f"rlivit:{sequence_id}:960"],
            }
        )
        video_rel = f"jobs/{sequence_id}/video.mp4"
        video_path = mp4 / video_rel
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"video-{sequence_id}".encode())
        video_pin = _root_pin(mp4, video_rel)
        mapping = [
            {
                "video_frame_index": index,
                "source_frame_index": ([3, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46] if bad_mapping and number == 0 else [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46])[index],
                "rgb_sha256": hashlib.sha256(f"{sequence_id}-{index}".encode()).hexdigest(),
            }
            for index in range(12)
        ]
        receipt = {
            "schema_version": "deepsafe.rlivit-mp4-job-receipt/v1",
            "status": "complete_verified_atomic",
            "sequence_id": sequence_id,
            "split": "test",
            "source": {"ground_truth": gt_pin, "frame_map": map_pin, "frame_mapping": mapping},
            "output": video_pin,
        }
        receipt["fingerprint_sha256"] = canonical_sha256(receipt)
        receipt_rel = f"jobs/{sequence_id}/receipt.json"
        _write_json(mp4 / receipt_rel, receipt)
        receipt_pin = _root_pin(mp4, receipt_rel)
        job_pins.append(receipt_pin)
        outputs.append(video_pin)
        campaign_jobs.append(
            {
                "sequence_id": sequence_id,
                "split": "test",
                "daytime": sequences[-1]["daytime"],
                "location": sequences[-1]["location"],
                "source_plan_job_id": f"rlivit-mp4:{sequence_id}",
                "logical_source_plan_video": f"derived/videos/{sequence_id}.mp4",
                "materialized_video": video_pin,
                "job_receipt": receipt_pin,
                "ground_truth": gt_pin,
                "frame_map": map_pin,
                "profiles": [640, 960],
                "gpu_job_ids": sequences[-1]["gpu_job_ids"],
                "gpu_job_status": "blocked_pending_mp4_gpu_reentry_and_model_binding",
            }
        )
    campaign = {
        "schema_version": "deepsafe.rlivit-mp4-campaign-binding/v1",
        "status": "test_mp4_materialized_gpu_jobs_still_blocked",
        "sequence_count": 40,
        "profiles": [640, 960],
        "gpu_job_count": 80,
        "source_campaign_plan_mutated": False,
        "jobs": campaign_jobs,
    }
    campaign["fingerprint_sha256"] = canonical_sha256(campaign)
    _write_json(mp4 / "campaign-video-binding.json", campaign)
    _write_json(mp4 / "run-binding.json", {"status": "synthetic"})
    receipt = {
        "schema_version": "deepsafe.rlivit-mp4-batch-receipt/v1",
        "status": "complete_verified_cpu_only",
        "sequence_count": 40,
        "expected_sequence_count": 40,
        "video_count": 40,
        "expected_video_count": 40,
        "frame_count_per_video": 12,
        "total_frames": 480,
        "fps": "5/4",
        "run_binding": _root_pin(mp4, "run-binding.json"),
        "campaign_video_binding": _root_pin(mp4, "campaign-video-binding.json"),
        "campaign_video_binding_fingerprint_sha256": campaign["fingerprint_sha256"],
        "job_receipts": job_pins,
        "outputs": outputs,
        "outputs_fingerprint_sha256": canonical_sha256(outputs),
        "gpu_docker_inference_executed": False,
        "source_campaign_gpu_jobs_mutated": False,
        "acceptance_effect": "mp4_input_materialization_only_gpu_jobs_remain_blocked",
    }
    receipt["fingerprint_sha256"] = canonical_sha256(receipt)
    _write_json(mp4 / "batch-receipt.json", receipt)
    return sequences, mp4 / "batch-receipt.json"


def test_mp4_receipt_chain_accepts_exact_40_and_rejects_mapping_drift(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(single, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(batch, "PROJECT_ROOT", tmp_path)
    sequences, receipt = _synthetic_mp4_chain(tmp_path)
    result = batch.validate_mp4_batch(tmp_path / "mp4", receipt, sequences)
    assert len(result["videos"]) == 40

    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    monkeypatch.setattr(single, "PROJECT_ROOT", bad_root)
    monkeypatch.setattr(batch, "PROJECT_ROOT", bad_root)
    bad_sequences, bad_receipt = _synthetic_mp4_chain(bad_root, bad_mapping=True)
    with pytest.raises(ValueError, match="exact 12-frame mapping differs"):
        batch.validate_mp4_batch(bad_root / "mp4", bad_receipt, bad_sequences)


def test_single_runner_without_execute_is_inert(monkeypatch) -> None:
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("GPU path must remain unreachable")

    monkeypatch.setattr(single, "run_job", forbidden)
    assert single.main(["--batch-plan", "missing.json", "--job-id", "rlivit:002:640", "--session-claim", "missing-claim.json"]) == 2
    assert called is False


def test_public_status_is_bounded_pathless_and_has_stable_matrix() -> None:
    plan = json.loads(
        Path("validation/results/rlivit/deepstream-dry-run/batch-plan.json").read_text()
    )
    status = batch.build_public_status(plan, phase="blocked")
    encoded = json.dumps(status, sort_keys=True)

    assert status["schema_version"] == "deepsafe.rlivit-public-status/v1"
    assert status["matrix"] == {
        "sequence_count": 40,
        "profiles": [640, 960],
        "job_count": 80,
        "frames_per_job": 12,
        "ground_truth_frames_per_profile": 480,
    }
    assert status["ground_truth"]["person_count"] == 4318
    assert status["progress"]["completed_jobs"] == 0
    assert status["gpu_docker_inference_executed"] is False
    assert status["fingerprint_sha256"] == canonical_sha256(
        {key: value for key, value in status.items() if key != "fingerprint_sha256"}
    )
    expected_codes = sorted({reason.split(":", 1)[0] for reason in plan["blockers"]})
    assert status["blocker_codes"] == expected_codes
    assert "operator_authorization_missing" in status["blocker_codes"]
    assert all(":" not in value and "/" not in value for value in status["blocker_codes"])
    assert set(status["evidence"]["source_plan"]) == {"size_bytes", "sha256"}
    assert len(encoded.encode()) < 512 * 1024
    for forbidden in ("/home/", "runs/", "campaign_nonce", "operator_identity", "authorized_results_root", '"command"'):
        assert forbidden not in encoded.casefold()


def test_public_complete_status_exposes_only_profile_metric_projection() -> None:
    plan = json.loads(
        Path("validation/results/rlivit/deepstream-dry-run/batch-plan.json").read_text()
    )
    plan["blockers"] = []
    metric = {
        "overall": {"ground_truth": 4318, "tp": 4000},
        "daytime": {"day": {}, "night": {}},
        "locations": {str(index): {} for index in range(8)},
        "coco_area": {"small": {}, "medium": {}, "large": {}},
        "height_bands": {"lt32": {}, "32to95": {}, "gte96": {}},
        "inputs": {"private": "/home/private/predictions.jsonl"},
        "sequences": {"002": {"private": True}},
    }
    aggregate = {"profiles": {"640": metric, "960": metric}}
    state = {
        "completed_jobs": [
            {"job_id": f"rlivit:{sequence:03d}:{profile}", "receipt": {}}
            for sequence in range(40)
            for profile in (640, 960)
        ],
        "launched_jobs": [
            f"rlivit:{sequence:03d}:{profile}"
            for sequence in range(40)
            for profile in (640, 960)
        ],
        "gpu_process_started_jobs": [
            f"rlivit:{sequence:03d}:{profile}"
            for sequence in range(40)
            for profile in (640, 960)
        ],
        "failed_job": None,
    }
    pin = {"path": "private/runs/nonce/file.json", "size_bytes": 123, "sha256": "a" * 64}
    status = batch.build_public_status(
        plan,
        phase="complete",
        state=state,
        aggregate=aggregate,
        aggregate_pin=pin,
        batch_receipt_pin=pin,
    )
    encoded = json.dumps(status, sort_keys=True)

    assert status["progress"]["completed_jobs"] == 80
    assert status["progress"]["remaining_jobs"] == 0
    assert status["gpu_docker_inference_executed"] is True
    assert status["fingerprint_sha256"] == canonical_sha256(
        {key: value for key, value in status.items() if key != "fingerprint_sha256"}
    )
    assert status["profiles"]["640"]["status"] == "complete"
    assert set(status["profiles"]["640"]["metrics"]) == {
        "overall",
        "daytime",
        "locations",
        "coco_area",
        "height_bands",
    }
    assert "/home/" not in encoded
    assert "runs/" not in encoded


def test_authorization_is_exact_current_immutable_and_single_use(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(single, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(batch, "PROJECT_ROOT", tmp_path)
    results_root = tmp_path / "validation/results/rlivit/runs" / ("1" * 64)
    results_root.mkdir(parents=True)
    definition = "2" * 64
    now = datetime.now(timezone.utc)
    authorization = {
        "schema_version": "deepsafe.rlivit-execution-authorization/v1",
        "status": "approved",
        "operator_identity": "operator@example.test",
        "campaign_nonce": "1" * 64,
        "issued_at_utc": (now - timedelta(minutes=1)).isoformat(),
        "expires_at_utc": (now + timedelta(hours=1)).isoformat(),
        "authorized_results_root": results_root.relative_to(tmp_path).as_posix(),
        "campaign_definition_sha256": definition,
        "single_use": True,
    }
    auth_path = tmp_path / "authorization.json"
    _write_json(auth_path, authorization)
    auth_path.chmod(0o440)

    loaded = batch._authorization_payload(auth_path, definition, results_root)
    assert loaded["artifact"]["sha256"] == hashlib.sha256(auth_path.read_bytes()).hexdigest()

    plan = {
        "schema_version": "deepsafe.rlivit-deepstream-batch-plan/v1",
        "campaign": {
            "results_root": results_root.relative_to(tmp_path).as_posix(),
            "campaign_nonce": "1" * 64,
            "execution_authorization": loaded,
        },
    }
    plan["fingerprint_sha256"] = canonical_sha256(plan)
    plan_path = results_root / "batch-plan.json"
    _write_json(plan_path, plan)
    claim = batch.claim_session(plan_path, plan)
    assert claim == results_root / "session-claim.json"
    assert stat_mode(claim) == 0o440
    with pytest.raises(ValueError, match="already been consumed"):
        batch.claim_session(plan_path, plan)


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777


@pytest.mark.parametrize("change,match", [
    (lambda value: value.__setitem__("campaign_definition_sha256", "3" * 64), "definition differs"),
    (lambda value: value.__setitem__("single_use", False), "fields/status differ"),
    (lambda value: value.__setitem__("expires_at_utc", (datetime.now(timezone.utc) + timedelta(hours=25)).isoformat()), "exceeds 24h"),
])
def test_authorization_adversarial_mutations_are_rejected(tmp_path: Path, monkeypatch, change, match: str) -> None:
    monkeypatch.setattr(single, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(batch, "PROJECT_ROOT", tmp_path)
    results_root = tmp_path / "validation/results/rlivit/runs" / ("1" * 64)
    results_root.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    value = {
        "schema_version": "deepsafe.rlivit-execution-authorization/v1",
        "status": "approved",
        "operator_identity": "operator@example.test",
        "campaign_nonce": "1" * 64,
        "issued_at_utc": now.isoformat(),
        "expires_at_utc": (now + timedelta(hours=1)).isoformat(),
        "authorized_results_root": results_root.relative_to(tmp_path).as_posix(),
        "campaign_definition_sha256": "2" * 64,
        "single_use": True,
    }
    change(value)
    path = tmp_path / "authorization.json"
    _write_json(path, value)
    path.chmod(0o440)
    with pytest.raises(ValueError, match=match):
        batch._authorization_payload(path, "2" * 64, results_root)
