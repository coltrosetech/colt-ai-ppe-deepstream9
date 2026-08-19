import copy
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from admin import validation as admin_validation
from admin.app import app
from validation import report_campaign as campaign
from tests.admin_lineage_fixtures import write_admin_endurance_lineage
from tests.test_campaign_report import _write_verified_synthetic_distance_v2_final


def _write_json(root: Path, relative: str, value: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_campaign_report_with_ppe_registry(
    root: Path,
    *,
    registry_state: str = "valid",
) -> dict:
    registry = root / "data/manifests/ppe-video-source-candidates.json"
    if registry_state != "missing":
        registry.parent.mkdir(parents=True, exist_ok=True)
        source = (
            Path(campaign.__file__).resolve().parents[1]
            / "data/manifests/ppe-video-source-candidates.json"
        )
        content = source.read_bytes()
        registry.write_bytes(
            content if registry_state == "valid" else content + b"\n"
        )
    report = campaign.build_campaign_report(
        project_root=root,
        results_root=root / "validation-results",
        hardware_incident_path=None,
    )
    _write_json(root, "campaign-report/report.json", report)
    return report


def _accepted_open_video_section() -> dict:
    rendered_scenes = [f"scene_{index:02d}" for index in range(11)]
    withheld_scene = "scene_11"
    pairs = []
    source_index = 0
    for scene_index, scene_id in enumerate(rendered_scenes):
        source_count = 2 if scene_index < 10 else 1
        source_ids = [
            f"{scene_id}.frame_{source_index + offset:02d}"
            for offset in range(source_count)
        ]
        source_index += source_count
        pairs.append(
            {
                "scene_id": scene_id,
                "comparison_kind": "hash_bound_ai_qualitative_visual_pair",
                "metric_claim_allowed": False,
                "source_review_ids": source_ids,
                "decision_ids": [
                    f"{source_id}:{profile}"
                    for source_id in source_ids
                    for profile in (640, 960)
                ],
            }
        )
    return {
        "evidence_kind": "tiered_gt_free_open_video_evidence",
        "ground_truth": False,
        "expected_candidate_jobs": 24,
        "candidate_job_records": 24,
        "validated_candidate_jobs": 24,
        "automatic_candidate_generation": {
            "evidence_level": "automatic_candidate_generation_only",
            "expected_jobs": 24,
            "validated_jobs": 24,
            "rendered_jobs": 22,
            "withheld_jobs": 2,
            "rendered_scene_ids": rendered_scenes,
            "withheld_scene_ids": [withheld_scene],
            "mixed_scene_ids": [],
            "candidate_assets": {
                "decision_count": 42,
                "expected_decision_count": 42,
                "asset_count": 168,
                "expected_asset_count": 168,
                "live_hash_verified_assets": 168,
                "index_contract_valid": True,
            },
            "accepted": True,
            "not_a_visual_review": True,
            "reasons": [],
        },
        "ai_qualitative_visual_audit": {
            "evidence_level": "hash_bound_ai_visual_audit_not_human_qa",
            "status": "complete",
            "source_frame_count": 21,
            "expected_source_frame_count": 21,
            "profile_decision_count": 42,
            "expected_profile_decision_count": 42,
            "distinct_reviewed_video_types": 11,
            "minimum_distinct_video_types": 10,
            "reviewed_scene_ids": rendered_scenes,
            "coverage": {
                "medium_close_source_frames": 11,
                "overhead_top_view_source_frames": 2,
                "high_oblique_source_frames": 6,
                "required_coverage_proven": True,
            },
            "input_hash_binding_valid": True,
            "exact_id_profile_count_binding_valid": True,
            "withheld_scenes_excluded": True,
            "accepted": True,
            "human_review_claimed": False,
            "ground_truth_claimed": False,
            "reasons": [],
        },
        "human_terminal_qa": {
            "evidence_level": "terminal_human_qa",
            "state": "pending",
            "expected_decisions": 42,
            "terminal_decisions": 0,
            "pending_decisions": 42,
            "artifact_contract_valid": True,
            "accepted": False,
            "required_for_ai_audit_acceptance": False,
            "reasons": [],
        },
        "paired_profile_comparison": {
            "policy": (
                "same non-sensitive GT-free source frame, exact 640/960 "
                "decision IDs, live asset hashes, and one hash-bound AI audit "
                "record; withheld scenes are excluded"
            ),
            "paired_scene_count": 11,
            "paired_source_frame_count": 21,
            "pairs": pairs,
        },
        "accepted": True,
        "metric_guardrail": {
            "precision_recall_ap_forbidden": True,
            "candidate_frames_are_not_fp_or_fn": True,
            "ai_estimates_are_not_ground_truth": True,
            "per_frame_preferences_must_not_be_aggregated": True,
            "human_qa_is_separate": True,
        },
        "evidence_ids": ["open_video_tiered_test_evidence"],
    }


def _write_accepted_open_video_report(root: Path) -> dict:
    report = _write_campaign_report_with_ppe_registry(root)
    report["campaigns"]["open_video_manual_review"] = (
        _accepted_open_video_section()
    )
    requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "open_video_ai_qualitative_audit"
    )
    requirement["state"] = "pass"
    _write_json(root, "campaign-report/report.json", report)
    return report


def _with_fingerprint(value: dict) -> dict:
    result = dict(value)
    encoded = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    result["fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def _rlivit_metric(ground_truth: int) -> dict:
    return {
        "ground_truth": ground_truth,
        "tp": ground_truth,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "ap_101_point": 1.0,
    }


def _rlivit_profile_metrics() -> dict:
    return {
        "overall": _rlivit_metric(4318),
        "daytime": {
            "day": _rlivit_metric(3422),
            "night": _rlivit_metric(896),
        },
        "locations": {
            key: _rlivit_metric(count)
            for key, count in {
                "0": 428,
                "1": 354,
                "2": 275,
                "3": 779,
                "4": 114,
                "5": 127,
                "6": 424,
                "7": 1817,
            }.items()
        },
        "coco_area": {
            "small": _rlivit_metric(2907),
            "medium": _rlivit_metric(1258),
            "large": _rlivit_metric(153),
        },
        "height_bands": {
            "lt32": _rlivit_metric(1634),
            "32to95": _rlivit_metric(2425),
            "gte96": _rlivit_metric(259),
        },
    }


def _rlivit_public_status(*, complete: bool = False) -> dict:
    pin = {"size_bytes": 4096, "sha256": "a" * 64}
    if complete:
        phase = "complete"
        progress = {
            "planned_jobs": 80,
            "launched_jobs": 80,
            "gpu_process_started_jobs": 80,
            "completed_jobs": 80,
            "failed_jobs": 0,
            "remaining_jobs": 0,
        }
        profiles = {
            profile: {
                "status": "complete",
                "planned_jobs": 40,
                "completed_jobs": 40,
                "metrics": _rlivit_profile_metrics(),
            }
            for profile in ("640", "960")
        }
        blockers: list[str] = []
        evidence = {key: pin for key in (
            "source_plan",
            "mp4_batch_receipt",
            "runtime_policy",
            "ds9_compatibility_receipt",
            "batch_aggregate",
            "batch_receipt",
        )}
    else:
        phase = "blocked"
        progress = {
            "planned_jobs": 80,
            "launched_jobs": 0,
            "gpu_process_started_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "remaining_jobs": 80,
        }
        profiles = {
            profile: {
                "status": "blocked",
                "planned_jobs": 40,
                "completed_jobs": 0,
                "metrics": None,
            }
            for profile in ("640", "960")
        }
        blockers = [
            "ds9_compatibility_not_ready",
            "mp4_batch_not_ready",
            "operator_authorization_missing",
        ]
        evidence = {
            "source_plan": pin,
            "mp4_batch_receipt": None,
            "runtime_policy": pin,
            "ds9_compatibility_receipt": None,
            "batch_aggregate": None,
            "batch_receipt": None,
        }
    return _with_fingerprint(
        {
            "schema_version": "deepsafe.rlivit-public-status/v1",
            "status": phase,
            "updated_at_utc": "2026-07-16T08:00:00+00:00",
            "dataset_id": "R-LiViT_RGB-T_v1.0",
            "gpu_docker_inference_executed": complete,
            "matrix": {
                "sequence_count": 40,
                "profiles": [640, 960],
                "job_count": 80,
                "frames_per_job": 12,
                "ground_truth_frames_per_profile": 480,
            },
            "ground_truth": {
                "status": "valid",
                "sequence_count": 40,
                "frame_count": 480,
                "person_count": 4318,
                "daytime": ["day", "night"],
                "locations": [str(index) for index in range(8)],
                "live_sources_verified": True,
            },
            "progress": progress,
            "profiles": profiles,
            "blocker_codes": blockers,
            "evidence": evidence,
        }
    )


def _write_rlivit_pinned_json(
    root: Path, relative: str, value: dict, *, display_path: str
) -> dict:
    _write_json(root, relative, value)
    content = (root / relative).read_bytes()
    return {
        "path": display_path,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _pathless(pin: dict) -> dict:
    return {"size_bytes": pin["size_bytes"], "sha256": pin["sha256"]}


def _rlivit_mp4_public_receipt(batch_pin: dict) -> dict:
    return _with_fingerprint(
        {
            "schema_version": "deepsafe.rlivit-mp4-admin-receipt/v2",
            "status": "complete_verified_cpu_only",
            "dataset_id": "R-LiViT_RGB-T_v1.0",
            "sequences": {"complete": 40, "expected": 40},
            "gpu_jobs": {
                "blocked": 80,
                "expected": 80,
                "status": "blocked_pending_mp4_gpu_reentry_and_model_binding",
            },
            "frames": {"per_video": 12, "total": 480},
            "coverage": {
                "daytime_sequence_counts": {"day": 27, "night": 13},
                "location_sequence_counts": {
                    "0": 5,
                    "1": 6,
                    "2": 2,
                    "3": 7,
                    "4": 4,
                    "5": 1,
                    "6": 4,
                    "7": 11,
                },
                "distinct_locations": 8,
            },
            "video": {
                "codec_name": "h264",
                "profile": "High",
                "pixel_format": "yuv420p",
                "width": 1280,
                "height": 720,
                "fps": "5/4",
                "duration_seconds": "9.600000",
            },
            "source_quality": {
                "status": "all_frames_pass_fixed_psnr_ssim_floors",
                "thresholds": {
                    "comparison_pixel_format": "yuv420p",
                    "minimum_frame_psnr_y_db": 45.0,
                    "minimum_frame_psnr_average_db": 45.0,
                    "minimum_frame_ssim_y": 0.99,
                    "minimum_frame_ssim_all": 0.99,
                    "software_decode_and_filter_only": True,
                },
                "minimum_observed": {
                    "psnr_y_db": "48.100000000",
                    "psnr_average_db": "47.900000000",
                    "ssim_y": "0.995000000",
                    "ssim_all": "0.994000000",
                },
                "software_decode_and_filter_only": True,
            },
            "thermal_policy_id": "workstation_managed",
            "maximum_cpu_platform_temperature_millidegrees_celsius": 74000,
            "gpu_executed": False,
            "docker_executed": False,
            "inference_executed": False,
            "batch_fingerprint_sha256": "b" * 64,
            "batch_receipt_pin": _pathless(batch_pin),
        }
    )


def _rlivit_complete_artifacts(root: Path) -> tuple[dict, dict]:
    nonce = "9" * 64
    run_relative = f"rlivit/runs/{nonce}"
    display_root = f"validation/results/{run_relative}"
    source_pin = {
        "path": "data/derived/r-livit/materialized-v1/derived/plans/person-campaign-plan.json",
        "size_bytes": 62020,
        "sha256": "1" * 64,
    }
    mp4_pin = {
        "path": "data/derived/r-livit/test-mp4-v1/batch-receipt.json",
        "size_bytes": 70000,
        "sha256": "2" * 64,
    }
    runtime_artifact = {
        "path": "validation/person-quality-policy.approved.json",
        "bytes": 17093,
        "sha256": "3" * 64,
    }
    runtime_policy = {
        "artifact": runtime_artifact,
        "campaign_authorization": {"status": "approved"},
        "policy_contract_sha256": "4" * 64,
        "policy_id": "person-r-livit-v1",
        "status": "approved",
    }
    ds9_pin = {
        "path": "validation/results/ds9-runtime-compatibility/current/receipt.json",
        "size_bytes": 8192,
        "sha256": "5" * 64,
    }
    authorization = {
        "schema_version": "deepsafe.rlivit-execution-authorization/v1",
        "status": "approved",
        "operator_identity": "test-operator",
        "campaign_nonce": nonce,
        "issued_at_utc": "2026-07-16T07:00:00+00:00",
        "expires_at_utc": "2026-07-16T19:00:00+00:00",
        "authorized_results_root": display_root,
        "campaign_definition_sha256": "6" * 64,
        "single_use": True,
        "artifact": {
            "path": f"{display_root}/authorization.json",
            "size_bytes": 1024,
            "sha256": "7" * 64,
        },
    }
    job_receipts: dict[str, dict] = {}
    for sequence in range(40):
        for profile in (640, 960):
            job_id = f"rlivit:{sequence:03d}:{profile}"
            job_receipt = _with_fingerprint(
                {
                    "schema_version": "deepsafe.rlivit-deepstream-job-receipt/v1",
                    "status": "complete_raw_replay_verified",
                    "created_at_utc": "2026-07-16T08:30:00+00:00",
                    "job_id": job_id,
                    "sequence_id": f"{sequence:03d}",
                    "model_input": profile,
                }
            )
            job_receipts[job_id] = _write_rlivit_pinned_json(
                root,
                f"{run_relative}/jobs/{sequence:03d}/{profile}/job-receipt.json",
                job_receipt,
                display_path=(
                    f"{display_root}/jobs/{sequence:03d}/{profile}/job-receipt.json"
                ),
            )
    campaign = {
        "campaign_nonce": nonce,
        "results_root": display_root,
        "sequence_count": 40,
        "profiles": [640, 960],
        "expected_jobs": 80,
        "expected_frames_per_job": 12,
        "expected_ground_truth_persons": 4318,
        "execution_authorization": authorization,
        "source_campaign": source_pin,
        "mp4_batch_receipt": mp4_pin,
        "runtime_policy": runtime_policy,
        "ds9_compatibility_receipt": ds9_pin,
    }
    plan = _with_fingerprint(
        {
            "schema_version": "deepsafe.rlivit-deepstream-batch-plan/v1",
            "created_at_utc": "2026-07-16T07:30:00+00:00",
            "status": "ready_for_authorized_execution",
            "blockers": [],
            "campaign": campaign,
            "ground_truth_validation": {
                "live_sources_verified": True,
                "frame_count": 480,
                "person_count": 4318,
            },
            "jobs": [{"job_id": key} for key in sorted(job_receipts)],
        }
    )
    plan_pin = _write_rlivit_pinned_json(
        root,
        f"{run_relative}/batch-plan.json",
        plan,
        display_path=f"{display_root}/batch-plan.json",
    )
    aggregate = _with_fingerprint(
        {
            "schema_version": "deepsafe.rlivit-batch-evaluation/v1",
            "status": "complete_independent_raw_replay",
            "created_at_utc": "2026-07-16T09:00:00+00:00",
            "plan_fingerprint_sha256": plan["fingerprint_sha256"],
            "matrix": {
                "sequence_count": 40,
                "profiles": [640, 960],
                "job_count": 80,
                "frames_per_job": 12,
            },
            "metrics_contract": {
                "iou": 0.5,
                "confidence": 0.25,
                "ap": "AP101@IoU0.5",
            },
            "profiles": {
                profile: _rlivit_profile_metrics() for profile in ("640", "960")
            },
            "job_receipts": job_receipts,
        }
    )
    aggregate_pin = _write_rlivit_pinned_json(
        root,
        f"{run_relative}/batch-aggregate.json",
        aggregate,
        display_path=f"{display_root}/batch-aggregate.json",
    )
    receipt = _with_fingerprint(
        {
            "schema_version": "deepsafe.rlivit-deepstream-batch-receipt/v1",
            "status": "complete_80_jobs_independent_raw_replay",
            "created_at_utc": "2026-07-16T09:01:00+00:00",
            "campaign_nonce": nonce,
            "plan": plan_pin,
            "session_claim": {
                "path": f"{display_root}/session-claim.json",
                "size_bytes": 1024,
                "sha256": "8" * 64,
            },
            "execution_authorization": authorization,
            "mp4_batch_receipt": mp4_pin,
            "runtime_policy": runtime_policy,
            "ds9_compatibility_receipt": ds9_pin,
            "aggregate": aggregate_pin,
            "state": {
                "path": f"{display_root}/batch-state.json",
                "size_bytes": 2048,
                "sha256": "a" * 64,
            },
            "sequence_count": 40,
            "profiles": [640, 960],
            "job_count": 80,
            "job_receipts": job_receipts,
        }
    )
    receipt_pin = _write_rlivit_pinned_json(
        root,
        f"{run_relative}/batch-receipt.json",
        receipt,
        display_path=f"{display_root}/batch-receipt.json",
    )
    status = _rlivit_public_status(complete=True)
    status["evidence"] = {
        "source_plan": _pathless(source_pin),
        "mp4_batch_receipt": _pathless(mp4_pin),
        "runtime_policy": {
            "size_bytes": runtime_artifact["bytes"],
            "sha256": runtime_artifact["sha256"],
        },
        "ds9_compatibility_receipt": _pathless(ds9_pin),
        "batch_aggregate": _pathless(aggregate_pin),
        "batch_receipt": _pathless(receipt_pin),
    }
    status = _with_fingerprint(
        {key: value for key, value in status.items() if key != "fingerprint_sha256"}
    )
    return status, _rlivit_mp4_public_receipt(mp4_pin)


def _verified_throughput_floor_binding() -> dict:
    return {
        "schema_version": "deepsafe.endurance-throughput-floor-binding/v1",
        "status": "verified",
        "artifact_schema": "deepsafe.endurance-throughput-floor/v1",
        "artifact_fingerprint": "a" * 64,
        "artifact_pin": {
            "path": "/host/private/throughput-floor.json",
            "size_bytes": 4096,
            "sha256": "b" * 64,
        },
        "profiles": {
            "640": {
                "aggregate_fps_floor": 80.0,
                "per_stream_fps_floor": 6.6,
            },
            "960": {
                "aggregate_fps_floor": 56.0,
                "per_stream_fps_floor": 4.6,
            },
        },
        "verification": {
            "status": "verified",
            "live_rederived": True,
            "verified_safe_runs": 24,
        },
        "source_runtime_identity": {
            "image": "nvcr.io/private/deepstream:9.0",
            "image_id": "sha256:" + "c" * 64,
            "gpu_index": 0,
            "gpu_identity": {
                "index": "0",
                "uuid": "GPU-private-uuid",
                "name": "NVIDIA RTX A5000 Laptop GPU",
                "driver_version": "590.48.01",
                "memory.total": "16384 MiB",
                "pci.bus_id": "00000000:01:00.0",
            },
            "power_profile": {"available": True, "value": "performance"},
            "max_temperature_c": 84.0,
            "power_safety_policy": {
                "operating_policy_mode": "workstation_managed",
                "hardware_protection_owner": (
                    "workstation_bios_ec_nvidia_driver"
                ),
                "static_signal_action": (
                    "record_measurement_quality_diagnostic"
                ),
                "power_limit_drop_tolerance_w": 5.0,
                "slowdown_consecutive_samples": 3,
                "preflight_samples": 3,
                "preflight_sample_interval_seconds": 1.0,
                "power_limit_fields": [
                    "power_requested_limit_w",
                    "power_current_limit_w",
                    "power_default_limit_w",
                ],
                "diagnostic_slowdown_flags": [
                    "clock_event_sw_thermal_slowdown",
                    "clock_event_hw_slowdown",
                    "clock_event_hw_thermal_slowdown",
                    "clock_event_hw_power_brake_slowdown",
                ],
                "abort_slowdown_flags": [],
                "required_telemetry_failure_action": "safety_abort",
            },
        },
        "source_inputs": {
            "summary": {
                "path": "/host/private/matrix-summary.json",
                "bytes": 2048,
                "sha256": "d" * 64,
            },
            "scene_manifest": {
                "path": "/host/private/scenes.json",
                "bytes": 1024,
                "sha256": "e" * 64,
            },
        },
    }


def test_ppe_video_source_registry_admin_projection_is_pathless_and_not_ready(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    report = _write_campaign_report_with_ppe_registry(tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/validation")
        assert response.status_code == 200
        card = response.json()["campaigns"]["ppe_video_source_registry"]
        assert card["state"] == "valid_metadata_only"
        assert card["available"] is True
        assert card["registry_valid"] is True
        assert card["registry_sha256"] == (
            campaign.EXPECTED_PPE_VIDEO_SOURCE_REGISTRY_SHA256
        )
        assert card["progress"] == {
            "completed": 12,
            "total": 12,
            "remaining": 0,
            "fraction": 1.0,
        }
        assert card["candidate_count"] == 12
        assert card["eligibility_counts"] == {
            key: 0 for key in campaign.PPE_VIDEO_SOURCE_ELIGIBILITY_KEYS
        }
        assert card["primary_plan"] == "user_owned_authorized_site_footage"
        assert card["candidate_groups"] == report["campaigns"][
            "ppe_video_source_registry"
        ]["candidate_groups"]
        assert card["registry_metadata_only"] is True
        assert card["media_or_annotations_downloaded"] is False
        assert card["reporter_downloaded_or_decoded_media"] is False
        assert card["reporter_external_execution_used"] is False
        assert card["quantitative_benchmark_ready"] is False
        assert card["ppe_model_ready"] is False
        assert card["acceptance_effect"] == (
            "planning_only_no_model_readiness"
        )
        assert card["raw_registry_download_available"] is False
        assert all("path" not in item and "href" not in item for item in card["evidence"])
        assert "data/manifests" not in json.dumps(card)
        assert "https://" not in json.dumps(card)
        assert client.get(
            "/api/validation",
            params={"artifact": "ppe_video_source_registry"},
        ).status_code == 404


def test_ppe_video_source_registry_admin_missing_and_tampered_fail_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    _write_campaign_report_with_ppe_registry(
        tmp_path,
        registry_state="missing",
    )

    with TestClient(app) as client:
        missing = client.get("/api/validation").json()["campaigns"][
            "ppe_video_source_registry"
        ]
        assert missing["state"] == "not_started"
        assert missing["registry_valid"] is False
        assert missing["registry_sha256"] is None
        assert missing["candidate_count"] is None
        assert missing["eligibility_counts"] == {}
        assert missing["primary_plan"] is None
        assert all(
            value == [] for value in missing["candidate_groups"].values()
        )
        assert missing["quantitative_benchmark_ready"] is False
        assert missing["ppe_model_ready"] is False

        _write_campaign_report_with_ppe_registry(
            tmp_path,
            registry_state="tampered",
        )
        tampered = client.get("/api/validation").json()["campaigns"][
            "ppe_video_source_registry"
        ]
        assert tampered["state"] == "artifact_error"
        assert tampered["registry_valid"] is False
        assert tampered["registry_sha256"] is None
        assert tampered["candidate_count"] is None
        assert tampered["eligibility_counts"] == {}
        assert tampered["primary_plan"] is None
        assert all(
            value == [] for value in tampered["candidate_groups"].values()
        )
        assert tampered["quantitative_benchmark_ready"] is False
        assert tampered["ppe_model_ready"] is False

        report_path = tmp_path / "campaign-report/report.json"
        overclaim = json.loads(report_path.read_text(encoding="utf-8"))
        overclaim["campaigns"]["ppe_video_source_registry"][
            "ppe_model_ready"
        ] = True
        report_path.write_text(json.dumps(overclaim), encoding="utf-8")
        rejected = client.get("/api/validation").json()["campaigns"][
            "ppe_video_source_registry"
        ]
        assert rejected["state"] == "artifact_error"
        assert rejected["ppe_model_ready"] is False
        assert rejected["quantitative_benchmark_ready"] is False
        assert rejected["candidate_count"] is None


def test_ppe_video_source_registry_ui_explicitly_says_model_not_ready():
    html = (
        Path(__file__).resolve().parents[1] / "admin/static/index.html"
    ).read_text(encoding="utf-8")
    assert "geçerli; yalnız metadata" in html
    assert "PPE model durumu" in html
    assert "hazır değil — yalnız kaynak planı" in html
    assert "Nicel PPE benchmarkı" in html
    assert "yalnız kaynak planlama; PPE model kabulü değil" in html


def test_stale_lineage_ui_is_localized_and_visually_fail_closed():
    html = (
        Path(__file__).resolve().parents[1] / "admin/static/index.html"
    ).read_text(encoding="utf-8")
    assert ".state-stale_lineage" in html
    assert "stale_lineage:'önceki koşuya ait; canlı koşuyla bağlı değil'" in html
    assert "stateLabels[campaign.reason]||campaign.reason" in html
    assert "stateLabels[item.artifact_state]||item.artifact_state" in html


def test_validation_projection_and_allowlisted_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_MAX_ARTIFACT_BYTES", "65536")

    _write_json(
        tmp_path,
        "scene-benchmark/matrix-summary.json",
        {
            "schema_version": "deepsafe.scene-benchmark-matrix/v1",
            "generated_at_utc": "2026-07-16T00:00:00+00:00",
            "streams": 12,
            "duration_seconds_per_run": 300,
            "warmup_seconds_per_run": 15,
            "selected_scenes": 10,
            "selected_sizes": [640, 960],
            "expected_runs": 20,
            "status_counts": {"complete": 3, "pending": 17},
            "runs": [{"status_path": "/host/private/should-not-leak"}],
        },
    )
    _write_json(
        tmp_path,
        "caviar/batch-aggregate.json",
        {
            "schema_version": "deepsafe.caviar-batch-aggregate/v1",
            "generated_at": "2026-07-16T00:01:00+00:00",
            "completeness": {
                "expected_jobs": 16,
                "complete_jobs": 2,
                "pending_jobs": 14,
            },
            "by_model_input": {
                "640": {
                    "complete_sequences": 1,
                    "expected_sequences": 8,
                    "micro": {"precision": 0.8, "recall": 0.1, "f1": 0.18},
                    "macro": {
                        "ap_101_point": 0.14,
                        "last_reported_average_fps": 360.0,
                    },
                }
            },
            "visual_audit_priority": {"secret_command": ["never", "project"]},
        },
    )
    _write_json(
        tmp_path,
        "open-video-review/campaign-plan.json",
        {
            "schema_version": "deepsafe.open-video-review-plan/v1",
            "created_at": "2026-07-16T00:02:00+00:00",
            "status": "planned",
            "campaign": {
                "scene_count": 12,
                "model_input_sizes": [640, 960],
                "expected_jobs": 24,
            },
            "jobs": [{"status": "planned"} for _ in range(24)],
        },
    )
    _write_json(
        tmp_path,
        "endurance/current/status.json",
        {
            "schema_version": "deepsafe.endurance-status/v1",
            "state": "running",
            "dry_run": False,
            "updated_at_utc": "2026-07-16T00:03:00+00:00",
            "target_validated_seconds": 604800,
            "validated_seconds": 3600,
            "profiles_validated_seconds": {"640": 1800, "960": 1800},
            "segments": {"total": 14, "status_counts": {"healthy": 1}},
            "unexpected_restarts": 0,
            "orphan_recoveries": 0,
            "campaign_health_gates": [],
            "checkpoint_path": "/host/private/checkpoint.json",
        },
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        assert response.status_code == 200
        payload = response.json()
        assert payload["read_only"] is True
        assert payload["execution_actions_available"] is False
        scene = payload["campaigns"]["scene_benchmark"]
        assert scene["state"] == "in_progress"
        assert scene["progress"] == {
            "completed": 3,
            "total": 20,
            "remaining": 17,
            "fraction": 0.15,
        }
        assert scene["scope"]["simulated_streams"] == 12
        assert payload["campaigns"]["caviar"]["state"] == "not_started"
        assert payload["campaigns"]["caviar"]["profiles"] == {}
        assert payload["campaigns"]["caviar"]["official_session"]["state"] is None
        open_video = payload["campaigns"]["open_video_review"]
        assert open_video["state"] == "not_started"
        assert open_video["available"] is False
        assert open_video["metric_context"] == {
            "ground_truth": False,
            "qualitative_review_only": True,
            "accuracy_metrics_forbidden": True,
            "automatic_candidates_are_not_visual_review": True,
            "ai_audit_is_not_human_qa": True,
        }
        endurance = payload["campaigns"]["endurance"]
        assert endurance["state"] == "attention"
        assert endurance["throughput_floor"] == {
            "status": "missing",
            "acceptance_safe": False,
            "artifact_fingerprint": None,
            "artifact_fingerprint_short": None,
            "verified_safe_runs": 0,
            "live_rederived": False,
            "profiles": {},
            "source_runtime": {"gpu_name": None, "driver_version": None},
        }
        assert "/host/private" not in response.text
        assert "secret_command" not in response.text

        evidence = client.get(
            "/api/validation", params={"artifact": "scene_benchmark_summary"}
        )
        assert evidence.status_code == 200
        assert evidence.headers["content-type"].startswith("application/json")
        assert evidence.headers["cache-control"] == "no-store"
        assert evidence.headers["x-content-type-options"] == "nosniff"
        assert "matrix-summary.json" in evidence.headers["content-disposition"]
        for projection_only in (
            "scene_benchmark_preflight",
            "caviar_batch_manifest",
            "open_video_plan",
            "endurance_status",
            "endurance_live",
        ):
            assert client.get(
                "/api/validation", params={"artifact": projection_only}
            ).status_code == 404
        assert client.post("/api/validation").status_code == 405


def test_caviar_admin_uses_bounded_report_projection_and_hides_private_session(
    tmp_path, monkeypatch
):
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path,
        hardware_incident_path=None,
    )
    nonce = "e" * 64
    caviar = report["campaigns"]["caviar_ground_truth"]
    caviar.update(
        {
            "expected_jobs": 16,
            "aggregate_result_jobs": 16,
            "validated_jobs": 16,
            "metrics_withheld": False,
            "accepted": True,
            "evidence_complete": True,
            "official_session": {
                "state": "resolved",
                "authorized_policy": True,
                "candidate_count": 1,
                "valid_candidate_count": 1,
                "conflicting_candidate_count": 0,
                "legacy_public_artifacts_ignored": True,
                "private_identity_projected": False,
                "private_nonce": nonce,
                "owner_identity": "private-owner@example.test",
                "authorized_results_root": f"/host/private/caviar/sessions/{nonce}",
            },
            "profiles": {
                "640": {
                    "complete_sequences": 8,
                    "expected_sequences": 8,
                    "precision": 0.92,
                    "recall": 0.91,
                    "f1": 0.915,
                    "ap_101_point_macro": 0.88,
                    "single_stream_offline_fps_macro": 310.0,
                },
                "960": {
                    "complete_sequences": 8,
                    "expected_sequences": 8,
                    "precision": 0.93,
                    "recall": 0.94,
                    "f1": 0.935,
                    "ap_101_point_macro": 0.9,
                    "single_stream_offline_fps_macro": 240.0,
                },
            },
        }
    )
    _write_json(tmp_path, "campaign-report/report.json", report)
    _write_json(
        tmp_path,
        "caviar/batch-aggregate.json",
        {
            "schema_version": "deepsafe.caviar-batch-aggregate/v1",
            "completeness": {
                "expected_jobs": 16,
                "complete_jobs": 16,
                "pending_jobs": 0,
            },
            "by_model_input": {
                "640": {
                    "complete_sequences": 8,
                    "expected_sequences": 8,
                    "micro": {"precision": 0.01, "recall": 0.01, "f1": 0.01},
                    "macro": {
                        "ap_101_point": 0.01,
                        "last_reported_average_fps": 1.0,
                    },
                }
            },
        },
    )
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/validation")

    assert response.status_code == 200
    projected = response.json()["campaigns"]["caviar"]
    assert projected["state"] == "complete"
    assert projected["profiles"]["640"]["recall"] == 0.91
    assert projected["official_session"] == {
        "state": "resolved",
        "candidate_count": 1,
        "valid_candidate_count": 1,
        "conflicting_candidate_count": 0,
        "legacy_public_artifacts_ignored": True,
    }
    assert nonce not in response.text
    assert "private-owner" not in response.text
    assert "/host/private" not in response.text


def test_open_video_projection_separates_automatic_ai_and_human_tiers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    _write_accepted_open_video_report(tmp_path)
    _write_json(
        tmp_path,
        "open-video-review/campaign-review.json",
        {
            "schema_version": "deepsafe.open-video-campaign-review/v1",
            "created_at": "2026-07-16T00:00:00+00:00",
            "jobs": [{"status": "rendered"} for _ in range(24)],
        },
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")

    assert response.status_code == 200
    projected = response.json()["campaigns"]["open_video_review"]
    assert projected["available"] is True
    assert projected["state"] == "complete"
    assert projected["progress_label"] == "Hash-bağlı AI audit"
    assert projected["progress"] == {
        "completed": 42,
        "total": 42,
        "remaining": 0,
        "fraction": 1.0,
    }
    assert projected["automatic_candidate_generation"] == {
        "state": "complete",
        "accepted": True,
        "progress": {
            "completed": 24,
            "total": 24,
            "remaining": 0,
            "fraction": 1.0,
        },
        "rendered_jobs": 22,
        "withheld_jobs": 2,
        "candidate_assets": {
            "decisions": {
                "completed": 42,
                "total": 42,
                "remaining": 0,
                "fraction": 1.0,
            },
            "assets": {
                "completed": 168,
                "total": 168,
                "remaining": 0,
                "fraction": 1.0,
            },
            "index_contract_valid": True,
        },
        "not_a_visual_review": True,
    }
    ai_audit = projected["ai_qualitative_visual_audit"]
    assert ai_audit["accepted"] is True
    assert ai_audit["source_frames"]["completed"] == 21
    assert ai_audit["profile_decisions"]["completed"] == 42
    assert ai_audit["distinct_video_types"] == 11
    assert ai_audit["coverage"] == {
        "medium_close_source_frames": 11,
        "overhead_top_view_source_frames": 2,
        "high_oblique_source_frames": 6,
    }
    assert ai_audit["withheld_scenes_excluded"] is True
    assert ai_audit["human_review_claimed"] is False
    assert projected["human_terminal_qa"] == {
        "state": "pending",
        "accepted": False,
        "progress": {
            "completed": 0,
            "total": 42,
            "remaining": 42,
            "fraction": 0.0,
        },
        "pending_decisions": 42,
        "artifact_contract_valid": True,
        "required_for_ai_audit_acceptance": False,
    }
    assert projected["paired_profile_comparison"] == {
        "paired_scene_count": 11,
        "paired_source_frame_count": 21,
    }


@pytest.mark.parametrize(
    "mutation",
    ["withheld_promoted_to_ai", "ai_promoted_to_human"],
)
def test_open_video_projection_rejects_tier_overclaim_fail_closed(
    tmp_path, monkeypatch, mutation
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    report = _write_accepted_open_video_report(tmp_path)
    section = report["campaigns"]["open_video_manual_review"]
    if mutation == "withheld_promoted_to_ai":
        section["ai_qualitative_visual_audit"]["reviewed_scene_ids"][0] = (
            "scene_11"
        )
    else:
        section["ai_qualitative_visual_audit"]["human_review_claimed"] = True
    _write_json(tmp_path, "campaign-report/report.json", report)

    with TestClient(app) as client:
        response = client.get("/api/validation")

    assert response.status_code == 200
    projected = response.json()["campaigns"]["open_video_review"]
    assert projected["available"] is False
    assert projected["state"] == "artifact_error"
    assert projected["automatic_candidate_generation"]["accepted"] is False
    assert projected["ai_qualitative_visual_audit"]["accepted"] is False
    assert projected["human_terminal_qa"]["accepted"] is False


def test_rlivit_blocked_projection_is_bounded_and_fail_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    _write_json(
        tmp_path,
        "rlivit/current/status.json",
        _rlivit_public_status(),
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        assert response.status_code == 200
        campaign = response.json()["campaigns"]["rlivit"]
        assert campaign["state"] == "blocked"
        assert campaign["progress"] == {
            "completed": 0,
            "total": 80,
            "remaining": 80,
            "fraction": 0.0,
        }
        assert campaign["scope"] == {
            "split": "test",
            "sequences": 40,
            "frames": 480,
            "targets": 4318,
            "jobs": 80,
            "model_input_sizes": [640, 960],
            "camera_geometry": "fixed_high_oblique_road_surveillance",
            "day_night": True,
            "locations": 8,
        }
        assert campaign["ground_truth"] == {
            "valid": True,
            "live_sources_verified": True,
            "person_boxes_per_profile": 4318,
        }
        assert campaign["accelerated_execution_occurred"] is False
        assert campaign["profiles"]["640"]["overall"] is None
        assert campaign["evidence_readiness"] == {
            "source_plan": True,
            "mp4_batch_receipt": False,
            "runtime_policy": True,
            "ds9_compatibility_receipt": False,
            "batch_aggregate": False,
            "batch_receipt": False,
        }
        assert "/home/" not in response.text
        assert "campaign_nonce" not in response.text
        assert client.get(
            "/api/validation", params={"artifact": "rlivit_public_status"}
        ).status_code == 404

        path = tmp_path / "rlivit/current/status.json"
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["progress"]["completed_jobs"] = 1
        path.write_text(json.dumps(tampered), encoding="utf-8")
        rejected = client.get("/api/validation").json()["campaigns"]["rlivit"]
        assert rejected["state"] == "artifact_error"
        assert rejected["profiles"] == {}
        assert rejected["progress"]["total"] == 0


def test_rlivit_complete_metrics_and_mp4_receipt_require_exact_contract(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    status, mp4_receipt = _rlivit_complete_artifacts(tmp_path)
    _write_json(
        tmp_path,
        "rlivit/current/status.json",
        status,
    )
    _write_json(tmp_path, "rlivit/mp4-batch-receipt.json", mp4_receipt)

    with TestClient(app) as client:
        response = client.get("/api/validation")
        campaign = response.json()["campaigns"]["rlivit"]
        assert campaign["state"] == "complete"
        assert campaign["progress"]["completed"] == 80
        assert campaign["profiles"]["640"]["overall"] == _rlivit_metric(4318)
        assert campaign["profiles"]["960"]["overall"]["ap_101_point"] == 1.0
        assert campaign["metric_context"]["coco_map"] is False
        assert campaign["input_materialization"] == {
            "status": "complete_verified_cpu_only",
            "complete_sequences": 40,
            "expected_sequences": 40,
            "frames": 480,
            "codec": "H.264 High / yuv420p",
            "fps": "5/4",
            "minimum_psnr_db": 47.9,
            "minimum_ssim": 0.994,
            "maximum_cpu_platform_temperature_c": 74.0,
            "thermal_policy_id": "workstation_managed",
            "gpu_executed": False,
            "batch_receipt_pin": status["evidence"]["mp4_batch_receipt"],
            "cross_bound_to_campaign": True,
        }
        assert campaign["proof_binding"] == {
            "status_fingerprint_role": "self_hash_integrity_only",
            "mp4_receipt_status_pin_cross_bound": True,
            "final_receipt_live_pin_verified": True,
            "aggregate_live_pin_verified": True,
            "job_receipts_live_pins_verified": True,
            "aggregate_status_metrics_cross_bound": True,
            "complete_claim_verified": True,
            "status_self_hash_valid": True,
        }
        assert all(
            item["raw_download_allowed"] is False
            for item in campaign["evidence"]
        )
        assert "test-operator" not in response.text
        assert "9" * 64 not in response.text
        assert "validation/results/rlivit/runs" not in response.text
        assert client.get(
            "/api/validation", params={"artifact": "rlivit_mp4_receipt"}
        ).status_code == 404

        status_path = tmp_path / "rlivit/current/status.json"
        inconsistent = json.loads(status_path.read_text(encoding="utf-8"))
        inconsistent["profiles"]["960"]["metrics"]["overall"]["ground_truth"] = 4317
        inconsistent["profiles"]["960"]["metrics"]["overall"]["tp"] = 4317
        inconsistent = _with_fingerprint(
            {key: value for key, value in inconsistent.items() if key != "fingerprint_sha256"}
        )
        status_path.write_text(json.dumps(inconsistent), encoding="utf-8")
        rejected = client.get("/api/validation").json()["campaigns"]["rlivit"]
        assert rejected["state"] == "artifact_error"
        assert rejected["profiles"] == {}


def test_rlivit_complete_requires_all_live_canonical_job_receipts(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    status, mp4_receipt = _rlivit_complete_artifacts(tmp_path)
    _write_json(tmp_path, "rlivit/current/status.json", status)
    _write_json(tmp_path, "rlivit/mp4-batch-receipt.json", mp4_receipt)
    nonce = "9" * 64
    job_path = (
        tmp_path
        / f"rlivit/runs/{nonce}/jobs/000/640/job-receipt.json"
    )
    original = job_path.read_bytes()

    def assert_rejected(client: TestClient) -> None:
        campaign = client.get("/api/validation").json()["campaigns"]["rlivit"]
        assert campaign["state"] == "artifact_error"
        assert campaign["profiles"] == {}
        assert campaign["proof_binding"]["job_receipts_live_pins_verified"] is False
        assert "jobs/000/640" not in json.dumps(campaign)

    with TestClient(app) as client:
        accepted = client.get("/api/validation").json()["campaigns"]["rlivit"]
        assert accepted["state"] == "complete"
        assert accepted["proof_binding"]["job_receipts_live_pins_verified"] is True

        job_path.write_bytes(original + b"tamper")
        assert_rejected(client)
        job_path.write_bytes(original)

        job_path.unlink()
        assert_rejected(client)
        job_path.write_bytes(original)

        job_path.unlink()
        job_path.symlink_to(Path("..") / "960" / "job-receipt.json")
        assert_rejected(client)
        job_path.unlink()
        job_path.write_bytes(original)

        restored = client.get("/api/validation").json()["campaigns"]["rlivit"]
        assert restored["state"] == "complete"
        assert restored["proof_binding"]["job_receipts_live_pins_verified"] is True


def test_rlivit_adversarial_semantics_and_cross_artifact_binding_fail_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    status, mp4_receipt = _rlivit_complete_artifacts(tmp_path)
    status_path = tmp_path / "rlivit/current/status.json"
    _write_json(tmp_path, "rlivit/mp4-batch-receipt.json", mp4_receipt)

    def publish(value: dict) -> None:
        _write_json(
            tmp_path,
            "rlivit/current/status.json",
            _with_fingerprint(
                {
                    key: item
                    for key, item in value.items()
                    if key != "fingerprint_sha256"
                }
            ),
        )

    def assert_artifact_error(client: TestClient) -> dict:
        campaign = client.get("/api/validation").json()["campaigns"]["rlivit"]
        assert campaign["state"] == "artifact_error"
        assert campaign["profiles"] == {}
        assert campaign["accelerated_execution_occurred"] is None
        assert campaign["metric_context"]["withheld_until_complete"] is True
        return campaign

    with TestClient(app) as client:
        null_metric = copy.deepcopy(status)
        null_metric["profiles"]["640"]["metrics"]["overall"]["precision"] = None
        publish(null_metric)
        assert_artifact_error(client)

        unknown_metric_field = copy.deepcopy(status)
        unknown_metric_field["profiles"]["640"]["metrics"]["overall"][
            "private_source"
        ] = "/home/private/runs/secret"
        publish(unknown_metric_field)
        assert_artifact_error(client)

        unverified_sources = copy.deepcopy(status)
        unverified_sources["ground_truth"]["live_sources_verified"] = False
        publish(unverified_sources)
        assert_artifact_error(client)

        contradictory_day_partition = copy.deepcopy(status)
        day = contradictory_day_partition["profiles"]["640"]["metrics"][
            "daytime"
        ]["day"]
        day.update(
            {
                "tp": 0,
                "fp": 0,
                "fn": day["ground_truth"],
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap_101_point": 0.0,
            }
        )
        publish(contradictory_day_partition)
        assert_artifact_error(client)

        wrong_mp4_pin = copy.deepcopy(status)
        wrong_mp4_pin["evidence"]["mp4_batch_receipt"]["sha256"] = "e" * 64
        publish(wrong_mp4_pin)
        rejected = assert_artifact_error(client)
        assert rejected["proof_binding"]["mp4_receipt_status_pin_cross_bound"] is False

        mismatched_public_mp4 = copy.deepcopy(mp4_receipt)
        mismatched_public_mp4["batch_receipt_pin"]["sha256"] = "d" * 64
        mismatched_public_mp4 = _with_fingerprint(
            {
                key: value
                for key, value in mismatched_public_mp4.items()
                if key != "fingerprint_sha256"
            }
        )
        _write_json(
            tmp_path,
            "rlivit/mp4-batch-receipt.json",
            mismatched_public_mp4,
        )
        publish(status)
        rejected = assert_artifact_error(client)
        assert rejected["proof_binding"]["mp4_receipt_status_pin_cross_bound"] is False
        _write_json(tmp_path, "rlivit/mp4-batch-receipt.json", mp4_receipt)

        legacy_mp4 = copy.deepcopy(mp4_receipt)
        legacy_mp4["schema_version"] = "deepsafe.rlivit-mp4-admin-receipt/v1"
        legacy_mp4 = _with_fingerprint(
            {
                key: value
                for key, value in legacy_mp4.items()
                if key != "fingerprint_sha256"
            }
        )
        _write_json(tmp_path, "rlivit/mp4-batch-receipt.json", legacy_mp4)
        publish(status)
        assert_artifact_error(client)
        _write_json(tmp_path, "rlivit/mp4-batch-receipt.json", mp4_receipt)

        self_hashed_without_final_receipt = _rlivit_public_status(complete=True)
        publish(self_hashed_without_final_receipt)
        rejected = assert_artifact_error(client)
        assert rejected["proof_binding"]["status_self_hash_valid"] is True
        assert rejected["proof_binding"]["complete_claim_verified"] is False

        awaiting_without_prerequisites = _rlivit_public_status()
        awaiting_without_prerequisites["status"] = "awaiting_execution"
        awaiting_without_prerequisites["blocker_codes"] = []
        awaiting_without_prerequisites["evidence"] = {
            key: None for key in awaiting_without_prerequisites["evidence"]
        }
        for profile in awaiting_without_prerequisites["profiles"].values():
            profile["status"] = "pending"
        publish(awaiting_without_prerequisites)
        assert_artifact_error(client)

        awaiting_with_partial_final_evidence = copy.deepcopy(
            awaiting_without_prerequisites
        )
        awaiting_with_partial_final_evidence["evidence"] = {
            key: copy.deepcopy(status["evidence"][key])
            if key
            in {
                "source_plan",
                "mp4_batch_receipt",
                "runtime_policy",
                "ds9_compatibility_receipt",
                "batch_aggregate",
            }
            else None
            for key in awaiting_with_partial_final_evidence["evidence"]
        }
        publish(awaiting_with_partial_final_evidence)
        assert_artifact_error(client)

        failed_without_launch = copy.deepcopy(awaiting_without_prerequisites)
        failed_without_launch["status"] = "failed"
        failed_without_launch["progress"]["failed_jobs"] = 1
        failed_without_launch["evidence"] = {
            key: copy.deepcopy(status["evidence"][key])
            if key
            in {
                "source_plan",
                "mp4_batch_receipt",
                "runtime_policy",
                "ds9_compatibility_receipt",
            }
            else None
            for key in failed_without_launch["evidence"]
        }
        publish(failed_without_launch)
        assert_artifact_error(client)

        assert status_path.is_file()


def test_rlivit_awaiting_execution_requires_exact_prerequisite_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    complete, _mp4 = _rlivit_complete_artifacts(tmp_path)
    awaiting = _rlivit_public_status()
    awaiting["status"] = "awaiting_execution"
    awaiting["blocker_codes"] = []
    awaiting["evidence"] = {
        key: copy.deepcopy(complete["evidence"][key])
        if key
        in {
            "source_plan",
            "mp4_batch_receipt",
            "runtime_policy",
            "ds9_compatibility_receipt",
        }
        else None
        for key in awaiting["evidence"]
    }
    for profile in awaiting["profiles"].values():
        profile["status"] = "pending"
    awaiting = _with_fingerprint(
        {key: value for key, value in awaiting.items() if key != "fingerprint_sha256"}
    )
    _write_json(tmp_path, "rlivit/current/status.json", awaiting)

    with TestClient(app) as client:
        campaign = client.get("/api/validation").json()["campaigns"]["rlivit"]
    assert campaign["state"] == "planned"
    assert campaign["progress"]["completed"] == 0
    assert campaign["accelerated_execution_occurred"] is False
    assert campaign["metric_context"]["withheld_until_complete"] is True


def test_endurance_floor_projection_is_bounded_and_live_evaluation_is_bound(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    binding = _verified_throughput_floor_binding()
    _write_json(
        tmp_path,
        "endurance/current/status.json",
        {
            "schema_version": "deepsafe.endurance-status/v1",
            "state": "running",
            "dry_run": False,
            "updated_at_utc": "2026-07-16T09:00:00+00:00",
            "target_validated_seconds": 604800,
            "validated_seconds": 21600,
            "profiles_validated_seconds": {"640": 21600, "960": 0},
            "segments": {"total": 28, "status_counts": {"healthy": 1}},
            "unexpected_restarts": 0,
            "orphan_recoveries": 0,
            "campaign_health_gates": [],
            "throughput_floor": binding,
            "checkpoint_path": "/host/private/checkpoint.json",
            "command": ["docker", "run", "private-command"],
            "private_uri": "rtsp://operator:secret@private-camera/live",
        },
    )
    _write_json(
        tmp_path,
        "endurance/current/live.json",
        {
            "schema_version": "deepsafe.endurance-live/v1",
            "state": "healthy",
            "updated_at_utc": "2026-07-16T09:01:00+00:00",
            "profile": 640,
            "throughput_floor_evaluation": {
                "schema_version": (
                    "deepsafe.endurance-throughput-floor-evidence/v1"
                ),
                "status": "passed",
                "artifact_fingerprint": "a" * 64,
                "profile": 640,
                "aggregate_fps_floor": 80.0,
                "per_stream_fps_floor": 6.6,
                "observed_aggregate_fps_p05": 81.25,
            },
            "throughput_floor_live_verification": {
                "status": "verified",
                "artifact_fingerprint": "a" * 64,
            },
            "health_gates": [{"detail": "/host/private/error"}],
        },
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        assert response.status_code == 200
        projected = response.json()["campaigns"]["endurance"]
        assert projected["state"] == "running"
        assert projected["throughput_floor"] == {
            "status": "verified",
            "acceptance_safe": True,
            "artifact_fingerprint": "a" * 64,
            "artifact_fingerprint_short": "aaaaaaaaaaaa…aaaaaaaa",
            "verified_safe_runs": 24,
            "live_rederived": True,
            "profiles": {
                "640": {
                    "aggregate_fps_floor": 80.0,
                    "per_stream_fps_floor": 6.6,
                },
                "960": {
                    "aggregate_fps_floor": 56.0,
                    "per_stream_fps_floor": 4.6,
                },
            },
            "source_runtime": {
                "gpu_name": "NVIDIA RTX A5000 Laptop GPU",
                "driver_version": "590.48.01",
            },
        }
        assert projected["heartbeat"]["throughput_floor_evaluation"] == {
            "status": "passed",
            "acceptance_safe": True,
            "profile": 640,
            "aggregate_fps_floor": 80.0,
            "per_stream_fps_floor": 6.6,
            "observed_aggregate_fps_p05": 81.25,
        }
        assert all("path" not in item for item in projected["evidence"])
        for forbidden in (
            "/host/private",
            "nvcr.io/private",
            "GPU-private-uuid",
            "private-command",
            "rtsp://",
            "operator:secret",
        ):
            assert forbidden not in response.text
        assert client.get(
            "/api/validation", params={"artifact": "endurance_status"}
        ).status_code == 404
        assert client.get(
            "/api/validation", params={"artifact": "endurance_live"}
        ).status_code == 404

        live = json.loads(
            (tmp_path / "endurance/current/live.json").read_text(encoding="utf-8")
        )
        live["throughput_floor_evaluation"]["artifact_fingerprint"] = "f" * 64
        _write_json(tmp_path, "endurance/current/live.json", live)
        mismatched = client.get("/api/validation").json()["campaigns"][
            "endurance"
        ]
        assert mismatched["heartbeat"]["throughput_floor_evaluation"] == {
            "status": "invalid",
            "acceptance_safe": False,
        }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda binding: binding["source_runtime_identity"]["gpu_identity"].pop(
            "pci.bus_id"
        ),
        lambda binding: binding["source_runtime_identity"][
            "power_safety_policy"
        ].update({"operating_policy_mode": "unknown"}),
        lambda binding: binding["source_runtime_identity"][
            "power_safety_policy"
        ].update({"static_signal_action": "safety_abort"}),
        lambda binding: binding["source_runtime_identity"][
            "power_safety_policy"
        ].update(
            {
                "abort_slowdown_flags": [
                    "clock_event_sw_thermal_slowdown",
                    "clock_event_hw_slowdown",
                    "clock_event_hw_thermal_slowdown",
                    "clock_event_hw_power_brake_slowdown",
                ]
            }
        ),
    ),
)
def test_endurance_floor_projection_rejects_runtime_policy_drift(mutation):
    binding = _verified_throughput_floor_binding()
    mutation(binding)
    projected = admin_validation._throughput_floor_projection(binding)
    assert projected["status"] == "invalid"
    assert projected["acceptance_safe"] is False


def test_endurance_missing_invalid_and_synthetic_floors_fail_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    invalid = _verified_throughput_floor_binding()
    invalid["verification"]["verified_safe_runs"] = 23
    synthetic = {
        "schema_version": "deepsafe.endurance-throughput-floor-binding/v1",
        "status": "synthetic_pending",
        "artifact_schema": None,
        "artifact_fingerprint": None,
        "artifact_pin": None,
        "profiles": {
            "640": {
                "aggregate_fps_floor": None,
                "per_stream_fps_floor": None,
            },
            "960": {
                "aggregate_fps_floor": None,
                "per_stream_fps_floor": None,
            },
        },
        "verification": {
            "status": "not_required_for_synthetic_dry_run",
            "live_rederived": False,
            "verified_safe_runs": 0,
        },
        "source_runtime_identity": None,
        "source_inputs": {"summary": None, "scene_manifest": None},
    }

    with TestClient(app) as client:
        for binding, expected in (
            (None, "missing"),
            (invalid, "invalid"),
            (synthetic, "synthetic_pending"),
        ):
            status = {
                "schema_version": "deepsafe.endurance-status/v1",
                "state": "complete",
                "dry_run": binding is synthetic,
                "updated_at_utc": "2026-07-16T09:00:00+00:00",
                "target_validated_seconds": 604800,
                "validated_seconds": 604800,
                "throughput_floor": binding,
            }
            _write_json(tmp_path, "endurance/current/status.json", status)
            projected = client.get("/api/validation").json()["campaigns"][
                "endurance"
            ]
            assert projected["state"] == "attention"
            assert projected["throughput_floor"]["status"] == expected
            assert projected["throughput_floor"]["acceptance_safe"] is False
            assert projected["throughput_floor"]["profiles"] == {}
            assert projected["heartbeat"]["throughput_floor_evaluation"] == {
                "status": "not_reported",
                "acceptance_safe": False,
            }


def test_gpu_and_loaf_projections_are_compact_and_raw_artifacts_stay_private(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(tmp_path))
    _write_json(
        tmp_path,
        "gpu-reentry/current/evidence.json",
        {
            "schema_version": "deepsafe.gpu-reentry-evidence/v1",
            "status": "blocked",
            "collected_at_utc": "2026-07-16T02:30:00+00:00",
            "gpu_index": 0,
            "collection_policy": {
                "read_only": True,
                "benchmark_started": False,
                "gpu_stress_performed": False,
                "settings_changed": False,
                "sudo_executed": False,
            },
            "idle_gpu_telemetry": {"samples": [{}, {}]},
            "verification": {
                "schema_version": "deepsafe.gpu-reentry-verification/v1",
                "status": "blocked",
                "all_required_evidence_present": False,
                "sustained_load_authorized": False,
                "failed_gate_ids": ["adapter_declaration"],
                "gates": [
                    {"id": "idle", "required": True, "passed": True},
                    {
                        "id": "adapter_declaration",
                        "required": True,
                        "passed": False,
                    },
                ],
            },
            "operator_declaration_file": {"path": "/host/private/operator.json"},
            "read_only_command": ["nvidia-smi", "--query-gpu=name"],
        },
    )
    _write_json(
        tmp_path,
        "gpu-reentry/current/verification.json",
        {
            "schema_version": "deepsafe.gpu-reentry-verification/v1",
            "status": "blocked",
            "verified_at_utc": "2026-07-16T02:31:00+00:00",
            "all_required_evidence_present": False,
            "sustained_load_authorized": False,
            "failed_gate_ids": ["adapter_declaration"],
            "gates": [
                {"id": "idle", "required": True, "passed": True},
                {
                    "id": "adapter_declaration",
                    "required": True,
                    "passed": False,
                    "detail": "run /host/private/unsafe-command",
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "loaf/val-20-25m/deepstream/batch-aggregate.json",
        {
            "schema_version": "deepsafe.loaf-deepstream-batch-aggregate/v1",
            "generated_at": "2026-07-16T02:32:00+00:00",
            "plan_fingerprint": "a" * 64,
            "aggregation_status": "complete",
            "completeness": {
                "expected_jobs": 16,
                "complete_jobs": 16,
                "pending_jobs": 0,
                "is_complete": True,
                "state_counts": {"complete": 16},
                "job_states": [{"command": ["docker", "run"]}],
            },
            "results": [{} for _ in range(16)],
            "profiles": {
                "640": {
                    "overall": {
                        "ground_truth": 100,
                        "tp": 70,
                        "fp": 5,
                        "fn": 30,
                        "precision": 0.933333,
                        "recall": 0.7,
                        "f1": 0.8,
                        "ap_101_point": 0.72,
                    },
                    "predictions": {"path": "/host/private/predictions.jsonl"},
                },
                "960": {
                    "overall": {
                        "ground_truth": 100,
                        "tp": 80,
                        "fp": 4,
                        "fn": 20,
                        "precision": 0.952381,
                        "recall": 0.8,
                        "f1": 0.869565,
                        "ap_101_point": 0.82,
                    },
                    "predictions": {"path": "/host/private/960.jsonl"},
                },
            },
        },
    )
    _write_json(
        tmp_path,
        "loaf/val-20-25m/deepstream/dry-run-plan.json",
        {
            "schema_version": "deepsafe.loaf-deepstream-batch-plan/v1",
            "created_at": "2026-07-16T02:31:30+00:00",
            "status": "execution-finished",
            "gpu_execution_requested": True,
            "plan_fingerprint": "a" * 64,
            "padding": "x" * (admin_validation.HARD_MAX_ARTIFACT_BYTES + 1),
            "source_contract": {
                "metric_geometry": "axis_aligned_envelope_of_rotated_box",
                "sequence_count": 8,
                "frame_count": 2948,
            },
            "campaign": {
                "split": "val",
                "distance_band_m": {
                    "minimum_inclusive": 20.0,
                    "maximum_exclusive": 25.0,
                },
                "sequence_count": 8,
                "frame_count": 2948,
                "model_input_sizes": [640, 960],
                "expected_jobs": 16,
            },
            "jobs": [
                {"command": ["python", "--execute", "/host/private/job"]}
                for _ in range(16)
            ],
        },
    )
    _write_json(
        tmp_path,
        "loaf/val-20-25m/distance-bins/preparation-manifest.json",
        {
            "schema_version": "deepsafe.loaf-distance-bin-preparation/v1",
            "status": "prepared_not_evaluated",
            "split": "val",
            "test_unseen_opened": False,
            "gpu_or_inference_executed": False,
            "sequence_count": 8,
            "frame_count": 2948,
            "base_target_count": 7539,
            "partition_target_count": 7539,
            "metric_geometry": "axis_aligned_envelope_of_rotated_box",
            "base_ground_truth": {"path": "/host/private/ground-truth.json"},
            "distance_bins": [
                {"bin_id": "20-21m", "target_annotation_count": 2298},
                {"bin_id": "21-22m", "target_annotation_count": 1702},
                {"bin_id": "22-23m", "target_annotation_count": 1333},
                {"bin_id": "23-24m", "target_annotation_count": 1210},
                {"bin_id": "24-25m", "target_annotation_count": 996},
            ],
        },
    )
    _write_json(
        tmp_path,
        "loaf/val-20-25m/distance-bins/evaluation-plan.json",
        {
            "schema_version": "deepsafe.loaf-distance-bin-evaluation-plan/v1",
            "status": "waiting_for_predictions",
            "split": "val",
            "dry_run": True,
            "gpu_or_inference_executed": False,
            "test_unseen_opened": False,
            "expected_profiles": [640, 960],
            "expected_distance_bins": [
                "20-21m",
                "21-22m",
                "22-23m",
                "23-24m",
                "24-25m",
            ],
            "expected_evaluations": 10,
            "metric": {
                "name": "AP101@IoU0.5",
                "explicitly_not": "not COCO mAP@[.50:.95]",
            },
            "profiles": [
                {
                    "model_input": profile,
                    "status": "waiting_for_complete_profile_merge",
                    "predictions": f"/host/private/{profile}/predictions.jsonl",
                }
                for profile in (640, 960)
            ],
        },
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        assert response.status_code == 200
        payload = response.json()
        gpu = payload["campaigns"]["gpu_reentry"]
        assert gpu["state"] == "blocked"
        assert gpu["progress"] == {
            "completed": 1,
            "total": 2,
            "remaining": 1,
            "fraction": 0.5,
        }
        assert gpu["sustained_load_authorized"] is False
        assert gpu["collection_policy"]["read_only"] is True

        loaf = payload["campaigns"]["loaf_deepstream"]
        assert loaf["state"] == "complete"
        assert loaf["progress"]["completed"] == 16
        assert loaf["scope"]["frames"] == 2948
        assert loaf["profiles"]["640"]["recall"] == 0.7
        assert loaf["metric_context"]["withheld_until_complete"] is False
        loaf_plan_evidence = next(
            item for item in loaf["evidence"] if item["id"] == "loaf_batch_plan"
        )
        assert loaf_plan_evidence["artifact_state"] == "ok"
        assert loaf_plan_evidence["size_bytes"] > (
            admin_validation.HARD_MAX_ARTIFACT_BYTES
        )

        bins = payload["campaigns"]["loaf_distance_bins"]
        assert bins["state"] == "waiting_for_predictions"
        assert bins["progress"] == {
            "completed": 0,
            "total": 10,
            "remaining": 10,
            "fraction": 0.0,
        }
        assert bins["partition_complete"] is True
        assert bins["bin_target_counts"]["24-25m"] == 996
        assert bins["safety"] == {
            "dry_run": True,
            "gpu_or_inference_executed": False,
            "test_unseen_opened": False,
        }
        assert bins["metric_context"]["coco_map"] is False

        rendered = response.text
        assert "/host/private" not in rendered
        assert "nvidia-smi" not in rendered
        assert "--execute" not in rendered
        assert "docker" not in rendered
        for campaign in (gpu, loaf, bins):
            assert all("href" not in item for item in campaign["evidence"])
            assert all(
                item["raw_download_allowed"] is False
                for item in campaign["evidence"]
            )

        plan_path = tmp_path / "loaf/val-20-25m/deepstream/dry-run-plan.json"
        mismatched_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        mismatched_plan["plan_fingerprint"] = "b" * 64
        plan_path.write_text(json.dumps(mismatched_plan), encoding="utf-8")
        mismatch = client.get("/api/validation").json()["campaigns"][
            "loaf_deepstream"
        ]
        assert mismatch["state"] == "artifact_error"
        assert mismatch["profiles"] == {}

        bin_counts = {
            "20-21m": 2298,
            "21-22m": 1702,
            "22-23m": 1333,
            "23-24m": 1210,
            "24-25m": 996,
        }
        _write_json(
            tmp_path,
            "loaf/val-20-25m/distance-bins/aggregate.json",
            {
                "schema_version": "deepsafe.loaf-distance-bin-aggregate/v1",
                "status": "complete",
                "split": "val",
                "test_unseen_opened": False,
                "metric": {
                    "name": "AP101@IoU0.5",
                    "explicitly_not": "not COCO mAP@[.50:.95]",
                },
                "completeness": {
                    "expected_evaluations": 10,
                    "complete_evaluations": 10,
                    "is_complete": True,
                },
                "rows": [
                    {
                        "model_input": profile,
                        "distance_bin_id": bin_id,
                        "ground_truth": count,
                        "tp": count,
                        "fp": 0,
                        "fn": 0,
                        "ignored_predictions": 2,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "ap101_iou_0_5": 1.0,
                        "evaluation": {"path": "/host/private/evaluation.json"},
                    }
                    for profile in (640, 960)
                    for bin_id, count in bin_counts.items()
                ],
            },
        )
        completed_bins_response = client.get("/api/validation")
        completed_bins = completed_bins_response.json()["campaigns"][
            "loaf_distance_bins"
        ]
        assert completed_bins["state"] == "complete"
        assert completed_bins["progress"] == {
            "completed": 10,
            "total": 10,
            "remaining": 0,
            "fraction": 1.0,
        }
        assert len(completed_bins["results"]) == 10
        assert completed_bins["results"][0] == {
            "model_input": 640,
            "distance_bin_id": "20-21m",
            "ground_truth": 2298,
            "tp": 2298,
            "fp": 0,
            "fn": 0,
            "ignored_predictions": 2,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "ap101_iou_0_5": 1.0,
        }
        assert "/host/private" not in completed_bins_response.text
        for artifact in (
            "gpu_reentry_evidence",
            "gpu_reentry_verification",
            "loaf_batch_aggregate",
            "loaf_batch_plan",
            "loaf_distance_bin_preparation",
            "loaf_distance_bin_evaluation_plan",
            "loaf_distance_bin_aggregate",
        ):
            assert client.get(
                "/api/validation", params={"artifact": artifact}
            ).status_code == 404


def test_gpu_reentry_projection_rejects_malformed_or_mismatched_verification(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(tmp_path))
    blocked = {
        "schema_version": "deepsafe.gpu-reentry-verification/v1",
        "status": "blocked",
        "all_required_evidence_present": False,
        "sustained_load_authorized": False,
        "failed_gate_ids": ["adapter"],
        "gates": [
            {"id": "idle", "required": True, "passed": True},
            {"id": "adapter", "required": True, "passed": False},
        ],
    }
    evidence = {
        "schema_version": "deepsafe.gpu-reentry-evidence/v1",
        "status": "blocked",
        "collected_at_utc": "2026-07-16T02:30:00+00:00",
        "collection_policy": {"read_only": True},
        "verification": blocked,
    }
    _write_json(tmp_path, "gpu-reentry/current/evidence.json", evidence)
    malformed = dict(blocked)
    malformed["gates"] = "not-an-array"
    _write_json(
        tmp_path, "gpu-reentry/current/verification.json", malformed
    )

    with TestClient(app) as client:
        malformed_projection = client.get("/api/validation").json()["campaigns"][
            "gpu_reentry"
        ]
        assert malformed_projection["state"] == "artifact_error"
        assert malformed_projection["progress"]["total"] == 0
        assert malformed_projection["sustained_load_authorized"] is None

        ready = {
            "schema_version": "deepsafe.gpu-reentry-verification/v1",
            "status": "ready_for_operator_review",
            "all_required_evidence_present": True,
            "sustained_load_authorized": False,
            "failed_gate_ids": [],
            "gates": [{"id": "idle", "required": True, "passed": True}],
        }
        mismatched_evidence = dict(evidence)
        mismatched_evidence["status"] = "ready_for_operator_review"
        mismatched_evidence["verification"] = ready
        _write_json(
            tmp_path, "gpu-reentry/current/evidence.json", mismatched_evidence
        )
        _write_json(
            tmp_path, "gpu-reentry/current/verification.json", blocked
        )
        mismatch = client.get("/api/validation").json()["campaigns"][
            "gpu_reentry"
        ]
        assert mismatch["state"] == "artifact_error"
        assert mismatch["failed_gate_ids"] == []

        _write_json(tmp_path, "gpu-reentry/current/evidence.json", evidence)
        matching = client.get("/api/validation").json()["campaigns"][
            "gpu_reentry"
        ]
        assert matching["state"] == "blocked"
        assert matching["progress"]["completed"] == 1
        assert matching["failed_gate_ids"] == ["adapter"]


def test_validation_missing_invalid_and_oversized_are_graceful(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_MAX_ARTIFACT_BYTES", "160")

    scene_path = tmp_path / "scene-benchmark/matrix-summary.json"
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text("{" + '"padding":"' + ("x" * 300) + '"}', encoding="utf-8")
    caviar_path = tmp_path / "caviar/batch-aggregate.json"
    caviar_path.parent.mkdir(parents=True)
    caviar_path.write_text("{broken", encoding="utf-8")

    with TestClient(app) as client:
        response = client.get("/api/validation")
        assert response.status_code == 200
        campaigns = response.json()["campaigns"]
        assert campaigns["scene_benchmark"]["state"] == "artifact_error"
        assert campaigns["scene_benchmark"]["evidence"][0]["artifact_state"] == "too_large"
        assert campaigns["caviar"]["state"] == "not_started"
        assert campaigns["caviar"]["evidence"][0]["artifact_state"] == "missing"
        assert campaigns["gpu_reentry"]["state"] == "not_started"
        assert campaigns["rlivit"]["state"] == "not_started"
        assert campaigns["rlivit"]["accelerated_execution_occurred"] is None
        assert campaigns["rlivit"]["metric_context"]["withheld_until_complete"] is True
        assert campaigns["loaf_deepstream"]["state"] == "not_started"
        assert campaigns["loaf_distance_bins"]["state"] == "not_started"
        assert campaigns["open_video_review"]["state"] == "not_started"
        assert campaigns["endurance"]["state"] == "not_started"

        assert client.get(
            "/api/validation", params={"artifact": "scene_benchmark_summary"}
        ).status_code == 413
        assert client.get(
            "/api/validation", params={"artifact": "caviar_batch_aggregate"}
        ).status_code == 404
        assert client.get(
            "/api/validation", params={"artifact": "../../etc/passwd"}
        ).status_code == 404


def test_scene_card_prefers_fail_closed_acceptance_safe_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    _write_json(
        tmp_path,
        "scene-benchmark/matrix-summary.json",
        {
            "schema_version": "deepsafe.scene-benchmark-matrix/v1",
            "expected_runs": 24,
            "status_counts": {"complete": 2, "pending": 22},
        },
    )
    _write_json(
        tmp_path,
        "campaign-report/report.json",
        {
            "schema_version": "deepsafe.validation-campaign-report/v1",
            "decision": {
                "status": "blocked_by_hardware",
                "accepted": False,
                "final_claim_allowed": False,
            },
            "requirements": [],
            "campaigns": {
                "scene_benchmark": {
                    "acceptance_safe_complete_runs": 0,
                    "operational_complete_runs": 2,
                }
            },
        },
    )

    with TestClient(app) as client:
        scene = client.get("/api/validation").json()["campaigns"][
            "scene_benchmark"
        ]

    assert scene["state"] == "planned"
    assert scene["progress"] == {
        "completed": 0,
        "total": 24,
        "remaining": 24,
        "fraction": 0.0,
    }
    assert scene["historical_source_status_counts"]["complete"] == 2
    assert scene["progress_basis"] == (
        "fail_closed_campaign_report_acceptance_safe_runs"
    )


def test_validation_rejects_allowlisted_symlink_outside_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path / "root"))
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps({"schema_version": "deepsafe.scene-benchmark-matrix/v1"}),
        encoding="utf-8",
    )
    link = tmp_path / "root/scene-benchmark/matrix-summary.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)

    with TestClient(app) as client:
        response = client.get("/api/validation")
        artifact = response.json()["campaigns"]["scene_benchmark"]["evidence"][0]
        assert artifact["artifact_state"] == "unsafe_path"
        assert client.get(
            "/api/validation", params={"artifact": "scene_benchmark_summary"}
        ).status_code == 404


def test_admin_page_contains_read_only_validation_card():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert 'id="validationGrid"' in response.text
    assert "fetch('/api/validation')" in response.text
    assert "panel test başlatmaz" in response.text
    assert "ready_for_operator_review" in response.text
    assert "raw_download_allowed" in response.text
    assert "values.ground_truth!==undefined" in response.text
    assert "values.ap_101_point_macro" in response.text
    assert "values.ap_101_point" in response.text
    assert "R-LiViT MP4 girdileri" in response.text
    assert "R-LiViT_RGB-T_v1.0" in response.text
    assert "hazır değil" in response.text
    assert "AP101@IoU0.5" in response.text
    assert "conf 0.25, IoU 0.5" in response.text
    assert "yalnız bozulma/bütünlük kontrolü" in response.text
    assert "Canlı job receipt pinleri" in response.text
    assert "Otomatik aday üretimi" in response.text
    assert "withheld görsel audit veya insan QA sayılmaz" in response.text
    assert "Hash-bağlı AI görsel audit" in response.text
    assert "AI audit insan QA mı" in response.text
    assert "Terminal insan QA" in response.text
    assert "insan tamamlanmışlığı iddia edilmiyor" in response.text


def test_person_quality_projection_is_compact_and_report_bound(tmp_path, monkeypatch):
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path,
        hardware_incident_path=None,
    )
    quality = report["campaigns"]["person_detection_quality"]
    quality.update(
        {
            "state": "passed",
            "accepted": True,
            "evidence_complete": True,
            "caviar_evidence_complete": False,
            "live_cpu_recomputed": True,
            "policy": {
                "policy_id": "owner-approved-person-v1",
                "status": "approved",
                "contract_sha256": "a" * 64,
                "task": "person_detection",
                "dataset": "CAVIAR",
                "approved_at_utc": "2026-07-15T23:00:00Z",
                "approval_strictly_before_campaign": True,
            },
            "metrics_by_profile": {
                profile: {
                    "sequences": 8,
                    "ground_truth": 800,
                    "tp": 720,
                    "fp": 80,
                    "fn": 80,
                    "micro_precision": 0.9,
                    "micro_recall": 0.9,
                    "micro_f1": 0.9,
                    "macro_ap50": 0.88,
                }
                for profile in ("640", "960")
            },
            "rules": [
                {
                    "rule_id": "micro-recall-each-profile",
                    "metric": "micro_recall",
                    "operator": "gte",
                    "threshold": 0.8,
                    "scope": {"kind": "each_profile", "profiles": [640, 960]},
                    "profile_values": {"640": 0.9, "960": 0.9},
                    "status": "pass",
                }
            ],
            "ap50_definition": "macro_mean_of_8_sequence_ap_101_point_at_iou_0.5",
            "evaluator": {
                "invoked": True,
                "execution_class": "cpu_only_read_only_inputs",
                "prewritten_decision_used": False,
                "gpu_or_docker_executed": False,
            },
            "reasons": [],
            "evidence_fingerprint_sha256": "b" * 64,
        }
    )
    report["campaigns"]["caviar_ground_truth"]["quality_outcome"] = {
        "state": "passed",
        "accepted": True,
        "separate_required_gate_id": "person_detection_quality",
    }
    quality_requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "person_detection_quality"
    )
    quality_requirement["state"] = "pass"
    report["decision"]["failed_required_gates"].remove("person_detection_quality")
    report["evidence"][0]["path"] = "/host/private/quality-owner-approval.json"
    _write_json(tmp_path, "campaign-report/report.json", report)
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/validation")
        raw = client.get(
            "/api/validation", params={"artifact": "campaign_report_json"}
        )

    assert response.status_code == 200
    projected = response.json()["campaigns"]["person_detection_quality"]
    assert projected["state"] == "passed"
    assert projected["accepted"] is True
    assert projected["live_cpu_recomputed"] is True
    assert projected["caviar_evidence_complete"] is False
    assert projected["policy"] == {
        "policy_id": "owner-approved-person-v1",
        "status": "approved",
        "contract_sha256": "a" * 64,
        "task": "person_detection",
        "dataset": "CAVIAR",
        "approved_at_utc": "2026-07-15T23:00:00Z",
        "approval_strictly_before_campaign": True,
    }
    assert projected["metrics_by_profile"]["960"]["macro_ap50"] == 0.88
    assert projected["rules"][0]["threshold"] == 0.8
    assert projected["prewritten_decision_used"] is False
    assert "/host/private" not in response.text
    assert "quality-owner-approval" not in response.text
    assert raw.status_code == 404


def test_campaign_acceptance_report_is_additive_and_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    endurance_evidence = write_admin_endurance_lineage(tmp_path)
    _write_json(
        tmp_path,
        "campaign-report/report.json",
        {
            "schema_version": "deepsafe.validation-campaign-report/v1",
            "generated_at_utc": "2026-07-16T04:00:00+00:00",
            "decision": {
                "status": "blocked_by_hardware",
                "accepted": False,
                "final_claim_allowed": False,
            },
            "requirement_summary": {
                "total": 9,
                "passed": 2,
                "state_counts": {
                    "pass": 2,
                    "blocked": 2,
                    "incomplete": 4,
                    "unproven": 1,
                },
            },
            "requirements": [
                {"id": f"gate-{index}", "state": "pass" if index < 2 else "incomplete"}
                for index in range(9)
            ],
            "evidence": list(endurance_evidence.values()),
            "campaigns": {
                "loaf_preparation": {
                    "state": "prepared_not_evaluated",
                    "acceptance_effect": "none",
                    "can_satisfy_calibrated_25m_detection": False,
                    "splits": {
                        "val": {
                            "state": "prepared_not_evaluated",
                            "target_people": 7539,
                            "frames": 2948,
                            "sequences": 8,
                            "media_status": "encoded_and_verified",
                        },
                        "test_unseen": {
                            "state": "prepared_not_evaluated",
                            "target_people": 5544,
                            "frames": 2255,
                            "sequences": 8,
                            "media_status": "planned_not_encoded",
                        },
                    },
                }
            },
            "private_host_path": "/host/private/must-not-project",
            "private_command": ["nvidia-smi", "--query-gpu=uuid"],
        },
    )
    markdown = tmp_path / "campaign-report/report.md"
    markdown.write_text(
        "# report\n\n/private/operator-only.md\n`nvidia-smi --query-gpu=uuid`\n",
        encoding="utf-8",
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        assert response.status_code == 200
        projected = response.json()["campaigns"]["campaign_report"]
        assert projected["state"] == "blocked_by_hardware"
        assert projected["accepted"] is False
        assert projected["final_claim_allowed"] is False
        assert projected["progress"] == {
            "completed": 2,
            "total": 9,
            "remaining": 7,
            "fraction": 0.222222,
        }
        assert projected["loaf_preparation"] == {
            "state": "prepared_not_evaluated",
            "acceptance_effect": "none",
            "can_satisfy_calibrated_25m_detection": False,
            "splits": {
                "val": {
                    "state": "prepared_not_evaluated",
                    "target_people": 7539,
                    "frames": 2948,
                    "sequences": 8,
                    "media_status": "encoded_and_verified",
                },
                "test_unseen": {
                    "state": "prepared_not_evaluated",
                    "target_people": 5544,
                    "frames": 2255,
                    "sequences": 8,
                    "media_status": "planned_not_encoded",
                },
            },
        }
        assert "/host/private" not in response.text
        assert "/private/operator-only.md" not in response.text
        assert "nvidia-smi" not in response.text
        assert "--query-gpu" not in response.text
        for evidence in projected["evidence"]:
            assert evidence["available"] is True
            assert evidence["raw_download_allowed"] is False
            assert "href" not in evidence

        raw_json = client.get(
            "/api/validation", params={"artifact": "campaign_report_json"}
        )
        raw_markdown = client.get(
            "/api/validation", params={"artifact": "campaign_report_markdown"}
        )
        assert raw_json.status_code == 404
        assert raw_markdown.status_code == 404
        assert "/host/private" not in raw_json.text
        assert "/private/operator-only.md" not in raw_markdown.text
        assert "nvidia-smi" not in raw_json.text + raw_markdown.text
        assert client.post(
            "/api/validation", params={"artifact": "campaign_report_json"}
        ).status_code == 405


def test_metadata_only_campaign_projection_cannot_make_an_unbound_acceptance_claim(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path,
        "campaign-report/report.json",
        {
            "schema_version": "deepsafe.validation-campaign-report/v1",
            "generated_at_utc": "2026-07-16T04:00:00+00:00",
            "decision": {
                "status": "accepted",
                "accepted": True,
                "final_claim_allowed": True,
            },
            "requirement_summary": {"state_counts": {}},
            "requirements": [],
            "campaigns": {
                "loaf_preparation": {
                    "state": "prepared_not_evaluated",
                    "acceptance_effect": "none",
                    "can_satisfy_calibrated_25m_detection": False,
                    "splits": {},
                }
            },
            "evidence": [],
        },
    )

    projected = admin_validation._campaign_report(
        admin_validation.ArtifactReader(tmp_path)
    )

    assert projected["available"] is True
    assert projected["artifact_state"] == "ok"
    assert projected["state"] == "artifact_error"
    assert projected["reason"] == "unbound_acceptance_claim"
    assert projected["accepted"] is False
    assert projected["final_claim_allowed"] is False
    assert projected["loaf_preparation"]["state"] == "prepared_not_evaluated"


def test_admin_rlivit_metric_row_validates_runner_diagnostics():
    row = {
        **_rlivit_metric(10),
        "evaluated_predictions": 10,
        "ignored_predictions": 2,
        "serialized_predictions_at_or_above_confidence": 12,
        "ap_serialized_predictions": 20,
        "ap_ignored_predictions": 5,
    }
    assert admin_validation._rlivit_metric_row(row) == row

    invalid = copy.deepcopy(row)
    invalid["ignored_predictions"] = 3
    assert admin_validation._rlivit_metric_row(invalid) is None

    partial = copy.deepcopy(row)
    partial.pop("ap_ignored_predictions")
    assert admin_validation._rlivit_metric_row(partial) is None


def _distance_v2_production_projection() -> dict:
    return {
        "workspace_scoped_inputs": True,
        "fixture_markers_absent": True,
        "minimum_source_bytes": campaign.DISTANCE_V2_MINIMUM_SOURCE_BYTES,
        "minimum_per_bin_instances": campaign.DISTANCE_V2_MINIMUM_BIN_INSTANCES,
        "minimum_per_bin_independent_events": campaign.DISTANCE_V2_MINIMUM_BIN_EVENTS,
        "minimum_per_bin_unambiguous_events": (
            campaign.DISTANCE_V2_MINIMUM_UNAMBIGUOUS_EVENTS
        ),
        "minimum_endpoint_independent_events": (
            campaign.DISTANCE_V2_MINIMUM_ENDPOINT_EVENTS
        ),
        "minimum_exact_25m_instances": (
            campaign.DISTANCE_V2_MINIMUM_EXACT_25_INSTANCES
        ),
    }


def test_admin_withholds_live_verified_synthetic_v2_receipt(
    tmp_path, monkeypatch
):
    _write_verified_synthetic_distance_v2_final(tmp_path)
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_ROOT", str(tmp_path / "validation/results")
    )
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT",
        str(campaign.PROJECT_ROOT / "validation/schemas"),
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        raw = client.get(
            "/api/validation",
            params={"artifact": "site_distance_evaluation_v2"},
        )

    assert response.status_code == 200
    projected = response.json()["campaigns"]["site_distance_25m"]
    assert projected["state"] == "unproven"
    assert projected["accepted"] is False
    assert projected["verification"]["live_semantic_replay"] is True
    assert projected["verification"]["production_evidence_contract"] is False
    assert projected["profiles"] == {}
    assert projected["distance_bins"] == []
    assert "project-owner" not in response.text
    assert "approved_by" not in response.text
    assert str(tmp_path) not in response.text
    assert raw.status_code == 404


def test_admin_v2_projection_requires_live_replay_and_report_hash_binding(
    tmp_path, monkeypatch
):
    final, documents = _write_verified_synthetic_distance_v2_final(tmp_path)
    production = _distance_v2_production_projection()
    monkeypatch.setattr(
        campaign,
        "_distance_v2_production_contract",
        lambda payload, root: (True, production, documents),
    )
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )
    report_path = tmp_path / "validation/results/campaign-report/report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        admin_validation,
        "_site_distance_v2_production_contract",
        lambda reader, payload: (True, production, documents),
    )
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_ROOT", str(tmp_path / "validation/results")
    )
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT",
        str(campaign.PROJECT_ROOT / "validation/schemas"),
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")

    assert response.status_code == 200
    projected = response.json()["campaigns"]["site_distance_25m"]
    assert projected["state"] == "proven"
    assert projected["accepted"] is True
    assert projected["scope"]["distance_m"] == {
        "minimum_inclusive": 20,
        "maximum_inclusive": 25,
    }
    assert projected["exact_25m_instances"] == 1
    assert [row["bin_id"] for row in projected["distance_bins"]] == list(
        admin_validation.SITE_DISTANCE_V2_BIN_IDS
    )
    assert set(projected["profiles"]) == {"640", "960"}
    assert projected["verification"] == {
        "live_semantic_replay": True,
        "production_evidence_contract": True,
        "campaign_report_hash_binding": True,
        "failure_codes": [],
    }
    assert final["receipt_sha256"] not in response.text
    assert "project-owner" not in response.text
    assert "approved_by" not in response.text
    assert str(tmp_path) not in response.text


def test_admin_v2_and_legacy_conflict_never_projects_metrics(
    tmp_path, monkeypatch
):
    _write_verified_synthetic_distance_v2_final(tmp_path)
    _write_json(
        tmp_path,
        "validation/results/distance-25m/evaluation.json",
        {"schema_version": "deepsafe.distance-validation/v1", "status": "complete"},
    )
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_ROOT", str(tmp_path / "validation/results")
    )
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT",
        str(campaign.PROJECT_ROOT / "validation/schemas"),
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")

    projected = response.json()["campaigns"]["site_distance_25m"]
    assert projected["state"] == "unproven"
    assert projected["selection"] == {
        "legacy_v1_present": True,
        "inclusive_v2_final_candidates": 1,
        "conflict": True,
    }
    assert projected["profiles"] == {}
    assert projected["distance_bins"] == []

def test_admin_rlivit_mp4_temperature_preserves_legacy_strict_boundary():
    batch_pin = {"size_bytes": 15423, "sha256": "a" * 64}
    managed = _rlivit_mp4_public_receipt(batch_pin)
    managed["maximum_cpu_platform_temperature_millidegrees_celsius"] = 100000
    managed = _with_fingerprint(
        {key: value for key, value in managed.items() if key != "fingerprint_sha256"}
    )
    assert admin_validation._rlivit_mp4_contract(managed)[
        "thermal_policy_id"
    ] == "workstation_managed"

    strict = copy.deepcopy(managed)
    strict["thermal_policy_id"] = "legacy_strict"
    strict = _with_fingerprint(
        {key: value for key, value in strict.items() if key != "fingerprint_sha256"}
    )
    assert admin_validation._rlivit_mp4_contract(strict) is None

    legacy = copy.deepcopy(managed)
    legacy.pop("thermal_policy_id")
    legacy["maximum_cpu_platform_temperature_millidegrees_celsius"] = 84999
    legacy = _with_fingerprint(
        {key: value for key, value in legacy.items() if key != "fingerprint_sha256"}
    )
    assert admin_validation._rlivit_mp4_contract(legacy)[
        "thermal_policy_id"
    ] == "legacy_strict"
    legacy["maximum_cpu_platform_temperature_millidegrees_celsius"] = 85000
    legacy = _with_fingerprint(
        {key: value for key, value in legacy.items() if key != "fingerprint_sha256"}
    )
    assert admin_validation._rlivit_mp4_contract(legacy) is None
