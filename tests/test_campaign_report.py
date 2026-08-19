import csv
import copy
import hashlib
import json
import shutil
import zlib
from pathlib import Path
from unittest import mock

import pytest

from evaluation.model import FrameKey
from validation import report_campaign as campaign
from validation import site_distance_evaluation as site_distance
from validation import site_distance_evaluator_v2 as site_distance_v2
from validation.endurance import supervisor as endurance_supervisor
from validation.scene_benchmark import run_matrix as scene_run_matrix
from tests.test_site_distance_evaluator_v2 import (
    _add_exact_25m_and_unambiguous_endpoint,
    _write_gt_predictions,
)
from tests.test_site_distance_readiness_v2 import (
    _pair as _distance_v2_pair,
    _preflight as _distance_v2_preflight,
    _profile_pair_fixture as _distance_v2_profile_pair_fixture,
    _write_json as _distance_v2_write_json,
)


_RLIVIT_FIXTURE_PNG_CACHE: dict[tuple[int, int], bytes] = {}


def _fixture_rgb_png(width: int, height: int) -> bytes:
    """Return a small-on-disk, fully decodable 8-bit RGB PNG fixture."""

    cached = _RLIVIT_FIXTURE_PNG_CACHE.get((width, height))
    if cached is not None:
        return cached

    def chunk(kind: bytes, value: bytes) -> bytes:
        crc = zlib.crc32(kind)
        crc = zlib.crc32(value, crc) & 0xFFFFFFFF
        return (
            len(value).to_bytes(4, "big")
            + kind
            + value
            + crc.to_bytes(4, "big")
        )

    header = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )
    scanline = b"\x00" + (b"\x00\x00\x00" * width)
    content = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanline * height, level=9))
        + chunk(b"IEND", b"")
    )
    _RLIVIT_FIXTURE_PNG_CACHE[(width, height)] = content
    return content


def _write_json(root: Path, relative: str, value: dict) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _populate_open_video_evidence_layers(
    root: Path,
    *,
    plan_path: Path,
    reviewed_scene_ids: list[str],
) -> None:
    """Create a small but fully hash-bound 21-frame/42-decision fixture."""

    source_records_path = (
        root / "validation/open_video_review/source-frame-reviews-v1.jsonl"
    )
    source_records_path.parent.mkdir(parents=True, exist_ok=True)

    def source_observation(index: int) -> dict:
        return {
            "visible_person_count_range": {"min": 0, "max": 0},
            "scorable_person_count_range": {"min": 0, "max": 0},
            "medium_close": index == 0,
            "top_view": index == 1,
            "high_oblique": index == 2,
        }

    source_records = [
        {
            "schema_version": "deepsafe.open-video-source-frame-review/v1",
            "record_id": f"{reviewed_scene_ids[index % len(reviewed_scene_ids)]}.f{index:06d}",
            "scene_id": reviewed_scene_ids[index % len(reviewed_scene_ids)],
            "sensitive": False,
            "frame": {"index": index},
            "observation": source_observation(index),
        }
        for index in range(campaign.EXPECTED_OPEN_AI_SOURCE_REVIEWS)
    ]
    source_records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in source_records),
        encoding="utf-8",
    )
    source_records_sha256 = hashlib.sha256(
        source_records_path.read_bytes()
    ).hexdigest()
    campaign_plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    bundle_id = "b_" + hashlib.sha256(
        (campaign_plan_sha256 + source_records_sha256).encode("ascii")
    ).hexdigest()

    decisions: list[dict] = []
    assets: dict[str, dict] = {}
    results_root = root / "validation/results"
    for index, source_record in enumerate(source_records):
        source_review_id = source_record["record_id"]
        scene_id = source_record["scene_id"]
        observation = source_record["observation"]
        for profile in campaign.PROFILES:
            decision_id = f"{source_review_id}:{profile}"
            evidence: dict[str, str] = {}
            for kind, (media_type, extension) in campaign.OPEN_MANUAL_ASSET_KINDS.items():
                asset_id = "a_" + hashlib.sha256(
                    f"{decision_id}:{kind}".encode("utf-8")
                ).hexdigest()
                relative_path = (
                    f"open-video-review/manual-assets/objects/{bundle_id}/"
                    f"{asset_id}.{extension}"
                )
                asset_path = results_root / relative_path
                asset_path.parent.mkdir(parents=True, exist_ok=True)
                if kind == "predictions":
                    content = b"\n"
                elif extension == "png":
                    content = b"\x89PNG\r\n\x1a\n" + f"{decision_id}:{kind}".encode()
                else:
                    content = json.dumps(
                        {"decision_id": decision_id, "kind": kind},
                        sort_keys=True,
                    ).encode("utf-8")
                asset_path.write_bytes(content)
                assets[asset_id] = {
                    "decision_id": decision_id,
                    "source_review_id": source_review_id,
                    "scene_id": scene_id,
                    "frame_index": index,
                    "model_input": profile,
                    "kind": kind,
                    "relative_path": relative_path,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "media_type": media_type,
                    "size_bytes": len(content),
                }
                evidence[kind] = asset_id
            decisions.append(
                {
                    "decision_id": decision_id,
                    "source_review_id": source_review_id,
                    "scene_id": scene_id,
                    "frame_index": index,
                    "model_input": profile,
                    "source_observation": observation,
                    "evidence": evidence,
                }
            )

    provenance_jobs = []
    for scene_id in reviewed_scene_ids:
        for profile in campaign.PROFILES:
            job_id = f"{scene_id}:{profile}"
            provenance_jobs.append(
                {
                    "job_id": job_id,
                    "scene_id": scene_id,
                    "model_input": profile,
                    **{
                        key: hashlib.sha256(f"{job_id}:{key}".encode()).hexdigest()
                        for key in (
                            "predictions_sha256",
                            "gpu_guard_sha256",
                            "run_manifest_sha256",
                            "conversion_sha256",
                        )
                    },
                }
            )
    index_path = _write_json(
        root,
        "validation/results/open-video-review/manual-assets/index.json",
        {
            "schema_version": campaign.OPEN_MANUAL_ASSET_SCHEMA,
            "status": "complete",
            "bundle_id": bundle_id,
            "decision_count": campaign.EXPECTED_OPEN_MANUAL_DECISIONS,
            "asset_count": campaign.EXPECTED_OPEN_MANUAL_ASSETS,
            "profiles": list(campaign.PROFILES),
            "review_confidence_threshold": 0.25,
            "sensitive_media_included": False,
            "source_records_sha256": source_records_sha256,
            "campaign_plan_sha256": campaign_plan_sha256,
            "decisions": decisions,
            "assets": assets,
            "input_provenance": {
                "videos": {
                    scene_id: {
                        "sha256": hashlib.sha256(scene_id.encode()).hexdigest(),
                        "size_bytes": 1,
                    }
                    for scene_id in reviewed_scene_ids
                },
                "jobs": provenance_jobs,
            },
            "metric_guardrail": "Sparse qualitative candidates are not ground truth or accuracy metrics.",
        },
    )
    index_content = index_path.read_bytes()
    index_value = json.loads(index_content)
    audit_reviews = []
    for source_record in source_records:
        source_review_id = source_record["record_id"]
        audit_reviews.append(
            {
                "source_review_id": source_review_id,
                "decision_ids": [
                    f"{source_review_id}:{profile}" for profile in campaign.PROFILES
                ],
                "estimated_visible_persons": {
                    "min": 0,
                    "max": 0,
                    "qualifier": "fixture",
                },
                "estimated_scorable_persons": {
                    "min": 0,
                    "max": 0,
                    "qualifier": "fixture",
                },
                "observed_box_count": {"640": 0, "960": 0},
                "severity": "none",
                "issues": [],
                "preferred_profile": "tie",
                "preference_strength": "none",
                "finding": "Fixture pair reviewed without a metric claim.",
                "human_followup_recommended": False,
            }
        )
    _write_json(
        root,
        "validation/results/open-video-review/ai-qualitative-audit.json",
        {
            "schema_version": campaign.OPEN_AI_QUALITATIVE_AUDIT_SCHEMA,
            "audit_id": "fixture-open-video-ai-audit",
            "created_at_utc": "2026-07-16T02:00:00+00:00",
            "status": "complete",
            "task": "gt_free_pairwise_person_detection_visual_audit",
            "ground_truth_available": False,
            "reviewer": {
                "reviewer_type": "AI",
                "reviewer_id": "fixture-ai",
                "visual_tool": "fixture-visual-tool",
            },
            "input_bindings": {
                "manual_assets_index": {
                    "path": "validation/results/open-video-review/manual-assets/index.json",
                    "size_bytes": len(index_content),
                    "sha256": hashlib.sha256(index_content).hexdigest(),
                    "schema_version": campaign.OPEN_MANUAL_ASSET_SCHEMA,
                    "bundle_id": index_value["bundle_id"],
                },
                "campaign_plan_sha256": campaign_plan_sha256,
                "source_records_sha256": source_records_sha256,
            },
            "scope": {
                "source_review_count": campaign.EXPECTED_OPEN_AI_SOURCE_REVIEWS,
                "profile_decision_count": campaign.EXPECTED_OPEN_MANUAL_DECISIONS,
                "profiles": list(campaign.PROFILES),
                "rendered_prediction_confidence_threshold": 0.25,
                "selection": "sparse_preselected_manual_review_frames",
            },
            "guardrail": {
                "allowed_use": ["qualitative_triage", "failure_mode_discovery"],
                "prohibited_interpretations": [
                    "accuracy_metric",
                    "dataset_level_profile_ranking",
                    "calibration_claim",
                    "production_acceptance",
                ],
                "canonical_text": "AI qualitative evidence is not dense ground truth.",
            },
            "methodology": {
                "source_first_review": True,
                "pairwise_profiles": list(campaign.PROFILES),
                "same_frame_source_required": True,
                "dense_annotation_performed": False,
                "counting_policy": "physical_person_instances",
                "reflections_screens_and_photos": "recorded_as_ambiguity_not_physical_person",
                "profile_preference_scope": "single_review_frame_only",
            },
            "reviews": audit_reviews,
            "qualitative_synthesis": {
                "recurring_patterns": ["fixture qualitative pattern"],
                "limitations": ["fixture is sparse and not ground truth"],
                "preference_tallies_intentionally_omitted": True,
            },
        },
    )

    human_rows = []
    for decision in decisions:
        human_rows.append(
            {
                "schema_version": campaign.OPEN_OVERLAY_DECISION_SCHEMA,
                **{
                    key: decision[key]
                    for key in (
                        "decision_id",
                        "source_review_id",
                        "scene_id",
                        "frame_index",
                        "model_input",
                    )
                },
                "review_visibility": "project_internal",
                "overlay_evidence": {
                    "review_report_path": None,
                    "review_report_sha256": None,
                    "overlay_image_path": None,
                    "overlay_image_sha256": None,
                    "predictions_path": None,
                    "predictions_sha256": None,
                },
                "decision": {
                    "status": "pending_review",
                    "detection_count_reviewed": None,
                    "visible_person_count_confirmed": None,
                    "scorable_person_count_confirmed": None,
                    "true_positive_count": None,
                    "false_positive_count": None,
                    "false_negative_count": None,
                    "ignored_detection_count": None,
                    "unscorable_visible_person_count": None,
                    "reasons": ["Awaiting terminal human QA."],
                },
                "review": {
                    "reviewer_id": None,
                    "reviewer_type": None,
                    "reviewed_at": None,
                },
                "metric_guardrail": "Human sampled-frame QA is not dense ground truth.",
            }
        )
    human_path = root / "validation/results/open-video-review/overlay-decisions-v1.jsonl"
    human_path.parent.mkdir(parents=True, exist_ok=True)
    human_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in human_rows),
        encoding="utf-8",
    )


def _copy_checked_in_ppe_video_source_registry(root: Path) -> Path:
    source = (
        Path(campaign.__file__).resolve().parents[1]
        / "data/manifests/ppe-video-source-candidates.json"
    )
    target = root / "data/manifests/ppe-video-source-candidates.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _pin(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _content_pin(root: Path, path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _throughput_source_pin(root: Path, path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _fixture_throughput_floor_verifier(
    artifact_path: Path,
    *,
    summary_path: Path,
    scene_manifest_path: Path,
    project_root: Path,
) -> dict:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert summary_path.is_file() and scene_manifest_path.is_file()
    assert artifact_path.resolve().is_relative_to(project_root.resolve())
    fingerprint = artifact["fingerprint"]
    return {
        "schema_version": "deepsafe.endurance-throughput-floor-verification/v1",
        "generated_at_utc": "2026-07-16T00:00:00Z",
        "status": "verified",
        "artifact_fingerprint": fingerprint,
        "rederived_fingerprint": fingerprint,
        "verified_safe_runs": 24,
        "profiles": {
            profile: {
                "aggregate_fps_floor": artifact["profiles"][profile][
                    "aggregate_fps_floor"
                ],
                "per_stream_fps_floor": artifact["profiles"][profile][
                    "per_stream_fps_floor"
                ],
            }
            for profile in ("640", "960")
        },
        "live_rederived": True,
        "gpu_or_docker_executed_by_verifier": False,
    }


def _fixture_endurance_raw_replay(
    attempt_dir: Path,
    *,
    duration_seconds: int,
    perf_interval_seconds: float,
    max_log_bytes: int,
    minimum_coverage_fraction: float,
    startup_grace_seconds: float,
    perf_stall_timeout_seconds: float,
) -> dict:
    """Small fixture adapter; production reporter never installs this hook."""

    del (
        max_log_bytes,
        minimum_coverage_fraction,
        startup_grace_seconds,
        perf_stall_timeout_seconds,
    )
    status = json.loads((attempt_dir / "status.json").read_text(encoding="utf-8"))
    pins = status["artifact_pins"]
    requested = int(duration_seconds / perf_interval_seconds)
    return {
        "schema_version": "deepsafe.endurance-raw-attempt-replay/v1",
        "status": "verified",
        "throughput": status["throughput"],
        "gpu": status["gpu"],
        "perf_csv": {
            "header": [
                "elapsed_seconds",
                "aggregate_fps",
                "per_stream_mean_fps",
            ],
            "rows": requested,
            "requested_window_rows": requested,
            "aggregate_current_fps": status["throughput"][
                "aggregate_current_fps"
            ],
        },
        "file_observations": {
            name: {
                "size_bytes": pins[name]["size_bytes"],
                "sha256": pins[name]["sha256"],
            }
            for name in ("log", "perf_csv", "gpu_csv")
        },
    }


def _write_accepted_site_distance(root: Path) -> None:
    base = root / "validation/inputs/distance-25m"
    source = base / "source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"campaign-report-site-source")
    calibration_doc = base / "calibration-verification.md"
    annotation_doc = base / "annotation-qa.md"
    acceptance_doc = base / "acceptance-approval.md"
    calibration_doc.write_text("survey pass\n", encoding="utf-8")
    annotation_doc.write_text("all visible people reviewed\n", encoding="utf-8")
    acceptance_doc.write_text("owner approved criterion\n", encoding="utf-8")

    calibration = _write_json(
        root,
        "validation/inputs/distance-25m/calibration.json",
        {
            "schema_version": "deepsafe.site-ground-plane-calibration/v1",
            "status": "verified",
            "calibration_id": "cal-accepted",
            "site_id": "site-accepted",
            "camera_id": "cam-accepted",
            "camera_configuration_id": "cam-accepted-fixed-v1",
            "model": "planar_homography_image_to_ground",
            "distance_unit": "m",
            "image": {"width": 40, "height": 30},
            "image_to_ground_homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "camera_ground_position_m": [20, 0],
            "valid_image_polygon_px": [[0, 0], [40, 0], [40, 30], [0, 30]],
            "verification": {
                "status": "pass",
                "method": "survey_control_points",
                "verified_by": "fixture-surveyor",
                "verified_at": "2026-07-15T10:00:00Z",
                "reference_point_count": 8,
                "maximum_ground_error_m": 0.05,
                "allowed_ground_error_m": 0.10,
                "document": _pin(calibration_doc),
            },
        },
    )
    ground_truth = _write_json(
        root,
        "validation/inputs/distance-25m/ground-truth.json",
        {
            "schema_version": "deepsafe.site-distance-ground-truth/v1",
            "status": "complete",
            "evidence_kind": "deployment_site_calibrated_person_ground_truth",
            "dataset_id": "site-accepted-gt",
            "site_id": "site-accepted",
            "camera_id": "cam-accepted",
            "camera_configuration_id": "cam-accepted-fixed-v1",
            "calibration_id": "cal-accepted",
            "sequence_id": "site-accepted-sequence",
            "source_asset": _pin(source),
            "image": {"width": 40, "height": 30},
            "annotation": {
                "status": "verified",
                "all_visible_people_in_calibrated_roi_annotated": True,
                "box_format": "pixel_xywh",
                "distance_point": "bbox_bottom_center_ground_contact",
                "reviewed_by": "fixture-annotator",
                "reviewed_at": "2026-07-15T11:00:00Z",
                "document": _pin(annotation_doc),
            },
            "frames": [
                {
                    "frame_index": 0,
                    "persons": [
                        {
                            "object_id": "person-0",
                            "bbox_pixel_xywh": [18, 10, 4, 12],
                            "ignored": False,
                        }
                    ],
                }
            ],
        },
    )
    acceptance = _write_json(
        root,
        "validation/inputs/distance-25m/acceptance.json",
        {
            "schema_version": "deepsafe.site-distance-acceptance/v1",
            "status": "approved",
            "criterion_id": "owner-distance-gate-accepted",
            "task": "person_detection",
            "evidence_kind": "deployment_site_calibrated_ground_plane",
            "distance_unit": "m",
            "distance_bin_m": [20, 25],
            "boundary": "lower_inclusive_upper_exclusive",
            "profiles": [640, 960],
            "evaluation_config": {
                "iou_threshold": 0.5,
                "confidence_threshold": 0.25,
                "distance_point": "bbox_bottom_center_ground_contact",
            },
            "minimum_ground_truth_instances_per_profile": 1,
            "rules": [
                {
                    "metric": "recall",
                    "operator": "gte",
                    "threshold": 0.5,
                    "applies_to": "each_profile",
                }
            ],
            "approval": {
                "approved_by": "fixture-site-owner",
                "approved_at": "2026-07-15T12:00:00Z",
                "document": _pin(acceptance_doc),
            },
        },
    )

    manifests: dict[int, Path] = {}
    frame_hash = site_distance._frame_key_sha256(
        {FrameKey("site-accepted-sequence", 0)}
    )
    for profile in (640, 960):
        prediction_path = base / f"profiles/{profile}/predictions.jsonl"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(
            json.dumps(
                {
                    "schema_version": "deepsafe.person-detections/v1",
                    "sequence_id": "site-accepted-sequence",
                    "frame_index": 0,
                    "image_width": 40,
                    "image_height": 30,
                    "source_uri": "file:///site/accepted.mp4",
                    "model_id": f"site-person-{profile}",
                    "detections": [
                        {
                            "class_name": "person",
                            "confidence": 0.9,
                            "bbox_norm_xywh": [18 / 40, 10 / 30, 4 / 40, 12 / 30],
                        }
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifests[profile] = _write_json(
            root,
            f"validation/inputs/distance-25m/profiles/{profile}/run-manifest.json",
            {
                "schema_version": "deepsafe.site-distance-profile-run/v1",
                "status": "complete",
                "evidence_kind": "deployment_site_profile_inference",
                "profile": profile,
                "dataset_id": "site-accepted-gt",
                "site_id": "site-accepted",
                "camera_id": "cam-accepted",
                "camera_configuration_id": "cam-accepted-fixed-v1",
                "calibration_id": "cal-accepted",
                "sequence_id": "site-accepted-sequence",
                "source_asset_sha256": _pin(source)["sha256"],
                "ground_truth": _pin(ground_truth),
                "calibration": _pin(calibration),
                "predictions": _pin(prediction_path),
                "frame_contract": {
                    "status": "exact",
                    "expected_frames": 1,
                    "serialized_frames": 1,
                    "frame_key_sha256": frame_hash,
                },
                "inference": {
                    "exit_code": 0,
                    "safety_guard_status": "pass",
                    "model_id": f"site-person-{profile}",
                    "model_sha256": "a" * 64,
                    "config_sha256": "b" * 64,
                    "completed_at": "2026-07-15T13:00:00Z",
                },
            },
        )

    result, attempt = site_distance.evaluate_site_distance(
        calibration_path=calibration,
        ground_truth_path=ground_truth,
        acceptance_path=acceptance,
        profile_640_manifest=manifests[640],
        profile_960_manifest=manifests[960],
    )
    assert result is not None and attempt["status"] == "accepted"
    _write_json(root, "validation/results/distance-25m/evaluation.json", result)


def _prepare_accepted_site_distance(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Create a self-contained result pinned to a project-local evaluator."""

    implementation = root / "validation/site_distance_evaluation.py"
    implementation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(site_distance.__file__), implementation)
    monkeypatch.setattr(site_distance, "__file__", str(implementation))
    _write_accepted_site_distance(root)


def _scene_status(scene: str, profile: int, fingerprint: str) -> dict:
    return {
        "schema_version": "deepsafe.scene-benchmark-run/v2",
        "status": "complete",
        "fingerprint": fingerprint,
        "scene": {"id": scene},
        "model": {"size": profile},
        "simulation": {
            "streams": 12,
            "requested_duration_seconds": 300,
        },
        "timing": {
            "measurement_complete": True,
            "measurement_wall_time_seconds": 300.1,
        },
        "throughput": {
            "status": "ok",
            "active_streams": 12,
            "per_stream_current_fps_analyzed_samples": {
                "mean": 48.0 if profile == 640 else 28.0
            },
            "aggregate_current_fps": {
                "mean": 576.0 if profile == 640 else 336.0,
                "p05": 570.0 if profile == 640 else 330.0,
                "p95": 582.0 if profile == 640 else 342.0,
            },
        },
        "gpu": {
            "status": "ok",
            "metrics": {
                "gpu_utilization_percent": {"mean": 70.0},
                "memory_used_mib": {"mean": 1500.0},
            },
        },
        "failure_reasons": [],
    }


def _populate_accepted_endurance(root: Path) -> None:
    current = root / "validation/results/endurance/current"
    power_limit_fields = [
        "power_requested_limit_w",
        "power_current_limit_w",
        "power_default_limit_w",
    ]
    diagnostic_slowdown_flags = [
        "clock_event_sw_thermal_slowdown",
        "clock_event_hw_slowdown",
        "clock_event_hw_thermal_slowdown",
        "clock_event_hw_power_brake_slowdown",
    ]
    power_policy = {
        "operating_policy_mode": "workstation_managed",
        "hardware_protection_owner": "workstation_bios_ec_nvidia_driver",
        "static_signal_action": "record_measurement_quality_diagnostic",
        "power_limit_drop_tolerance_w": 5.0,
        "slowdown_consecutive_samples": 2,
        "preflight_samples": 2,
        "preflight_sample_interval_seconds": 1.0,
        "power_limit_fields": power_limit_fields,
        "diagnostic_slowdown_flags": diagnostic_slowdown_flags,
        "clock_event_telemetry_fields": [
            "pstate",
            "clock_event_reasons_active_mask",
            "clock_event_sw_power_cap",
            *diagnostic_slowdown_flags,
        ],
        "abort_slowdown_flags": [],
        "required_telemetry_failure_action": "safety_abort",
        "power_limit_telemetry_required": True,
        "power_limit_drop_action": "record_measurement_quality_diagnostic",
        "sustained_slowdown_action": "record_measurement_quality_diagnostic",
        "sw_power_cap_semantics": "record_only_unless_power_limit_below_default",
    }
    gpu_identity = {
        "index": "0",
        "uuid": "GPU-fixture-uuid",
        "name": "NVIDIA fixture production GPU",
        "driver_version": "fixture-driver",
        "memory.total": "16384 MiB",
        "pci.bus_id": "00000000:01:00.0",
    }
    floor_runtime_identity = {
        "image": "deepsafe-deepstream:9.0",
        "image_id": "sha256:fixture-resolved-image",
        "gpu_index": 0,
        "gpu_identity": gpu_identity,
        "power_profile": {"available": True, "value": "performance"},
        "max_temperature_c": 86.0,
        "power_safety_policy": {
            "operating_policy_mode": "workstation_managed",
            "hardware_protection_owner": "workstation_bios_ec_nvidia_driver",
            "static_signal_action": "record_measurement_quality_diagnostic",
            "power_limit_drop_tolerance_w": 5.0,
            "slowdown_consecutive_samples": 2,
            "preflight_samples": 2,
            "preflight_sample_interval_seconds": 1.0,
            "power_limit_fields": power_limit_fields,
            "diagnostic_slowdown_flags": diagnostic_slowdown_flags,
            "abort_slowdown_flags": [],
            "required_telemetry_failure_action": "safety_abort",
        },
    }
    source_media = []
    sources = []
    for index in range(12):
        relative = f"data/fixture/endurance-scene-{index:02d}.mp4"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-h264-camera-{index:02d}".encode())
        pin = _content_pin(root, path)
        metadata = {
            "codec": "h264",
            "pixel_format": "yuv420p",
            "width": 1920,
            "height": 1080,
            "fps_fraction": "30/1",
            "duration_seconds": 30.0,
        }
        scene_id = f"endurance-scene-{index:02d}"
        sources.append(
            {
                "camera_id": index,
                "scene_id": scene_id,
                "benchmark_type": f"endurance-camera-type-{index:02d}",
                "video_path": relative,
                "video_size_bytes": pin["size_bytes"],
                "video_sha256": pin["sha256"],
                **metadata,
            }
        )
        source_media.append(
            {
                "camera_id": index,
                "scene_id": scene_id,
                **pin,
                **metadata,
            }
        )
    model_profiles = {}
    for profile in (640, 960):
        infer = root / f"models/person/{profile}/config_infer_primary.txt"
        engine = root / f"models/person/{profile}/fixture.engine"
        infer.parent.mkdir(parents=True, exist_ok=True)
        infer.write_text(
            f"model-engine-file=/models/person/{profile}/fixture.engine\n",
            encoding="utf-8",
        )
        engine.write_bytes(f"fixture-tensorrt-engine-{profile}".encode())
        infer_pin = _content_pin(root, infer)
        engine_pin = _content_pin(root, engine)
        model_profiles[str(profile)] = {
            "size": profile,
            "infer_config": infer_pin["path"],
            "engine": engine_pin["path"],
            "infer_config_sha256": infer_pin["sha256"],
            "engine_size_bytes": engine_pin["size_bytes"],
            "engine_sha256": engine_pin["sha256"],
            "person_only_classes": [0],
        }
    control_relatives = [
        "validation/endurance/campaign.json",
        "validation/scene_benchmark/scenes.json",
        "validation/scene_benchmark/source-catalog.json",
        "validation/endurance/supervisor.py",
        "validation/scene_benchmark/run_matrix.py",
        "validation/gpu_reentry_evidence.py",
        "benchmark/summarize.py",
        "deepstream/Dockerfile",
    ]
    control_pins = []
    for relative in control_relatives:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "benchmark/summarize.py":
            path.write_bytes(Path(campaign.benchmark_summarize.__file__).read_bytes())
        else:
            path.write_text(f"fixture control: {relative}\n", encoding="utf-8")
        control_pins.append(_content_pin(root, path))
    input_pins = {
        "schema_version": "deepsafe.endurance-input-pins/v1",
        "source_media": source_media,
        "model_profiles": model_profiles,
        "control_files": control_pins,
    }
    summary_path = root / "validation/results/scene-benchmark/matrix-summary.json"
    if not summary_path.is_file():
        _write_json(
            root,
            "validation/results/scene-benchmark/matrix-summary.json",
            {"schema_version": "fixture.scene-matrix/v1", "safe_runs": 24},
        )
    scene_manifest_path = root / "validation/scene_benchmark/scenes.json"
    floor_profiles = {
        "640": {"aggregate_fps_floor": 450.0, "per_stream_fps_floor": 37.5},
        "960": {"aggregate_fps_floor": 260.0, "per_stream_fps_floor": 21.6},
    }
    floor_inputs = {
        "summary": _throughput_source_pin(root, summary_path),
        "scene_manifest": _throughput_source_pin(root, scene_manifest_path),
        "runtime_identity": floor_runtime_identity,
    }
    floor_projection = {
        "fixture_contract": "24-safe-v2-runs",
        "inputs": floor_inputs,
        "profiles": floor_profiles,
    }
    floor_artifact = {
        "schema_version": "deepsafe.endurance-throughput-floor/v1",
        "status": "frozen",
        "inputs": floor_inputs,
        "profiles": floor_profiles,
        "fingerprint_input": floor_projection,
        "fingerprint": campaign._endurance_json_sha256(floor_projection),
    }
    floor_path = _write_json(
        root,
        "validation/results/endurance/throughput-floor.json",
        floor_artifact,
    )
    throughput_floor_binding = {
        "schema_version": "deepsafe.endurance-throughput-floor-binding/v1",
        "status": "verified",
        "artifact_schema": "deepsafe.endurance-throughput-floor/v1",
        "artifact_fingerprint": floor_artifact["fingerprint"],
        "artifact_pin": _content_pin(root, floor_path),
        "profiles": floor_profiles,
        "verification": {
            "status": "verified",
            "live_rederived": True,
            "verified_safe_runs": 24,
        },
        "source_runtime_identity": floor_runtime_identity,
        "source_inputs": {
            "summary": floor_inputs["summary"],
            "scene_manifest": floor_inputs["scene_manifest"],
        },
    }
    resolved = {
        "schema_version": "deepsafe.endurance-campaign/v1",
        "name": "fixture-12-camera-seven-day",
        "campaign_path": "validation/endurance/campaign.json",
        "scene_manifest_path": "validation/scene_benchmark/scenes.json",
        "duration_seconds": 604800,
        "segment_seconds": 21600,
        "streams": 12,
        "profiles": [640, 960],
        "profile_strategy": "alternate_segments",
        "streammux": {"width": 1920, "height": 1080},
        "perf_interval_seconds": 5,
        "telemetry_interval_seconds": 1,
        "rss_interval_seconds": 10,
        "latency_bucket_seconds": 60,
        "startup_grace_seconds": 120,
        "perf_stall_timeout_seconds": 180,
        "min_telemetry_coverage_fraction": 0.95,
        "max_temperature_c": 86.0,
        "max_campaign_disk_bytes": 5368709120,
        "min_free_disk_bytes": 10737418240,
        "max_log_bytes_per_segment": 8388608,
        "power_safety": power_policy,
        "drift": {
            "baseline_segments_per_profile": 3,
            "max_fps_drop_fraction": 0.2,
            "max_latency_p95_increase_fraction": 0.5,
            "max_latency_p95_increase_ms": 50.0,
            "max_vram_growth_mib_per_hour": 128.0,
            "max_rss_growth_mib_per_hour": 128.0,
        },
        "sources": sources,
        "input_pins": input_pins,
        "throughput_floor": throughput_floor_binding,
        "execution_request": {
            "image": floor_runtime_identity["image"],
            "gpu_index": 0,
        },
    }
    static_campaign = dict(resolved)
    static_fingerprint = campaign._endurance_json_sha256(
        {"campaign": static_campaign, "source_files": []}
    )
    resolved["static_input_fingerprint"] = static_fingerprint
    fingerprint = campaign._endurance_json_sha256(
        {"campaign": resolved, "source_files": []}
    )
    resolved["config_fingerprint"] = fingerprint
    _write_json(
        root,
        "validation/results/endurance/current/campaign-resolved.json",
        resolved,
    )
    planned_segments = campaign._endurance_expected_plan()
    _write_json(
        root,
        "validation/results/endurance/current/plan.json",
        {
            "schema_version": "deepsafe.endurance-plan/v1",
            "config_fingerprint": fingerprint,
            "static_input_fingerprint": static_fingerprint,
            "input_pins_sha256": campaign._endurance_json_sha256(input_pins),
            "throughput_floor": throughput_floor_binding,
            "execution_request": resolved["execution_request"],
            "power_safety_policy": power_policy,
            "segments": planned_segments,
        },
    )
    session_id = "session-0123456789abcdef"
    session_dir = current / "sessions" / session_id
    preflight = {
        "schema_version": "deepsafe.endurance-session-preflight/v1",
        "session_id": session_id,
        "status": "ok",
        "image": resolved["execution_request"]["image"],
        "image_id": "sha256:fixture-resolved-image",
        "gpu_index": 0,
        "gpu_identity": gpu_identity,
        "xid": {"available": True, "lines": []},
        "kernel_oom": {"available": True, "lines": []},
        "power_profile": {"available": True, "value": "performance"},
        "max_temperature_c": floor_runtime_identity["max_temperature_c"],
        "power_safety_policy": floor_runtime_identity["power_safety_policy"],
    }
    preflight_path = _write_json(
        root,
        f"validation/results/endurance/current/sessions/{session_id}/preflight.json",
        preflight,
    )
    reentry = {
        "schema_version": "deepsafe.gpu-reentry-evidence/v1",
        "status": "ready_for_operator_review",
        "verification": {"status": "ready_for_operator_review"},
    }
    reentry_source_path = _write_json(
        root,
        "validation/inputs/endurance/fixture-reentry-source.json",
        reentry,
    )
    reentry_copy_path = _write_json(
        root,
        (
            "validation/results/endurance/current/sessions/"
            f"{session_id}/reentry-evidence.json"
        ),
        reentry,
    )
    runtime_identity = {
        "requested_image": resolved["execution_request"]["image"],
        "resolved_image_id": "sha256:fixture-resolved-image",
        "gpu_index": 0,
        "gpu_identity": gpu_identity,
    }
    session_receipt = {
        "schema_version": "deepsafe.endurance-session-receipt/v1",
        "session_id": session_id,
        "created_at_utc": "2026-07-16T00:00:00+00:00",
        "dry_run": False,
        "config_fingerprint": fingerprint,
        "static_input_fingerprint": static_fingerprint,
        "input_pins_sha256": campaign._endurance_json_sha256(input_pins),
        "throughput_floor": throughput_floor_binding,
        "throughput_floor_runtime": {
            "status": "exact_match",
            "source_runtime_identity": floor_runtime_identity,
        },
        "execution_request": resolved["execution_request"],
        "runtime_identity": runtime_identity,
        "guard_contract": {
            "reentry_verification_status": "ready_for_operator_review",
            "preflight_status": "ok",
            "telemetry_interval_seconds": 1,
            "power_safety_policy": power_policy,
            "reentry_source": _content_pin(root, reentry_source_path),
        },
        "campaign_artifacts": {
            "campaign_resolved": _content_pin(
                root, current / "campaign-resolved.json"
            ),
            "plan": _content_pin(root, current / "plan.json"),
            "preflight": _content_pin(root, preflight_path),
            "reentry_evidence": _content_pin(root, reentry_copy_path),
        },
    }
    session_receipt_path = _write_json(
        root,
        (
            "validation/results/endurance/current/sessions/"
            f"{session_id}/session-receipt.json"
        ),
        session_receipt,
    )
    session_binding = {
        "session_id": session_id,
        "receipt": _content_pin(root, session_receipt_path),
    }
    checkpoint_segments = []
    expected_sweep_actual = campaign._endurance_expected_input_sweep_actual(
        campaign.EvidenceStore(root), resolved
    )
    assert expected_sweep_actual is not None
    expected_input_sha = campaign._endurance_json_sha256(input_pins)
    expected_sweep_fingerprint = campaign._endurance_json_sha256(
        expected_sweep_actual
    )
    latency_sources = {
        str(source_id): {
            "buckets": 360,
            "samples": 21600,
            "coverage_fraction": 1.0,
        }
        for source_id in range(12)
    }
    for planned in planned_segments:
        attempt_number = 1
        profile = planned["profile"]
        attempt_dir = (
            current
            / "segments"
            / planned["segment_id"]
            / f"attempt-{attempt_number:02d}"
        )
        artifact_files = {
            "config": "deepstream.txt",
            "log": "deepstream.log",
            "perf_csv": "perf.csv",
            "gpu_csv": "gpu.csv",
            "platform_thermal_csv": "platform-thermal.csv",
            "latency_csv": "latency-1m.csv",
            "rss_csv": "rss.csv",
            "runtime_container": "runtime-container.json",
        }
        artifact_pins = {}
        for name, filename in artifact_files.items():
            if name == "runtime_container":
                continue
            artifact_path = attempt_dir / filename
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            if name == "config":
                artifact_path.write_text(
                    campaign._endurance_render_config_text(
                        resolved, profile, root
                    ),
                    encoding="utf-8",
                )
            else:
                artifact_path.write_text(
                    f"fixture {planned['segment_id']} {name}\n", encoding="utf-8"
                )
            artifact_pins[name] = _content_pin(root, artifact_path)
        input_sweeps = {
            "pre": {
                "schema_version": "deepsafe.endurance-input-pin-sweep/v1",
                "status": "ok",
                "checked_at_utc": "2026-07-15T23:59:59+00:00",
                "input_pins_sha256": expected_input_sha,
                "live_pins_fingerprint": expected_sweep_fingerprint,
                "actual": expected_sweep_actual,
                "mismatches": [],
            },
            "post": {
                "schema_version": "deepsafe.endurance-input-pin-sweep/v1",
                "status": "ok",
                "checked_at_utc": "2026-07-16T05:59:59+00:00",
                "input_pins_sha256": expected_input_sha,
                "live_pins_fingerprint": expected_sweep_fingerprint,
                "actual": expected_sweep_actual,
                "mismatches": [],
            },
        }
        command_contract = campaign._endurance_expected_command_contract(
            project_root=root,
            current=current,
            resolved=resolved,
            planned=planned,
            attempt_number=attempt_number,
            attempt_dir=attempt_dir,
            session_binding=session_binding,
            runtime_identity=runtime_identity,
        )
        expected_cmd = command_contract["command"][
            command_contract["command"].index(command_contract["resolved_image_id"])
            + 1 :
        ]
        runtime_container_attestation = {
            "schema_version": "deepsafe.endurance-runtime-container-attestation/v1",
            "status": "verified_running",
            "captured_at_utc": "2026-07-16T00:00:01+00:00",
            "command_contract_sha256": campaign._endurance_json_sha256(
                command_contract
            ),
            "container_id": "c" * 64,
            "container_name": command_contract["container_name"],
            "image_id": command_contract["resolved_image_id"],
            "config_image": command_contract["resolved_image_id"],
            "entrypoint": ["/opt/nvidia/nvidia_entrypoint.sh"],
            "cmd": expected_cmd,
            "actual_process": {
                "path": "/opt/nvidia/nvidia_entrypoint.sh",
                "args": expected_cmd,
            },
            "working_dir": "/workspace",
            "required_environment": sorted(
                [
                    "GST_DEBUG=1",
                    "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
                    "NVDS_ENABLE_LATENCY_MEASUREMENT=1",
                ]
            ),
            "expected_container_labels": dict(
                sorted(command_contract["container_labels"].items())
            ),
            "gpu_device_request": {
                "driver": "nvidia",
                "count": 0,
                "device_ids": [str(command_contract["gpu_index"])],
                "capabilities": [["gpu"]],
                "options": {},
            },
            "mounts": sorted(
                [
                    {
                        "type": "bind",
                        "source": str(root.resolve()),
                        "destination": "/workspace",
                        "read_write": False,
                    },
                    {
                        "type": "bind",
                        "source": str((root / "models").resolve()),
                        "destination": "/models",
                        "read_write": False,
                    },
                ],
                key=lambda mount: (mount["destination"], mount["source"]),
            ),
            "host_config": {"auto_remove": True, "ipc_mode": "host"},
            "state": {
                "running": True,
                "started_at_utc": "2026-07-16T00:00:00.000000000Z",
            },
        }
        runtime_container_path = _write_json(
            root,
            str((attempt_dir / "runtime-container.json").relative_to(root)),
            runtime_container_attestation,
        )
        artifact_pins["runtime_container"] = _content_pin(
            root, runtime_container_path
        )
        aggregate_p05 = 550.0 if profile == 640 else 320.0
        floor_evidence = command_contract["throughput_floor"]
        attempt = {
            "schema_version": "deepsafe.endurance-segment/v1",
            "status": "healthy",
            "config_fingerprint": fingerprint,
            "static_input_fingerprint": static_fingerprint,
            "throughput_floor": floor_evidence,
            "throughput_floor_evaluation": {
                **floor_evidence,
                "status": "passed",
                "observed_aggregate_fps_p05": aggregate_p05,
            },
            "throughput_floor_live_verification": {
                phase: {
                    "status": "verified",
                    "artifact_fingerprint": floor_evidence[
                        "artifact_fingerprint"
                    ],
                }
                for phase in ("pre", "post")
            },
            "session": session_binding,
            "segment": planned,
            "attempt": attempt_number,
            "attempt_finalization": {
                "state": "finalized",
                "method": "cross_segment_drift",
                "cross_segment_drift_gates": [],
                "finalized_at_utc": "2026-07-16T06:00:01+00:00",
            },
            "input_pin_sweeps": input_sweeps,
            "validated_seconds": 21600,
            "dry_run": False,
            "timing": {
                "started_at_utc": "2026-07-16T00:00:00+00:00",
                "finished_at_utc": "2026-07-16T06:00:00+00:00",
                "wall_seconds": 21600.0,
                "requested_seconds": 21600,
                "synthetic": False,
            },
            "process": {
                "container_name": command_contract["container_name"],
                "container_labels": command_contract["container_labels"],
                "command_contract": command_contract,
                "command": command_contract["command"],
                "exit_code": 0,
                "termination_method": None,
                "runtime_container_id": runtime_container_attestation[
                    "container_id"
                ],
            },
            "runtime_container_attestation": runtime_container_attestation,
            "runtime_container_attestation_error": None,
            "container_cleanup": {
                "verified_absent": True,
                "removed": False,
                "container_id": None,
                "container_name": command_contract["container_name"],
                "checked_at_utc": "2026-07-16T05:59:59+00:00",
            },
            "health_gates": [],
            "retriable": False,
            "throughput": {
                "status": "ok",
                "raw_parser_status": "ok",
                "expected_streams": 12,
                "header_streams": 12,
                "header_stream_ids": list(range(12)),
                "requested_duration_seconds": 21600,
                "perf_interval_seconds": 5,
                "requested_perf_intervals": 4320,
                "perf_intervals_raw_active": 4320,
                "perf_intervals_in_measurement_bounds": 4320,
                "perf_intervals_analyzed": 4320,
                "malformed_perf_lines": 0,
                "active_streams": 12,
                "inactive_stream_ids": [],
                "streams_with_nonpositive_fps": [],
                "aggregate_current_fps": {
                    "samples": 4320,
                    "mean": 564.0 if profile == 640 else 336.0,
                    "p05": aggregate_p05,
                },
                "window_contract": {
                    "schema_version": "deepsafe.endurance-perf-window/v1",
                    "status": "ok",
                    "csv_header": [
                        "elapsed_seconds",
                        "aggregate_fps",
                        "per_stream_mean_fps",
                    ],
                    "requested_duration_seconds": 21600,
                    "perf_interval_seconds": 5,
                    "requested_perf_intervals": 4320,
                    "valid_complete_rows": 4320,
                    "analyzed_rows": 4320,
                    "malformed_log_rows": 0,
                    "row_coverage_fraction": 1.0,
                    "temporal_slots_covered": 4320,
                    "temporal_slot_coverage_fraction": 1.0,
                    "minimum_required_coverage_fraction": 0.95,
                    "first_elapsed_seconds": 5.0,
                    "last_elapsed_seconds": 21600.0,
                    "observed_span_seconds": 21595.0,
                    "minimum_required_span_seconds": 21300.0,
                    "startup_grace_seconds": 120.0,
                    "tail_tolerance_seconds": 180.0,
                    "maximum_observed_gap_seconds": 5.0,
                    "maximum_allowed_gap_seconds": 180.0,
                    "raw_parser_status": "ok",
                    "csv_errors_first_20": [],
                    "checks": {
                        "raw_parser_status_eligible": True,
                        "parser_stream_contract": True,
                        "parser_window_contract": True,
                        "csv_well_formed": True,
                        "log_csv_row_count_crosscheck": True,
                        "log_csv_aggregate_crosscheck": True,
                        "row_coverage": True,
                        "temporal_slot_coverage": True,
                        "startup_endpoint": True,
                        "tail_endpoint": True,
                        "elapsed_span": True,
                        "maximum_gap": True,
                    },
                    "failed_checks": [],
                },
            },
            "fps_trend": {
                "status": "ok",
                "change_fraction": 0.0,
                "growth_per_hour": 0.0,
            },
            "latency": {
                "status": "ok",
                "expected_streams": 12,
                "expected_source_ids": list(range(12)),
                "observed_source_ids": list(range(12)),
                "missing_source_ids": [],
                "unexpected_source_ids": [],
                "malformed_rows": 0,
                "expected_buckets": 360,
                "aggregate_buckets": 360,
                "per_source": latency_sources,
                "p95_ms_across_buckets": {
                    "mean": 32.0 if profile == 640 else 52.0,
                },
                "p95_drift": {
                    "status": "ok",
                    "change_fraction": 0.0,
                    "delta": 0.0,
                },
            },
            "gpu": {
                "status": "ok",
                "sample_interval_seconds": 1.0,
                "requested_samples": 21600,
                "samples_raw": 21600,
                "samples_analyzed": 21600,
                "missing_numeric_samples": {},
                "gpu_names": ["NVIDIA fixture production GPU"],
                "metrics": {
                    "gpu_utilization_percent": {
                        "samples": 21600,
                        "mean": 75.0,
                        "max": 99.0,
                    },
                    "memory_used_mib": {
                        "samples": 21600,
                        "mean": 2048.0,
                        "max": 2048.0,
                    },
                    "power_draw_w": {"samples": 21600, "mean": 100.0},
                    "temperature_c": {"samples": 21600, "mean": 70.0, "max": 76.0},
                },
            },
            "memory_trends": {
                "vram_mib": {"status": "ok", "growth_per_hour": 0.0},
                "rss_mib": {"status": "ok", "growth_per_hour": 0.0},
            },
            "rss": {"samples": 2160, "monitor_errors": []},
            "telemetry_coverage": {
                "perf": 1.0,
                "gpu": 1.0,
                "latency": 1.0,
                "rss": 1.0,
                "latency_per_source": {
                    str(source_id): 1.0 for source_id in range(12)
                },
                "minimum_required": 0.95,
            },
            "xid": {
                "before": {"available": True},
                "after": {"available": True},
                "new_lines": [],
            },
            "oom": {
                "kernel_before": {"available": True},
                "kernel_after": {"available": True},
                "new_kernel_lines": [],
                "log_matches": {},
            },
            "gpu_safety": {
                "policy": power_policy,
                "abort_reason_code": None,
                "abort_reason": None,
                "event": None,
                "samples": 21600,
                "power_limit_drop_samples": 0,
                "slowdown_active_samples": 0,
                "max_consecutive_slowdown_samples": 0,
            },
            "power_profile_after": {"available": True, "value": "performance"},
            "disk": {
                "campaign_bytes_start": planned["index"] * 100,
                "campaign_bytes_end": (planned["index"] + 1) * 100,
                "growth_bytes": 100,
                "max_campaign_bytes": 5368709120,
                "free_bytes_after": 21474836480,
            },
            "log_filter": {
                "raw_latency_lines_suppressed": 21600,
                "batch_markers_suppressed": 0,
                "bytes_written": 1024,
                "truncated": False,
                "max_bytes": 8388608,
                "fatal_matches": {},
                "perf_rows_rejected_nonpositive": 0,
                "collector_exception": None,
            },
            "artifacts": {
                **{name: pin["path"] for name, pin in artifact_pins.items()},
                "status": str(
                    (attempt_dir / "status.json").resolve().relative_to(root.resolve())
                ),
            },
            "artifact_pins": artifact_pins,
        }
        status_path = _write_json(
            root,
            (
                "validation/results/endurance/current/segments/"
                f"{planned['segment_id']}/attempt-01/status.json"
            ),
            attempt,
        )
        attempt_receipt = {
            "schema_version": "deepsafe.endurance-attempt-receipt/v1",
            "created_at_utc": "2026-07-16T06:00:01+00:00",
            "segment": planned,
            "attempt": attempt_number,
            "status": "healthy",
            "validated_seconds": 21600,
            "config_fingerprint": fingerprint,
            "static_input_fingerprint": static_fingerprint,
            "throughput_floor": floor_evidence,
            "session": session_binding,
            "status_pin": _content_pin(root, status_path),
            "artifact_pins": artifact_pins,
            "artifact_pins_sha256": campaign._endurance_json_sha256(
                artifact_pins
            ),
            "input_pin_sweep_fingerprints": {
                "pre": expected_sweep_fingerprint,
                "post": expected_sweep_fingerprint,
            },
        }
        attempt_receipt_path = _write_json(
            root,
            (
                "validation/results/endurance/current/segments/"
                f"{planned['segment_id']}/attempt-01/attempt-receipt.json"
            ),
            attempt_receipt,
        )
        checkpoint_segments.append(
            {
                **planned,
                "status": "healthy",
                "validated_seconds": 21600,
                "attempts": [attempt],
                "attempt_receipts": [_content_pin(root, attempt_receipt_path)],
            }
        )
    checkpoint = {
        "schema_version": "deepsafe.endurance-checkpoint/v1",
        "campaign_name": resolved["name"],
        "config_fingerprint": fingerprint,
        "static_input_fingerprint": static_fingerprint,
        "input_pins": input_pins,
        "throughput_floor": throughput_floor_binding,
        "power_safety_policy": power_policy,
        "state": "complete",
        "dry_run": False,
        "started_at_utc": "2026-07-16T00:00:00+00:00",
        "finished_at_utc": "2026-07-23T00:00:00+00:00",
        "active": None,
        "campaign_health_gates": [],
        "target_validated_seconds": 604800,
        "validated_seconds": 604800,
        "unexpected_restarts": 0,
        "orphan_recoveries": 0,
        "sessions": [session_binding],
        "segments": checkpoint_segments,
    }
    _write_json(
        root,
        "validation/results/endurance/current/checkpoint.json",
        checkpoint,
    )
    _write_json(
        root,
        "validation/results/endurance/current/status.json",
        {
            "schema_version": "deepsafe.endurance-status/v1",
            "available": True,
            "state": "complete",
            "dry_run": False,
            "campaign_name": resolved["name"],
            "config_fingerprint": fingerprint,
            "static_input_fingerprint": static_fingerprint,
            "throughput_floor": throughput_floor_binding,
            "power_safety_policy": power_policy,
            "updated_at_utc": "2026-07-23T01:00:00+00:00",
            "target_validated_seconds": 604800,
            "validated_seconds": 604800,
            "progress_fraction": 1.0,
            "profiles_validated_seconds": {"640": 302400, "960": 302400},
            "segments": {"total": 28, "status_counts": {"healthy": 28}},
            "active": None,
            "unexpected_restarts": 0,
            "orphan_recoveries": 0,
            "scheduled_profile_rotations_completed": 27,
            "scheduled_profile_rotations_target": 27,
            "sessions": {"count": 1, "latest": session_binding},
            "campaign_health_gates": [],
            "artifact_bytes": 1000000,
            "max_artifact_bytes": 5368709120,
        },
    )
    for day in range(1, 8):
        summary = campaign._endurance_day_expected(checkpoint_segments, day)
        summary["generated_at_utc"] = f"2026-07-{16 + day:02d}T01:00:00+00:00"
        _write_json(
            root,
            f"validation/results/endurance/current/reports/day-{day:02d}.json",
            summary,
        )


def _populate_accepted_campaign(root: Path) -> Path:
    results = root / "validation/results"
    runs = []
    benchmark_types = [
        "medium_close_static_group",
        "near_vertical_overhead_security",
        *[f"distinct_camera_type_{index}" for index in range(2, 12)],
    ]
    for scene_index, benchmark_type in enumerate(benchmark_types):
        scene = f"scene_{scene_index:02d}"
        for profile in (640, 960):
            fingerprint = hashlib.sha256(f"{scene}:{profile}".encode()).hexdigest()
            _write_json(
                root,
                f"validation/results/scene-benchmark/{scene}/{profile}/status.json",
                _scene_status(scene, profile, fingerprint),
            )
            runs.append(
                {
                    "scene_id": scene,
                    "benchmark_type": benchmark_type,
                    "model_input_size": profile,
                    "status": "complete",
                    "fingerprint": fingerprint,
                }
            )
    _write_json(
        root,
        "validation/results/scene-benchmark/matrix-summary.json",
        {
            "schema_version": "deepsafe.scene-benchmark-matrix/v1",
            "generated_at_utc": "2026-07-16T00:00:00+00:00",
            "streams": 12,
            "duration_seconds_per_run": 300,
            "selected_scenes": 12,
            "selected_sizes": [640, 960],
            "expected_runs": 24,
            "status_counts": {"complete": 24},
            "runs": runs,
        },
    )

    caviar_results = []
    caviar_jobs = []
    job_states = []
    for sequence_index in range(8):
        sequence = f"sequence_{sequence_index:02d}"
        for profile in (640, 960):
            job_id = f"{sequence}:{profile}"
            caviar_jobs.append({"job_id": job_id})
            job_states.append({"job_id": job_id, "state": "complete", "reasons": []})
            metric = {
                "ground_truth": 100,
                "tp": 80 if profile == 640 else 85,
                "fp": 10,
                "fn": 20 if profile == 640 else 15,
                "precision": 0.888889 if profile == 640 else 0.894737,
                "recall": 0.8 if profile == 640 else 0.85,
                "f1": 0.842105 if profile == 640 else 0.871795,
                "ap_101_point": 0.82 if profile == 640 else 0.87,
            }
            caviar_results.append(
                {
                    "job_id": job_id,
                    "sequence_id": sequence,
                    "model_input": profile,
                    "status": "complete",
                    **metric,
                    "small_recall": 0.5,
                    "medium_recall": 0.9,
                    "large_recall": None,
                    "last_reported_average_fps": 300.0,
                }
            )
            _write_json(
                root,
                f"validation/results/caviar/{sequence}/{profile}/run-manifest.json",
                {
                    "status": "complete",
                    "sequence_id": sequence,
                    "model_input": profile,
                },
            )
            _write_json(
                root,
                f"validation/results/caviar/{sequence}/{profile}/evaluation.json",
                {
                    "schema_version": "deepsafe.person-evaluation/v1",
                    "overall": metric,
                    "diagnostics": {"prediction_only_frames": []},
                },
            )
    _write_json(
        root,
        "validation/results/caviar/batch-manifest.json",
        {
            "schema_version": "deepsafe.caviar-batch-plan/v1",
            "campaign": {
                "expected_jobs": 16,
                "sequence_count": 8,
                "model_input_sizes": [640, 960],
            },
            "jobs": caviar_jobs,
        },
    )
    _write_json(
        root,
        "validation/results/caviar/batch-aggregate.json",
        {
            "schema_version": "deepsafe.caviar-batch-aggregate/v1",
            "generated_at": "2026-07-16T01:00:00+00:00",
            "completeness": {
                "expected_jobs": 16,
                "complete_jobs": 16,
                "pending_jobs": 0,
                "is_complete": True,
                "job_states": job_states,
            },
            "results": caviar_results,
        },
    )

    open_jobs = []
    open_review_jobs = []
    open_scene_ids = [f"scene_{scene_index:02d}" for scene_index in range(12)]
    withheld_scene_id = open_scene_ids[-1]
    for scene_index, scene in enumerate(open_scene_ids):
        for profile in (640, 960):
            job_id = f"{scene}:{profile}"
            review_path = (
                f"validation/results/open-video-review/{scene}/{profile}/"
                "qualitative-review/review.json"
            )
            open_jobs.append(
                {
                    "job_id": job_id,
                    "scene_id": scene,
                    "model_input": profile,
                    "sensitive": scene == withheld_scene_id,
                }
            )
            open_review_jobs.append(
                {
                    "job_id": job_id,
                    "status": (
                        "closed-review-withheld"
                        if scene == withheld_scene_id
                        else "rendered"
                    ),
                    "review": review_path,
                }
            )
            _write_json(
                root,
                review_path,
                {
                    "schema_version": "deepsafe.open-video-review/v1",
                    "task": "gt_free_person_detection_qualitative_review",
                    "ground_truth_available": False,
                    "scene_id": scene,
                    "model_input": profile,
                },
            )
    open_plan_path = _write_json(
        root,
        "validation/results/open-video-review/campaign-plan.json",
        {
            "schema_version": "deepsafe.open-video-review-plan/v1",
            "campaign": {
                "expected_jobs": 24,
                "scene_count": 12,
                "model_input_sizes": [640, 960],
            },
            "accuracy_guardrail": {"ground_truth": False},
            "jobs": open_jobs,
        },
    )
    _write_json(
        root,
        "validation/results/open-video-review/campaign-review.json",
        {
            "schema_version": "deepsafe.open-video-review/v1",
            "ground_truth_available": False,
            "jobs": open_review_jobs,
        },
    )
    _populate_open_video_evidence_layers(
        root,
        plan_path=open_plan_path,
        reviewed_scene_ids=open_scene_ids[:-1],
    )

    _populate_accepted_endurance(root)

    _write_accepted_site_distance(root)
    _populate_rlivit_complete_evidence(root)
    _copy_checked_in_ppe_video_source_registry(root)
    return results


def _rlivit_metric_row(ground_truth: int) -> dict:
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
        "overall": _rlivit_metric_row(4318),
        "daytime": {
            "day": _rlivit_metric_row(3422),
            "night": _rlivit_metric_row(896),
        },
        "locations": {
            str(index): _rlivit_metric_row(count)
            for index, count in enumerate((428, 354, 275, 779, 114, 127, 424, 1817))
        },
        "coco_area": {
            "small": _rlivit_metric_row(2907),
            "medium": _rlivit_metric_row(1258),
            "large": _rlivit_metric_row(153),
        },
        "height_bands": {
            "lt32": _rlivit_metric_row(1634),
            "32to95": _rlivit_metric_row(2425),
            "gte96": _rlivit_metric_row(259),
        },
    }


def _rlivit_repin(value: dict) -> dict:
    value.pop("fingerprint_sha256", None)
    value["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return value


def _rlivit_public_status_value(phase: str = "complete") -> dict:
    pin_names = (
        "source_plan",
        "mp4_batch_receipt",
        "runtime_policy",
        "ds9_compatibility_receipt",
        "batch_aggregate",
        "batch_receipt",
    )
    value = {
        "schema_version": "deepsafe.rlivit-public-status/v1",
        "status": phase,
        "updated_at_utc": "2026-07-16T02:00:00+00:00",
        "dataset_id": "R-LiViT_RGB-T_v1.0",
        "gpu_docker_inference_executed": phase in {"running", "failed", "complete"},
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
        "progress": {
            "planned_jobs": 80,
            "launched_jobs": 80,
            "gpu_process_started_jobs": 80,
            "completed_jobs": 80,
            "failed_jobs": 0,
            "remaining_jobs": 0,
        },
        "profiles": {
            profile: {
                "status": "complete",
                "planned_jobs": 40,
                "completed_jobs": 40,
                "metrics": _rlivit_profile_metrics(),
            }
            for profile in ("640", "960")
        },
        "blocker_codes": [],
        "evidence": {
            name: {"size_bytes": index + 1, "sha256": f"{index + 1:x}" * 64}
            for index, name in enumerate(pin_names)
        },
    }
    if phase == "blocked":
        value["gpu_docker_inference_executed"] = False
        value["progress"].update(
            launched_jobs=0,
            gpu_process_started_jobs=0,
            completed_jobs=0,
            remaining_jobs=80,
        )
        value["blocker_codes"] = ["operator_authorization_missing"]
        for profile in value["profiles"].values():
            profile.update(status="blocked", completed_jobs=0, metrics=None)
        value["evidence"]["batch_aggregate"] = None
        value["evidence"]["batch_receipt"] = None
    elif phase == "awaiting_execution":
        value["gpu_docker_inference_executed"] = False
        value["progress"].update(
            launched_jobs=0,
            gpu_process_started_jobs=0,
            completed_jobs=0,
            remaining_jobs=80,
        )
        for profile in value["profiles"].values():
            profile.update(status="pending", completed_jobs=0, metrics=None)
        value["evidence"]["batch_aggregate"] = None
        value["evidence"]["batch_receipt"] = None
    elif phase in {"running", "failed"}:
        value["progress"].update(
            launched_jobs=5,
            gpu_process_started_jobs=5,
            completed_jobs=4,
            failed_jobs=1 if phase == "failed" else 0,
            remaining_jobs=76,
        )
        value["profiles"]["640"].update(
            status="running", completed_jobs=4, metrics=None
        )
        value["profiles"]["960"].update(
            status="pending", completed_jobs=0, metrics=None
        )
        value["evidence"]["batch_aggregate"] = None
        value["evidence"]["batch_receipt"] = None
    return _rlivit_repin(value)


def _rlivit_pathless(pin: dict) -> dict:
    return {
        "size_bytes": pin.get("size_bytes", pin.get("bytes")),
        "sha256": pin["sha256"],
    }


def _rlivit_mp4_public_receipt(batch_pin: dict) -> dict:
    return _rlivit_repin(
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
            "batch_receipt_pin": _rlivit_pathless(batch_pin),
        }
    )


def _populate_rlivit_complete_evidence(
    root: Path,
    *,
    profile_metrics: dict | None = None,
    profile_metrics_by_profile: dict[str, dict] | None = None,
) -> dict:
    """Materialize a small but fully pinned 40x2 private completion chain."""

    default_metrics = profile_metrics or _rlivit_profile_metrics()
    metrics_by_profile = {
        profile: copy.deepcopy(
            profile_metrics_by_profile[profile]
            if profile_metrics_by_profile is not None
            else default_metrics
        )
        for profile in ("640", "960")
    }
    nonce = "9" * 64
    run_relative = f"validation/results/rlivit/runs/{nonce}"
    display_root = run_relative
    run_directory = root / run_relative
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
    authorization_payload = {
        "schema_version": "deepsafe.rlivit-execution-authorization/v1",
        "status": "approved",
        "operator_identity": "fixture-operator",
        "campaign_nonce": nonce,
        "issued_at_utc": "2026-07-16T07:00:00+00:00",
        "expires_at_utc": "2026-07-16T19:00:00+00:00",
        "authorized_results_root": display_root,
        "campaign_definition_sha256": "6" * 64,
        "single_use": True,
    }
    authorization_path = _write_json(
        root,
        f"validation/authorizations/rlivit-{nonce}.json",
        authorization_payload,
    )
    authorization_path.chmod(0o440)
    authorization = {
        **authorization_payload,
        "artifact": _content_pin(root, authorization_path),
    }
    sequence_ids = list(campaign.EXPECTED_RLIVIT_SEQUENCE_IDS)
    ground_truth = _rlivit_repin(
        {
            "schema_version": "deepsafe.rlivit-gt-validation/v1",
            "status": "valid",
            "sequence_count": 40,
            "frame_count": 480,
            "person_count": 4318,
            "daytime": ["day", "night"],
            "locations": [str(index) for index in range(8)],
            "sequence_ids": sequence_ids,
            "live_sources_verified": True,
            "sequence_source_fingerprints": {
                sequence: hashlib.sha256(f"source:{sequence}".encode()).hexdigest()
                for sequence in sequence_ids
            },
        }
    )
    jobs = [
        {
            "job_id": f"rlivit:{sequence}:{profile}",
            "sequence_id": sequence,
            "model_input": profile,
            "expected_frames": 12,
            "run_root": f"{display_root}/jobs/{sequence}/{profile}",
            "status": "ready",
            "blockers": [],
        }
        for sequence in sequence_ids
        for profile in (640, 960)
    ]
    campaign_value = {
        "dataset_id": "R-LiViT_RGB-T_v1.0",
        "split": "test",
        "sequence_count": 40,
        "profiles": [640, 960],
        "expected_jobs": 80,
        "expected_frames_per_job": 12,
        "expected_ground_truth_persons": 4318,
        "materialized_root": "data/derived/r-livit/materialized-v1",
        "mp4_root": "data/derived/r-livit/test-mp4-v1",
        "results_root": display_root,
        "source_campaign": source_pin,
        "mp4_batch_receipt": mp4_pin,
        "mp4_campaign_binding": {
            "path": "data/derived/r-livit/test-mp4-v1/campaign-video-binding.json",
            "size_bytes": 4096,
            "sha256": "7" * 64,
        },
        "runtime_policy": runtime_policy,
        "model_runtime_contract": {"status": "bound"},
        "ds9_compatibility_receipt": ds9_pin,
        "rlivit_control_artifacts": {},
        "campaign_definition_sha256": authorization[
            "campaign_definition_sha256"
        ],
        "execution_authorization": authorization,
        "campaign_nonce": nonce,
        "session_claim": f"{display_root}/session-claim.json",
        "batch_receipt": f"{display_root}/batch-receipt.json",
        "gpu_execution_policy": "fixture-bound single-use execution",
    }
    plan = _rlivit_repin(
        {
            "schema_version": "deepsafe.rlivit-deepstream-batch-plan/v1",
            "created_at_utc": "2026-07-16T07:30:00+00:00",
            "status": "ready_for_authorized_execution",
            "gpu_docker_inference_requested": False,
            "blockers": [],
            "campaign": campaign_value,
            "ground_truth_validation": ground_truth,
            "sequences": [{"sequence_id": sequence} for sequence in sequence_ids],
            "jobs": jobs,
        }
    )
    plan_path = _write_json(root, f"{run_relative}/batch-plan.json", plan)
    plan_pin = _content_pin(root, plan_path)
    session_claim = {
        "schema_version": "deepsafe.rlivit-execution-session-claim/v1",
        "claimed_at_utc": "2026-07-16T07:45:00+00:00",
        "campaign_nonce": nonce,
        "authorized_results_root": display_root,
        "single_use": True,
        "plan": plan_pin,
        "authorization": authorization["artifact"],
    }
    session_path = _write_json(
        root, f"{run_relative}/session-claim.json", session_claim
    )
    session_pin = _content_pin(root, session_path)
    job_paths: dict[str, Path] = {}
    job_receipts: dict[str, dict] = {}
    for job in jobs:
        job_id = job["job_id"]
        job_receipt = _rlivit_repin(
            {
                "schema_version": "deepsafe.rlivit-deepstream-job-receipt/v1",
                "status": "complete_raw_replay_verified",
                "created_at_utc": "2026-07-16T08:30:00+00:00",
                "job_id": job_id,
                "sequence_id": job["sequence_id"],
                "model_input": job["model_input"],
                "plan": plan_pin,
                "session_claim": session_pin,
                "execution_authorization": authorization,
                "mp4_batch_receipt": mp4_pin,
                "runtime_policy": runtime_policy,
                "ds9_compatibility_receipt": ds9_pin,
            }
        )
        path = _write_json(
            root,
            f"{job['run_root']}/job-receipt.json",
            job_receipt,
        )
        job_paths[job_id] = path
        job_receipts[job_id] = _content_pin(root, path)
    aggregate = _rlivit_repin(
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
                profile: copy.deepcopy(metrics_by_profile[profile])
                for profile in ("640", "960")
            },
            "job_receipts": job_receipts,
        }
    )
    aggregate_path = _write_json(
        root, f"{run_relative}/batch-aggregate.json", aggregate
    )
    aggregate_pin = _content_pin(root, aggregate_path)
    canonical_job_ids = sorted(job_receipts)
    state = {
        "schema_version": "deepsafe.rlivit-deepstream-batch-state/v1",
        "status": "complete",
        "started_at_utc": "2026-07-16T08:00:00+00:00",
        "finished_at_utc": "2026-07-16T09:01:00+00:00",
        "plan": plan_pin,
        "session_claim": session_pin,
        "launched_jobs": canonical_job_ids,
        "gpu_process_started_jobs": canonical_job_ids,
        "completed_jobs": [
            {"job_id": job_id, "receipt": job_receipts[job_id]}
            for job_id in canonical_job_ids
        ],
        "failed_job": None,
    }
    state_path = _write_json(root, f"{run_relative}/batch-state.json", state)
    state_pin = _content_pin(root, state_path)
    final_receipt = _rlivit_repin(
        {
            "schema_version": "deepsafe.rlivit-deepstream-batch-receipt/v1",
            "status": "complete_80_jobs_independent_raw_replay",
            "created_at_utc": "2026-07-16T09:02:00+00:00",
            "campaign_nonce": nonce,
            "plan": plan_pin,
            "session_claim": session_pin,
            "execution_authorization": authorization,
            "mp4_batch_receipt": mp4_pin,
            "runtime_policy": runtime_policy,
            "ds9_compatibility_receipt": ds9_pin,
            "aggregate": aggregate_pin,
            "state": state_pin,
            "sequence_count": 40,
            "profiles": [640, 960],
            "job_count": 80,
            "job_receipts": job_receipts,
        }
    )
    final_path = _write_json(
        root, f"{run_relative}/batch-receipt.json", final_receipt
    )
    final_pin = _content_pin(root, final_path)
    public_mp4_path = _write_json(
        root,
        "validation/results/rlivit/mp4-batch-receipt.json",
        _rlivit_mp4_public_receipt(mp4_pin),
    )
    status = _rlivit_public_status_value("complete")
    status["updated_at_utc"] = "2026-07-16T09:03:00+00:00"
    for profile in ("640", "960"):
        status["profiles"][profile]["metrics"] = copy.deepcopy(
            metrics_by_profile[profile]
        )
    status["evidence"] = {
        "source_plan": _rlivit_pathless(source_pin),
        "mp4_batch_receipt": _rlivit_pathless(mp4_pin),
        "runtime_policy": _rlivit_pathless(runtime_artifact),
        "ds9_compatibility_receipt": _rlivit_pathless(ds9_pin),
        "batch_aggregate": _rlivit_pathless(aggregate_pin),
        "batch_receipt": _rlivit_pathless(final_pin),
    }
    _rlivit_repin(status)
    status_path = _write_json(
        root, "validation/results/rlivit/current/status.json", status
    )
    return {
        "run_directory": run_directory,
        "authorization": authorization_path,
        "plan": plan_path,
        "session_claim": session_path,
        "job_receipts": job_paths,
        "aggregate": aggregate_path,
        "state": state_path,
        "final_receipt": final_path,
        "mp4_public_receipt": public_mp4_path,
        "status": status_path,
    }


def _populate_rlivit_paired_audit_fixture(root: Path) -> dict:
    """Create a tiny live-pin fixture from the checked-in semantic audit."""

    source_root = Path(campaign.__file__).resolve().parents[1]
    existing_authorization = (
        root / f"validation/authorizations/rlivit-{'9' * 64}.json"
    )
    if existing_authorization.exists():
        existing_authorization.chmod(0o600)
    source_audit = json.loads(
        (
            source_root
            / "validation/results/rlivit/paired-error-audit/paired-audit.json"
        ).read_text(encoding="utf-8")
    )
    profile_metrics_by_profile = {
        profile: {
            "overall": copy.deepcopy(
                source_audit["paired_metrics"]["overall"][profile]
            ),
            "daytime": {
                key: copy.deepcopy(value[profile])
                for key, value in source_audit["paired_metrics"]["strata"][
                    "daytime"
                ].items()
            },
            "locations": {
                key: copy.deepcopy(value[profile])
                for key, value in source_audit["paired_metrics"]["strata"][
                    "locations"
                ].items()
            },
            "coco_area": {
                key: copy.deepcopy(value[profile])
                for key, value in source_audit["paired_metrics"]["strata"][
                    "coco_area"
                ].items()
            },
            "height_bands": {
                key: copy.deepcopy(value[profile])
                for key, value in source_audit["paired_metrics"]["strata"][
                    "height_bands"
                ].items()
            },
        }
        for profile in ("640", "960")
    }
    private = _populate_rlivit_complete_evidence(
        root,
        profile_metrics_by_profile=profile_metrics_by_profile,
    )
    package_root = root / "validation/results/rlivit/paired-error-audit"
    package_root.mkdir(parents=True, exist_ok=True)

    implementation = root / "validation/rlivit_paired_audit.py"
    implementation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "validation/rlivit_paired_audit.py", implementation)
    schema = root / "validation/schemas/rlivit-paired-error-audit-v1.schema.json"
    schema.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_root
        / "validation/schemas/rlivit-paired-error-audit-v1.schema.json",
        schema,
    )
    controls = {
        "implementation": _content_pin(root, implementation),
        "schema": _content_pin(root, schema),
    }

    audit = copy.deepcopy(source_audit)
    final_pin = _content_pin(root, private["final_receipt"])
    aggregate_pin = _content_pin(root, private["aggregate"])
    final_value = json.loads(private["final_receipt"].read_text(encoding="utf-8"))
    aggregate_value = json.loads(private["aggregate"].read_text(encoding="utf-8"))
    audit["lineage"].update(
        {
            "campaign_nonce": "9" * 64,
            "batch_receipt": final_pin,
            "batch_receipt_fingerprint_sha256": final_value[
                "fingerprint_sha256"
            ],
            "batch_aggregate": aggregate_pin,
            "batch_aggregate_fingerprint_sha256": aggregate_value[
                "fingerprint_sha256"
            ],
            "job_receipts": {
                job_id: _content_pin(root, path)
                for job_id, path in private["job_receipts"].items()
            },
            "controls": controls,
        }
    )

    fixture_dimensions = {
        "source": (4, 3),
        "overlay": (4, 4),
        "contact_sheet": (8, 6),
    }

    def materialize_visual(relative_path: str) -> dict:
        path = package_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative_path.startswith("assets/source/"):
            dimensions = fixture_dimensions["source"]
        elif relative_path.startswith("assets/overlays/"):
            dimensions = fixture_dimensions["overlay"]
        else:
            assert relative_path.startswith("assets/contact-sheets/")
            dimensions = fixture_dimensions["contact_sheet"]
        path.write_bytes(_fixture_rgb_png(*dimensions))
        content = path.read_bytes()
        return {
            "path": relative_path,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    first_visual: Path | None = None
    for frame in audit["assets"]["frames"].values():
        source = materialize_visual(frame["source"]["path"])
        if first_visual is None:
            first_visual = package_root / source["path"]
        frame["source"] = source
        frame["origin"] = {
            "path": frame["origin"]["path"],
            "size_bytes": source["size_bytes"],
            "sha256": source["sha256"],
        }
        for profile in ("640", "960"):
            frame["overlays"][profile] = materialize_visual(
                frame["overlays"][profile]["path"]
            )
    for sheet in audit["assets"]["contact_sheets"].values():
        sheet["artifact"] = materialize_visual(sheet["artifact"]["path"])
    audit["lineage"]["controls"] = controls
    _rlivit_repin(audit)
    audit_path = _write_json(
        root,
        "validation/results/rlivit/paired-error-audit/paired-audit.json",
        audit,
    )
    asset_manifest_fingerprint = campaign._rlivit_paired_canonical_sha256(
        audit["assets"]
    )
    assert asset_manifest_fingerprint is not None
    receipt = _rlivit_repin(
        {
            "schema_version": campaign.RLIVIT_PAIRED_RECEIPT_SCHEMA,
            "status": "complete",
            "audit": {
                **_content_pin(package_root, audit_path),
                "path": "paired-audit.json",
            },
            "audit_fingerprint_sha256": audit["fingerprint_sha256"],
            "input_batch_receipt": final_pin,
            "controls": controls,
            "rendered": True,
            "asset_manifest_fingerprint_sha256": asset_manifest_fingerprint,
        }
    )
    receipt_path = _write_json(
        root,
        "validation/results/rlivit/paired-error-audit/receipt.json",
        receipt,
    )
    expected_contract = {
        "EXPECTED_RLIVIT_PAIRED_AUDIT_PIN": {
            **_content_pin(package_root, audit_path),
            "path": "paired-audit.json",
        },
        "EXPECTED_RLIVIT_PAIRED_RECEIPT_PIN": {
            **_content_pin(package_root, receipt_path),
            "path": "receipt.json",
        },
        "EXPECTED_RLIVIT_PAIRED_AUDIT_FINGERPRINT": audit[
            "fingerprint_sha256"
        ],
        "EXPECTED_RLIVIT_PAIRED_RECEIPT_FINGERPRINT": receipt[
            "fingerprint_sha256"
        ],
        "EXPECTED_RLIVIT_PAIRED_ASSET_MANIFEST_FINGERPRINT": (
            asset_manifest_fingerprint
        ),
        "RLIVIT_PAIRED_SOURCE_IMAGE_DIMENSIONS": fixture_dimensions[
            "source"
        ],
        "RLIVIT_PAIRED_OVERLAY_IMAGE_DIMENSIONS": fixture_dimensions[
            "overlay"
        ],
        "RLIVIT_PAIRED_CONTACT_SHEET_DIMENSIONS": fixture_dimensions[
            "contact_sheet"
        ],
    }
    assert first_visual is not None
    return {
        **private,
        "paired_root": package_root,
        "paired_audit": audit_path,
        "paired_receipt": receipt_path,
        "paired_implementation": implementation,
        "paired_schema": schema,
        "paired_first_visual": first_visual,
        "paired_expected_contract": expected_contract,
    }


def _rebind_rlivit_paired_audit_fixture(root: Path, artifacts: dict) -> None:
    """Re-pin a deliberately edited audit and its downstream receipt."""

    audit = json.loads(artifacts["paired_audit"].read_text(encoding="utf-8"))
    _rlivit_repin(audit)
    artifacts["paired_audit"].write_text(json.dumps(audit), encoding="utf-8")
    receipt = json.loads(
        artifacts["paired_receipt"].read_text(encoding="utf-8")
    )
    receipt["audit"] = {
        **_content_pin(artifacts["paired_root"], artifacts["paired_audit"]),
        "path": "paired-audit.json",
    }
    receipt["audit_fingerprint_sha256"] = audit["fingerprint_sha256"]
    receipt["asset_manifest_fingerprint_sha256"] = (
        campaign._rlivit_paired_canonical_sha256(audit["assets"])
    )
    _rlivit_repin(receipt)
    artifacts["paired_receipt"].write_text(
        json.dumps(receipt), encoding="utf-8"
    )


def _refresh_rlivit_paired_fixture_contract(artifacts: dict) -> None:
    """Test-only allowlist refresh for exercising a newly signed bad package."""

    audit = json.loads(artifacts["paired_audit"].read_text(encoding="utf-8"))
    receipt = json.loads(
        artifacts["paired_receipt"].read_text(encoding="utf-8")
    )
    artifacts["paired_expected_contract"] = {
        "EXPECTED_RLIVIT_PAIRED_AUDIT_PIN": {
            **_content_pin(
                artifacts["paired_root"], artifacts["paired_audit"]
            ),
            "path": "paired-audit.json",
        },
        "EXPECTED_RLIVIT_PAIRED_RECEIPT_PIN": {
            **_content_pin(
                artifacts["paired_root"], artifacts["paired_receipt"]
            ),
            "path": "receipt.json",
        },
        "EXPECTED_RLIVIT_PAIRED_AUDIT_FINGERPRINT": audit[
            "fingerprint_sha256"
        ],
        "EXPECTED_RLIVIT_PAIRED_RECEIPT_FINGERPRINT": receipt[
            "fingerprint_sha256"
        ],
        "EXPECTED_RLIVIT_PAIRED_ASSET_MANIFEST_FINGERPRINT": receipt[
            "asset_manifest_fingerprint_sha256"
        ],
        "RLIVIT_PAIRED_SOURCE_IMAGE_DIMENSIONS": artifacts[
            "paired_expected_contract"
        ]["RLIVIT_PAIRED_SOURCE_IMAGE_DIMENSIONS"],
        "RLIVIT_PAIRED_OVERLAY_IMAGE_DIMENSIONS": artifacts[
            "paired_expected_contract"
        ]["RLIVIT_PAIRED_OVERLAY_IMAGE_DIMENSIONS"],
        "RLIVIT_PAIRED_CONTACT_SHEET_DIMENSIONS": artifacts[
            "paired_expected_contract"
        ]["RLIVIT_PAIRED_CONTACT_SHEET_DIMENSIONS"],
    }


def _rebind_rlivit_job_chain(root: Path, artifacts: dict, job_id: str) -> None:
    """Re-pin a deliberately mutated job through every downstream artifact."""

    job_pin = _content_pin(root, artifacts["job_receipts"][job_id])
    aggregate = json.loads(artifacts["aggregate"].read_text(encoding="utf-8"))
    aggregate["job_receipts"][job_id] = job_pin
    _rlivit_repin(aggregate)
    artifacts["aggregate"].write_text(json.dumps(aggregate), encoding="utf-8")
    aggregate_pin = _content_pin(root, artifacts["aggregate"])

    state = json.loads(artifacts["state"].read_text(encoding="utf-8"))
    completed = next(
        item for item in state["completed_jobs"] if item["job_id"] == job_id
    )
    completed["receipt"] = job_pin
    artifacts["state"].write_text(json.dumps(state), encoding="utf-8")
    state_pin = _content_pin(root, artifacts["state"])

    final = json.loads(artifacts["final_receipt"].read_text(encoding="utf-8"))
    final["job_receipts"][job_id] = job_pin
    final["aggregate"] = aggregate_pin
    final["state"] = state_pin
    _rlivit_repin(final)
    artifacts["final_receipt"].write_text(json.dumps(final), encoding="utf-8")
    final_pin = _content_pin(root, artifacts["final_receipt"])

    status = json.loads(artifacts["status"].read_text(encoding="utf-8"))
    status["evidence"]["batch_aggregate"] = _rlivit_pathless(aggregate_pin)
    status["evidence"]["batch_receipt"] = _rlivit_pathless(final_pin)
    _rlivit_repin(status)
    artifacts["status"].write_text(json.dumps(status), encoding="utf-8")


def _quality_pin(root: Path, path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _install_live_quality_decision(
    root: Path,
    results: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    passed: bool,
) -> Path:
    monkeypatch.setattr(
        campaign,
        "_live_verify_endurance_throughput_floor",
        _fixture_throughput_floor_verifier,
    )
    monkeypatch.setattr(
        campaign,
        "_endurance_replay_attempt_raw",
        _fixture_endurance_raw_replay,
    )
    policy = json.loads(
        (campaign.PROJECT_ROOT / "validation/person-quality-policy.draft.json").read_text(
            encoding="utf-8"
        )
    )
    nonce = "c" * 64
    session_relative = f"validation/results/caviar/sessions/{nonce}"
    session_root = root / session_relative
    policy["status"] = "approved"
    policy["policy_owner"] = {"identity": "fixture-owner@example.test"}
    policy["disclaimer"] = "Owner-approved fixture policy; not whole-campaign acceptance."
    approval = _write_json(
        root,
        "validation/inputs/person-quality-approval.json",
        {
            "schema_version": "deepsafe.person-quality-policy-approval/v1",
            "decision": "approved",
            "policy_id": policy["policy_id"],
            "policy_contract_sha256": campaign.policy_contract_sha256(policy),
            "owner_identity": policy["policy_owner"]["identity"],
            "approved_at_utc": "2026-07-15T23:00:00Z",
            "campaign_nonce": nonce,
            "expires_at_utc": "2026-07-16T22:00:00Z",
            "authorized_results_root": session_relative,
            "single_use": True,
        },
    )
    policy["approval_artifact"] = _quality_pin(root, approval)
    policy_path = _write_json(
        root, "validation/inputs/person-quality-policy.json", policy
    )
    policy_pin = _quality_pin(root, policy_path)
    authorization = {
        "campaign_nonce": nonce,
        "expires_at_utc": "2026-07-16T22:00:00Z",
        "authorized_results_root": session_relative,
        "single_use": True,
    }
    quality_binding = {
        "artifact": policy_pin,
        "policy_id": policy["policy_id"],
        "policy_contract_sha256": campaign.policy_contract_sha256(policy),
        "status": "approved",
        "campaign_authorization": authorization,
    }

    shutil.copytree(results / "caviar", session_root, dirs_exist_ok=True)
    claim_path = _write_json(
        root,
        f"{session_relative}/session-claim.json",
        {
            "schema_version": "deepsafe.caviar-session-claim/v1",
            "claimed_at_utc": "2026-07-16T00:00:00Z",
            "campaign_nonce": nonce,
            "authorized_results_root": session_relative,
            "single_use": True,
            "quality_policy": quality_binding,
        },
    )
    claim_path.chmod(0o440)
    claim_pin = _quality_pin(root, claim_path)
    legacy_plan = json.loads((results / "caviar/batch-manifest.json").read_text())
    jobs = []
    job_receipts = {}
    for raw_job in legacy_plan["jobs"]:
        job_id = raw_job["job_id"]
        sequence, rendered_profile = job_id.rsplit(":", 1)
        profile = int(rendered_profile)
        run_relative = f"{session_relative}/{sequence}/{profile}"
        receipt_path = _write_json(
            root,
            f"{run_relative}/job-receipt.json",
            {
                "schema_version": "deepsafe.caviar-job-receipt/v1",
                "created_at_utc": "2026-07-16T00:30:00Z",
                "job_id": job_id,
                "quality_policy": quality_binding,
                "campaign_authorization": authorization,
                "session_claim": claim_pin,
            },
        )
        receipt_path.chmod(0o440)
        receipt_pin = _quality_pin(root, receipt_path)
        job_receipts[job_id] = receipt_pin
        jobs.append(
            {
                "job_id": job_id,
                "sequence_id": sequence,
                "model_input": profile,
                "run_root": run_relative,
                "status": "complete",
                "started_at": "2026-07-16T00:01:00Z",
                "finished_at": "2026-07-16T00:30:00Z",
                "job_receipt": receipt_pin,
            }
        )
    manifest_path = _write_json(
        root,
        f"{session_relative}/batch-manifest.json",
        {
            "schema_version": "deepsafe.caviar-batch-plan/v1",
            "created_at": "2026-07-16T00:00:00Z",
            "status": "complete",
            "gpu_execution_requested": True,
            "started_at": "2026-07-16T00:00:00Z",
            "finished_at": "2026-07-16T01:00:00Z",
            "launched_jobs": 16,
            "failed_jobs": 0,
            "campaign": {
                "sequence_count": 8,
                "expected_jobs": 16,
                "model_input_sizes": [640, 960],
                "results_root": session_relative,
                "batch_manifest": f"{session_relative}/batch-manifest.json",
                "batch_receipt": f"{session_relative}/batch-receipt.json",
                "session_claim": f"{session_relative}/session-claim.json",
                "session_claim_artifact": claim_pin,
                "campaign_nonce": nonce,
                "quality_policy": quality_binding,
            },
            "jobs": jobs,
        },
    )
    aggregate_path = session_root / "batch-aggregate.json"
    batch_receipt_path = _write_json(
        root,
        f"{session_relative}/batch-receipt.json",
        {
            "schema_version": "deepsafe.caviar-batch-receipt/v1",
            "created_at_utc": "2026-07-16T01:00:00Z",
            "quality_policy": quality_binding,
            "campaign_authorization": authorization,
            "campaign_nonce": nonce,
            "session_claim": claim_pin,
            "plan": _quality_pin(root, manifest_path),
            "aggregate": _quality_pin(root, aggregate_path),
            "job_receipts": dict(sorted(job_receipts.items())),
        },
    )
    batch_receipt_path.chmod(0o440)
    metric_value = 0.9 if passed else 0.5
    metrics = {
        profile: {
            "sequences": 8,
            "ground_truth": 800,
            "tp": 720 if passed else 400,
            "fp": 80,
            "fn": 80 if passed else 400,
            "micro_precision": metric_value,
            "micro_recall": metric_value,
            "micro_f1": metric_value,
            "macro_ap50": metric_value,
        }
        for profile in ("640", "960")
    }
    rules = []
    for rule in policy["rules"]:
        selected = (
            rule["scope"]["profiles"]
            if rule["scope"]["kind"] == "each_profile"
            else [rule["scope"]["profile"]]
        )
        values = {
            str(profile): metrics[str(profile)][rule["metric"]]
            for profile in selected
        }
        rules.append(
            {
                **rule,
                "profile_values": values,
                "status": "pass" if passed else "fail",
            }
        )
    pins = sorted(
        [
            _quality_pin(root, policy_path),
            _quality_pin(root, approval),
            _quality_pin(root, manifest_path),
            _quality_pin(root, aggregate_path),
        ],
        key=lambda item: item["path"],
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            pins,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    calls = []

    def live_evaluator(**kwargs):
        calls.append(kwargs)
        return {
            "schema_version": "deepsafe.person-quality-decision/v1",
            "generated_at_utc": "2026-07-16T01:00:00Z",
            "status": "quality_gate_passed" if passed else "quality_gate_failed",
            "acceptance_effect": "person_quality_gate_only" if passed else "none",
            "policy_status": "approved",
            "policy": {
                "artifact": _quality_pin(root, policy_path),
                "policy_id": policy["policy_id"],
                "contract_sha256": campaign.policy_contract_sha256(policy),
            },
            "approval": {
                "artifact": _quality_pin(root, approval),
                "owner_identity": policy["policy_owner"]["identity"],
                "approved_at_utc": "2026-07-15T23:00:00Z",
                "strictly_before_campaign": True,
            },
            "campaign_completeness": {
                "status": "complete",
                "completed_jobs": 16,
                "expected_jobs": 16,
                "sequences": 8,
                "profiles": [640, 960],
                "all_jobs_newly_launched": True,
                "started_at_utc": "2026-07-16T00:00:00Z",
                "finished_at_utc": "2026-07-16T01:00:00Z",
            },
            "quality_decision": {
                "status": "pass" if passed else "fail",
                "metrics_by_profile": metrics,
                "rules": rules,
                "ap50_definition": "macro_mean_of_8_sequence_ap_101_point_at_iou_0.5",
            },
            "evidence": {
                "artifacts": pins,
                "artifact_count": len(pins),
                "fingerprint_sha256": fingerprint,
            },
            "gpu_or_docker_executed_by_evaluator": False,
        }

    monkeypatch.setattr(campaign, "evaluate_quality_policy", live_evaluator)
    monkeypatch.setattr(campaign, "_fixture_quality_calls", calls, raising=False)
    monkeypatch.setattr(
        campaign,
        "_fixture_caviar_session_root",
        session_root,
        raising=False,
    )
    return policy_path


def test_missing_campaign_is_preliminary_and_never_final(tmp_path):
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )

    assert report["decision"]["status"] == "preliminary"
    assert report["decision"]["accepted"] is False
    assert report["decision"]["final_claim_allowed"] is False
    assert "scene_benchmark_24_runs" in report["decision"]["failed_required_gates"]
    assert "calibrated_25m_detection" in report["decision"]["failed_required_gates"]


def test_ppe_video_source_registry_is_pathless_planning_only_metadata(tmp_path):
    registry = _copy_checked_in_ppe_video_source_registry(tmp_path)
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )

    section = report["campaigns"]["ppe_video_source_registry"]
    assert section == {
        "evidence_kind": "ppe_video_source_registry_metadata_only",
        "state": "valid_metadata_only",
        "registry_valid": True,
        "registry_sha256": campaign.EXPECTED_PPE_VIDEO_SOURCE_REGISTRY_SHA256,
        "expected_candidate_count": 12,
        "candidate_count": 12,
        "eligibility_counts": {
            key: 0 for key in campaign.PPE_VIDEO_SOURCE_ELIGIBILITY_KEYS
        },
        "primary_plan": "user_owned_authorized_site_footage",
        "candidate_groups": {
            "written_license_contact_ids": [
                "al_azani_kfupm_ppe_cctv",
                "mobiusi_helmet_action",
            ],
            "qualitative_video_ids": [
                "foundation_pit_v2",
                "pixabay_construction_worker_348896",
                "mixkit_two_construction_workers_1436",
            ],
            "static_diagnostic_ids": [
                "tcrsf_sfchd",
                "ppe_cctv_topdown",
                "put_your_ppe_on",
            ],
            "ml_restricted_ids": [
                "pexels_construction_worker_roof_16393893",
                "pexels_construction_site_7448386",
            ],
        },
        "acquisition": {
            "registry_metadata_only": True,
            "media_or_annotations_downloaded": False,
            "reporter_downloaded_or_decoded_media": False,
            "reporter_network_gpu_docker_or_inference_used": False,
        },
        "quantitative_benchmark_ready": False,
        "ppe_model_ready": False,
        "acceptance_effect": "planning_only_no_model_readiness",
        "reasons": [],
        "evidence_ids": ["ppe_video_source_registry"],
    }
    assert not any("path" in key for key in section)
    assert "https://" not in json.dumps(section)
    assert hashlib.sha256(registry.read_bytes()).hexdigest() == section[
        "registry_sha256"
    ]
    requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "ppe_video_source_registry_integrity"
    )
    assert requirement["state"] == "pass"
    assert requirement["required_for_acceptance"] is False

    markdown = campaign.render_markdown(
        report,
        tmp_path / "validation/results/campaign-report/report.md",
        tmp_path,
    )
    assert "PPE video source registry (metadata only)" in markdown
    assert "Quantitative PPE benchmark readiness and PPE model readiness are both **false**" in markdown
    assert "12/12" in markdown
    assert "https://" not in markdown


def test_ppe_video_source_registry_missing_withholds_all_candidate_claims(
    tmp_path,
):
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )

    section = report["campaigns"]["ppe_video_source_registry"]
    assert section["state"] == "missing"
    assert section["registry_valid"] is False
    assert section["registry_sha256"] is None
    assert section["candidate_count"] is None
    assert all(value is None for value in section["eligibility_counts"].values())
    assert section["primary_plan"] is None
    assert all(value == [] for value in section["candidate_groups"].values())
    assert section["quantitative_benchmark_ready"] is False
    assert section["ppe_model_ready"] is False
    requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "ppe_video_source_registry_integrity"
    )
    assert requirement["state"] == "incomplete"
    assert requirement["required_for_acceptance"] is False


@pytest.mark.parametrize(
    ("payload", "evidence_state"),
    [
        (None, "pin_sha256_mismatch"),
        (b"{broken", "invalid_json"),
        (
            b'{"schema_version":"deepsafe.ppe-video-source-candidates/v1",'
            b'"schema_version":"deepsafe.ppe-video-source-candidates/v1"}',
            "invalid_json",
        ),
        (b"x" * (1024 * 1024 + 1), "too_large"),
    ],
)
def test_ppe_video_source_registry_tampered_invalid_or_oversized_fails_closed(
    tmp_path, payload, evidence_state
):
    registry = _copy_checked_in_ppe_video_source_registry(tmp_path)
    if payload is None:
        registry.write_bytes(registry.read_bytes() + b"\n")
    else:
        registry.write_bytes(payload)

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )
    section = report["campaigns"]["ppe_video_source_registry"]
    assert section["state"] == "invalid"
    assert section["registry_valid"] is False
    assert section["candidate_count"] is None
    assert all(value is None for value in section["eligibility_counts"].values())
    assert section["primary_plan"] is None
    assert all(value == [] for value in section["candidate_groups"].values())
    assert section["quantitative_benchmark_ready"] is False
    assert section["ppe_model_ready"] is False
    evidence = next(
        item
        for item in report["evidence"]
        if item["id"] == "ppe_video_source_registry"
    )
    assert evidence["state"] == evidence_state


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":"fixture/v1","nested":{"value":1,"value":2}}',
        b'{"schema_version":"fixture/v1","value":NaN}',
        b'{"schema_version":"fixture/v1","value":Infinity}',
        b'{"schema_version":"fixture/v1","value":1e9999}',
    ],
)
def test_evidence_store_json_reader_rejects_ambiguous_or_nonfinite_json(
    tmp_path, payload
):
    artifact = tmp_path / "integrity.json"
    artifact.write_bytes(payload)
    store = campaign.EvidenceStore(tmp_path)

    assert store.read_json("integrity_json", artifact) is None
    assert store.entries["integrity_json"]["state"] == "invalid_json"


@pytest.mark.parametrize(
    "payload",
    [
        b'{"nested":{"value":1,"value":2}}\n',
        b'{"value":NaN}\n',
        b'{"value":-Infinity}\n',
        b'{"value":1e9999}\n',
    ],
)
def test_integrity_jsonl_reader_rejects_duplicate_or_nonfinite_rows(
    tmp_path, payload
):
    artifact = tmp_path / "integrity.jsonl"
    artifact.write_bytes(payload)
    store = campaign.EvidenceStore(tmp_path)

    assert campaign._open_read_jsonl(store, "integrity_jsonl", artifact) is None
    assert store.entries["integrity_jsonl"]["state"] == "invalid_jsonl"


def test_ppe_video_source_registry_rejects_symlinked_ancestor(tmp_path):
    project_root = tmp_path / "project"
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copy2(
        Path(campaign.__file__).resolve().parents[1]
        / "data/manifests/ppe-video-source-candidates.json",
        outside / "ppe-video-source-candidates.json",
    )
    data = project_root / "data"
    data.mkdir(parents=True)
    (data / "manifests").symlink_to(outside, target_is_directory=True)

    report = campaign.build_campaign_report(
        project_root=project_root,
        results_root=project_root / "validation/results",
        hardware_incident_path=None,
    )
    section = report["campaigns"]["ppe_video_source_registry"]
    assert section["state"] == "invalid"
    assert section["registry_valid"] is False
    assert section["candidate_count"] is None
    evidence = next(
        item
        for item in report["evidence"]
        if item["id"] == "ppe_video_source_registry"
    )
    assert evidence["state"] == "unsafe_symlink"


def test_ppe_video_source_registry_report_schema_rejects_readiness_overclaim(
    tmp_path,
):
    _copy_checked_in_ppe_video_source_registry(tmp_path)
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )
    overclaim = copy.deepcopy(report)
    overclaim["campaigns"]["ppe_video_source_registry"][
        "ppe_model_ready"
    ] = True

    with pytest.raises(ValueError, match="invalid PPE video source registry"):
        campaign.validate_report_shape(overclaim)

    schema = json.loads(
        (
            campaign.PROJECT_ROOT
            / "validation/schemas/validation-campaign-report-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(overclaim)


def test_full_consistent_campaign_is_accepted_and_schema_valid(tmp_path, monkeypatch):
    implementation = tmp_path / "validation/site_distance_evaluation.py"
    implementation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(site_distance.__file__), implementation)
    monkeypatch.setattr(site_distance, "__file__", str(implementation))
    results = _populate_accepted_campaign(tmp_path)
    policy_path = _install_live_quality_decision(
        tmp_path, results, monkeypatch, passed=True
    )
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
        person_quality_policy_path=policy_path,
    )

    assert report["decision"] == {
        "status": "accepted",
        "accepted": True,
        "final_claim_allowed": True,
        "failed_required_gates": [],
        "statement": "All campaign acceptance gates are proven.",
    }
    assert report["campaigns"]["scene_benchmark"]["paired_profile_comparison"][
        "paired_scene_count"
    ] == 12
    assert report["campaigns"]["caviar_ground_truth"]["paired_profile_comparison"][
        "paired_sequence_count"
    ] == 8
    assert report["campaigns"]["open_video_manual_review"]["paired_profile_comparison"][
        "paired_scene_count"
    ] == 11
    open_review = report["campaigns"]["open_video_manual_review"]
    assert open_review["automatic_candidate_generation"]["validated_jobs"] == 24
    assert open_review["automatic_candidate_generation"]["withheld_scene_ids"] == [
        "scene_11"
    ]
    assert open_review["ai_qualitative_visual_audit"]["accepted"] is True
    assert open_review["ai_qualitative_visual_audit"][
        "distinct_reviewed_video_types"
    ] == 11
    assert open_review["human_terminal_qa"]["terminal_decisions"] == 0
    assert open_review["human_terminal_qa"]["pending_decisions"] == 42
    assert open_review["human_terminal_qa"]["state"] == "pending"
    endurance = report["campaigns"]["endurance"]
    assert endurance["evidence_complete"] is True
    assert endurance["verified_latest_attempt_status_files"] == 28
    assert endurance["verified_attempt_receipts"] == 28
    assert endurance["verified_production_sessions"] == 1
    assert endurance["performance_quality_threshold_applied"] is True
    assert endurance["performance_quality_outcome"] == "passed"
    assert endurance["throughput_floor"]["status"] == "verified"
    assert endurance["throughput_floor"]["live_rederived"] is True
    assert endurance["throughput_floor"]["passing_endurance_attempts"] == 28
    assert endurance["throughput_floor"]["proven_floor_violations"] == 0
    assert report["campaigns"]["gpu_reentry"]["state"] == "missing"
    quality_section = report["campaigns"]["person_detection_quality"]
    assert quality_section["state"] == "passed"
    assert quality_section["live_cpu_recomputed"] is True
    assert quality_section["accepted"] is True
    rlivit = report["campaigns"]["rlivit_ground_truth"]
    assert rlivit["state"] == "complete"
    assert rlivit["evidence_complete"] is True
    assert rlivit["matrix"] == {
        "sequence_count": 40,
        "frame_count": 480,
        "person_count": 4318,
        "expected_jobs": 80,
        "completed_jobs": 80,
        "remaining_jobs": 0,
        "profiles": [640, 960],
        "live_sources_verified": True,
    }
    assert set(rlivit["pathless_evidence_pins"]) == set(
        campaign.RLIVIT_EVIDENCE_PIN_IDS
    )
    assert all(rlivit["pathless_evidence_pins"].values())
    assert all(rlivit["private_completion_proof"].values())
    assert rlivit["quality_interpretation"] == {
        "threshold_policy_applied": False,
        "quality_accepted": False,
        "person_quality_gate_unchanged": True,
        "statement": "R-LiViT completeness and metrics do not replace the owner-approved CAVIAR quality gate.",
    }
    rlivit_requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "rlivit_ground_truth_80_jobs"
    )
    assert rlivit_requirement["required_for_acceptance"] is True
    assert rlivit_requirement["state"] == "pass"
    markdown = campaign.render_markdown(
        report, tmp_path / "validation/results/campaign-report/report.md", tmp_path
    )
    assert "R-LiViT high/oblique security-camera ground truth" in markdown
    assert "80/80" in markdown
    assert "Pathless private completion proof" in markdown
    assert "9" * 64 not in markdown
    assert "fixture-operator" not in markdown
    assert "cannot replace or alter the CAVIAR person-quality decision" in markdown
    assert "Open-video automatic candidates" in markdown
    assert "Open-video hash-bound AI visual audit" in markdown
    assert "Terminal human QA remains **0/42**" in markdown
    assert "Withheld scenes are not counted as visually audited: `scene_11`" in markdown
    assert "Twelve GT-free open videos manually reviewed" not in markdown
    assert campaign._fixture_quality_calls == [
        {
            "policy_path": policy_path,
            "campaign_plan_path": campaign._fixture_caviar_session_root
            / "batch-manifest.json",
            "aggregate_path": campaign._fixture_caviar_session_root
            / "batch-aggregate.json",
            "project_root": tmp_path.resolve(),
        }
    ]

    schema = json.loads(
        (campaign.PROJECT_ROOT / "validation/schemas/validation-campaign-report-v1.schema.json").read_text()
    )
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == campaign.REPORT_SCHEMA
    assert schema["properties"]["campaigns"]["properties"][
        "open_video_manual_review"
    ] == {"$ref": "#/$defs/open_video_manual_review"}
    try:
        import jsonschema
    except ImportError:
        campaign.validate_report_shape(report)
    else:
        jsonschema.Draft202012Validator(schema).validate(report)


def test_caviar_official_nonce_session_wins_and_legacy_public_root_is_ignored(
    tmp_path, monkeypatch
):
    implementation = tmp_path / "validation/site_distance_evaluation.py"
    implementation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(site_distance.__file__), implementation)
    monkeypatch.setattr(site_distance, "__file__", str(implementation))
    results = _populate_accepted_campaign(tmp_path)
    policy_path = _install_live_quality_decision(
        tmp_path, results, monkeypatch, passed=True
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
        person_quality_policy_path=policy_path,
    )

    caviar = report["campaigns"]["caviar_ground_truth"]
    assert caviar["accepted"] is True
    assert caviar["metrics_withheld"] is False
    assert caviar["official_session"] == {
        "state": "resolved",
        "authorized_policy": True,
        "candidate_count": 1,
        "valid_candidate_count": 1,
        "conflicting_candidate_count": 0,
        "legacy_public_artifacts_ignored": True,
        "private_identity_projected": False,
    }
    caviar_paths = {
        item["path"]
        for item in report["evidence"]
        if item["id"] in caviar["evidence_ids"]
    }
    assert str(campaign._fixture_caviar_session_root.relative_to(tmp_path)) in "\n".join(
        sorted(caviar_paths)
    )
    assert "validation/results/caviar/batch-manifest.json" not in caviar_paths
    assert "validation/results/caviar/batch-aggregate.json" not in caviar_paths


def test_caviar_stale_public_root_cannot_replace_missing_authorized_session(
    tmp_path, monkeypatch
):
    results = _populate_accepted_campaign(tmp_path)
    policy_path = _install_live_quality_decision(
        tmp_path, results, monkeypatch, passed=True
    )
    shutil.rmtree(campaign._fixture_caviar_session_root)

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
        person_quality_policy_path=policy_path,
    )

    caviar = report["campaigns"]["caviar_ground_truth"]
    assert caviar["accepted"] is False
    assert caviar["evidence_complete"] is False
    assert caviar["metrics_withheld"] is True
    assert caviar["aggregate_result_jobs"] == 0
    assert caviar["profiles"] == {}
    assert caviar["paired_profile_comparison"]["pairs"] == []
    assert caviar["official_session"]["state"] == "missing"
    assert caviar["official_session"]["candidate_count"] == 0
    assert caviar["official_session"]["legacy_public_artifacts_ignored"] is True
    separation = next(
        item
        for item in report["requirements"]
        if item["id"] == "ground_truth_manual_separation"
    )
    assert separation["state"] == "pass"


def test_caviar_second_session_targeting_current_nonce_is_a_conflict(
    tmp_path, monkeypatch
):
    results = _populate_accepted_campaign(tmp_path)
    policy_path = _install_live_quality_decision(
        tmp_path, results, monkeypatch, passed=True
    )
    conflicting = results / "caviar/sessions" / ("d" * 64) / "batch-receipt.json"
    conflicting.parent.mkdir(parents=True)
    shutil.copy2(campaign._fixture_caviar_session_root / "batch-receipt.json", conflicting)
    conflicting.chmod(0o440)

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
        person_quality_policy_path=policy_path,
    )

    official = report["campaigns"]["caviar_ground_truth"]["official_session"]
    assert official["state"] == "conflict"
    assert official["candidate_count"] == 2
    assert official["valid_candidate_count"] == 1
    assert official["conflicting_candidate_count"] == 1
    assert report["campaigns"]["caviar_ground_truth"]["accepted"] is False


def test_caviar_session_aggregate_pin_tamper_fails_closed(tmp_path, monkeypatch):
    results = _populate_accepted_campaign(tmp_path)
    policy_path = _install_live_quality_decision(
        tmp_path, results, monkeypatch, passed=True
    )
    aggregate = campaign._fixture_caviar_session_root / "batch-aggregate.json"
    aggregate.write_bytes(aggregate.read_bytes() + b"\n")

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
        person_quality_policy_path=policy_path,
    )

    caviar = report["campaigns"]["caviar_ground_truth"]
    assert caviar["accepted"] is False
    assert caviar["official_session"]["state"] == "invalid"
    assert caviar["metrics_withheld"] is True
    assert caviar["aggregate_result_jobs"] == 0
    assert caviar["profiles"] == {}
    separation = next(
        item
        for item in report["requirements"]
        if item["id"] == "ground_truth_manual_separation"
    )
    assert separation["state"] == "pass"
    evidence = next(
        item for item in report["evidence"] if item["id"] == "caviar_batch_aggregate"
    )
    assert evidence["state"] == "pin_size_mismatch"


def test_caviar_claim_outside_approval_window_fails_with_coherent_live_pins(
    tmp_path, monkeypatch
):
    results = _populate_accepted_campaign(tmp_path)
    policy_path = _install_live_quality_decision(
        tmp_path, results, monkeypatch, passed=True
    )
    session_root = campaign._fixture_caviar_session_root
    claim_path = session_root / "session-claim.json"
    claim_path.chmod(0o640)
    claim = json.loads(claim_path.read_text())
    claim["claimed_at_utc"] = "2026-07-16T22:30:00Z"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    claim_path.chmod(0o440)
    claim_pin = _quality_pin(tmp_path, claim_path)

    manifest_path = session_root / "batch-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rewritten_job_pins = {}
    for job in manifest["jobs"]:
        receipt_path = tmp_path / job["job_receipt"]["path"]
        receipt_path.chmod(0o640)
        receipt = json.loads(receipt_path.read_text())
        receipt["session_claim"] = claim_pin
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        receipt_path.chmod(0o440)
        pin = _quality_pin(tmp_path, receipt_path)
        job["job_receipt"] = pin
        rewritten_job_pins[job["job_id"]] = pin
    manifest["campaign"]["session_claim_artifact"] = claim_pin
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    batch_receipt_path = session_root / "batch-receipt.json"
    batch_receipt_path.chmod(0o640)
    batch_receipt = json.loads(batch_receipt_path.read_text())
    batch_receipt["session_claim"] = claim_pin
    batch_receipt["plan"] = _quality_pin(tmp_path, manifest_path)
    batch_receipt["job_receipts"] = dict(sorted(rewritten_job_pins.items()))
    batch_receipt_path.write_text(json.dumps(batch_receipt), encoding="utf-8")
    batch_receipt_path.chmod(0o440)

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
        person_quality_policy_path=policy_path,
    )

    caviar = report["campaigns"]["caviar_ground_truth"]
    assert caviar["accepted"] is False
    assert caviar["official_session"]["state"] == "invalid"
    assert set(caviar["official_session"]) == {
        "state",
        "authorized_policy",
        "candidate_count",
        "valid_candidate_count",
        "conflicting_candidate_count",
        "legacy_public_artifacts_ignored",
        "private_identity_projected",
    }
    semantic_evidence = {
        item["id"]: item["state"]
        for item in report["evidence"]
        if item["id"]
        in {
            "caviar_batch_receipt",
            "caviar_session_claim",
        }
    }
    assert semantic_evidence == {
        "caviar_batch_receipt": "ok",
        "caviar_session_claim": "ok",
    }


def test_open_video_evidence_levels_exclude_withheld_and_keep_human_qa_pending(
    tmp_path,
):
    results = _populate_accepted_campaign(tmp_path)

    section, facts = campaign._build_open_review_section(
        campaign.EvidenceStore(tmp_path), results
    )

    automatic = section["automatic_candidate_generation"]
    ai_audit = section["ai_qualitative_visual_audit"]
    human = section["human_terminal_qa"]
    assert automatic["validated_jobs"] == 24
    assert automatic["rendered_jobs"] == 22
    assert automatic["withheld_jobs"] == 2
    assert automatic["withheld_scene_ids"] == ["scene_11"]
    assert automatic["not_a_visual_review"] is True
    assert automatic["candidate_assets"] == {
        "decision_count": 42,
        "expected_decision_count": 42,
        "asset_count": 168,
        "expected_asset_count": 168,
        "live_hash_verified_assets": 168,
        "index_contract_valid": True,
    }
    assert ai_audit["accepted"] is True
    assert ai_audit["source_frame_count"] == 21
    assert ai_audit["profile_decision_count"] == 42
    assert ai_audit["distinct_reviewed_video_types"] == 11
    assert ai_audit["coverage"] == {
        "medium_close_source_frames": 1,
        "overhead_top_view_source_frames": 1,
        "high_oblique_source_frames": 1,
        "required_coverage_proven": True,
    }
    assert "scene_11" not in ai_audit["reviewed_scene_ids"]
    assert ai_audit["withheld_scenes_excluded"] is True
    assert ai_audit["human_review_claimed"] is False
    assert human == {
        "evidence_level": "terminal_human_qa",
        "state": "pending",
        "expected_decisions": 42,
        "terminal_decisions": 0,
        "pending_decisions": 42,
        "artifact_contract_valid": True,
        "accepted": False,
        "required_for_ai_audit_acceptance": False,
        "reasons": [],
    }
    assert section["paired_profile_comparison"]["paired_scene_count"] == 11
    assert section["paired_profile_comparison"]["paired_source_frame_count"] == 21
    assert section["accepted"] is True
    assert facts["accepted"] is True


def test_report_shape_rejects_ai_review_promoted_to_human_or_withheld_review(
    tmp_path,
):
    results = _populate_accepted_campaign(tmp_path)
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
    )

    ai_as_human = copy.deepcopy(report)
    ai_as_human["campaigns"]["open_video_manual_review"][
        "ai_qualitative_visual_audit"
    ]["human_review_claimed"] = True
    with pytest.raises(ValueError, match="invalid AI qualitative"):
        campaign.validate_report_shape(ai_as_human)

    withheld_as_reviewed = copy.deepcopy(report)
    audit = withheld_as_reviewed["campaigns"]["open_video_manual_review"][
        "ai_qualitative_visual_audit"
    ]
    audit["reviewed_scene_ids"][0] = "scene_11"
    with pytest.raises(ValueError, match="withheld scene promoted"):
        campaign.validate_report_shape(withheld_as_reviewed)


def test_open_video_ai_audit_rejects_stale_manual_index_hash(tmp_path):
    results = _populate_accepted_campaign(tmp_path)
    audit_path = results / "open-video-review/ai-qualitative-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["input_bindings"]["manual_assets_index"]["sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    section, facts = campaign._build_open_review_section(
        campaign.EvidenceStore(tmp_path), results
    )

    assert section["automatic_candidate_generation"]["accepted"] is True
    assert section["ai_qualitative_visual_audit"]["accepted"] is False
    assert "ai_audit_input_hash_binding_invalid" in section[
        "ai_qualitative_visual_audit"
    ]["reasons"]
    assert section["accepted"] is False
    assert facts["accepted"] is False


def test_open_video_ai_audit_rejects_estimate_outside_source_review_range(
    tmp_path,
):
    results = _populate_accepted_campaign(tmp_path)
    audit_path = results / "open-video-review/ai-qualitative-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["reviews"][0]["estimated_visible_persons"]["max"] = 1
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    section, facts = campaign._build_open_review_section(
        campaign.EvidenceStore(tmp_path), results
    )

    ai_audit = section["ai_qualitative_visual_audit"]
    assert ai_audit["accepted"] is False
    assert "ai_audit_estimate_outside_source_review_range" in ai_audit[
        "reasons"
    ]
    assert section["accepted"] is False
    assert facts["accepted"] is False


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("schema", "ai_audit_schema_invalid"),
        ("count", "ai_audit_scope_contract_invalid"),
        ("pair", "ai_audit_exact_id_profile_or_count_binding_invalid"),
    ],
)
def test_open_video_ai_audit_schema_count_and_exact_profile_pair_fail_closed(
    tmp_path, mutation, expected_reason
):
    results = _populate_accepted_campaign(tmp_path)
    audit_path = results / "open-video-review/ai-qualitative-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if mutation == "schema":
        audit["schema_version"] = "deepsafe.open-video-ai-qualitative-audit/v2"
    elif mutation == "count":
        audit["scope"]["profile_decision_count"] = 41
    else:
        first = audit["reviews"][0]
        first["decision_ids"][1] = first["decision_ids"][0]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    section, facts = campaign._build_open_review_section(
        campaign.EvidenceStore(tmp_path), results
    )

    assert section["ai_qualitative_visual_audit"]["accepted"] is False
    assert expected_reason in section["ai_qualitative_visual_audit"]["reasons"]
    assert section["accepted"] is False
    assert facts["accepted"] is False


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("schema", "manual_assets_schema_invalid"),
        ("count", "manual_assets_header_contract_invalid"),
        ("pair", "manual_assets_decision_identity_invalid"),
    ],
)
def test_open_video_manual_index_schema_count_and_profile_pair_fail_closed(
    tmp_path, mutation, expected_reason
):
    results = _populate_accepted_campaign(tmp_path)
    index_path = results / "open-video-review/manual-assets/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if mutation == "schema":
        index["schema_version"] = "deepsafe.open-video-manual-assets/v2"
    elif mutation == "count":
        index["decision_count"] = 41
    else:
        index["decisions"][1]["model_input"] = 640
    index_path.write_text(json.dumps(index), encoding="utf-8")

    section, facts = campaign._build_open_review_section(
        campaign.EvidenceStore(tmp_path), results
    )

    automatic = section["automatic_candidate_generation"]
    assert automatic["accepted"] is False
    assert expected_reason in automatic["reasons"]
    assert section["ai_qualitative_visual_audit"]["accepted"] is False
    assert section["accepted"] is False
    assert facts["index_contract"] is False


def test_open_video_manual_index_rejects_non_authoritative_source_observation(
    tmp_path,
):
    results = _populate_accepted_campaign(tmp_path)
    index_path = results / "open-video-review/manual-assets/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    source_id = index["decisions"][0]["source_review_id"]
    for decision in index["decisions"]:
        if decision["source_review_id"] == source_id:
            decision["source_observation"]["visible_person_count_range"]["max"] = 1
    index_path.write_text(json.dumps(index), encoding="utf-8")

    section, facts = campaign._build_open_review_section(
        campaign.EvidenceStore(tmp_path), results
    )

    automatic = section["automatic_candidate_generation"]
    assert automatic["accepted"] is False
    assert "manual_assets_source_observation_binding_mismatch" in automatic[
        "reasons"
    ]
    assert section["accepted"] is False
    assert facts["index_contract"] is False


def test_open_video_manual_asset_live_hash_tamper_fails_candidate_and_ai_levels(
    tmp_path,
):
    results = _populate_accepted_campaign(tmp_path)
    index_path = results / "open-video-review/manual-assets/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    first_asset = next(iter(index["assets"].values()))
    asset_path = results / first_asset["relative_path"]
    asset_path.write_bytes(asset_path.read_bytes() + b"tampered")

    section, facts = campaign._build_open_review_section(
        campaign.EvidenceStore(tmp_path), results
    )

    assert section["automatic_candidate_generation"]["accepted"] is False
    assert "manual_assets_asset_live_hash_mismatch" in section[
        "automatic_candidate_generation"
    ]["reasons"]
    assert section["ai_qualitative_visual_audit"]["accepted"] is False
    assert section["accepted"] is False
    assert facts["index_contract"] is False


@pytest.mark.parametrize(
    ("phase", "expected_requirement_state"),
    [
        ("blocked", "blocked"),
        ("awaiting_execution", "incomplete"),
        ("running", "incomplete"),
        ("failed", "unproven"),
    ],
)
def test_rlivit_noncomplete_phases_never_pass_requirement_or_expose_metrics(
    tmp_path, phase, expected_requirement_state
):
    results = tmp_path / "validation/results"
    _write_json(
        tmp_path,
        "validation/results/rlivit/current/status.json",
        _rlivit_public_status_value(phase),
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
    )

    section = report["campaigns"]["rlivit_ground_truth"]
    requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "rlivit_ground_truth_80_jobs"
    )
    assert section["state"] == phase
    assert section["evidence_complete"] is False
    assert section["metrics_withheld"] is True
    assert all(item["metrics"] is None for item in section["profiles"].values())
    assert section["quality_interpretation"]["quality_accepted"] is False
    assert requirement["state"] == expected_requirement_state
    assert report["decision"]["accepted"] is False


def test_rlivit_blocked_gate_does_not_change_caviar_quality_outcome(
    tmp_path, monkeypatch
):
    implementation = tmp_path / "validation/site_distance_evaluation.py"
    implementation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(site_distance.__file__), implementation)
    monkeypatch.setattr(site_distance, "__file__", str(implementation))
    results = _populate_accepted_campaign(tmp_path)
    _write_json(
        tmp_path,
        "validation/results/rlivit/current/status.json",
        _rlivit_public_status_value("blocked"),
    )
    policy_path = _install_live_quality_decision(
        tmp_path, results, monkeypatch, passed=True
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
        person_quality_policy_path=policy_path,
    )

    assert report["campaigns"]["person_detection_quality"]["state"] == "passed"
    assert report["campaigns"]["person_detection_quality"]["accepted"] is True
    assert report["campaigns"]["caviar_ground_truth"]["quality_outcome"] == {
        "state": "passed",
        "accepted": True,
        "separate_required_gate_id": "person_detection_quality",
    }
    assert report["campaigns"]["rlivit_ground_truth"]["state"] == "blocked"
    assert report["decision"]["accepted"] is False
    assert "rlivit_ground_truth_80_jobs" in report["decision"][
        "failed_required_gates"
    ]


def _build_paired_fixture_projection(root: Path, artifacts: dict) -> tuple[dict, dict]:
    store = campaign.EvidenceStore(root)
    rlivit, rlivit_facts = campaign._build_rlivit_section(
        store, root / "validation/results"
    )
    assert rlivit_facts["accepted"] is True
    with mock.patch.multiple(
        campaign, **artifacts["paired_expected_contract"]
    ):
        section, facts = campaign._build_rlivit_paired_error_audit_section(
            store,
            root / "validation/results",
            rlivit,
        )
    return section, facts


def test_rlivit_paired_audit_fixture_is_fully_live_verified(tmp_path):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert facts == {
        "accepted": True,
        "paired_sequences": 40,
        "state": "complete",
    }
    assert section["state"] == "complete"
    assert section["valid_for_paired_comparison"] is True
    assert section["coverage"] == {
        "expected_sequences": 40,
        "verified_sequences": 40,
        "expected_paired_frames": 480,
        "verified_paired_frames": 480,
        "expected_ground_truth_persons": 4318,
        "verified_ground_truth_persons": 4318,
    }
    assert section["visual_assets"]["verified_source_images"] == 31
    assert section["visual_assets"]["verified_overlays"] == 62
    assert section["visual_assets"]["verified_contact_sheets"] == 5
    assert section["visual_assets"]["verified_visual_pins"] == 98
    assert section["lineage_scope"] == {
        "audit_attested_unique_pinned_files": 763,
        "reporter_live_verified_receipt_pins": 82,
        "reporter_rehashed_all_763_raw_inputs": False,
    }
    assert all(section["proof"].values())
    assert section["quality_interpretation"]["quality_accepted"] is False
    assert section["quality_interpretation"][
        "evidence_completeness_accepted"
    ] is False
    with mock.patch.multiple(
        campaign, **artifacts["paired_expected_contract"]
    ):
        assert campaign._rlivit_paired_projection_valid(section)


def test_rlivit_paired_audit_is_valid_gt_alternative_only_for_paired_gate(
    tmp_path,
):
    results = _populate_accepted_campaign(tmp_path)
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)

    with mock.patch.multiple(
        campaign, **artifacts["paired_expected_contract"]
    ):
        report = campaign.build_campaign_report(
            project_root=tmp_path,
            results_root=results,
            hardware_incident_path=None,
        )

    paired_requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "paired_640_960_comparison"
    )
    assert paired_requirement["state"] == "pass"
    assert "CAVIAR GT=0/8" in paired_requirement["detail"]
    assert "R-LiViT GT=40/40" in paired_requirement["detail"]
    assert report["campaigns"]["caviar_ground_truth"]["accepted"] is False
    assert report["campaigns"]["person_detection_quality"]["accepted"] is False
    assert report["campaigns"]["rlivit_ground_truth"]["evidence_complete"] is True
    assert report["campaigns"]["rlivit_ground_truth"]["paired_error_audit"][
        "quality_interpretation"
    ]["visual_error_audit_quality_accepted"] is False
    assert report["decision"]["accepted"] is False
    assert "caviar_ground_truth_16_jobs" in report["decision"][
        "failed_required_gates"
    ]
    assert "person_detection_quality" in report["decision"][
        "failed_required_gates"
    ]


def test_rlivit_paired_audit_missing_is_fail_closed(tmp_path):
    _populate_rlivit_complete_evidence(tmp_path)
    store = campaign.EvidenceStore(tmp_path)
    rlivit, _facts = campaign._build_rlivit_section(
        store, tmp_path / "validation/results"
    )

    section, facts = campaign._build_rlivit_paired_error_audit_section(
        store,
        tmp_path / "validation/results",
        rlivit,
    )

    assert section["state"] == "missing"
    assert section["valid_for_paired_comparison"] is False
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_visual_asset_tamper(tmp_path):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    artifacts["paired_first_visual"].write_bytes(
        artifacts["paired_first_visual"].read_bytes() + b"tamper"
    )

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_visual_asset_live_pin_invalid"
    ]
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_symlinked_visual_asset(tmp_path):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    target = next(
        path
        for path in (artifacts["paired_root"] / "assets/overlays/640").iterdir()
        if path != artifacts["paired_first_visual"]
    )
    artifacts["paired_first_visual"].unlink()
    artifacts["paired_first_visual"].symlink_to(target)

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_visual_asset_live_pin_invalid"
    ]
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_signature_only_png_when_resigned(
    tmp_path,
):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    artifacts["paired_first_visual"].write_bytes(
        b"\x89PNG\r\n\x1a\nnot-a-decoded-image"
    )
    bad_pin = _content_pin(
        artifacts["paired_root"], artifacts["paired_first_visual"]
    )
    audit = json.loads(
        artifacts["paired_audit"].read_text(encoding="utf-8")
    )
    frame = next(
        value
        for value in audit["assets"]["frames"].values()
        if value["source"]["path"] == bad_pin["path"]
    )
    frame["source"] = bad_pin
    frame["origin"] = {**bad_pin, "path": frame["origin"]["path"]}
    artifacts["paired_audit"].write_text(
        json.dumps(audit), encoding="utf-8"
    )
    _rebind_rlivit_paired_audit_fixture(tmp_path, artifacts)
    _refresh_rlivit_paired_fixture_contract(artifacts)

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_visual_asset_decode_contract_invalid"
    ]
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_intermediate_package_symlink(tmp_path):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    store = campaign.EvidenceStore(tmp_path)
    rlivit, facts = campaign._build_rlivit_section(
        store, tmp_path / "validation/results"
    )
    assert facts["accepted"] is True
    rlivit_root = tmp_path / "validation/results/rlivit"
    outside = tmp_path / "outside-rlivit"
    rlivit_root.rename(outside)
    rlivit_root.symlink_to(outside, target_is_directory=True)

    with mock.patch.multiple(
        campaign, **artifacts["paired_expected_contract"]
    ):
        section, paired_facts = campaign._build_rlivit_paired_error_audit_section(
            store,
            tmp_path / "validation/results",
            rlivit,
        )

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_package_root_unsafe"
    ]
    assert paired_facts["accepted"] is False


def test_rlivit_paired_audit_holds_package_fd_across_rename_swap(
    tmp_path, monkeypatch
):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    original = campaign._rlivit_paired_secure_bytes
    swapped = False

    def rename_after_receipt(root, relative_path, **kwargs):
        nonlocal swapped
        result = original(root, relative_path, **kwargs)
        if (
            not swapped
            and root == artifacts["paired_root"]
            and relative_path == "receipt.json"
        ):
            swapped = True
            held_package = tmp_path / "held-paired-package"
            redirect = tmp_path / "redirect-paired-package"
            artifacts["paired_root"].rename(held_package)
            redirect.mkdir()
            artifacts["paired_root"].symlink_to(
                redirect, target_is_directory=True
            )
        return result

    monkeypatch.setattr(
        campaign, "_rlivit_paired_secure_bytes", rename_after_receipt
    )

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert swapped is True
    assert section["state"] == "complete"
    assert section["proof"]["package_root_fd_snapshot_verified"] is True
    assert facts["accepted"] is True


def test_rlivit_paired_audit_rejects_duplicate_json_key(tmp_path):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    raw = artifacts["paired_receipt"].read_text(encoding="utf-8")
    artifacts["paired_receipt"].write_text(
        raw[:-1] + ', "status": "complete"}', encoding="utf-8"
    )

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert section["state"] == "invalid"
    assert section["reasons"] == ["rlivit_paired_audit_receipt_invalid"]
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_nonfinite_json_number(tmp_path):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    raw = artifacts["paired_receipt"].read_text(encoding="utf-8")
    assert '"rendered": true' in raw
    artifacts["paired_receipt"].write_text(
        raw.replace('"rendered": true', '"rendered": NaN'), encoding="utf-8"
    )

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert section["state"] == "invalid"
    assert section["reasons"] == ["rlivit_paired_audit_receipt_invalid"]
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_path_traversal_even_when_rehashed(tmp_path):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    audit = json.loads(artifacts["paired_audit"].read_text(encoding="utf-8"))
    first_frame = audit["assets"]["frames"][sorted(audit["assets"]["frames"])[0]]
    first_frame["source"]["path"] = "../escape.png"
    artifacts["paired_audit"].write_text(json.dumps(audit), encoding="utf-8")
    _rebind_rlivit_paired_audit_fixture(tmp_path, artifacts)

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_receipt_contract_invalid"
    ]
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_schema_contract_change_when_rehashed(
    tmp_path,
):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    audit = json.loads(artifacts["paired_audit"].read_text(encoding="utf-8"))
    audit["schema_version"] = "deepsafe.rlivit-paired-error-audit/v2"
    artifacts["paired_audit"].write_text(json.dumps(audit), encoding="utf-8")
    _rebind_rlivit_paired_audit_fixture(tmp_path, artifacts)

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_receipt_contract_invalid"
    ]
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_live_schema_pin_tamper(tmp_path):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    artifacts["paired_schema"].write_bytes(
        artifacts["paired_schema"].read_bytes() + b"\n"
    )

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_control_live_pin_invalid"
    ]
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_live_implementation_pin_tamper(tmp_path):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    artifacts["paired_implementation"].write_bytes(
        artifacts["paired_implementation"].read_bytes() + b"\n# tamper\n"
    )

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_control_live_pin_invalid"
    ]
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_oversize_control_even_when_repinned(
    tmp_path,
):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    with artifacts["paired_implementation"].open("wb") as handle:
        handle.truncate(campaign.RLIVIT_PAIRED_CONTROL_MAX_BYTES + 1)
    implementation_pin = _content_pin(
        tmp_path, artifacts["paired_implementation"]
    )
    audit = json.loads(artifacts["paired_audit"].read_text(encoding="utf-8"))
    audit["lineage"]["controls"]["implementation"] = implementation_pin
    artifacts["paired_audit"].write_text(json.dumps(audit), encoding="utf-8")
    receipt = json.loads(
        artifacts["paired_receipt"].read_text(encoding="utf-8")
    )
    receipt["controls"]["implementation"] = implementation_pin
    artifacts["paired_receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    _rebind_rlivit_paired_audit_fixture(tmp_path, artifacts)

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_receipt_contract_invalid"
    ]
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_receipt_fingerprint_tamper(tmp_path):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    receipt = json.loads(
        artifacts["paired_receipt"].read_text(encoding="utf-8")
    )
    receipt["fingerprint_sha256"] = "0" * 64
    artifacts["paired_receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_receipt_contract_invalid"
    ]
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_asset_manifest_tamper_when_rehashed(
    tmp_path,
):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    receipt = json.loads(
        artifacts["paired_receipt"].read_text(encoding="utf-8")
    )
    receipt["asset_manifest_fingerprint_sha256"] = "0" * 64
    _rlivit_repin(receipt)
    artifacts["paired_receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    section, facts = _build_paired_fixture_projection(tmp_path, artifacts)

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_receipt_contract_invalid"
    ]
    assert facts["accepted"] is False


def test_rlivit_paired_audit_rejects_public_private_metric_crossbind_tamper(
    tmp_path,
):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    store = campaign.EvidenceStore(tmp_path)
    rlivit, facts = campaign._build_rlivit_section(
        store, tmp_path / "validation/results"
    )
    assert facts["accepted"] is True
    rlivit["profiles"]["640"]["metrics"]["overall"]["ap_101_point"] += 0.01

    with mock.patch.multiple(
        campaign, **artifacts["paired_expected_contract"]
    ):
        section, paired_facts = campaign._build_rlivit_paired_error_audit_section(
            store,
            tmp_path / "validation/results",
            rlivit,
        )

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_profile_metric_cross_bind_invalid"
    ]
    assert paired_facts["accepted"] is False


def test_rlivit_paired_audit_rejects_live_batch_job_map_mismatch(tmp_path):
    artifacts = _populate_rlivit_paired_audit_fixture(tmp_path)
    store = campaign.EvidenceStore(tmp_path)
    rlivit, facts = campaign._build_rlivit_section(
        store, tmp_path / "validation/results"
    )
    assert facts["accepted"] is True

    batch = json.loads(
        artifacts["final_receipt"].read_text(encoding="utf-8")
    )
    batch["job_receipts"]["rlivit:002:640"] = batch["job_receipts"][
        "rlivit:002:960"
    ]
    _rlivit_repin(batch)
    artifacts["final_receipt"].write_text(
        json.dumps(batch), encoding="utf-8"
    )
    batch_pin = _content_pin(tmp_path, artifacts["final_receipt"])

    audit = json.loads(
        artifacts["paired_audit"].read_text(encoding="utf-8")
    )
    audit["lineage"]["batch_receipt"] = batch_pin
    audit["lineage"]["batch_receipt_fingerprint_sha256"] = batch[
        "fingerprint_sha256"
    ]
    artifacts["paired_audit"].write_text(
        json.dumps(audit), encoding="utf-8"
    )
    receipt = json.loads(
        artifacts["paired_receipt"].read_text(encoding="utf-8")
    )
    receipt["input_batch_receipt"] = batch_pin
    artifacts["paired_receipt"].write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    _rebind_rlivit_paired_audit_fixture(tmp_path, artifacts)
    _refresh_rlivit_paired_fixture_contract(artifacts)

    with mock.patch.multiple(
        campaign, **artifacts["paired_expected_contract"]
    ):
        section, paired_facts = campaign._build_rlivit_paired_error_audit_section(
            store,
            tmp_path / "validation/results",
            rlivit,
        )

    assert section["state"] == "invalid"
    assert section["reasons"] == [
        "rlivit_paired_audit_batch_receipt_contract_invalid"
    ]
    assert paired_facts["accepted"] is False


def test_rlivit_canonical_fingerprint_is_recomputed_live(tmp_path):
    status = _rlivit_public_status_value("complete")
    status["progress"]["completed_jobs"] = 79
    _write_json(
        tmp_path,
        "validation/results/rlivit/current/status.json",
        status,
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )

    section = report["campaigns"]["rlivit_ground_truth"]
    assert section["state"] == "invalid"
    assert section["schema_contract_valid"] is True
    assert section["canonical_fingerprint_valid"] is False
    assert section["evidence_complete"] is False


def test_rlivit_complete_requires_all_six_pathless_pins_and_two_metric_profiles(
    tmp_path,
):
    status = _rlivit_public_status_value("complete")
    status["evidence"]["batch_receipt"] = None
    status["profiles"]["960"]["metrics"] = None
    _rlivit_repin(status)
    _write_json(
        tmp_path,
        "validation/results/rlivit/current/status.json",
        status,
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )

    section = report["campaigns"]["rlivit_ground_truth"]
    requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "rlivit_ground_truth_80_jobs"
    )
    assert section["state"] == "invalid"
    assert section["evidence_complete"] is False
    assert requirement["state"] == "invalid"


def test_rlivit_complete_rejects_contradictory_partition_confusion_counts(
    tmp_path,
):
    status = _rlivit_public_status_value("complete")
    day = status["profiles"]["640"]["metrics"]["daytime"]["day"]
    day.update(
        tp=3421,
        fp=0,
        fn=1,
        precision=1.0,
        recall=round(3421 / 3422, 6),
    )
    day["f1"] = round(
        2 * day["precision"] * day["recall"]
        / (day["precision"] + day["recall"]),
        6,
    )
    _rlivit_repin(status)
    _write_json(
        tmp_path,
        "validation/results/rlivit/current/status.json",
        status,
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )

    section = report["campaigns"]["rlivit_ground_truth"]
    assert section["state"] == "invalid"
    assert section["evidence_complete"] is False


def test_rlivit_profile_metrics_accept_nonadditive_area_and_height_strata(
    tmp_path,
):
    metrics = _rlivit_profile_metrics()
    for partition_name in ("coco_area", "height_bands"):
        for row in metrics[partition_name].values():
            ground_truth = row["ground_truth"]
            row.update(
                tp=ground_truth - 1,
                fp=1,
                fn=1,
                precision=round((ground_truth - 1) / ground_truth, 6),
                recall=round((ground_truth - 1) / ground_truth, 6),
                ap_101_point=0.75,
            )
            row["f1"] = row["precision"]
    _populate_rlivit_complete_evidence(tmp_path, profile_metrics=metrics)

    section, _facts = campaign._build_rlivit_section(
        campaign.EvidenceStore(tmp_path),
        tmp_path / "validation/results",
    )

    assert section["state"] == "complete"
    assert section["evidence_complete"] is True
    assert all(section["private_completion_proof"].values())


def test_rlivit_complete_rejects_contradictory_location_confusion_counts(
    tmp_path,
):
    status = _rlivit_public_status_value("complete")
    location = status["profiles"]["640"]["metrics"]["locations"]["0"]
    location.update(
        tp=427,
        fp=0,
        fn=1,
        precision=1.0,
        recall=round(427 / 428, 6),
    )
    location["f1"] = round(
        2 * location["precision"] * location["recall"]
        / (location["precision"] + location["recall"]),
        6,
    )
    _rlivit_repin(status)
    _write_json(
        tmp_path,
        "validation/results/rlivit/current/status.json",
        status,
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )

    assert report["campaigns"]["rlivit_ground_truth"]["state"] == "invalid"


def test_rlivit_public_only_rehashed_complete_claim_cannot_pass(tmp_path):
    _write_json(
        tmp_path,
        "validation/results/rlivit/current/status.json",
        _rlivit_public_status_value("complete"),
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )
    section = report["campaigns"]["rlivit_ground_truth"]
    requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "rlivit_ground_truth_80_jobs"
    )

    assert section["state"] == "invalid"
    assert section["reasons"] == ["rlivit_private_completion_proof_invalid"]
    assert section["private_completion_proof"]["complete_claim_verified"] is False
    assert requirement["state"] == "invalid"


def test_rlivit_complete_private_proof_is_pathless_and_all_true(tmp_path):
    _populate_rlivit_complete_evidence(tmp_path)

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )
    section = report["campaigns"]["rlivit_ground_truth"]
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert section["state"] == "complete"
    assert all(section["private_completion_proof"].values())
    assert "fixture-operator" not in encoded
    assert "9" * 64 not in encoded
    assert "rlivit/runs/" not in encoded


def test_rlivit_missing_live_job_receipt_invalidates_completion(tmp_path):
    artifacts = _populate_rlivit_complete_evidence(tmp_path)
    artifacts["job_receipts"]["rlivit:002:640"].unlink()

    section, _facts = campaign._build_rlivit_section(
        campaign.EvidenceStore(tmp_path), tmp_path / "validation/results"
    )

    assert section["state"] == "invalid"
    assert section["private_completion_proof"][
        "job_receipts_live_pins_verified"
    ] is False


def test_rlivit_post_run_job_receipt_tamper_invalidates_completion(tmp_path):
    artifacts = _populate_rlivit_complete_evidence(tmp_path)
    path = artifacts["job_receipts"]["rlivit:002:640"]
    path.write_bytes(path.read_bytes() + b" ")

    section, _facts = campaign._build_rlivit_section(
        campaign.EvidenceStore(tmp_path), tmp_path / "validation/results"
    )

    assert section["state"] == "invalid"
    assert section["private_completion_proof"][
        "job_receipts_live_pins_verified"
    ] is False


def test_rlivit_symlinked_live_job_receipt_is_not_followed(tmp_path):
    artifacts = _populate_rlivit_complete_evidence(tmp_path)
    path = artifacts["job_receipts"]["rlivit:002:640"]
    target = artifacts["job_receipts"]["rlivit:002:960"]
    path.unlink()
    path.symlink_to(target)

    section, _facts = campaign._build_rlivit_section(
        campaign.EvidenceStore(tmp_path), tmp_path / "validation/results"
    )

    assert section["state"] == "invalid"
    assert section["private_completion_proof"][
        "job_receipts_live_pins_verified"
    ] is False


def test_rlivit_symlinked_private_runs_component_is_not_followed(tmp_path):
    _populate_rlivit_complete_evidence(tmp_path)
    runs = tmp_path / "validation/results/rlivit/runs"
    outside = tmp_path / "private-runs-target"
    runs.rename(outside)
    runs.symlink_to(outside, target_is_directory=True)

    section, _facts = campaign._build_rlivit_section(
        campaign.EvidenceStore(tmp_path), tmp_path / "validation/results"
    )

    assert section["state"] == "invalid"
    assert section["private_completion_proof"][
        "final_receipt_live_pin_verified"
    ] is False


def test_rlivit_symlinked_authorization_parent_is_not_followed(tmp_path):
    _populate_rlivit_complete_evidence(tmp_path)
    authorizations = tmp_path / "validation/authorizations"
    outside = tmp_path / "private-authorization-target"
    authorizations.rename(outside)
    authorizations.symlink_to(outside, target_is_directory=True)

    section, _facts = campaign._build_rlivit_section(
        campaign.EvidenceStore(tmp_path), tmp_path / "validation/results"
    )

    assert section["state"] == "invalid"
    assert section["private_completion_proof"][
        "authorization_session_state_live_pins_verified"
    ] is False


def test_rlivit_rehashed_final_receipt_cannot_repin_one_job(tmp_path):
    artifacts = _populate_rlivit_complete_evidence(tmp_path)
    job_id = "rlivit:002:640"
    job_path = artifacts["job_receipts"][job_id]
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["post_run_forgery"] = True
    _rlivit_repin(job)
    job_path.write_text(json.dumps(job), encoding="utf-8")
    forged_job_pin = _content_pin(tmp_path, job_path)

    final = json.loads(artifacts["final_receipt"].read_text(encoding="utf-8"))
    final["job_receipts"][job_id] = forged_job_pin
    _rlivit_repin(final)
    artifacts["final_receipt"].write_text(json.dumps(final), encoding="utf-8")
    final_pin = _content_pin(tmp_path, artifacts["final_receipt"])
    status = json.loads(artifacts["status"].read_text(encoding="utf-8"))
    status["evidence"]["batch_receipt"] = _rlivit_pathless(final_pin)
    _rlivit_repin(status)
    artifacts["status"].write_text(json.dumps(status), encoding="utf-8")

    section, _facts = campaign._build_rlivit_section(
        campaign.EvidenceStore(tmp_path), tmp_path / "validation/results"
    )

    assert section["state"] == "invalid"
    assert section["private_completion_proof"]["complete_claim_verified"] is False


def test_rlivit_live_authorization_must_retain_immutable_mode(tmp_path):
    artifacts = _populate_rlivit_complete_evidence(tmp_path)
    artifacts["authorization"].chmod(0o640)

    section, _facts = campaign._build_rlivit_section(
        campaign.EvidenceStore(tmp_path), tmp_path / "validation/results"
    )

    assert section["state"] == "invalid"
    assert section["private_completion_proof"][
        "authorization_session_state_live_pins_verified"
    ] is False


def test_rlivit_rebound_state_cannot_hide_missing_gpu_process_start(tmp_path):
    artifacts = _populate_rlivit_complete_evidence(tmp_path)
    state = json.loads(artifacts["state"].read_text(encoding="utf-8"))
    state["gpu_process_started_jobs"].pop()
    artifacts["state"].write_text(json.dumps(state), encoding="utf-8")
    state_pin = _content_pin(tmp_path, artifacts["state"])
    final = json.loads(artifacts["final_receipt"].read_text(encoding="utf-8"))
    final["state"] = state_pin
    _rlivit_repin(final)
    artifacts["final_receipt"].write_text(json.dumps(final), encoding="utf-8")
    final_pin = _content_pin(tmp_path, artifacts["final_receipt"])
    status = json.loads(artifacts["status"].read_text(encoding="utf-8"))
    status["evidence"]["batch_receipt"] = _rlivit_pathless(final_pin)
    _rlivit_repin(status)
    artifacts["status"].write_text(json.dumps(status), encoding="utf-8")

    section, _facts = campaign._build_rlivit_section(
        campaign.EvidenceStore(tmp_path), tmp_path / "validation/results"
    )

    assert section["state"] == "invalid"
    assert section["private_completion_proof"][
        "authorization_session_state_live_pins_verified"
    ] is False


def test_rlivit_rebound_job_outside_authorization_window_is_rejected(tmp_path):
    artifacts = _populate_rlivit_complete_evidence(tmp_path)
    job_id = "rlivit:199:960"
    job_path = artifacts["job_receipts"][job_id]
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["created_at_utc"] = "2026-07-16T20:00:00+00:00"
    _rlivit_repin(job)
    job_path.write_text(json.dumps(job), encoding="utf-8")
    _rebind_rlivit_job_chain(tmp_path, artifacts, job_id)

    section, _facts = campaign._build_rlivit_section(
        campaign.EvidenceStore(tmp_path), tmp_path / "validation/results"
    )

    assert section["state"] == "invalid"
    assert section["private_completion_proof"]["authorized_timeline_verified"] is False


def test_rlivit_mp4_public_receipt_pin_must_match_status(tmp_path):
    artifacts = _populate_rlivit_complete_evidence(tmp_path)
    public_mp4 = json.loads(
        artifacts["mp4_public_receipt"].read_text(encoding="utf-8")
    )
    public_mp4["batch_receipt_pin"]["sha256"] = "d" * 64
    _rlivit_repin(public_mp4)
    artifacts["mp4_public_receipt"].write_text(
        json.dumps(public_mp4), encoding="utf-8"
    )

    section, _facts = campaign._build_rlivit_section(
        campaign.EvidenceStore(tmp_path), tmp_path / "validation/results"
    )

    assert section["state"] == "invalid"
    assert section["private_completion_proof"][
        "mp4_receipt_status_pin_cross_bound"
    ] is False


def test_rlivit_noncomplete_status_does_not_scan_private_runs(tmp_path):
    _write_json(
        tmp_path,
        "validation/results/rlivit/current/status.json",
        _rlivit_public_status_value("blocked"),
    )
    outside = tmp_path / "outside-private-runs"
    outside.mkdir()
    runs = tmp_path / "validation/results/rlivit/runs"
    runs.parent.mkdir(parents=True, exist_ok=True)
    runs.symlink_to(outside, target_is_directory=True)

    store = campaign.EvidenceStore(tmp_path)
    section, _facts = campaign._build_rlivit_section(
        store, tmp_path / "validation/results"
    )

    assert section["state"] == "blocked"
    assert section["evidence_ids"] == ["rlivit_public_status"]
    assert "rlivit_mp4_public_receipt" not in store.entries
    assert section["private_completion_proof"]["complete_claim_verified"] is False


def test_rlivit_private_reader_enforces_file_and_total_byte_caps(tmp_path):
    small = _write_json(tmp_path, "private/small.json", {})
    oversized = tmp_path / "private/oversized.json"
    oversized.write_bytes(b"{" + b" " * (512 * 1024) + b"}")

    small_budget = {
        "bytes": campaign.RLIVIT_PRIVATE_TOTAL_JSON_BYTES - 1,
        "exceeded": False,
    }
    assert campaign._rlivit_private_json(
        small,
        containment_root=tmp_path / "private",
        budget=small_budget,
    ) is None
    assert small_budget["exceeded"] is True
    assert campaign._rlivit_private_json(
        oversized,
        containment_root=tmp_path / "private",
        budget={"bytes": 0, "exceeded": False},
    ) is None


def test_rlivit_private_run_search_is_bounded_to_256_directories(tmp_path):
    _populate_rlivit_complete_evidence(tmp_path)
    runs = tmp_path / "validation/results/rlivit/runs"
    for index in range(256):
        candidate = runs / f"{index:064x}"
        if not candidate.exists():
            candidate.mkdir()

    section, _facts = campaign._build_rlivit_section(
        campaign.EvidenceStore(tmp_path), tmp_path / "validation/results"
    )

    assert section["state"] == "invalid"
    assert section["private_completion_proof"][
        "final_receipt_live_pin_verified"
    ] is False


def test_rlivit_private_fields_are_rejected_without_report_leakage(tmp_path):
    secret = "/home/private/runs/secret-nonce"
    status = _rlivit_public_status_value("complete")
    status["profiles"]["640"]["metrics"]["overall"]["private_source"] = secret
    status["profiles"]["640"]["metrics"]["overall"]["command"] = [
        "docker",
        "run",
        secret,
    ]
    _rlivit_repin(status)
    _write_json(
        tmp_path,
        "validation/results/rlivit/current/status.json",
        status,
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert report["campaigns"]["rlivit_ground_truth"]["state"] == "invalid"
    assert secret not in encoded
    assert "operator_identity" not in encoded
    assert '"command"' not in encoded


def test_rlivit_public_status_symlink_is_not_followed(tmp_path):
    target = _write_json(
        tmp_path,
        "validation/results/rlivit/private-status.json",
        _rlivit_public_status_value("complete"),
    )
    public = tmp_path / "validation/results/rlivit/current/status.json"
    public.parent.mkdir(parents=True, exist_ok=True)
    public.symlink_to(target)

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )

    section = report["campaigns"]["rlivit_ground_truth"]
    evidence = next(
        item for item in report["evidence"] if item["id"] == "rlivit_public_status"
    )
    assert section["state"] == "invalid"
    assert section["reasons"] == ["rlivit_status_unsafe_symlink"]
    assert evidence["state"] == "unsafe_symlink"


def test_complete_caviar_with_draft_policy_is_not_a_quality_pass(
    tmp_path, monkeypatch
):
    implementation = tmp_path / "validation/site_distance_evaluation.py"
    implementation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(site_distance.__file__), implementation)
    monkeypatch.setattr(site_distance, "__file__", str(implementation))
    results = _populate_accepted_campaign(tmp_path)
    draft = json.loads(
        (campaign.PROJECT_ROOT / "validation/person-quality-policy.draft.json").read_text(
            encoding="utf-8"
        )
    )
    _write_json(
        tmp_path, "validation/person-quality-policy.draft.json", draft
    )
    _write_json(
        tmp_path,
        "validation/results/caviar/person-quality-decision.json",
        {
            "schema_version": "deepsafe.person-quality-decision/v1",
            "status": "quality_gate_passed",
            "private_path": "/host/forged/decision.json",
        },
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
    )

    quality_section = report["campaigns"]["person_detection_quality"]
    caviar = report["campaigns"]["caviar_ground_truth"]
    assert caviar["evidence_complete"] is False
    assert caviar["metrics_withheld"] is True
    assert caviar["aggregate_result_jobs"] == 0
    assert caviar["profiles"] == {}
    assert caviar["official_session"]["state"] == "not_authorized"
    assert caviar["official_session"]["legacy_public_artifacts_ignored"] is True
    assert quality_section["state"] == "draft_unapproved"
    assert quality_section["accepted"] is False
    assert quality_section["live_cpu_recomputed"] is False
    assert quality_section["evaluator"]["prewritten_decision_used"] is False
    separation = next(
        item
        for item in report["requirements"]
        if item["id"] == "ground_truth_manual_separation"
    )
    assert separation["state"] == "pass"
    assert "person_detection_quality" in report["decision"]["failed_required_gates"]
    assert report["decision"]["final_claim_allowed"] is False
    assert all(
        "person-quality-decision" not in item["path"]
        for item in report["evidence"]
    )


def test_approved_quality_threshold_failure_stays_unproven(tmp_path, monkeypatch):
    implementation = tmp_path / "validation/site_distance_evaluation.py"
    implementation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(site_distance.__file__), implementation)
    monkeypatch.setattr(site_distance, "__file__", str(implementation))
    results = _populate_accepted_campaign(tmp_path)
    policy_path = _install_live_quality_decision(
        tmp_path, results, monkeypatch, passed=False
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
        person_quality_policy_path=policy_path,
    )

    quality_section = report["campaigns"]["person_detection_quality"]
    requirement = next(
        item for item in report["requirements"] if item["id"] == "person_detection_quality"
    )
    assert quality_section["state"] == "failed"
    assert quality_section["evidence_complete"] is True
    assert quality_section["live_cpu_recomputed"] is True
    assert quality_section["accepted"] is False
    assert requirement["state"] == "unproven"
    assert report["decision"]["final_claim_allowed"] is False


def _build_endurance_fixture_section(
    root: Path,
    *,
    verifier=_fixture_throughput_floor_verifier,
    raw_verifier=_fixture_endurance_raw_replay,
) -> tuple[dict, dict]:
    with mock.patch.object(
        campaign, "_live_verify_endurance_throughput_floor", verifier
    ), mock.patch.object(
        campaign, "_endurance_replay_attempt_raw", raw_verifier
    ):
        section, _hardware, facts = campaign._build_endurance_section(
            campaign.EvidenceStore(root),
            root / "validation/results",
        )
    return section, facts


def test_endurance_workstation_managed_policy_matches_current_producer_contract(
    tmp_path,
):
    _populate_accepted_endurance(tmp_path)
    resolved = json.loads(
        (
            tmp_path
            / "validation/results/endurance/current/campaign-resolved.json"
        ).read_text(encoding="utf-8")
    )

    assert campaign._endurance_resolved_reasons(resolved) == []
    assert campaign._endurance_floor_runtime_reasons(
        resolved["throughput_floor"]["source_runtime_identity"]
    ) == []


def test_reporter_replay_renderer_byte_matches_active_supervisor_topology(
    tmp_path, monkeypatch
):
    _populate_accepted_endurance(tmp_path)
    resolved = json.loads(
        (
            tmp_path
            / "validation/results/endurance/current/campaign-resolved.json"
        ).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(scene_run_matrix, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        endurance_supervisor, "validate_person_profile", lambda _profile: None
    )

    reporter_text = campaign._endurance_render_config_text(resolved, 640, tmp_path)
    supervisor_text = endurance_supervisor.render_deepstream_config_text(resolved, 640)

    assert reporter_text == supervisor_text
    assert "num-sources=" not in reporter_text
    assert reporter_text.count("\ntype=2\n") == 12
    assert reporter_text.count("\ntype=1\n") == 12
    for source_id in range(12):
        assert f"[source{source_id}]" in reporter_text
        assert f"camera-id={source_id}" in reporter_text
        assert f"[sink{source_id}]" in reporter_text
        assert f"source-id={source_id}" in reporter_text


def test_endurance_floor_runtime_requires_pci_bus_identity(tmp_path):
    _populate_accepted_endurance(tmp_path)
    resolved = json.loads(
        (
            tmp_path
            / "validation/results/endurance/current/campaign-resolved.json"
        ).read_text(encoding="utf-8")
    )
    runtime = copy.deepcopy(
        resolved["throughput_floor"]["source_runtime_identity"]
    )
    runtime["gpu_identity"].pop("pci.bus_id")

    assert "throughput_floor_runtime_gpu_identity_invalid" in (
        campaign._endurance_floor_runtime_reasons(runtime)
    )


@pytest.mark.parametrize(
    ("field", "legacy_value", "reason"),
    [
        (
            "power_limit_drop_action",
            "immediate_safety_abort",
            "campaign_resolved_power_limit_action_invalid",
        ),
        (
            "sustained_slowdown_action",
            "safety_abort",
            "campaign_resolved_slowdown_action_invalid",
        ),
    ],
)
def test_endurance_workstation_policy_rejects_legacy_actions(
    tmp_path, field, legacy_value, reason
):
    _populate_accepted_endurance(tmp_path)
    resolved = json.loads(
        (
            tmp_path
            / "validation/results/endurance/current/campaign-resolved.json"
        ).read_text(encoding="utf-8")
    )
    resolved["power_safety"][field] = legacy_value

    reasons = campaign._endurance_resolved_reasons(resolved)

    assert "campaign_resolved_power_safety_contract_invalid" in reasons
    assert reason in reasons


def test_endurance_attempt_diagnostic_counters_are_mode_aware(tmp_path):
    _populate_accepted_endurance(tmp_path)
    current = tmp_path / "validation/results/endurance/current"
    resolved = json.loads(
        (current / "campaign-resolved.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads(
        (current / "checkpoint.json").read_text(encoding="utf-8")
    )
    attempt = copy.deepcopy(checkpoint["segments"][0]["attempts"][0])
    attempt["gpu_safety"].update(
        power_limit_drop_samples=2,
        slowdown_active_samples=3,
        max_consecutive_slowdown_samples=2,
    )
    planned = campaign._endurance_expected_plan()[0]
    gpu_name = resolved["throughput_floor"]["source_runtime_identity"][
        "gpu_identity"
    ]["name"]

    workstation_reasons = campaign._endurance_attempt_reasons(
        attempt, planned, resolved, 1, gpu_name
    )

    assert "attempt_gpu_power_limit_drop_seen" not in workstation_reasons
    assert "attempt_gpu_slowdown_seen" not in workstation_reasons
    assert (
        "attempt_gpu_safety_diagnostic_counters_invalid" not in workstation_reasons
    )

    legacy_resolved = copy.deepcopy(resolved)
    legacy_resolved["power_safety"]["operating_policy_mode"] = "legacy_strict"
    legacy_attempt = copy.deepcopy(attempt)
    legacy_attempt["gpu_safety"]["policy"] = legacy_resolved["power_safety"]
    legacy_reasons = campaign._endurance_attempt_reasons(
        legacy_attempt, planned, legacy_resolved, 1, gpu_name
    )

    assert "attempt_gpu_power_limit_drop_seen" in legacy_reasons
    assert "attempt_gpu_slowdown_seen" in legacy_reasons


def test_reporter_accepts_zero_byte_attempt_artifact_only_with_explicit_empty_pin(
    tmp_path,
):
    artifact = tmp_path / "validation/results/endurance/empty.log"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"")
    pin = {
        "path": str(artifact.relative_to(tmp_path)),
        "size_bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }

    assert campaign._endurance_verify_pin(
        campaign.EvidenceStore(tmp_path),
        "empty_without_declaration",
        pin,
        expected_path=artifact,
        allow_explicit_empty=True,
    ) is False
    assert campaign._endurance_verify_pin(
        campaign.EvidenceStore(tmp_path),
        "declared_stable_empty",
        {**pin, "allow_empty": True},
        expected_path=artifact,
        allow_explicit_empty=True,
    ) is True
    assert campaign._endurance_verify_pin(
        campaign.EvidenceStore(tmp_path),
        "empty_not_allowed_for_healthy_attempt",
        {**pin, "allow_empty": True},
        expected_path=artifact,
        allow_explicit_empty=False,
    ) is False


def _mutate_latest_endurance_attempt(
    root: Path, segment_index: int, mutation
) -> None:
    checkpoint_path = root / "validation/results/endurance/current/checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record = checkpoint["segments"][segment_index]
    attempt = record["attempts"][-1]
    mutation(attempt)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    status_path = (
        root
        / "validation/results/endurance/current/segments"
        / record["segment_id"]
        / f"attempt-{attempt['attempt']:02d}/status.json"
    )
    status_path.write_text(json.dumps(attempt), encoding="utf-8")


def _coherently_repin_endurance_attempt(
    root: Path,
    segment_index: int,
    mutation,
) -> tuple[Path, dict]:
    """Rewrite the full status -> receipt -> checkpoint pin chain after mutation."""

    checkpoint_path = root / "validation/results/endurance/current/checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record = checkpoint["segments"][segment_index]
    status = record["attempts"][-1]
    attempt_dir = (
        root
        / "validation/results/endurance/current/segments"
        / record["segment_id"]
        / f"attempt-{status['attempt']:02d}"
    )
    mutation(status, attempt_dir)
    status_path = attempt_dir / "status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    receipt_path = attempt_dir / "attempt-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_pins"] = status["artifact_pins"]
    receipt["artifact_pins_sha256"] = campaign._endurance_json_sha256(
        status["artifact_pins"]
    )
    receipt["status_pin"] = _content_pin(root, status_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    record["attempts"][-1] = status
    record["attempt_receipts"][-1] = _content_pin(root, receipt_path)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    return attempt_dir, status


def _write_small_raw_endurance_attempt(attempt_dir: Path) -> None:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    header = "**PERF:  " + "\t".join(
        f"FPS {stream_id} (Avg)" for stream_id in range(12)
    )
    rows = []
    for value in (10.0, 12.0):
        rows.append(
            "**PERF:  "
            + "\t".join(f"{value:.2f} ({value:.2f})" for _ in range(12))
        )
    (attempt_dir / "deepstream.log").write_text(
        "\n".join([header, *rows]) + "\n", encoding="utf-8"
    )
    with (attempt_dir / "perf.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["elapsed_seconds", "aggregate_fps", "per_stream_mean_fps"]
        )
        writer.writerow([5, 120, 10])
        writer.writerow([10, 144, 12])
    with (attempt_dir / "gpu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "gpu_name",
                "gpu_utilization_percent",
                "memory_used_mib",
                "temperature_c",
                "power_draw_w",
            ]
        )
        for sample in range(10):
            writer.writerow(
                [
                    f"2026/07/16 00:00:{sample:02d}.000",
                    "fixture GPU",
                    60,
                    1200,
                    65,
                    85,
                ]
            )


def _write_raw_perf_window_attempt(
    attempt_dir: Path,
    *,
    elapsed_values: list[float],
    duration_seconds: int,
) -> None:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    header = "**PERF:  " + "\t".join(
        f"FPS {stream_id} (Avg)" for stream_id in range(12)
    )
    log_rows: list[str] = []
    csv_rows: list[list[float]] = []
    for index, elapsed in enumerate(elapsed_values):
        per_stream = float(10 + index % 5)
        aggregate = per_stream * 12
        log_rows.append(
            "**PERF:  "
            + "\t".join(
                f"{per_stream:.2f} ({per_stream:.2f})" for _ in range(12)
            )
        )
        csv_rows.append([elapsed, aggregate, per_stream])
    (attempt_dir / "deepstream.log").write_text(
        "\n".join([header, *log_rows]) + "\n", encoding="utf-8"
    )
    with (attempt_dir / "perf.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["elapsed_seconds", "aggregate_fps", "per_stream_mean_fps"]
        )
        writer.writerows(csv_rows)
    with (attempt_dir / "gpu.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "gpu_name",
                "gpu_utilization_percent",
                "memory_used_mib",
                "temperature_c",
                "power_draw_w",
            ]
        )
        for sample in range(duration_seconds):
            writer.writerow(
                [
                    f"fixture-{sample:06d}",
                    "fixture GPU",
                    60,
                    1200,
                    65,
                    85,
                ]
            )


def _replay_perf_window(attempt_dir: Path, *, duration_seconds: int) -> dict:
    return campaign._endurance_replay_attempt_raw(
        attempt_dir,
        duration_seconds=duration_seconds,
        perf_interval_seconds=5,
        max_log_bytes=1024 * 1024,
        minimum_coverage_fraction=0.95,
        startup_grace_seconds=10,
        perf_stall_timeout_seconds=20,
    )


def test_endurance_production_raw_replay_parses_log_perf_and_gpu_csv(tmp_path):
    attempt_dir = tmp_path / "attempt"
    _write_small_raw_endurance_attempt(attempt_dir)

    replay = campaign._endurance_replay_attempt_raw(
        attempt_dir,
        duration_seconds=10,
        perf_interval_seconds=5,
        max_log_bytes=1024 * 1024,
        minimum_coverage_fraction=0.95,
        startup_grace_seconds=120,
        perf_stall_timeout_seconds=180,
    )

    assert replay["status"] == "verified"
    assert replay["throughput"]["status"] == "ok"
    assert replay["gpu"]["status"] == "ok"
    assert replay["perf_csv"]["rows"] == 2
    assert replay["throughput"]["aggregate_current_fps"]["p05"] > 0


def test_endurance_production_raw_replay_rejects_perf_csv_projection_drift(tmp_path):
    attempt_dir = tmp_path / "attempt"
    _write_small_raw_endurance_attempt(attempt_dir)
    perf_path = attempt_dir / "perf.csv"
    perf_path.write_text(
        "elapsed_seconds,aggregate_fps,per_stream_mean_fps\n"
        "5,132,11\n"
        "10,144,12\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="projection differs"):
        campaign._endurance_replay_attempt_raw(
            attempt_dir,
            duration_seconds=10,
            perf_interval_seconds=5,
            max_log_bytes=1024 * 1024,
            minimum_coverage_fraction=0.95,
            startup_grace_seconds=120,
            perf_stall_timeout_seconds=180,
        )


def test_endurance_raw_replay_status_comparison_is_exact(tmp_path):
    attempt_dir = tmp_path / "attempt"
    _write_small_raw_endurance_attempt(attempt_dir)
    replay = campaign._endurance_replay_attempt_raw(
        attempt_dir,
        duration_seconds=10,
        perf_interval_seconds=5,
        max_log_bytes=1024 * 1024,
        minimum_coverage_fraction=0.95,
        startup_grace_seconds=120,
        perf_stall_timeout_seconds=180,
    )
    status = {
        "throughput": copy.deepcopy(replay["throughput"]),
        "gpu": copy.deepcopy(replay["gpu"]),
        "artifact_pins": {
            name: {
                "size_bytes": observation["size_bytes"],
                "sha256": observation["sha256"],
            }
            for name, observation in replay["file_observations"].items()
        },
        "throughput_floor_evaluation": {
            "observed_aggregate_fps_p05": replay["throughput"][
                "aggregate_current_fps"
            ]["p05"]
        },
    }
    assert campaign._endurance_attempt_raw_reasons(status, replay) == []

    status["throughput"]["aggregate_current_fps"]["p05"] += 1
    status["gpu"]["metrics"]["memory_used_mib"]["mean"] += 1
    reasons = campaign._endurance_attempt_raw_reasons(status, replay)
    assert "attempt_status_throughput_differs_from_raw_log" in reasons
    assert "attempt_status_gpu_differs_from_raw_csv" in reasons


def test_endurance_raw_replay_accepts_95_percent_only_with_full_time_window(
    tmp_path,
):
    attempt_dir = tmp_path / "attempt"
    _write_raw_perf_window_attempt(
        attempt_dir,
        elapsed_values=[float(value) for value in range(5, 100, 5)],
        duration_seconds=100,
    )

    replay = _replay_perf_window(attempt_dir, duration_seconds=100)
    throughput = replay["throughput"]
    window = throughput["window_contract"]

    assert throughput["raw_parser_status"] == "insufficient_perf_window"
    assert throughput["status"] == "ok"
    assert window["valid_complete_rows"] == 19
    assert window["row_coverage_fraction"] == 0.95
    assert window["temporal_slot_coverage_fraction"] == 0.95
    assert window["first_elapsed_seconds"] == 5.0
    assert window["last_elapsed_seconds"] == 95.0
    assert window["maximum_observed_gap_seconds"] == 5.0
    assert window["failed_checks"] == []


@pytest.mark.parametrize(
    ("elapsed_values", "failed_check"),
    [
        ([float(value) for value in range(5, 95, 5)], "row_coverage"),
        (
            [5.0 + index * 0.01 for index in range(18)] + [95.0],
            "temporal_slot_coverage",
        ),
        (
            [5.0 + index * 0.01 for index in range(19)] + [95.0],
            "temporal_slot_coverage",
        ),
        (
            [5.0 + index * 0.2 for index in range(19)],
            "tail_endpoint",
        ),
        (
            [float(value) for value in range(5, 55, 5)]
            + [float(value) for value in range(75, 125, 5)],
            "maximum_gap",
        ),
    ],
    ids=[
        "below-count-coverage",
        "sparse",
        "clustered",
        "missing-tail",
        "stalled-gap",
    ],
)
def test_endurance_perf_window_rejects_count_or_time_distribution_forgeries(
    tmp_path, elapsed_values, failed_check
):
    attempt_dir = tmp_path / "attempt"
    _write_raw_perf_window_attempt(
        attempt_dir,
        elapsed_values=elapsed_values,
        duration_seconds=100,
    )
    parsed = campaign.benchmark_summarize.parse_perf(
        attempt_dir / "deepstream.log", 12, 100, 5
    )

    normalized = campaign._endurance_apply_perf_window_contract(
        parsed,
        attempt_dir / "perf.csv",
        expected_streams=12,
        duration_seconds=100,
        perf_interval_seconds=5,
        minimum_coverage_fraction=0.95,
        startup_grace_seconds=10,
        perf_stall_timeout_seconds=20,
    )

    assert normalized["status"] == "invalid_perf_window"
    assert failed_check in normalized["window_contract"]["failed_checks"]
    with pytest.raises(ValueError, match="telemetry window"):
        _replay_perf_window(attempt_dir, duration_seconds=100)


@pytest.mark.parametrize(
    "bad_perf_csv",
    [
        (
            "elapsed_seconds,aggregate_fps,per_stream_mean_fps\n"
            "5,120,10\n5,132,11\n"
        ),
        (
            "elapsed_seconds,aggregate_fps,per_stream_mean_fps\n"
            "NaN,120,10\n10,132,11\n"
        ),
        (
            "elapsed_seconds,aggregate_fps,per_stream_mean_fps\n"
            "5,Infinity,10\n10,132,11\n"
        ),
        (
            "elapsed_seconds,aggregate_fps,per_stream_mean_fps\n"
            "5,120,-Infinity\n10,132,11\n"
        ),
        (
            "elapsed_seconds,aggregate_fps,per_stream_mean_fps\n"
            "5,120,10,unexpected\n10,132,11\n"
        ),
        (
            "elapsed_seconds,aggregate_fps,aggregate_fps\n"
            "5,120,10\n10,132,11\n"
        ),
    ],
    ids=[
        "duplicate-elapsed",
        "nan-elapsed",
        "infinite-aggregate",
        "negative-infinite-mean",
        "extra-column",
        "duplicate-header",
    ],
)
def test_endurance_raw_replay_rejects_duplicate_nonfinite_or_malformed_csv(
    tmp_path, bad_perf_csv
):
    attempt_dir = tmp_path / "attempt"
    _write_small_raw_endurance_attempt(attempt_dir)
    (attempt_dir / "perf.csv").write_text(bad_perf_csv, encoding="utf-8")

    with pytest.raises(ValueError, match="telemetry window"):
        campaign._endurance_replay_attempt_raw(
            attempt_dir,
            duration_seconds=10,
            perf_interval_seconds=5,
            max_log_bytes=1024 * 1024,
            minimum_coverage_fraction=0.95,
            startup_grace_seconds=120,
            perf_stall_timeout_seconds=180,
        )


def test_reporter_perf_window_replay_matches_supervisor_contract_exactly(tmp_path):
    attempt_dir = tmp_path / "attempt"
    _write_raw_perf_window_attempt(
        attempt_dir,
        elapsed_values=[float(value) for value in range(5, 100, 5)],
        duration_seconds=100,
    )
    parsed = campaign.benchmark_summarize.parse_perf(
        attempt_dir / "deepstream.log", 12, 100, 5
    )
    kwargs = {
        "expected_streams": 12,
        "duration_seconds": 100,
        "perf_interval_seconds": 5,
        "minimum_coverage_fraction": 0.95,
        "startup_grace_seconds": 10,
        "perf_stall_timeout_seconds": 20,
    }

    reporter = campaign._endurance_apply_perf_window_contract(
        copy.deepcopy(parsed), attempt_dir / "perf.csv", **kwargs
    )
    supervisor = endurance_supervisor.apply_perf_window_contract(
        copy.deepcopy(parsed), attempt_dir / "perf.csv", **kwargs
    )

    assert reporter == supervisor


def test_reporter_raw_replay_preserves_supervisor_integer_parser_interval(
    tmp_path,
):
    attempt_dir = tmp_path / "attempt"
    _write_small_raw_endurance_attempt(attempt_dir)
    supervisor_parsed = endurance_supervisor.parse_perf(
        attempt_dir / "deepstream.log",
        12,
        10,
        int(5),
    )
    supervisor_throughput = endurance_supervisor.apply_perf_window_contract(
        supervisor_parsed,
        attempt_dir / "perf.csv",
        expected_streams=12,
        duration_seconds=10,
        perf_interval_seconds=5.0,
        minimum_coverage_fraction=0.95,
        startup_grace_seconds=120.0,
        perf_stall_timeout_seconds=180.0,
    )

    replay = campaign._endurance_replay_attempt_raw(
        attempt_dir,
        duration_seconds=10,
        perf_interval_seconds=5.0,
        max_log_bytes=1024 * 1024,
        minimum_coverage_fraction=0.95,
        startup_grace_seconds=120.0,
        perf_stall_timeout_seconds=180.0,
    )

    assert type(supervisor_throughput["perf_interval_seconds"]) is int
    assert type(replay["throughput"]["perf_interval_seconds"]) is int
    assert replay["throughput"] == supervisor_throughput
    status = {
        "throughput": copy.deepcopy(supervisor_throughput),
        "gpu": copy.deepcopy(replay["gpu"]),
        "artifact_pins": {
            name: {
                "size_bytes": observation["size_bytes"],
                "sha256": observation["sha256"],
            }
            for name, observation in replay["file_observations"].items()
        },
        "throughput_floor_evaluation": {
            "observed_aggregate_fps_p05": supervisor_throughput[
                "aggregate_current_fps"
            ]["p05"]
        },
    }
    assert campaign._endurance_attempt_raw_reasons(status, replay) == []


@pytest.mark.parametrize("bad_interval", [None, "5", True, float("nan"), float("inf")])
def test_reporter_perf_window_bad_parser_interval_fails_with_supervisor_parity(
    tmp_path, bad_interval
):
    attempt_dir = tmp_path / "attempt"
    _write_small_raw_endurance_attempt(attempt_dir)
    parsed = campaign.benchmark_summarize.parse_perf(
        attempt_dir / "deepstream.log", 12, 10, 5
    )
    parsed["perf_interval_seconds"] = bad_interval
    kwargs = {
        "expected_streams": 12,
        "duration_seconds": 10,
        "perf_interval_seconds": 5,
        "minimum_coverage_fraction": 0.95,
        "startup_grace_seconds": 120,
        "perf_stall_timeout_seconds": 180,
    }

    reporter = campaign._endurance_apply_perf_window_contract(
        copy.deepcopy(parsed), attempt_dir / "perf.csv", **kwargs
    )
    supervisor = endurance_supervisor.apply_perf_window_contract(
        copy.deepcopy(parsed), attempt_dir / "perf.csv", **kwargs
    )

    assert reporter["status"] == supervisor["status"] == "invalid_perf_window"
    assert reporter["window_contract"] == supervisor["window_contract"]
    assert reporter["window_contract"]["checks"]["parser_window_contract"] is False


def test_reporter_perf_window_csv_parser_error_fails_with_supervisor_parity(
    tmp_path,
):
    attempt_dir = tmp_path / "attempt"
    _write_small_raw_endurance_attempt(attempt_dir)
    parsed = campaign.benchmark_summarize.parse_perf(
        attempt_dir / "deepstream.log", 12, 10, 5
    )
    (attempt_dir / "perf.csv").write_text(
        "elapsed_seconds,aggregate_fps,per_stream_mean_fps\n"
        + "5,"
        + "1" * 200
        + ",10\n",
        encoding="utf-8",
    )
    kwargs = {
        "expected_streams": 12,
        "duration_seconds": 10,
        "perf_interval_seconds": 5,
        "minimum_coverage_fraction": 0.95,
        "startup_grace_seconds": 120,
        "perf_stall_timeout_seconds": 180,
    }
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(64)
        reporter = campaign._endurance_apply_perf_window_contract(
            copy.deepcopy(parsed), attempt_dir / "perf.csv", **kwargs
        )
        supervisor = endurance_supervisor.apply_perf_window_contract(
            copy.deepcopy(parsed), attempt_dir / "perf.csv", **kwargs
        )
    finally:
        csv.field_size_limit(previous_limit)

    assert reporter == supervisor
    assert reporter["status"] == "invalid_perf_window"
    assert reporter["window_contract"]["csv_errors_first_20"] == [
        "unreadable:Error"
    ]


def test_endurance_imported_raw_parser_must_match_project_control_pin(tmp_path):
    parser_path = tmp_path / "benchmark/summarize.py"
    parser_path.parent.mkdir(parents=True, exist_ok=True)
    parser_path.write_bytes(
        Path(campaign.benchmark_summarize.__file__).read_bytes()
        + b"\n# coherently pinned substitute\n"
    )
    resolved = {
        "input_pins": {"control_files": [_content_pin(tmp_path, parser_path)]}
    }

    valid, reasons, _evidence = campaign._endurance_raw_parser_contract(
        campaign.EvidenceStore(tmp_path), resolved
    )

    assert valid is False
    assert "endurance_imported_raw_parser_differs_from_project_pin" in reasons
    assert "endurance_raw_parser_project_pin_failed" not in reasons


def test_minimal_endurance_summaries_cannot_fake_seven_day_acceptance(tmp_path):
    current = "validation/results/endurance/current"
    planned = campaign._endurance_expected_plan()
    _write_json(
        tmp_path,
        f"{current}/plan.json",
        {"schema_version": "deepsafe.endurance-plan/v1", "segments": planned},
    )
    _write_json(
        tmp_path,
        f"{current}/checkpoint.json",
        {
            "schema_version": "deepsafe.endurance-checkpoint/v1",
            "state": "complete",
            "dry_run": False,
            "target_validated_seconds": 604800,
            "validated_seconds": 604800,
            "segments": [
                {
                    **item,
                    "status": "healthy",
                    "validated_seconds": 21600,
                    "attempts": [
                        {
                            "schema_version": "deepsafe.endurance-segment/v1",
                            "status": "healthy",
                            "validated_seconds": 21600,
                            "health_gates": [],
                        }
                    ],
                }
                for item in planned
            ],
        },
    )
    _write_json(
        tmp_path,
        f"{current}/status.json",
        {
            "schema_version": "deepsafe.endurance-status/v1",
            "state": "complete",
            "dry_run": False,
            "target_validated_seconds": 604800,
            "validated_seconds": 604800,
            "profiles_validated_seconds": {"640": 302400, "960": 302400},
            "segments": {"total": 28, "status_counts": {"healthy": 28}},
            "active": None,
            "campaign_health_gates": [],
        },
    )
    for day in range(1, 8):
        _write_json(
            tmp_path,
            f"{current}/reports/day-{day:02d}.json",
            {
                "schema_version": "deepsafe.endurance-daily-summary/v1",
                "campaign_day": day,
                "state": "complete",
                "planned_segments": 4,
                "healthy_segments": 4,
                "validated_seconds": 86400,
            },
        )

    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert section["verified_latest_attempt_status_files"] == 0
    assert facts["checkpoint_contract"] is False
    assert any("campaign_resolved" in reason for reason in section["reasons"])


def test_running_endurance_does_not_count_absent_attempt_status_files_as_verified(
    tmp_path,
):
    _populate_accepted_endurance(tmp_path)
    checkpoint_path = (
        tmp_path / "validation/results/endurance/current/checkpoint.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    for segment in checkpoint["segments"]:
        segment["status"] = "pending"
        segment["attempts"] = []
        segment["attempt_receipts"] = []
        segment["validated_seconds"] = 0
    first = checkpoint["segments"][0]
    checkpoint.update(
        {
            "state": "running",
            "finished_at_utc": None,
            "validated_seconds": 0,
            "active": {
                "segment_id": first["segment_id"],
                "profile": first["profile"],
                "attempt": 1,
            },
        }
    )
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert section["verified_latest_attempt_status_files"] == 0
    assert section["verified_attempt_receipts"] == 0
    assert section["raw_attempt_telemetry_replays_verified"] == 0
    assert facts["checkpoint_contract"] is False


def test_malformed_endurance_checkpoint_fails_closed_without_reporter_crash(tmp_path):
    _populate_accepted_endurance(tmp_path)
    checkpoint_path = tmp_path / "validation/results/endurance/current/checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["segments"][0]["attempts"] = None
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["checkpoint_contract"] is False
    assert any("attempts_missing" in reason for reason in section["reasons"])


@pytest.mark.parametrize("mode", ["missing", "tampered"])
def test_endurance_latest_status_file_is_required_and_snapshot_bound(tmp_path, mode):
    _populate_accepted_endurance(tmp_path)
    path = (
        tmp_path
        / "validation/results/endurance/current/segments/segment-000-640/attempt-01/status.json"
    )
    if mode == "missing":
        path.unlink()
    else:
        status = json.loads(path.read_text(encoding="utf-8"))
        status["validated_seconds"] = 1
        path.write_text(json.dumps(status), encoding="utf-8")

    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert section["verified_latest_attempt_status_files"] == 27
    assert facts["checkpoint_contract"] is False
    assert any("status_snapshot_mismatch" in reason for reason in section["reasons"])


def test_endurance_resolved_plan_checkpoint_fingerprint_mismatch_is_rejected(tmp_path):
    _populate_accepted_endurance(tmp_path)
    plan_path = tmp_path / "validation/results/endurance/current/plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["config_fingerprint"] = hashlib.sha256(b"different-config").hexdigest()
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["plan_contract"] is False
    assert "plan_resolved_fingerprint_or_matrix_mismatch" in section["reasons"]


def test_endurance_floor_must_live_rederive_from_the_24_run_matrix(tmp_path):
    _populate_accepted_endurance(tmp_path)

    def reject_live_rederivation(*_args, **_kwargs):
        raise ValueError("fixture matrix changed after floor freeze")

    section, facts = _build_endurance_fixture_section(
        tmp_path, verifier=reject_live_rederivation
    )

    assert section["accepted"] is False
    assert section["throughput_floor"]["status"] == "unproven"
    assert section["performance_quality_threshold_applied"] is False
    assert section["performance_quality_outcome"] == "unproven"
    assert facts["throughput_floor_contract"] is False
    assert any(
        "throughput_floor_live_rederivation_failed" in reason
        for reason in section["reasons"]
    )


def test_endurance_incomplete_campaign_without_proven_floor_violation_is_pending(
    tmp_path,
):
    _populate_accepted_endurance(tmp_path)
    checkpoint_path = tmp_path / "validation/results/endurance/current/checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    active = checkpoint["segments"][0]
    active["status"] = "running"
    active["validated_seconds"] = 0
    active["attempts"] = []
    active["attempt_receipts"] = []
    checkpoint["state"] = "running"
    checkpoint["finished_at_utc"] = None
    checkpoint["validated_seconds"] -= campaign.EXPECTED_SEGMENT_SECONDS
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert section["performance_quality_threshold_applied"] is True
    assert section["performance_quality_outcome"] == "pending"
    assert section["throughput_floor"]["passing_endurance_attempts"] == 27
    assert section["throughput_floor"]["proven_floor_violations"] == 0
    assert facts["throughput_floor_attempts_passed"] == 27
    assert facts["throughput_floor_proven_violations"] == 0

    schema = json.loads(
        (
            campaign.PROJECT_ROOT
            / "validation/schemas/validation-campaign-report-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    try:
        import jsonschema
    except ImportError:
        return
    jsonschema.Draft202012Validator(schema["$defs"]["endurance"]).validate(section)
    forged = copy.deepcopy(section)
    forged["performance_quality_outcome"] = "failed"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema["$defs"]["endurance"]).validate(
            forged
        )


def test_endurance_attempt_below_frozen_profile_floor_is_rejected(tmp_path):
    _populate_accepted_endurance(tmp_path)

    def forge_below_floor(attempt):
        attempt["throughput"]["aggregate_current_fps"]["p05"] = 449.9
        attempt["throughput_floor_evaluation"][
            "observed_aggregate_fps_p05"
        ] = 449.9

    _mutate_latest_endurance_attempt(tmp_path, 0, forge_below_floor)
    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert section["throughput_floor"]["passing_endurance_attempts"] == 27
    assert section["performance_quality_threshold_applied"] is True
    assert section["performance_quality_outcome"] == "failed"
    assert section["throughput_floor"]["proven_floor_violations"] == 1
    assert facts["throughput_floor_attempts_passed"] == 27
    assert facts["throughput_floor_proven_violations"] == 1
    assert any(
        "attempt_throughput_floor_evaluation_failed" in reason
        for reason in section["reasons"]
    )


def test_endurance_coherent_status_receipt_checkpoint_repin_cannot_forge_raw_p05(
    tmp_path,
):
    _populate_accepted_endurance(tmp_path)
    base = (
        tmp_path
        / "validation/results/endurance/current/segments/segment-000-640/attempt-01"
    )
    original = json.loads((base / "status.json").read_text(encoding="utf-8"))
    raw_throughput = copy.deepcopy(original["throughput"])
    raw_gpu = copy.deepcopy(original["gpu"])

    def forge(status, _attempt_dir):
        status["throughput"]["aggregate_current_fps"]["p05"] = 9999.0
        status["throughput_floor_evaluation"][
            "observed_aggregate_fps_p05"
        ] = 9999.0

    _coherently_repin_endurance_attempt(tmp_path, 0, forge)

    def replay_original_raw(
        attempt_dir,
        *,
        duration_seconds,
        perf_interval_seconds,
        max_log_bytes,
        minimum_coverage_fraction,
        startup_grace_seconds,
        perf_stall_timeout_seconds,
    ):
        if attempt_dir.resolve() != base.resolve():
            return _fixture_endurance_raw_replay(
                attempt_dir,
                duration_seconds=duration_seconds,
                perf_interval_seconds=perf_interval_seconds,
                max_log_bytes=max_log_bytes,
                minimum_coverage_fraction=minimum_coverage_fraction,
                startup_grace_seconds=startup_grace_seconds,
                perf_stall_timeout_seconds=perf_stall_timeout_seconds,
            )
        status = json.loads((attempt_dir / "status.json").read_text(encoding="utf-8"))
        pins = status["artifact_pins"]
        requested = int(duration_seconds / perf_interval_seconds)
        return {
            "schema_version": "deepsafe.endurance-raw-attempt-replay/v1",
            "status": "verified",
            "throughput": copy.deepcopy(raw_throughput),
            "gpu": copy.deepcopy(raw_gpu),
            "perf_csv": {
                "header": [
                    "elapsed_seconds",
                    "aggregate_fps",
                    "per_stream_mean_fps",
                ],
                "rows": requested,
                "requested_window_rows": requested,
                "aggregate_current_fps": copy.deepcopy(
                    raw_throughput["aggregate_current_fps"]
                ),
            },
            "file_observations": {
                name: {
                    "size_bytes": pins[name]["size_bytes"],
                    "sha256": pins[name]["sha256"],
                }
                for name in ("log", "perf_csv", "gpu_csv")
            },
        }

    section, facts = _build_endurance_fixture_section(
        tmp_path, raw_verifier=replay_original_raw
    )

    assert section["accepted"] is False
    assert section["raw_attempt_telemetry_replays_verified"] == 27
    assert section["throughput_floor"]["passing_endurance_attempts"] == 27
    assert facts["raw_attempts_verified"] == 27
    assert any(
        "attempt_status_throughput_differs_from_raw_log" in reason
        for reason in section["reasons"]
    )


@pytest.mark.parametrize(
    ("artifact_name", "filename"),
    [("log", "deepstream.log"), ("gpu_csv", "gpu.csv")],
)
def test_endurance_coherently_repinned_raw_telemetry_mismatch_is_rejected(
    tmp_path, artifact_name, filename
):
    _populate_accepted_endurance(tmp_path)
    marker = b"coherently-repinned-raw-mismatch"

    def rewrite_raw_and_pin(status, attempt_dir):
        path = attempt_dir / filename
        path.write_bytes(path.read_bytes() + marker)
        status["artifact_pins"][artifact_name] = _content_pin(tmp_path, path)

    _coherently_repin_endurance_attempt(tmp_path, 0, rewrite_raw_and_pin)

    def reject_changed_raw(attempt_dir, **kwargs):
        del kwargs
        if marker in (attempt_dir / filename).read_bytes():
            raise ValueError(f"raw {artifact_name} projection mismatch")
        return _fixture_endurance_raw_replay(
            attempt_dir,
            duration_seconds=21600,
            perf_interval_seconds=5,
            max_log_bytes=8388608,
            minimum_coverage_fraction=0.95,
            startup_grace_seconds=120,
            perf_stall_timeout_seconds=180,
        )

    section, facts = _build_endurance_fixture_section(
        tmp_path, raw_verifier=reject_changed_raw
    )

    assert section["accepted"] is False
    assert section["raw_attempt_telemetry_replays_verified"] == 27
    assert facts["raw_attempts_verified"] == 27
    assert any(
        "attempt_raw_telemetry_replay_failed:ValueError:raw" in reason
        for reason in section["reasons"]
    )


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("image", "attempt_runtime_container_command_identity_mismatch"),
        ("argv", "attempt_runtime_container_process_mismatch"),
        ("gpu", "attempt_runtime_container_gpu_request_mismatch"),
        ("mount", "attempt_runtime_container_mount_or_host_config_mismatch"),
        ("labels", "attempt_runtime_container_config_projection_mismatch"),
        ("timeline", "attempt_runtime_container_timeline_invalid"),
    ],
)
def test_endurance_coherently_repinned_runtime_container_attestation_is_rejected(
    tmp_path, mutation, expected_reason
):
    _populate_accepted_endurance(tmp_path)

    def forge(status, attempt_dir):
        path = attempt_dir / "runtime-container.json"
        attestation = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "image":
            forged = "sha256:" + "9" * 64
            attestation["image_id"] = forged
            attestation["config_image"] = forged
        elif mutation == "argv":
            attestation["cmd"][-1] = "/workspace/attacker.txt"
            attestation["actual_process"]["args"][-1] = "/workspace/attacker.txt"
        elif mutation == "gpu":
            attestation["gpu_device_request"]["device_ids"] = ["1"]
        elif mutation == "mount":
            attestation["mounts"][0]["read_write"] = True
        elif mutation == "labels":
            attestation["expected_container_labels"][
                "io.deepsafe.endurance.segment-id"
            ] = "segment-attacker"
        else:
            attestation["captured_at_utc"] = "2026-07-17T00:00:00+00:00"
        path.write_text(json.dumps(attestation), encoding="utf-8")
        status["runtime_container_attestation"] = attestation
        status["artifact_pins"]["runtime_container"] = _content_pin(
            tmp_path, path
        )

    _coherently_repin_endurance_attempt(tmp_path, 0, forge)
    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["checkpoint_contract"] is False
    assert any(expected_reason in reason for reason in section["reasons"])


def test_endurance_floor_runtime_must_match_the_session_preflight(tmp_path):
    _populate_accepted_endurance(tmp_path)
    preflight_path = (
        tmp_path
        / "validation/results/endurance/current/sessions/"
        "session-0123456789abcdef/preflight.json"
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["gpu_identity"]["driver_version"] = "different-driver"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert section["throughput_floor"]["runtime_identity_exact"] is False
    assert facts["checkpoint_contract"] is False
    assert any(
        "preflight" in reason and "failed" in reason
        for reason in section["reasons"]
    )


def test_endurance_partial_stream_attempt_is_rejected_even_when_snapshot_matches(tmp_path):
    _populate_accepted_endurance(tmp_path)

    def remove_stream(attempt):
        attempt["throughput"]["active_streams"] = 11
        attempt["throughput"]["inactive_stream_ids"] = [11]

    _mutate_latest_endurance_attempt(tmp_path, 0, remove_stream)
    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["checkpoint_contract"] is False
    assert any("throughput_active_streams_not_12" in reason for reason in section["reasons"])
    assert any("throughput_partial_streams" in reason for reason in section["reasons"])


@pytest.mark.parametrize(
    ("mode", "expected_reason"),
    [
        ("command", "production_command_or_config_contract_failed"),
        ("command_contract", "production_command_or_config_contract_failed"),
        ("container_labels", "production_command_or_config_contract_failed"),
        ("floor_label", "production_command_or_config_contract_failed"),
        ("pre_input_pin", "attempt_input_pin_sweep_pre_actual_mismatch"),
        ("post_input_pin", "attempt_input_pin_sweep_post_actual_mismatch"),
        ("cleanup", "attempt_container_cleanup_contract_failed"),
        ("finalization", "attempt_cross_segment_finalization_contract_failed"),
    ],
)
def test_reporter_rejects_forged_endurance_execution_evidence(
    tmp_path, mode, expected_reason
):
    _populate_accepted_endurance(tmp_path)

    def forge(attempt):
        if mode == "command":
            attempt["process"]["command"][-1] = "/workspace/forged.txt"
        elif mode == "command_contract":
            attempt["process"]["command_contract"]["resolved_image_id"] = (
                "sha256:" + "f" * 64
            )
        elif mode == "container_labels":
            label = "io.deepsafe.endurance.gpu-index"
            attempt["process"]["container_labels"][label] = "7"
        elif mode == "floor_label":
            label = "io.deepsafe.endurance.throughput-floor-fingerprint"
            attempt["process"]["container_labels"][label] = "f" * 64
        elif mode == "pre_input_pin":
            attempt["input_pin_sweeps"]["pre"]["actual"]["source_media"][0][
                "sha256"
            ] = "f" * 64
        elif mode == "post_input_pin":
            attempt["input_pin_sweeps"]["post"]["actual"]["control_files"][0][
                "sha256"
            ] = "f" * 64
        elif mode == "cleanup":
            attempt["container_cleanup"]["verified_absent"] = False
        else:
            attempt["attempt_finalization"]["state"] = "pending"

    _mutate_latest_endurance_attempt(tmp_path, 0, forge)
    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["checkpoint_contract"] is False
    assert any(expected_reason in reason for reason in section["reasons"])


@pytest.mark.parametrize(
    ("mode", "expected_reason"),
    [
        ("wrong_gpu", "attempt_gpu_session_identity_mismatch"),
        ("zero_utilization", "attempt_gpu_active_load_unproven"),
        ("zero_memory", "attempt_gpu_active_load_unproven"),
    ],
)
def test_endurance_attempt_must_prove_active_load_on_session_gpu(
    tmp_path, mode, expected_reason
):
    _populate_accepted_endurance(tmp_path)

    def break_active_gpu_contract(attempt):
        if mode == "wrong_gpu":
            attempt["gpu"]["gpu_names"] = ["Unrelated GPU"]
        elif mode == "zero_utilization":
            attempt["gpu"]["metrics"]["gpu_utilization_percent"]["max"] = 0.0
        else:
            attempt["gpu"]["metrics"]["memory_used_mib"]["max"] = 0.0

    _mutate_latest_endurance_attempt(tmp_path, 0, break_active_gpu_contract)
    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["checkpoint_contract"] is False
    assert any(expected_reason in reason for reason in section["reasons"])


@pytest.mark.parametrize(
    ("segment_index", "metric_path", "value", "expected_reason"),
    [
        (
            6,
            ("throughput", "aggregate_current_fps", "mean"),
            100.0,
            "attempt_cross_segment_fps_drift_failed",
        ),
        (
            7,
            ("latency", "p95_ms_across_buckets", "mean"),
            120.0,
            "attempt_cross_segment_latency_drift_failed",
        ),
    ],
)
def test_reporter_independently_recomputes_cross_segment_drift(
    tmp_path, segment_index, metric_path, value, expected_reason
):
    _populate_accepted_endurance(tmp_path)

    def inject_cross_segment_drift(attempt):
        target = attempt
        for key in metric_path[:-1]:
            target = target[key]
        target[metric_path[-1]] = value

    _mutate_latest_endurance_attempt(
        tmp_path, segment_index, inject_cross_segment_drift
    )
    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["checkpoint_contract"] is False
    assert any(expected_reason in reason for reason in section["reasons"])


def test_endurance_segment_exceeding_campaign_disk_cap_is_rejected(tmp_path):
    _populate_accepted_endurance(tmp_path)

    def exceed_cap(attempt):
        maximum = attempt["disk"]["max_campaign_bytes"]
        attempt["disk"]["campaign_bytes_start"] = maximum - 99
        attempt["disk"]["campaign_bytes_end"] = maximum + 1
        attempt["disk"]["growth_bytes"] = 100

    _mutate_latest_endurance_attempt(tmp_path, 0, exceed_cap)
    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["checkpoint_contract"] is False
    assert any("attempt_disk_gate_failed" in reason for reason in section["reasons"])


def test_endurance_live_campaign_tree_exceeding_disk_cap_is_rejected(tmp_path):
    _populate_accepted_endurance(tmp_path)
    oversized = (
        tmp_path
        / "validation/results/endurance/current/unpinned-oversized-artifact.bin"
    )
    with oversized.open("wb") as handle:
        handle.truncate(5368709121)

    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert section["live_campaign_disk_contract_valid"] is False
    assert facts["status_contract"] is False
    assert "endurance_live_campaign_disk_cap_or_tree_contract_failed" in section[
        "reasons"
    ]


def test_endurance_live_source_media_pin_tamper_is_rejected(tmp_path):
    _populate_accepted_endurance(tmp_path)
    media = tmp_path / "data/fixture/endurance-scene-00.mp4"
    media.write_bytes(media.read_bytes() + b"tampered")

    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["checkpoint_contract"] is False
    assert "endurance_source_media_pin_00_failed" in section["reasons"]


@pytest.mark.parametrize(
    ("artifact_name", "filename"),
    [("log", "deepstream.log"), ("perf_csv", "perf.csv")],
)
def test_endurance_core_attempt_artifact_hash_tamper_is_rejected(
    tmp_path, artifact_name, filename
):
    _populate_accepted_endurance(tmp_path)
    artifact = (
        tmp_path
        / "validation/results/endurance/current/segments/segment-000-640/"
        f"attempt-01/{filename}"
    )
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["checkpoint_contract"] is False
    assert any(
        f"artifact_{artifact_name}_pin_failed" in reason
        for reason in section["reasons"]
    )


def test_endurance_attempt_profile_remains_exactly_bound_to_plan_and_receipt(
    tmp_path,
):
    _populate_accepted_endurance(tmp_path)

    def forge_profile(status, _attempt_dir):
        status["segment"]["profile"] = 960

    _coherently_repin_endurance_attempt(tmp_path, 0, forge_profile)
    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["checkpoint_contract"] is False
    assert any(
        "receipt_contract_failed" in reason
        or "attempt_segment_snapshot_mismatch" in reason
        for reason in section["reasons"]
    )


@pytest.mark.parametrize(
    ("canonical_text", "tampered_text"),
    [
        ("[primary-gie]\nenable=1", "[primary-gie]\nenable=0"),
        ("camera-id=0", "camera-id=11"),
    ],
)
def test_endurance_rejects_coherently_repinned_noncanonical_deepstream_config(
    tmp_path, canonical_text, tampered_text
):
    _populate_accepted_endurance(tmp_path)
    base = (
        tmp_path
        / "validation/results/endurance/current/segments/segment-000-640/attempt-01"
    )
    config_path = base / "deepstream.txt"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            canonical_text, tampered_text, 1
        ),
        encoding="utf-8",
    )
    forged_config_pin = _content_pin(tmp_path, config_path)

    checkpoint_path = tmp_path / "validation/results/endurance/current/checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record = checkpoint["segments"][0]
    status = record["attempts"][0]
    status["artifact_pins"]["config"] = forged_config_pin
    status["process"]["command_contract"]["rendered_config_sha256"] = (
        forged_config_pin["sha256"]
    )
    label = "io.deepsafe.endurance.rendered-config-sha256"
    status["process"]["container_labels"][label] = forged_config_pin["sha256"]
    status["process"]["command_contract"]["container_labels"][label] = (
        forged_config_pin["sha256"]
    )
    command = status["process"]["command"]
    for index, value in enumerate(command):
        if value.startswith(f"{label}="):
            command[index] = f"{label}={forged_config_pin['sha256']}"
    status["process"]["command_contract"]["command"] = list(command)
    status_path = base / "status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    receipt_path = base / "attempt-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_pins"] = status["artifact_pins"]
    receipt["artifact_pins_sha256"] = campaign._endurance_json_sha256(
        status["artifact_pins"]
    )
    receipt["status_pin"] = _content_pin(tmp_path, status_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    record["attempts"][0] = status
    record["attempt_receipts"][0] = _content_pin(tmp_path, receipt_path)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["checkpoint_contract"] is False
    assert any(
        "production_command_or_config_contract_failed" in reason
        for reason in section["reasons"]
    )


def test_endurance_attempt_receipt_content_tamper_is_rejected(tmp_path):
    _populate_accepted_endurance(tmp_path)
    receipt_path = (
        tmp_path
        / "validation/results/endurance/current/segments/segment-000-640/"
        "attempt-01/attempt-receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["validated_seconds"] = 1
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    section, facts = _build_endurance_fixture_section(tmp_path)

    assert section["accepted"] is False
    assert facts["checkpoint_contract"] is False
    assert any("receipt_contract_failed" in reason for reason in section["reasons"])


def test_minimal_distance_json_cannot_open_calibrated_gate(tmp_path):
    _write_json(
        tmp_path,
        "validation/results/distance-25m/evaluation.json",
        {
            "schema_version": "deepsafe.distance-validation/v1",
            "status": "complete",
        },
    )

    section, facts = campaign._build_distance_section(
        campaign.EvidenceStore(tmp_path),
        tmp_path / "validation/results",
    )

    assert section["accepted"] is False
    assert section["state"] == "unproven"
    assert section["schema_contract_valid"] is False
    assert section["pin_matrix_valid"] is False
    assert section["independent_cpu_recomputation_valid"] is False
    assert facts["accepted"] is False


def _write_verified_synthetic_distance_v2_final(root: Path) -> tuple[dict, dict]:
    fixture_root = root / "synthetic-fixture"
    fixture_root.mkdir(parents=True, exist_ok=True)
    paths = _distance_v2_profile_pair_fixture(fixture_root)
    _add_exact_25m_and_unambiguous_endpoint(paths)
    _write_gt_predictions(paths, confidence=0.9)
    results = root / "validation/results/distance-25m"
    preflight = _distance_v2_write_json(
        results / "preflight-receipt-v2.json",
        _distance_v2_preflight(paths),
    )
    pair = _distance_v2_write_json(
        results / "profile-pair-receipt-v2.json",
        _distance_v2_pair(paths),
    )
    attempt = results / "evaluation-attempt-v2-001.json"
    final = results / "evaluation-final-v2-001.json"
    _, final_value = site_distance_v2.evaluate_and_write(
        camera_configuration_path=paths["camera"],
        calibration_path=paths["calibration"],
        media_frame_ledger_path=paths["media"],
        ground_truth_path=paths["ground_truth"],
        acceptance_path=paths["acceptance"],
        preflight_receipt_path=preflight,
        profile_pair_receipt_path=pair,
        profile_640_manifest_path=paths["manifests"][640],
        profile_960_manifest_path=paths["manifests"][960],
        attempt_output_path=attempt,
        final_output_path=final,
    )
    assert final_value is not None
    documents = {
        "media_frame_ledger": json.loads(paths["media"].read_text(encoding="utf-8")),
        "profile_640_manifest": json.loads(
            paths["manifests"][640].read_text(encoding="utf-8")
        ),
        "profile_960_manifest": json.loads(
            paths["manifests"][960].read_text(encoding="utf-8")
        ),
    }
    return final_value, documents


def test_live_verified_v2_fixture_cannot_open_production_distance_gate(tmp_path):
    final, _ = _write_verified_synthetic_distance_v2_final(tmp_path)

    section, facts = campaign._build_distance_section(
        campaign.EvidenceStore(tmp_path),
        tmp_path / "validation/results",
    )

    assert final["status"] == "complete"
    assert section["schema_contract_valid"] is True
    assert section["independent_cpu_recomputation_valid"] is True
    assert section["production_evidence_contract_valid"] is False
    assert section["accepted"] is False
    assert section["profiles"] == {}
    assert section["distance_bins"] == []
    assert "inclusive_v2_production_evidence_contract_failed" in section["reasons"]
    assert facts["accepted"] is False


def test_v2_final_projects_only_replayed_aggregate_fields(
    tmp_path, monkeypatch
):
    final, documents = _write_verified_synthetic_distance_v2_final(tmp_path)
    production = {
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
    monkeypatch.setattr(
        campaign,
        "_distance_v2_production_contract",
        lambda payload, root: (True, production, documents),
    )

    section, facts = campaign._build_distance_section(
        campaign.EvidenceStore(tmp_path),
        tmp_path / "validation/results",
    )

    assert section["accepted"] is True
    assert section["evidence_version"] == "inclusive_v2"
    assert section["selection"] == {
        "state": "inclusive_v2",
        "legacy_v1_present": False,
        "inclusive_v2_final_candidates": 1,
        "conflict": False,
    }
    assert set(section["profiles"]) == {"640", "960"}
    assert [row["bin_id"] for row in section["distance_bins"]] == list(
        campaign.EXPECTED_LOAF_DISTANCE_BINS
    )
    assert section["exact_25m_instances"] == 1
    serialized = json.dumps(section, sort_keys=True)
    assert "project-owner" not in serialized
    assert "approved_by" not in serialized
    assert "synthetic-fixture" not in serialized
    assert final["receipt_sha256"] not in serialized
    assert facts["accepted"] is True


def test_legacy_and_v2_distance_finals_conflict_fail_closed(tmp_path):
    _write_verified_synthetic_distance_v2_final(tmp_path)
    _write_json(
        tmp_path,
        "validation/results/distance-25m/evaluation.json",
        {"schema_version": "deepsafe.distance-validation/v1", "status": "complete"},
    )

    section, facts = campaign._build_distance_section(
        campaign.EvidenceStore(tmp_path),
        tmp_path / "validation/results",
    )

    assert section["accepted"] is False
    assert section["selection"]["state"] == "conflict"
    assert section["selection"]["legacy_v1_present"] is True
    assert "legacy_v1_and_inclusive_v2_evidence_conflict" in section["reasons"]
    assert facts["accepted"] is False


def test_multiple_v2_distance_finals_conflict_even_when_bytes_match(tmp_path):
    _write_verified_synthetic_distance_v2_final(tmp_path)
    results = tmp_path / "validation/results/distance-25m"
    shutil.copy2(
        results / "evaluation-final-v2-001.json",
        results / "evaluation-final-v2-002.json",
    )

    section, facts = campaign._build_distance_section(
        campaign.EvidenceStore(tmp_path),
        tmp_path / "validation/results",
    )

    assert section["accepted"] is False
    assert section["selection"] == {
        "state": "conflict",
        "legacy_v1_present": False,
        "inclusive_v2_final_candidates": 2,
        "conflict": True,
    }
    assert "inclusive_v2_final_candidate_conflict" in section["reasons"]
    assert facts["accepted"] is False


def test_tampered_v2_final_fails_live_replay_without_leaking_detail(tmp_path):
    _write_verified_synthetic_distance_v2_final(tmp_path)
    final_path = (
        tmp_path
        / "validation/results/distance-25m/evaluation-final-v2-001.json"
    )
    value = json.loads(final_path.read_text(encoding="utf-8"))
    value["profiles"]["640"]["overall"]["tp"] -= 1
    final_path.write_text(json.dumps(value), encoding="utf-8")

    section, _ = campaign._build_distance_section(
        campaign.EvidenceStore(tmp_path),
        tmp_path / "validation/results",
    )

    assert section["accepted"] is False
    assert section["independent_cpu_recomputation_valid"] is False
    assert section["reasons"] == ["inclusive_v2_live_semantic_replay_failed"]
    assert str(tmp_path) not in json.dumps(section)


@pytest.mark.parametrize(
    "relative_path",
    [
        "validation/inputs/distance-25m/profiles/640/predictions.jsonl",
        "validation/inputs/distance-25m/acceptance-approval.md",
    ],
)
def test_tampered_pinned_distance_input_is_rejected(
    tmp_path, monkeypatch, relative_path
):
    _prepare_accepted_site_distance(tmp_path, monkeypatch)
    tampered = tmp_path / relative_path
    tampered.write_bytes(tampered.read_bytes() + b"tampered\n")

    section, facts = campaign._build_distance_section(
        campaign.EvidenceStore(tmp_path),
        tmp_path / "validation/results",
    )

    assert section["accepted"] is False
    assert section["schema_contract_valid"] is True
    assert section["pin_matrix_valid"] is False
    assert section["independent_cpu_recomputation_valid"] is False
    assert any("evidence pin failed" in reason for reason in section["reasons"])
    assert facts["accepted"] is False


def test_schema_valid_distance_metric_tamper_fails_independent_recomputation(
    tmp_path, monkeypatch
):
    _prepare_accepted_site_distance(tmp_path, monkeypatch)
    evaluation_path = tmp_path / "validation/results/distance-25m/evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["profiles"]["640"]["metrics"]["precision"] = 0.99
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    section, facts = campaign._build_distance_section(
        campaign.EvidenceStore(tmp_path),
        tmp_path / "validation/results",
    )

    assert section["accepted"] is False
    assert section["schema_contract_valid"] is True
    assert section["pin_matrix_valid"] is True
    assert section["independent_cpu_recomputation_valid"] is False
    assert "distance evaluation differs from independent CPU recomputation" in section[
        "reasons"
    ]
    assert facts["accepted"] is False


def test_comparison_excludes_unpaired_scene(tmp_path):
    results = _populate_accepted_campaign(tmp_path)
    missing = results / "scene-benchmark/scene_00/960/status.json"
    missing.unlink()

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
    )
    comparison = report["campaigns"]["scene_benchmark"]["paired_profile_comparison"]

    assert report["decision"]["accepted"] is False
    assert comparison["paired_scene_count"] == 11
    assert "scene_00" not in {item["scene_id"] for item in comparison["pairs"]}


def test_hash_pinned_legacy_incident_sets_blocked_by_hardware(tmp_path):
    status_path = _write_json(
        tmp_path,
        "validation/results/scene-benchmark/people_waiting_crosswalk/640/status.json",
        {
            "schema_version": "deepsafe.scene-benchmark-run/v1",
            "status": "interrupted",
        },
    )
    gpu_path = tmp_path / "validation/results/scene-benchmark/people_waiting_crosswalk/640/gpu.csv"
    log_path = tmp_path / "validation/results/scene-benchmark/people_waiting_crosswalk/640/deepstream.log"
    gpu_path.write_text("clock,power\n480,55\n", encoding="utf-8")
    log_path.write_text("fps drop\n", encoding="utf-8")
    hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (status_path, gpu_path, log_path)
    ]
    incident = tmp_path / "docs/gpu-power-throttle-incident.md"
    incident.parent.mkdir(parents=True)
    incident.write_text(
        "SW Power Cap and SW\nThermal were observed; kayip yaklasik %50.2.\n"
        + "\n".join(hashes),
        encoding="utf-8",
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=incident,
    )

    assert report["decision"]["status"] == "blocked_by_hardware"
    assert report["decision"]["final_claim_allowed"] is False
    assert report["hardware_blockers"][0]["code"] == (
        "verified_legacy_gpu_power_throttle_incident"
    )


def test_outputs_are_deterministic_and_require_accepted_is_nonzero(tmp_path):
    output = tmp_path / "validation/results/campaign-report"
    argv = [
        "--project-root",
        str(tmp_path),
        "--results-root",
        "validation/results",
        "--output-dir",
        str(output),
        "--no-hardware-incident",
    ]
    assert campaign.main(argv) == 0
    first_json = (output / "report.json").read_bytes()
    first_markdown = (output / "report.md").read_bytes()
    assert campaign.main(argv) == 0
    assert (output / "report.json").read_bytes() == first_json
    assert (output / "report.md").read_bytes() == first_markdown
    assert campaign.main([*argv, "--require-accepted"]) == 2


def test_new_optional_artifacts_fail_closed_on_valid_schema_malformed_shapes(tmp_path):
    results = tmp_path / "validation/results"
    _write_json(
        tmp_path,
        "validation/results/gpu-reentry/current/evidence.json",
        {
            "schema_version": "deepsafe.gpu-reentry-evidence/v1",
            "status": "blocked",
            "verification": [],
        },
    )
    _write_json(
        tmp_path,
        "validation/results/gpu-reentry/current/verification.json",
        {
            "schema_version": "deepsafe.gpu-reentry-verification/v1",
            "status": "blocked",
            "failed_gate_ids": [],
            "gates": {},
            "all_required_evidence_present": False,
            "sustained_load_authorized": False,
        },
    )
    _write_json(
        tmp_path,
        "validation/results/loaf/val-20-25m/deepstream/dry-run-plan.json",
        {
            "schema_version": "deepsafe.loaf-deepstream-batch-plan/v1",
            "status": "planned",
            "campaign": [],
            "source_contract": [],
            "jobs": {},
            "sequences": {},
        },
    )
    _write_json(
        tmp_path,
        "validation/results/loaf/val-20-25m/deepstream/batch-aggregate.json",
        {
            "schema_version": "deepsafe.loaf-deepstream-batch-aggregate/v1",
            "completeness": [],
        },
    )
    _write_json(
        tmp_path,
        "validation/results/loaf/val-20-25m/distance-bins/preparation-manifest.json",
        {
            "schema_version": "deepsafe.loaf-distance-bin-preparation/v1",
            "status": "prepared_not_evaluated",
            "distance_bins": {},
        },
    )
    _write_json(
        tmp_path,
        "validation/results/loaf/val-20-25m/distance-bins/evaluation-plan.json",
        {
            "schema_version": "deepsafe.loaf-distance-bin-evaluation-plan/v1",
            "profiles": {},
        },
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=results,
        hardware_incident_path=None,
    )

    assert report["campaigns"]["gpu_reentry"]["state"] == "invalid"
    assert report["campaigns"]["loaf_deepstream"]["state"] == "incomplete_or_invalid"
    assert report["campaigns"]["loaf_distance_bins"]["state"] == "incomplete_or_invalid"
    assert report["decision"]["accepted"] is False


def test_current_loaf_preparation_is_pinned_but_cannot_open_25m_gate():
    report = campaign.build_campaign_report(
        project_root=campaign.PROJECT_ROOT,
        results_root=campaign.DEFAULT_RESULTS_ROOT,
        hardware_incident_path=None,
    )
    loaf = report["campaigns"]["loaf_preparation"]
    loaf_deepstream = report["campaigns"]["loaf_deepstream"]
    loaf_bins = report["campaigns"]["loaf_distance_bins"]
    gpu_reentry = report["campaigns"]["gpu_reentry"]
    distance = report["campaigns"]["distance_25m"]

    assert loaf["state"] == "prepared_not_evaluated"
    assert loaf["artifacts_consistent"] is True
    assert loaf["provenance_sha256"] == (
        "2583e74fd620c81861d36dee117b8f25dbf9ca431465b0147e37e51b6e38970b"
    )
    assert loaf["metric_geometry"] == "axis_aligned_envelope_of_rotated_box"
    assert loaf["splits"]["val"] | {
        "target_people": 7539,
        "frames": 2948,
        "sequences": 8,
        "media_status": "encoded_and_verified",
    } == loaf["splits"]["val"]
    assert loaf["splits"]["test_unseen"] | {
        "target_people": 5544,
        "frames": 2255,
        "sequences": 8,
        "media_status": "encoded_and_verified",
    } == loaf["splits"]["test_unseen"]
    assert loaf["dataset_rights"] == {
        "license_status": "unverified",
        "internal_research_validation_only": True,
        "model_training_allowed": False,
        "redistribution_allowed": False,
        "written_rights_clearance_required": True,
        "guardrail_consistent": True,
    }
    assert loaf["can_satisfy_calibrated_25m_detection"] is False
    assert loaf_deepstream["state"] == "complete"
    assert loaf_deepstream["plan_contract_valid"] is True
    assert loaf_deepstream["aggregate_contract_valid"] is True
    assert loaf_deepstream["completion_evidence_valid"] is True
    assert loaf_deepstream["complete_jobs"] == 16
    assert loaf_deepstream["expected_jobs"] == 16
    assert loaf_deepstream["metrics_withheld"] is False
    assert loaf_bins["state"] == "complete"
    assert loaf_bins["preparation_contract_valid"] is True
    assert loaf_bins["evaluation_plan_contract_valid"] is True
    assert [item["target_people"] for item in loaf_bins["distance_bins"]] == [
        2298,
        1702,
        1333,
        1210,
        996,
    ]
    assert loaf_bins["aggregate_contract_valid"] is True
    assert loaf_bins["completion_evidence_valid"] is True
    assert loaf_bins["complete_evaluations"] == 10
    assert loaf_bins["expected_evaluations"] == 10
    assert gpu_reentry["state"] in {
        "blocked",
        "ready_for_operator_review",
        "stale",
    }
    assert gpu_reentry["verification_consistent"] is True
    assert gpu_reentry["sustained_load_authorized"] is False
    assert distance["state"] == "unproven"
    assert distance["accepted"] is False
    gate = next(
        item for item in report["requirements"] if item["id"] == "calibrated_25m_detection"
    )
    assert gate["state"] == "unproven"
    assert gate["evidence_ids"] == ["distance_25m_evaluation"]


def test_rlivit_metric_row_accepts_only_consistent_runner_diagnostics():
    row = {
        **_rlivit_metric_row(10),
        "evaluated_predictions": 10,
        "ignored_predictions": 2,
        "serialized_predictions_at_or_above_confidence": 12,
        "ap_serialized_predictions": 20,
        "ap_ignored_predictions": 5,
    }
    projected = campaign._rlivit_metric_row(row)
    assert projected == row
    assert list(projected) == [
        "ground_truth",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "ap_101_point",
        "evaluated_predictions",
        "ignored_predictions",
        "serialized_predictions_at_or_above_confidence",
        "ap_serialized_predictions",
        "ap_ignored_predictions",
    ]

    for field, invalid in (
        ("evaluated_predictions", 9),
        ("ignored_predictions", 3),
        ("serialized_predictions_at_or_above_confidence", 21),
        ("ap_serialized_predictions", 11),
        ("ap_ignored_predictions", 1),
    ):
        candidate = copy.deepcopy(row)
        candidate[field] = invalid
        assert campaign._rlivit_metric_row(candidate) is None

    partial = copy.deepcopy(row)
    partial.pop("ap_ignored_predictions")
    assert campaign._rlivit_metric_row(partial) is None

    unknown = copy.deepcopy(row)
    unknown["private_diagnostic"] = 1
    assert campaign._rlivit_metric_row(unknown) is None


def test_rlivit_complete_cross_binds_runner_diagnostics(tmp_path):
    metrics = _rlivit_profile_metrics()
    for group, value in metrics.items():
        rows = {"overall": value} if group == "overall" else value
        for row in rows.values():
            row.update(
                {
                    "evaluated_predictions": row["tp"] + row["fp"],
                    "ignored_predictions": 0,
                    "serialized_predictions_at_or_above_confidence": row["tp"]
                    + row["fp"],
                    "ap_serialized_predictions": row["tp"] + row["fp"],
                    "ap_ignored_predictions": 0,
                }
            )
    _populate_rlivit_complete_evidence(tmp_path, profile_metrics=metrics)
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )
    section = report["campaigns"]["rlivit_ground_truth"]
    assert section["state"] == "complete"
    assert section["private_completion_proof"][
        "aggregate_status_metrics_cross_bound"
    ] is True
    assert section["profiles"]["640"]["metrics"]["overall"][
        "evaluated_predictions"
    ] == 4318


def test_rlivit_mp4_temperature_is_record_only_only_for_workstation_policy():
    batch_pin = {"size_bytes": 15423, "sha256": "a" * 64}
    managed = _rlivit_mp4_public_receipt(batch_pin)
    managed["maximum_cpu_platform_temperature_millidegrees_celsius"] = 100000
    _rlivit_repin(managed)
    assert campaign._rlivit_mp4_public_contract(managed) == batch_pin

    explicit_strict = copy.deepcopy(managed)
    explicit_strict["thermal_policy_id"] = "legacy_strict"
    _rlivit_repin(explicit_strict)
    assert campaign._rlivit_mp4_public_contract(explicit_strict) is None

    legacy = copy.deepcopy(managed)
    legacy.pop("thermal_policy_id")
    legacy["maximum_cpu_platform_temperature_millidegrees_celsius"] = 84999
    _rlivit_repin(legacy)
    assert campaign._rlivit_mp4_public_contract(legacy) == batch_pin
    legacy["maximum_cpu_platform_temperature_millidegrees_celsius"] = 85000
    _rlivit_repin(legacy)
    assert campaign._rlivit_mp4_public_contract(legacy) is None

    unknown = copy.deepcopy(managed)
    unknown["thermal_policy_id"] = "unbounded"
    _rlivit_repin(unknown)
    assert campaign._rlivit_mp4_public_contract(unknown) is None
