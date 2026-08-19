import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

from evaluation.metrics import evaluate_person_predictions
from evaluation.readers import load_caviar, load_predictions_jsonl
from validation import person_quality_policy as quality
from validation import run_caviar
from validation.scene_benchmark.run_matrix import GPU_CSV_HEADER, assess_gpu_safety


FIXTURE_NONCE = "a" * 64
FIRST_SEQUENCE = "Fight_OneManDown"


def _first_run(root: Path) -> Path:
    return (
        root
        / "validation/results/caviar/sessions"
        / FIXTURE_NONCE
        / FIRST_SEQUENCE
        / "640"
    )


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _json(path: Path, value: dict) -> Path:
    return _write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _draft() -> dict:
    policy = json.loads(
        (quality.PROJECT_ROOT / "validation/person-quality-policy.draft.json").read_text()
    )
    runtime = policy["campaign_contract"]["runtime_binding"]
    placeholder = lambda path: {"path": path, "bytes": 1, "sha256": "0" * 64}
    runtime["control_artifacts"] = {
        name: runtime.get("control_artifacts", {}).get(name, placeholder(path))
        for name, path in quality.CONTROL_ARTIFACT_PATHS.items()
    }
    runtime["native_parser"] = {
        **copy.deepcopy(quality.NATIVE_PARSER_CONTRACT),
        "dockerfile": runtime["control_artifacts"]["deepstream_image_build"],
    }
    runtime["gpu_contract"] = copy.deepcopy(quality.TARGET_GPU_CONTRACT)
    runtime["execution_contract"] = {
        **copy.deepcopy(quality.DEFAULT_EXECUTION_CONTRACT),
        "reentry_evidence": placeholder(
            "validation/results/gpu-reentry/current/evidence.json"
        ),
        "ds9_compatibility_receipt": copy.deepcopy(
            quality.DS9_COMPATIBILITY_RECEIPT_CONTRACT
        ),
    }
    policy["campaign_contract"]["dataset_catalog"] = [
        {
            "sequence_id": sequence_id,
            "video": placeholder(paths["video"]),
            "ground_truth": placeholder(paths["ground_truth"]),
            "video_metadata": {"width": 100, "height": 100, "fps": 25.0, "frames": 1},
            "frame_mapping": {
                "index_base": 0,
                "decoded_frame_count": 1,
                "decoded_frame_range": [0, 0],
                "annotated_frame_count": 1,
                "annotated_frame_range": [0, 0],
                "annotated_frames_sha256": hashlib.sha256(b"0").hexdigest(),
                "unannotated_decoded_frame_count": 0,
                "unannotated_decoded_ranges": [],
                "xml_frames_outside_video": [],
                "prediction_export_policy": quality.PREDICTION_EXPORT_POLICY,
            },
        }
        for sequence_id, paths in quality.NATIVE_CAVIAR_DATASET_PATHS.items()
    ]
    return policy


def _fixture_runtime_contract(root: Path) -> dict:
    profiles = {}
    for profile in quality.PROFILES:
        pins = {}
        for name, relative in quality.MODEL_ARTIFACT_PATHS[profile].items():
            path = _write(root / relative, f"fixture-{profile}-{name}\n")
            pins[name] = quality.make_file_pin(path, project_root=root)
        profiles[str(profile)] = {
            "model_id": f"yolo11s-{profile}-fp16",
            "model_artifacts": pins,
        }
    controls = {}
    for name, relative in quality.CONTROL_ARTIFACT_PATHS.items():
        path = _write(root / relative, f"fixture-control-{name}\n")
        controls[name] = quality.make_file_pin(path, project_root=root)
    reentry_path = _json(
        root / "validation/results/gpu-reentry/current/evidence.json",
        {"schema_version": "fixture-reentry/v1", "status": "ready"},
    )
    return {
        "requested_container_image": "deepsafe-deepstream:9.0",
        "profiles": profiles,
        "control_artifacts": controls,
        "native_parser": {
            **copy.deepcopy(quality.NATIVE_PARSER_CONTRACT),
            "dockerfile": controls["deepstream_image_build"],
        },
        "gpu_contract": copy.deepcopy(quality.TARGET_GPU_CONTRACT),
        "execution_contract": {
            **copy.deepcopy(quality.DEFAULT_EXECUTION_CONTRACT),
            "reentry_evidence": quality.make_file_pin(
                reentry_path, project_root=root
            ),
            "ds9_compatibility_receipt": copy.deepcopy(
                quality.DS9_COMPATIBILITY_RECEIPT_CONTRACT
            ),
        },
    }


def _fixture_dataset_catalog(root: Path) -> list[dict]:
    catalog = []
    for sequence_id, paths in quality.NATIVE_CAVIAR_DATASET_PATHS.items():
        video = _write(root / paths["video"], f"fixture-video-{sequence_id}\n")
        ground_truth = _write(root / paths["ground_truth"], _xml())
        catalog.append(
            {
                "sequence_id": sequence_id,
                "video": quality.make_file_pin(video, project_root=root),
                "ground_truth": quality.make_file_pin(
                    ground_truth, project_root=root
                ),
                "video_metadata": {
                    "width": 100,
                    "height": 100,
                    "fps": 25.0,
                    "frames": 1,
                },
                "frame_mapping": {
                    "index_base": 0,
                    "decoded_frame_count": 1,
                    "decoded_frame_range": [0, 0],
                    "annotated_frame_count": 1,
                    "annotated_frame_range": [0, 0],
                    "annotated_frames_sha256": hashlib.sha256(b"0").hexdigest(),
                    "unannotated_decoded_frame_count": 0,
                    "unannotated_decoded_ranges": [],
                    "xml_frames_outside_video": [],
                    "prediction_export_policy": quality.PREDICTION_EXPORT_POLICY,
                },
            }
        )
    return catalog


def _immutable_json(path: Path, value: dict) -> Path:
    _json(path, value)
    path.chmod(0o440)
    return path


def _guard_pin(root: Path, path: Path, *, allow_empty: bool = False) -> dict:
    return {
        **quality.make_file_pin(path, project_root=root),
        "allow_empty": allow_empty,
    }


def _gpu_row(
    timestamp: str,
    gpu_identity: dict,
    *,
    static_diagnostics: bool = False,
    malformed: bool = False,
    identity_drift: bool = False,
) -> list[str]:
    row = {
        "timestamp": timestamp,
        "gpu_index": gpu_identity["index"],
        "gpu_name": (
            "Unexpected GPU" if identity_drift else gpu_identity["name"]
        ),
        "gpu_utilization_percent": "40",
        "memory_utilization_percent": "15",
        "memory_used_mib": "2048",
        "memory_total_mib": gpu_identity["memory.total"],
        "temperature_c": "90" if static_diagnostics else "60",
        "power_draw_w": "55",
        "sm_clock_mhz": "1200",
        "memory_clock_mhz": "6001",
        "power_requested_limit_w": "55" if static_diagnostics else "115",
        "power_current_limit_w": (
            "unreadable" if malformed else ("55" if static_diagnostics else "115")
        ),
        "power_default_limit_w": "115",
        "pstate": "P0",
        "clock_event_reasons_active_mask": "0x0",
        "clock_event_sw_power_cap": "Not Active",
        "clock_event_sw_thermal_slowdown": "Not Active",
        "clock_event_hw_slowdown": "Not Active",
        "clock_event_hw_thermal_slowdown": (
            "Active" if static_diagnostics else "Not Active"
        ),
        "clock_event_hw_power_brake_slowdown": "Not Active",
    }
    return [str(row[name]) for name in GPU_CSV_HEADER]


def _xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<dataset name="fixture">
  <frame number="0">
    <objectlist><object id="0"><box xc="50" yc="50" w="20" h="20"/></object></objectlist>
    <grouplist/>
  </frame>
</dataset>
"""


def _prediction(sequence: str, profile: int) -> dict:
    return {
        "schema_version": "deepsafe.person-detections/v1",
        "sequence_id": sequence,
        "frame_index": 0,
        "image_width": 100,
        "image_height": 100,
        "source_uri": f"file:///{sequence}.mp4",
        "model_id": f"yolo11s-{profile}-fp16",
        "detections": [
            {
                "class_name": "person",
                "confidence": 0.99,
                "bbox_norm_xywh": [0.4, 0.4, 0.2, 0.2],
            }
        ],
    }


def _approved_policy(
    root: Path,
    *,
    approved_at: str = "2026-07-16T08:00:00Z",
    expires_at: str = "2026-07-16T11:00:00Z",
    policy_mutator=None,
) -> Path:
    policy = _draft()
    policy["status"] = "approved"
    policy["policy_owner"] = {"identity": "owner@example.test"}
    policy["disclaimer"] = "Owner-approved quality gate; this is not whole-campaign acceptance."
    policy["campaign_contract"]["runtime_binding"] = _fixture_runtime_contract(root)
    policy["campaign_contract"]["dataset_catalog"] = _fixture_dataset_catalog(root)
    if policy_mutator is not None:
        policy_mutator(policy)
    approval = {
        "schema_version": quality.APPROVAL_SCHEMA,
        "decision": "approved",
        "policy_id": policy["policy_id"],
        "policy_contract_sha256": quality.policy_contract_sha256(policy),
        "owner_identity": policy["policy_owner"]["identity"],
        "approved_at_utc": approved_at,
        "campaign_nonce": FIXTURE_NONCE,
        "expires_at_utc": expires_at,
        "authorized_results_root": (
            f"validation/results/caviar/sessions/{FIXTURE_NONCE}"
        ),
        "single_use": True,
    }
    approval_path = _json(root / "inputs/person-quality-approval.json", approval)
    policy["approval_artifact"] = quality.make_file_pin(
        approval_path, project_root=root
    )
    return _json(root / "inputs/person-quality-policy.json", policy)


def _campaign(
    root: Path,
    policy_path: Path,
    *,
    operating_policy_id: str = quality.WORKSTATION_MANAGED_POLICY_ID,
    static_diagnostics: bool = False,
    malformed_telemetry: bool = False,
    runtime_identity_drift: bool = False,
    postflight_driver_drift: bool = False,
    insufficient_coverage: bool = False,
    guard_fault: str | None = None,
    coverage_override: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    runtime_contract = policy["campaign_contract"]["runtime_binding"]
    dataset_catalog = policy["campaign_contract"]["dataset_catalog"]
    approval = json.loads(
        (root / policy["approval_artifact"]["path"]).read_text(encoding="utf-8")
    )
    authorization = {
        key: approval[key]
        for key in (
            "campaign_nonce",
            "expires_at_utc",
            "authorized_results_root",
            "single_use",
        )
    }
    operating_policy = quality.operating_policy_contract(operating_policy_id)
    workstation_managed = (
        operating_policy_id == quality.WORKSTATION_MANAGED_POLICY_ID
    )
    results = root / authorization["authorized_results_root"]
    policy_pin = quality.make_file_pin(policy_path, project_root=root)
    quality_binding = {
        "artifact": policy_pin,
        "policy_id": policy["policy_id"],
        "policy_contract_sha256": quality.policy_contract_sha256(policy),
        "status": "approved",
        "campaign_authorization": authorization,
    }
    resolved_image_id = "sha256:" + "1" * 64
    runtime_control_manifest_sha256 = runtime_contract["control_artifacts"][
        "runtime_control_manifest"
    ]["sha256"]
    ds9_receipt_path = _immutable_json(
        root
        / runtime_contract["execution_contract"]["ds9_compatibility_receipt"][
            "path"
        ],
        {
            "schema_version": "fixture.ds9-runtime-compatibility-receipt/v1",
            "status": "production_ready",
            "resolved_image_id": resolved_image_id,
            "runtime_control_manifest_sha256": runtime_control_manifest_sha256,
        },
    )
    ds9_binding = {
        "status": "production_ready",
        "receipt": quality.make_file_pin(ds9_receipt_path, project_root=root),
        "resolved_image_id": resolved_image_id,
        "runtime_control_manifest_sha256": runtime_control_manifest_sha256,
        "gpu_smoke": {
            "schema_version": quality.GPU_SMOKE_SCHEMA_VERSION,
            "status": "pass",
            "resolved_image_id": resolved_image_id,
            "runtime_control_manifest_sha256": runtime_control_manifest_sha256,
            "checks": {
                name: "pass" for name in quality.REQUIRED_GPU_SMOKE_CHECKS
            },
        },
    }
    session_claim_path = _immutable_json(
        results / "session-claim.json",
        {
            "schema_version": "deepsafe.caviar-session-claim/v1",
            "claimed_at_utc": "2026-07-16T08:30:00Z",
            "campaign_nonce": FIXTURE_NONCE,
            "authorized_results_root": authorization["authorized_results_root"],
            "single_use": True,
            "quality_policy": quality_binding,
        },
    )
    session_claim_pin = quality.make_file_pin(session_claim_path, project_root=root)
    sequences = []
    jobs = []
    rows = {}
    aggregate_rows = []
    manifest_relative = (
        authorization["authorized_results_root"] + "/batch-manifest.json"
    )
    for dataset_item in dataset_catalog:
        sequence_id = dataset_item["sequence_id"]
        video = root / dataset_item["video"]["path"]
        gt = root / dataset_item["ground_truth"]["path"]
        sequence = {
            "sequence_id": sequence_id,
            "collection": "CAVIARDATA1" if "CAVIARDATA1" in str(video) else "CAVIARDATA2",
            "video": str(video.relative_to(root)),
            "ground_truth": str(gt.relative_to(root)),
            "video_artifact": dataset_item["video"],
            "ground_truth_artifact": dataset_item["ground_truth"],
            "video_metadata": copy.deepcopy(dataset_item["video_metadata"]),
            "frame_mapping": copy.deepcopy(dataset_item["frame_mapping"]),
        }
        sequences.append(sequence)
        for profile in quality.PROFILES:
            run_root = results / sequence_id / str(profile)
            job_id = f"{sequence_id}:{profile}"
            model_contract = copy.deepcopy(runtime_contract["profiles"][str(profile)])
            planned_command = [
                sys.executable,
                "-m",
                "validation.run_caviar",
                "--sequence",
                sequence_id,
                "--video",
                sequence["video"],
                "--ground-truth",
                sequence["ground_truth"],
                "--model-size",
                str(profile),
                "--export-threshold",
                "0.001",
                "--evaluation-confidence",
                "0.25",
                "--iou",
                "0.5",
                "--parser",
                "cuda",
                "--max-nvinfer-upscale",
                "1.0",
                "--streammux-policy",
                "no-nvinfer-upscale",
                "--container-image",
                runtime_contract["requested_container_image"],
                "--gpu",
                "0",
                "--reentry-evidence",
                "validation/results/gpu-reentry/current/evidence.json",
                "--ds9-compatibility-receipt",
                runtime_contract["execution_contract"][
                    "ds9_compatibility_receipt"
                ]["path"],
                "--max-temperature-c",
                "82.0",
                "--power-limit-drop-tolerance-w",
                "5.0",
                "--slowdown-consecutive-samples",
                "2",
                "--kill-grace",
                "15",
                "--run-root",
                str(run_root.relative_to(root)),
                "--batch-manifest",
                manifest_relative,
                "--job-id",
                job_id,
            ]
            predictions = _write(
                run_root / "predictions.jsonl",
                json.dumps(_prediction(sequence_id, profile)) + "\n",
            )
            evaluated = evaluate_person_predictions(
                load_caviar(
                    gt,
                    sequence_id=sequence_id,
                    image_width=100,
                    image_height=100,
                ),
                load_predictions_jsonl(predictions),
                iou_threshold=0.5,
                confidence_threshold=0.25,
            )
            _json(run_root / "evaluation.json", evaluated)
            conversion = {
                "sequence_id": sequence_id,
                "decoded_frame_files": 1,
                "exported_frame_records": 1,
                "skipped_unannotated_frames": 0,
            }
            _json(run_root / "conversion.json", conversion)
            (run_root / "kitti").mkdir(parents=True, exist_ok=True)
            infer_path = _write(
                run_root / "generated/config-infer-primary.txt",
                run_caviar.render_infer_config(profile, 0.001, parser="cuda"),
            )
            app_path = _write(
                run_root / "generated/deepstream-app.txt",
                run_caviar.render_deepstream_config_paths(
                    video_container_path="/workspace/" + sequence["video"],
                    kitti_container_path=(
                        "/workspace/" + str((run_root / "kitti").relative_to(root))
                    ),
                    infer_config_container_path=(
                        "/workspace/" + str(infer_path.relative_to(root))
                    ),
                    width=profile,
                    height=profile,
                ),
            )
            container_name = f"fixture-caviar-{sequence_id.lower()}-{profile}"
            requested_docker_command = run_caviar.build_docker_command(
                container_name=container_name,
                image=runtime_contract["requested_container_image"],
                gpu=runtime_contract["gpu_contract"]["uuid"],
                app_config=app_path,
                repo_root=root,
            )
            docker_command = list(requested_docker_command)
            docker_command[docker_command.index(runtime_contract["requested_container_image"])] = (
                resolved_image_id
            )
            gpu_identity = copy.deepcopy(runtime_contract["gpu_contract"])
            preflight_diagnostics = (
                [
                    {
                        "code": "temperature_threshold",
                        "reason": "fixture temperature reached the static threshold",
                        "operating_policy_mode": "workstation_managed",
                        "measurement_quality_signal": True,
                        "disposition": "record_only_workstation_hardware_managed",
                    },
                    {
                        "code": "power_limit_below_default",
                        "reason": "fixture current power limit is below default",
                        "operating_policy_mode": "workstation_managed",
                        "measurement_quality_signal": True,
                        "disposition": "record_only_workstation_hardware_managed",
                    },
                    {
                        "code": "sustained_clock_slowdown",
                        "reason": "fixture slowdown clock-event flag is active",
                        "operating_policy_mode": "workstation_managed",
                        "measurement_quality_signal": True,
                        "disposition": "record_only_workstation_hardware_managed",
                    },
                ]
                if static_diagnostics
                else []
            )
            preflight = {
                "status": "ok",
                "checked_at_utc": "2026-07-16T09:00:52Z",
                "image": runtime_contract["requested_container_image"],
                "image_id": resolved_image_id,
                "gpu_identity": gpu_identity,
                "power_profile": {"available": True, "value": "performance"},
                "power_safety_policy": {
                    "fail_closed": True,
                    "operating_policy_mode": (
                        "workstation_managed" if workstation_managed else "legacy_strict"
                    ),
                },
                "safety_events": [],
                "guard_operating_policy_id": operating_policy_id,
                "guard_disposition": (
                    "native_record_only_workstation_static_signals"
                    if preflight_diagnostics
                    else "clean"
                ),
            }
            if preflight_diagnostics:
                preflight.update(
                    {
                        "collector_status": "ok",
                        "diagnostic_events": preflight_diagnostics,
                    }
                )
            preflight_path = _json(
                run_root / "safety/gpu-preflight.json", preflight
            )
            deepstream_log = _write(
                run_root / "deepstream.log",
                (
                    "deserialized trt engine from :"
                    f"/workspace/models/person/{profile}/yolo11s_b12_gpu0_fp16.engine\n"
                    "Use deserialized engine model: "
                    f"/workspace/models/person/{profile}/yolo11s_b12_gpu0_fp16.engine\n"
                    "**PERF: 25.0 (25.0)\n"
                ),
            )
            timeline = {
                "guard_started_at_utc": "2026-07-16T09:00:50Z",
                "reentry_verified_at_utc": "2026-07-16T09:00:51Z",
                "preflight_checked_at_utc": "2026-07-16T09:00:52Z",
                "ds9_compatibility_verified_at_utc": (
                    "2026-07-16T09:00:52.500000Z"
                ),
                "process_started_at_utc": "2026-07-16T09:00:53Z",
                "process_finished_at_utc": "2026-07-16T09:00:55Z",
                "postflight_checked_at_utc": "2026-07-16T09:00:56Z",
                "guard_finished_at_utc": "2026-07-16T09:00:57Z",
            }
            gpu_csv_path = run_root / "safety/gpu.csv"
            gpu_csv_path.parent.mkdir(parents=True, exist_ok=True)
            sample_timestamps = (
                ("2026-07-16T09:00:53Z",)
                if insufficient_coverage
                else (
                    "2026-07-16T09:00:53Z",
                    "2026-07-16T09:00:54Z",
                )
            )
            gpu_rows = [
                _gpu_row(
                    timestamp,
                    gpu_identity,
                    static_diagnostics=static_diagnostics,
                    malformed=malformed_telemetry,
                    identity_drift=runtime_identity_drift,
                )
                for timestamp in sample_timestamps
            ]
            with gpu_csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(GPU_CSV_HEADER)
                writer.writerows(gpu_rows)
            platform_csv_path = _write(
                run_root / "safety/platform-thermal.csv",
                "timestamp\n" + "\n".join(sample_timestamps) + "\n",
            )
            reentry = {
                "report_path": runtime_contract["execution_contract"][
                    "reentry_evidence"
                ]["path"],
                "report_sha256": runtime_contract["execution_contract"][
                    "reentry_evidence"
                ]["sha256"],
                "collected_at_utc": "2026-07-16T09:00:45Z",
                "operating_policy": operating_policy,
                "verification": {
                    "status": "ready_for_operator_review",
                    "operating_policy": operating_policy,
                },
                "load_authority_granted_by_gate": False,
                "execution_authority": {
                    "source": "explicit_user_instruction",
                    "granted_by_this_evidence": False,
                    "cryptographic_identity_authentication": False,
                },
                "gpu_identity": {"available": True, "fields": gpu_identity},
            }
            sample_count = len(sample_timestamps)
            runtime_diagnostics = {
                "operating_policy_id": operating_policy_id,
                "signal_action": (
                    "record_only" if workstation_managed else "software_abort"
                ),
                "maximum_temperature_c": 90.0 if static_diagnostics else 60.0,
                "temperature_threshold_c": 82.0,
                "temperature_threshold_samples": (
                    sample_count if static_diagnostics else 0
                ),
                "power_limit_drop_tolerance_w": 5.0,
                "power_limit_drop_samples": sample_count if static_diagnostics else 0,
                "slowdown_active_samples": sample_count if static_diagnostics else 0,
                "max_consecutive_slowdown_samples": (
                    sample_count if static_diagnostics else 0
                ),
            }
            native_policy_mode = (
                "workstation_managed" if workstation_managed else "legacy_strict"
            )
            monitor_disposition = (
                "record_only_workstation_hardware_managed"
                if workstation_managed
                else "safety_abort"
            )
            runtime_monitor_events = []
            runtime_monitor_event_counts = {}
            consecutive_slowdown = 0
            for sample_number, row in enumerate(gpu_rows, start=1):
                snapshot = dict(zip(GPU_CSV_HEADER, row, strict=True))
                codes = []
                if static_diagnostics:
                    codes.extend(
                        ["temperature_threshold", "power_limit_below_default"]
                    )
                    consecutive_slowdown += 1
                    if consecutive_slowdown == 2:
                        codes.append("sustained_clock_slowdown")
                for code in codes:
                    runtime_monitor_event_counts[code] = (
                        runtime_monitor_event_counts.get(code, 0) + 1
                    )
                    runtime_monitor_events.append(
                        {
                            "schema_version": "deepsafe.gpu-safety-event/v1",
                            "detected_at_utc": sample_timestamps[sample_number - 1],
                            "code": code,
                            "reason": f"fixture {code.replace('_', ' ')}",
                            "sample_number": sample_number,
                            "gpu_csv_line_number": sample_number + 1,
                            "snapshot": snapshot,
                            "assessment": assess_gpu_safety(
                                snapshot, power_limit_drop_tolerance_w=5.0
                            ),
                            "operating_policy_mode": native_policy_mode,
                            "measurement_quality_signal": True,
                            "disposition": monitor_disposition,
                        }
                    )
            diagnostics = {
                "preflight_static_signals": preflight_diagnostics,
                "runtime_static_signals": runtime_diagnostics,
                "runtime_monitor_events": runtime_monitor_events,
                "runtime_monitor_event_counts": runtime_monitor_event_counts,
            }
            telemetry = {
                "samples": sample_count,
                "query_error_count": 0,
                "query_errors_first_50": [],
                "gpu_identity_checks": 1,
                "last_observed_gpu_identity": gpu_identity,
                "power_limit_drop_samples": runtime_diagnostics[
                    "power_limit_drop_samples"
                ],
                "slowdown_active_samples": runtime_diagnostics[
                    "slowdown_active_samples"
                ],
                "max_consecutive_slowdown_samples": runtime_diagnostics[
                    "max_consecutive_slowdown_samples"
                ],
                "platform_thermal_samples": sample_count,
                "platform_thermal_read_error_count": 0,
                "safety_event": None,
                "coverage": {
                    "interval_seconds": 1.0,
                    "endpoint_tolerance_seconds": 1.5,
                    "sample_count_grace_seconds": 0.5,
                    "process_duration_seconds": 2.0,
                    "expected_minimum_samples": 2,
                    "csv_samples": sample_count,
                    "coverage_satisfied": not insufficient_coverage,
                    "first_sample_at_utc": sample_timestamps[0],
                    "last_sample_at_utc": sample_timestamps[-1],
                    "telemetry_valid": True,
                    "malformed_sample_count": 0,
                    "malformed_samples_first_50": [],
                    "identity_drift_sample_count": 0,
                    "identity_drift_samples_first_50": [],
                    "quality_diagnostics": runtime_diagnostics,
                },
            }
            if coverage_override is not None:
                telemetry["coverage"].update(coverage_override)
            guard_receipt = {
                "schema_version": "deepsafe.gpu-guard-artifact-receipt/v1",
                "created_at_utc": timeline["guard_finished_at_utc"],
                "guard_status": "complete",
                "compatibility_mode": "production",
                "operating_policy_id": operating_policy_id,
                "operating_policy": operating_policy,
                "requested_image": runtime_contract["requested_container_image"],
                "resolved_image_id": resolved_image_id,
                "requested_command": requested_docker_command,
                "executed_command": docker_command,
                "running_container": {
                    "name": container_name,
                    "image_id": resolved_image_id,
                },
                "timeline": timeline,
                "reentry_evidence": reentry,
                "ds9_runtime_compatibility": ds9_binding,
                "diagnostics": diagnostics,
                "artifacts": {
                    "preflight": _guard_pin(root, preflight_path),
                    "gpu_csv": _guard_pin(root, gpu_csv_path),
                    "platform_thermal_csv": _guard_pin(root, platform_csv_path),
                    "deepstream_log": _guard_pin(root, deepstream_log),
                },
                "safety_event": {
                    "present": False,
                    "disposition": None,
                    "path": str(
                        (run_root / "safety/gpu-safety-event.json").relative_to(root)
                    ),
                },
                "record_only_diagnostic_event": {
                    "present": False,
                    "disposition": None,
                    "path": str(
                        (run_root / "safety/gpu-safety-event.json").relative_to(root)
                    ),
                },
            }
            guard_receipt_path = _immutable_json(
                run_root / "safety/gpu-guard-artifact-receipt.json",
                guard_receipt,
            )
            guard_receipt_pin = quality.make_file_pin(
                guard_receipt_path, project_root=root
            )
            guard_report = {
                "schema_version": "deepsafe.gpu-guarded-process/v1",
                "status": "complete",
                "compatibility_mode": "production",
                "operating_policy_id": operating_policy_id,
                "operating_policy": operating_policy,
                "started_at_utc": timeline["guard_started_at_utc"],
                "finished_at_utc": timeline["guard_finished_at_utc"],
                "gpu_index": 0,
                "container_name": container_name,
                "image": runtime_contract["requested_container_image"],
                "requested_image": runtime_contract["requested_container_image"],
                "resolved_image_id": resolved_image_id,
                "requested_command": requested_docker_command,
                "command": docker_command,
                "policy": {
                    "operating_policy_id": operating_policy_id,
                    "max_temperature_c": 82.0,
                    "power_limit_drop_tolerance_w": 5.0,
                    "slowdown_consecutive_samples": 2,
                    "gpu_sample_interval_seconds": 1.0,
                    "kill_grace_seconds": 15,
                    "fail_closed": True,
                    "temperature_power_slowdown_action": (
                        "record_only" if workstation_managed else "software_abort"
                    ),
                    "unreadable_telemetry_action": "abort",
                    "hardware_protection_owner": operating_policy[
                        "hardware_protection_owner"
                    ],
                },
                "diagnostics": diagnostics,
                "process": {
                    "started": True,
                    "command": docker_command,
                    "container_image_id": resolved_image_id,
                    "exit_code": 0,
                    "termination_method": None,
                },
                "timeline": timeline,
                "failure_reasons": [],
                "reentry_evidence": reentry,
                "ds9_runtime_compatibility": ds9_binding,
                "artifacts": {
                    "ds9_runtime_compatibility_receipt": (
                        runtime_contract["execution_contract"][
                            "ds9_compatibility_receipt"
                        ]["path"]
                    ),
                },
                "preflight": {
                    "status": "ok",
                    "checked_at_utc": preflight["checked_at_utc"],
                    "requested_image": runtime_contract[
                        "requested_container_image"
                    ],
                    "resolved_image_id": resolved_image_id,
                    "gpu_identity": gpu_identity,
                    "power_profile": preflight["power_profile"],
                    "power_safety_policy": preflight["power_safety_policy"],
                    "safety_events": [],
                },
                "pre_run": {"xid": {"available": True, "lines": [], "count": 0}},
                "telemetry": telemetry,
                "postflight": {
                    "xid": {"available": True, "lines": [], "count": 0},
                    "new_xid_lines": [],
                    "power_profile": {"available": True, "value": "performance"},
                    "gpu_identity": {
                        **gpu_identity,
                        **(
                            {"driver_version": "unexpected-driver"}
                            if postflight_driver_drift
                            else {}
                        ),
                    },
                    "gpu_identity_error": None,
                    "fatal_log_patterns": {
                        "gstreamer_error": 0,
                        "pipeline_failed": 0,
                        "cuda_error": 0,
                        "engine_deserialize_error": 0,
                        "out_of_memory": 0,
                    },
                },
                "artifact_receipt": guard_receipt_pin,
            }
            if guard_fault == "query_error":
                guard_report["telemetry"].update(
                    {
                        "query_error_count": 1,
                        "query_errors_first_50": ["fixture nvidia-smi failure"],
                    }
                )
            elif guard_fault == "identity_monitor_missing":
                guard_report["telemetry"].update(
                    {
                        "gpu_identity_checks": 0,
                        "last_observed_gpu_identity": None,
                    }
                )
            elif guard_fault == "xid":
                guard_report["pre_run"]["xid"] = {
                    "available": True,
                    "lines": ["NVRM: Xid fixture"],
                    "count": 1,
                }
            elif guard_fault == "fatal":
                guard_report["postflight"]["fatal_log_patterns"][
                    "cuda_error"
                ] = 1
            elif guard_fault == "process":
                guard_report["process"]["exit_code"] = 1
            elif guard_fault is not None:
                raise AssertionError(f"unknown guard fixture fault: {guard_fault}")
            guard_report_path = _json(
                run_root / "safety/gpu-guard-report.json", guard_report
            )
            manifest = {
                "status": "complete",
                "sequence_id": sequence_id,
                "model": f"yolo11s-{profile}-fp16",
                "model_input": profile,
                "video": sequence["video"],
                "ground_truth": sequence["ground_truth"],
                "bbox_parser": "cuda",
                "export_threshold": 0.001,
                "evaluation_confidence": 0.25,
                "iou": 0.5,
                "streammux": {
                    "width": profile,
                    "height": profile,
                    "source_to_mux_scale": profile / 100,
                    "max_nvinfer_upscale": 1.0,
                    "policy": "no-nvinfer-upscale",
                },
                "deepstream_config": str(app_path.relative_to(root)),
                "infer_config": str(infer_path.relative_to(root)),
                "deepstream_config_suffix_requirement": ".txt",
                "docker_requested_command": requested_docker_command,
                "docker_command": docker_command,
                "batch_binding": {
                    "job_id": job_id,
                    "planned_command": planned_command,
                    "quality_policy": quality_binding,
                    "model_contract": model_contract,
                    "dataset_item": dataset_item,
                    "campaign_authorization": authorization,
                    "session_claim": session_claim_pin,
                },
                "runtime_binding": {
                    "model_id": f"yolo11s-{profile}-fp16",
                    "model_artifacts_preflight": model_contract["model_artifacts"],
                    "model_artifacts_postflight": model_contract["model_artifacts"],
                    "control_artifacts_preflight": runtime_contract[
                        "control_artifacts"
                    ],
                    "control_artifacts_postflight": runtime_contract[
                        "control_artifacts"
                    ],
                    "input_artifacts_preflight": {
                        "video": dataset_item["video"],
                        "ground_truth": dataset_item["ground_truth"],
                    },
                    "input_artifacts_postflight": {
                        "video": dataset_item["video"],
                        "ground_truth": dataset_item["ground_truth"],
                    },
                    "generated_configs": {
                        "deepstream_app": quality.make_file_pin(
                            app_path, project_root=root
                        ),
                        "primary_inference": quality.make_file_pin(
                            infer_path, project_root=root
                        ),
                    },
                    "container": {
                        "requested_image": runtime_contract[
                            "requested_container_image"
                        ],
                        "resolved_image_id": resolved_image_id,
                        "container_name": container_name,
                        "requested_command": requested_docker_command,
                        "command": docker_command,
                    },
                },
                "gpu_safety": {
                    "status": "complete",
                    "preflight": quality.make_file_pin(
                        preflight_path, project_root=root
                    ),
                    "guard_report": quality.make_file_pin(
                        guard_report_path, project_root=root
                    ),
                    "guard_receipt": guard_receipt_pin,
                },
                "ds9_runtime_compatibility": {
                    "receipt": runtime_contract["execution_contract"][
                        "ds9_compatibility_receipt"
                    ]["path"],
                    "status": "production_ready",
                    "pending_report": str(
                        (
                            run_root
                            / "ds9-runtime-compatibility-pending.json"
                        ).relative_to(root)
                    ),
                    "binding": ds9_binding,
                },
                "engine_load_attestation": run_caviar.attest_engine_load(
                    deepstream_log, profile
                ),
            }
            manifest_path = _json(run_root / "run-manifest.json", manifest)
            job_receipt_path = _immutable_json(
                run_root / "job-receipt.json",
                {
                    "schema_version": "deepsafe.caviar-job-receipt/v1",
                    "created_at_utc": "2026-07-16T09:01:58Z",
                    "job_id": job_id,
                    "quality_policy": quality_binding,
                    "campaign_authorization": authorization,
                    "session_claim": session_claim_pin,
                    "model_contract": model_contract,
                    "dataset_item": dataset_item,
                    "artifacts": {
                        name: quality.make_file_pin(path, project_root=root)
                        for name, path in {
                            "run_manifest": manifest_path,
                            "conversion": run_root / "conversion.json",
                            "predictions": predictions,
                            "evaluation": run_root / "evaluation.json",
                            "deepstream_log": deepstream_log,
                            "deepstream_config": app_path,
                            "infer_config": infer_path,
                            "gpu_guard_report": guard_report_path,
                            "gpu_guard_receipt": guard_receipt_path,
                            "gpu_preflight": preflight_path,
                            "ds9_runtime_compatibility_receipt": (
                                ds9_receipt_path
                            ),
                        }.items()
                    },
                },
            )
            job_receipt_pin = quality.make_file_pin(
                job_receipt_path, project_root=root
            )
            jobs.append(
                {
                    "job_id": job_id,
                    "sequence_id": sequence_id,
                    "model_input": profile,
                    "run_root": str(run_root.relative_to(root)),
                    "model_contract": model_contract,
                    "command": planned_command,
                    "status": "complete",
                    "return_code": 0,
                    "started_at": "2026-07-16T09:00:40Z",
                    "finished_at": "2026-07-16T09:02:00Z",
                    "postflight": {"state": "complete", "reasons": []},
                    "job_receipt": job_receipt_pin,
                }
            )
            overall = evaluated["overall"]
            rows[(sequence_id, profile)] = overall
            aggregate_rows.append(
                {
                    "job_id": job_id,
                    "sequence_id": sequence_id,
                    "model_input": profile,
                    "status": "complete",
                    **{
                        key: overall[key]
                        for key in (
                            "ground_truth",
                            "tp",
                            "fp",
                            "fn",
                            "precision",
                            "recall",
                            "f1",
                            "ap_101_point",
                        )
                    },
                }
            )
    campaign = {
        "sequence_count": 8,
        "model_input_sizes": [640, 960],
        "expected_jobs": 16,
        "results_root": str(results.relative_to(root)),
        "export_threshold": 0.001,
        "evaluation_confidence": 0.25,
        "iou": 0.5,
        "bbox_parser": "cuda",
        "max_nvinfer_upscale": 1.0,
        "streammux_policy": "no-nvinfer-upscale",
        "container_image": runtime_contract["requested_container_image"],
        "gpu": 0,
        "quality_policy": quality_binding,
        "model_runtime_contract": runtime_contract,
        "dataset_catalog": dataset_catalog,
        "batch_manifest": manifest_relative,
        "batch_receipt": authorization["authorized_results_root"] + "/batch-receipt.json",
        "campaign_nonce": FIXTURE_NONCE,
        "session_claim": authorization["authorized_results_root"] + "/session-claim.json",
        "session_claim_artifact": session_claim_pin,
        "gpu_safety": {
            "reentry_evidence": "validation/results/gpu-reentry/current/evidence.json",
            "max_temperature_c": 82.0,
            "power_limit_drop_tolerance_w": 5.0,
            "slowdown_consecutive_samples": 2,
            "kill_grace_seconds": 15,
        },
        "ds9_runtime_compatibility": {
            "receipt": runtime_contract["execution_contract"][
                "ds9_compatibility_receipt"
            ]["path"],
            "status": "production_ready",
            "pending_report": (
                str(results.relative_to(root))
                + "/ds9-runtime-compatibility-pending.json"
            ),
            "prevalidated": ds9_binding,
        },
    }
    plan = {
        "schema_version": "deepsafe.caviar-batch-plan/v1",
        "status": "complete",
        "gpu_execution_requested": True,
        "started_at": "2026-07-16T09:00:00Z",
        "finished_at": "2026-07-16T10:00:00Z",
        "launched_jobs": 16,
        "failed_jobs": 0,
        "campaign": campaign,
        "sequences": sequences,
        "jobs": jobs,
    }
    plan_path = _json(results / "batch-manifest.json", plan)
    profiles = quality._profile_metrics(rows)
    by_profile = {}
    for profile in quality.PROFILES:
        values = profiles[str(profile)]
        by_profile[str(profile)] = {
            "complete_sequences": 8,
            "expected_sequences": 8,
            "micro": {
                key: values[key]
                for key in ("ground_truth", "tp", "fp", "fn")
            }
            | {
                "precision": values["micro_precision"],
                "recall": values["micro_recall"],
                "f1": values["micro_f1"],
            },
            "macro": {"ap_101_point": values["macro_ap50"]},
        }
    aggregate = {
        "schema_version": "deepsafe.caviar-batch-aggregate/v1",
        "generated_at": "2026-07-16T10:01:00Z",
        "completeness": {
            "expected_jobs": 16,
            "complete_jobs": 16,
            "pending_jobs": 0,
            "is_complete": True,
            "job_states": [
                {"job_id": job["job_id"], "state": "complete", "reasons": []}
                for job in jobs
            ],
        },
        "results": aggregate_rows,
        "by_model_input": by_profile,
    }
    aggregate_path = _json(results / "batch-aggregate.json", aggregate)
    _immutable_json(
        results / "batch-receipt.json",
        {
            "schema_version": "deepsafe.caviar-batch-receipt/v1",
            "created_at_utc": "2026-07-16T10:02:00Z",
            "quality_policy": quality_binding,
            "campaign_authorization": authorization,
            "campaign_nonce": FIXTURE_NONCE,
            "session_claim": session_claim_pin,
            "plan": quality.make_file_pin(plan_path, project_root=root),
            "aggregate": quality.make_file_pin(aggregate_path, project_root=root),
            "job_receipts": {
                job["job_id"]: job["job_receipt"]
                for job in sorted(jobs, key=lambda item: item["job_id"])
            },
        },
    )
    return plan_path, aggregate_path


def _evaluate(
    root: Path,
    policy: Path,
    plan: Path,
    aggregate: Path,
) -> dict:
    return quality.evaluate_quality_policy(
        policy_path=policy,
        campaign_plan_path=plan,
        aggregate_path=aggregate,
        project_root=root,
    )


def test_draft_never_reads_results_or_creates_acceptance(tmp_path: Path) -> None:
    policy = _json(tmp_path / "draft.json", _draft())
    result = quality.evaluate_quality_policy(
        policy_path=policy,
        campaign_plan_path=tmp_path / "missing-plan.json",
        aggregate_path=tmp_path / "missing-aggregate.json",
        project_root=tmp_path,
    )
    assert result["status"] == "draft_no_decision"
    assert result["acceptance_effect"] == "none"
    assert result["campaign_completeness"]["status"] == "not_evaluated"
    assert result == quality.evaluate_quality_policy(
        policy_path=policy,
        campaign_plan_path=tmp_path / "another-missing-plan.json",
        aggregate_path=tmp_path / "another-missing-aggregate.json",
        project_root=tmp_path,
    )


def test_repository_draft_has_exact_live_runtime_and_native_dataset_pins() -> None:
    binding = quality.load_policy_execution_binding(
        quality.DEFAULT_POLICY,
        project_root=quality.PROJECT_ROOT,
        require_approved=False,
    )
    assert binding["quality_policy"]["status"] == "draft"
    assert [
        item["sequence_id"] for item in binding["dataset_catalog"]
    ] == list(quality.NATIVE_CAVIAR_DATASET_PATHS)
    assert len(binding["verified_live_artifacts"]) == 44


@pytest.mark.parametrize(
    ("duration_seconds", "expected_samples"),
    ((1.5, 1), (1.500001, 2), (3.000165, 3), (3.5, 3), (3.500001, 4)),
)
def test_quality_replay_uses_the_same_half_interval_telemetry_boundary(
    duration_seconds: float, expected_samples: int
) -> None:
    assert quality._expected_telemetry_samples(duration_seconds) == expected_samples


def test_approved_pre_run_policy_passes_complete_quality_gate(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    result = _evaluate(tmp_path, policy, plan, aggregate)
    assert result["status"] == "quality_gate_passed"
    assert result["acceptance_effect"] == "person_quality_gate_only"
    assert result["campaign_completeness"]["completed_jobs"] == 16
    assert result["quality_decision"]["metrics_by_profile"]["640"] == {
        "sequences": 8,
        "ground_truth": 8,
        "tp": 8,
        "fp": 0,
        "fn": 0,
        "micro_precision": 1.0,
        "micro_recall": 1.0,
        "micro_f1": 1.0,
        "macro_ap50": 1.0,
    }
    assert all(item["status"] == "pass" for item in result["quality_decision"]["rules"])
    assert result["gpu_or_docker_executed_by_evaluator"] is False


def test_workstation_managed_record_only_static_diagnostics_pass_quality_gate(
    tmp_path: Path,
) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(
        tmp_path,
        policy,
        operating_policy_id=quality.WORKSTATION_MANAGED_POLICY_ID,
        static_diagnostics=True,
    )

    result = _evaluate(tmp_path, policy, plan, aggregate)

    assert result["status"] == "quality_gate_passed"


def test_clean_legacy_strict_policy_still_passes_quality_gate(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(
        tmp_path,
        policy,
        operating_policy_id=quality.LEGACY_STRICT_PHYSICAL_POLICY_ID,
    )

    result = _evaluate(tmp_path, policy, plan, aggregate)

    assert result["status"] == "quality_gate_passed"


def test_legacy_strict_policy_rejects_any_static_diagnostic(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(
        tmp_path,
        policy,
        operating_policy_id=quality.LEGACY_STRICT_PHYSICAL_POLICY_ID,
        static_diagnostics=True,
    )

    with pytest.raises(
        quality.QualityPolicyError,
        match="legacy strict policy cannot present record-only diagnostics",
    ):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_malformed_live_gpu_telemetry_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(
        tmp_path, policy, malformed_telemetry=True
    )

    with pytest.raises(
        quality.QualityPolicyError, match="must be finite numeric telemetry"
    ):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_runtime_gpu_identity_drift_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(
        tmp_path, policy, runtime_identity_drift=True
    )

    with pytest.raises(quality.QualityPolicyError, match="GPU identity differs"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_postflight_driver_identity_drift_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(
        tmp_path, policy, postflight_driver_drift=True
    )

    with pytest.raises(
        quality.QualityPolicyError, match="postflight GPU/driver identity differs"
    ):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_insufficient_one_hz_telemetry_coverage_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy, insufficient_coverage=True)

    with pytest.raises(
        quality.QualityPolicyError,
        match="1 Hz telemetry is missing, malformed, drifting, or incomplete",
    ):
        _evaluate(tmp_path, policy, plan, aggregate)


@pytest.mark.parametrize(
    ("coverage_override", "message"),
    (
        ({"endpoint_tolerance_seconds": None}, "1 Hz telemetry"),
        ({"endpoint_tolerance_seconds": 2.0}, "1 Hz telemetry"),
        ({"sample_count_grace_seconds": 1.0}, "1 Hz telemetry"),
        ({"expected_minimum_samples": 1}, "guarded process window"),
    ),
)
def test_one_hz_telemetry_boundary_contract_is_exact_and_replayed(
    tmp_path: Path, coverage_override: dict[str, object], message: str
) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(
        tmp_path, policy, coverage_override=coverage_override
    )

    with pytest.raises(quality.QualityPolicyError, match=message):
        _evaluate(tmp_path, policy, plan, aggregate)


@pytest.mark.parametrize(
    ("guard_fault", "message"),
    (
        ("query_error", "1 Hz telemetry is missing, malformed, drifting, or incomplete"),
        (
            "identity_monitor_missing",
            "1 Hz telemetry is missing, malformed, drifting, or incomplete",
        ),
        ("xid", "pre-run Xid evidence is unavailable or non-empty"),
        ("fatal", "Xid/OOM/fatal postflight differs"),
        ("process", "guard process receipt differs"),
    ),
)
def test_non_static_guard_faults_remain_fail_closed(
    tmp_path: Path, guard_fault: str, message: str
) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy, guard_fault=guard_fault)

    with pytest.raises(quality.QualityPolicyError, match=message):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_missing_evaluation_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    (_first_run(tmp_path) / "evaluation.json").unlink()
    with pytest.raises(quality.QualityPolicyError, match="missing evaluation"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_tampered_approval_pin_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    approval = tmp_path / "inputs/person-quality-approval.json"
    approval.write_text(approval.read_text() + " ", encoding="utf-8")
    with pytest.raises(quality.QualityPolicyError, match="live size/hash differs"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_post_hoc_approval_fails_even_with_current_hash_pin(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path, approved_at="2026-07-16T09:30:00Z")
    plan, aggregate = _campaign(tmp_path, policy)
    with pytest.raises(quality.QualityPolicyError, match="must predate"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_tampered_live_evaluation_fails_cpu_recomputation(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    path = _first_run(tmp_path) / "evaluation.json"
    evaluation = json.loads(path.read_text())
    evaluation["overall"]["ap_101_point"] = 0.5
    _json(path, evaluation)
    with pytest.raises(quality.QualityPolicyError, match="immutable receipt evaluation"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_approved_failing_threshold_is_not_accepted(tmp_path: Path) -> None:
    def make_failing(policy):
        rule = next(item for item in policy["rules"] if item["metric"] == "micro_precision")
        rule["operator"] = "lt"
        rule["threshold"] = 0.5

    policy = _approved_policy(tmp_path, policy_mutator=make_failing)
    plan, aggregate = _campaign(tmp_path, policy)
    result = _evaluate(tmp_path, policy, plan, aggregate)
    assert result["status"] == "quality_gate_failed"
    assert result["acceptance_effect"] == "none"
    assert any(item["status"] == "fail" for item in result["quality_decision"]["rules"])


def test_tampered_aggregate_binding_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    payload = json.loads(aggregate.read_text())
    payload["by_model_input"]["640"]["micro"]["recall"] = 0.5
    _json(aggregate, payload)
    with pytest.raises(quality.QualityPolicyError, match="batch receipt pin chain"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_approved_policy_requires_complete_metric_profile_coverage() -> None:
    policy = _draft()
    policy["status"] = "approved"
    policy["policy_owner"] = {"identity": "owner@example.test"}
    policy["approval_artifact"] = {"path": "x", "bytes": 1, "sha256": "0" * 64}
    policy["rules"] = policy["rules"][:-1]
    with pytest.raises(quality.QualityPolicyError, match="cover all four metrics"):
        quality._validate_policy(policy)


def test_plan_runtime_contract_must_equal_approved_policy(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["campaign"]["model_runtime_contract"]["profiles"]["640"][
        "model_id"
    ] = "different-model"
    _json(plan, payload)
    with pytest.raises(quality.QualityPolicyError, match="runtime contract differs"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_manifest_model_id_must_equal_profile_contract(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    path = _first_run(tmp_path) / "run-manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"] = "different-model"
    _json(path, payload)
    with pytest.raises(quality.QualityPolicyError, match="immutable receipt run_manifest"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_live_engine_hash_tamper_fails_before_metrics(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    engine = tmp_path / quality.MODEL_ARTIFACT_PATHS[640]["engine"]
    engine.write_text("tampered-engine\n", encoding="utf-8")
    with pytest.raises(quality.QualityPolicyError, match="live size/hash differs"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_policy_engine_path_must_be_canonical_profile_path(tmp_path: Path) -> None:
    def mutate(policy):
        policy["campaign_contract"]["runtime_binding"]["profiles"]["640"][
            "model_artifacts"
        ]["engine"]["path"] = "models/person/640/other.engine"

    policy = _approved_policy(tmp_path, policy_mutator=mutate)
    with pytest.raises(quality.QualityPolicyError, match="path differs"):
        quality.evaluate_quality_policy(
            policy_path=policy,
            campaign_plan_path=tmp_path / "missing-plan.json",
            aggregate_path=tmp_path / "missing-aggregate.json",
            project_root=tmp_path,
        )


def test_manifest_generated_config_hash_tamper_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    path = _first_run(tmp_path) / "run-manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runtime_binding"]["generated_configs"]["deepstream_app"][
        "sha256"
    ] = "0" * 64
    _json(path, payload)
    with pytest.raises(quality.QualityPolicyError, match="immutable receipt run_manifest"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_live_generated_config_content_tamper_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    config = _first_run(tmp_path) / "generated/config-infer-primary.txt"
    config.write_text(config.read_text(encoding="utf-8") + "# tamper\n", encoding="utf-8")
    manifest_path = _first_run(tmp_path) / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_binding"]["generated_configs"]["primary_inference"] = (
        quality.make_file_pin(config, project_root=tmp_path)
    )
    _json(manifest_path, manifest)
    with pytest.raises(quality.QualityPolicyError, match="immutable receipt run_manifest"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_guard_command_must_equal_manifest_command(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    path = _first_run(tmp_path) / "safety/gpu-guard-report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["command"][-1] = "/workspace/tampered-config.txt"
    _json(path, payload)
    with pytest.raises(quality.QualityPolicyError, match="immutable receipt gpu_guard_report"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_resolved_image_id_must_match_guard_preflight_identity(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    path = _first_run(tmp_path) / "safety/gpu-guard-report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["preflight"]["resolved_image_id"] = "sha256:" + "2" * 64
    _json(path, payload)
    with pytest.raises(quality.QualityPolicyError, match="immutable receipt gpu_guard_report"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_predictions_model_id_must_equal_exact_profile(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    path = _first_run(tmp_path) / "predictions.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_id"] = "yolo11s-960-fp16"
    _write(path, json.dumps(payload) + "\n")
    with pytest.raises(quality.QualityPolicyError, match="immutable receipt predictions"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_plan_command_and_manifest_binding_must_match_exactly(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["jobs"][0]["command"].extend(["--force"])
    _json(plan, payload)
    with pytest.raises(quality.QualityPolicyError, match="exact plan contract"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_native_caviar_sequence_substitution_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["sequences"][0]["sequence_id"] = "Synthetic_Clip"
    _json(plan, payload)
    with pytest.raises(quality.QualityPolicyError, match="non-native CAVIAR"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_approved_ffprobe_or_frame_map_projection_cannot_be_rewritten(
    tmp_path: Path,
) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["sequences"][0]["video_metadata"]["frames"] = 2
    _json(plan, payload)
    with pytest.raises(quality.QualityPolicyError, match="native video/GT provenance"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_transitive_evaluation_code_tamper_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    path = tmp_path / quality.CONTROL_ARTIFACT_PATHS["evaluation_metrics"]
    path.write_text("coherent replacement\n", encoding="utf-8")
    with pytest.raises(quality.QualityPolicyError, match="live size/hash differs"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_ds9_compatibility_receipt_repin_after_campaign_fails_closed(
    tmp_path: Path,
) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    receipt = tmp_path / quality.DS9_COMPATIBILITY_RECEIPT_CONTRACT["path"]
    receipt.chmod(0o600)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["coherent_repin_attempt"] = True
    _json(receipt, payload)
    receipt.chmod(0o440)
    with pytest.raises(
        quality.QualityPolicyError,
        match="CAVIAR DS9 compatibility receipt live size/hash differs",
    ):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_writable_job_receipt_is_not_immutable_evidence(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    (_first_run(tmp_path) / "job-receipt.json").chmod(0o640)
    with pytest.raises(quality.QualityPolicyError, match="write-protected"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_writable_batch_receipt_is_not_immutable_evidence(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    (plan.parent / "batch-receipt.json").chmod(0o640)
    with pytest.raises(quality.QualityPolicyError, match="write-protected"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_minimal_guard_report_cannot_replace_full_receipt(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    _json(
        _first_run(tmp_path) / "safety/gpu-guard-report.json",
        {"schema_version": "deepsafe.gpu-guarded-process/v1", "status": "complete"},
    )
    with pytest.raises(quality.QualityPolicyError, match="gpu_guard_report live pin"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_telemetry_gap_or_rewrite_breaks_guard_receipt(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    path = _first_run(tmp_path) / "safety/gpu.csv"
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    rows[-1][0] = "2026-07-16T09:01:30Z"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(quality.QualityPolicyError, match="receipt gpu_csv"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_gpu_target_substitution_in_plan_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["campaign"]["gpu"] = 1
    _json(plan, payload)
    with pytest.raises(quality.QualityPolicyError, match="approved GPU contract"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_engine_fallback_log_rewrite_breaks_job_receipt(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    path = _first_run(tmp_path) / "deepstream.log"
    path.write_text(path.read_text(encoding="utf-8") + "building from model.onnx\n")
    with pytest.raises(quality.QualityPolicyError, match="deepstream_log live pin"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_symlink_model_artifact_is_rejected_before_metrics(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    engine = tmp_path / quality.MODEL_ARTIFACT_PATHS[640]["engine"]
    replacement = _write(tmp_path / "replacement.engine", engine.read_text())
    engine.unlink()
    engine.symlink_to(replacement)
    with pytest.raises(quality.QualityPolicyError, match="symlink evidence path"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_approval_expiry_before_campaign_finish_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path, expires_at="2026-07-16T08:45:00Z")
    plan, aggregate = _campaign(tmp_path, policy)
    with pytest.raises(quality.QualityPolicyError, match="remain valid"):
        _evaluate(tmp_path, policy, plan, aggregate)


def test_campaign_nonce_mismatch_fails_closed(tmp_path: Path) -> None:
    policy = _approved_policy(tmp_path)
    plan, aggregate = _campaign(tmp_path, policy)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["campaign"]["campaign_nonce"] = "b" * 64
    _json(plan, payload)
    with pytest.raises(quality.QualityPolicyError, match="nonce/session"):
        _evaluate(tmp_path, policy, plan, aggregate)
