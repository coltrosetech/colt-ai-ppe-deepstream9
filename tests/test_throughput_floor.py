import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from validation.endurance import throughput_floor as floor
from validation.scene_benchmark import run_matrix as matrix


def _write(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _json(path: Path, value: dict) -> Path:
    return _write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _gpu_identity() -> dict:
    return {
        "index": "0",
        "uuid": "GPU-fixture",
        "name": "fixture GPU",
        "driver_version": "590.0",
        "memory.total": "16384 MiB",
        "pci.bus_id": "00000000:01:00.0",
    }


def _scene_power_policy(
    mode: str = matrix.DEFAULT_GPU_OPERATING_POLICY_MODE,
) -> dict:
    strict = mode == matrix.LEGACY_STRICT_GPU_OPERATING_POLICY_MODE
    flags = list(matrix.REQUIRED_SLOWDOWN_TELEMETRY_FIELDS)
    return {
        "operating_policy_mode": mode,
        "hardware_protection_owner": "workstation_bios_ec_nvidia_driver",
        "static_signal_action": (
            "safety_abort" if strict else "record_measurement_quality_diagnostic"
        ),
        "power_limit_drop_tolerance_w": 5.0,
        "slowdown_consecutive_samples": 2,
        "preflight_samples": matrix.PREFLIGHT_GPU_SAMPLES,
        "preflight_sample_interval_seconds": (
            matrix.PREFLIGHT_GPU_SAMPLE_INTERVAL_SECONDS
        ),
        "power_limit_fields": [
            "power_requested_limit_w",
            "power_current_limit_w",
            "power_default_limit_w",
        ],
        "diagnostic_slowdown_flags": flags,
        "abort_slowdown_flags": flags if strict else [],
        "required_telemetry_failure_action": "safety_abort",
    }


def _static_diagnostic_event(
    code: str,
    *,
    sample_number: int,
    mode: str = matrix.DEFAULT_GPU_OPERATING_POLICY_MODE,
) -> dict:
    return {
        "schema_version": "deepsafe.gpu-safety-event/v1",
        "detected_at_utc": f"2026-07-16T08:00:{sample_number:02d}Z",
        "code": code,
        "reason": f"fixture {code}",
        "sample_number": sample_number,
        "gpu_csv_line_number": sample_number + 1,
        "snapshot": {"temperature_c": "90"},
        "assessment": {
            "power_limit_telemetry_complete": True,
            "power_limit_drop_detected": code == "power_limit_below_default",
            "dangerous_slowdown_active": code == "sustained_clock_slowdown",
        },
        "operating_policy_mode": mode,
        "measurement_quality_signal": True,
        "disposition": (
            "safety_abort"
            if mode == matrix.LEGACY_STRICT_GPU_OPERATING_POLICY_MODE
            else "record_only_workstation_hardware_managed"
        ),
    }


def _campaign(root: Path) -> dict[str, Path]:
    # derive_floor pins a project-local copy and proves it is byte-identical to
    # the executing module.  This keeps temporary-project tests honest.
    tool_copy = root / "validation/endurance/throughput_floor.py"
    _write(tool_copy, Path(floor.__file__).read_bytes())
    runner = _write(
        root / "validation/scene_benchmark/run_matrix.py",
        Path(matrix.__file__).read_bytes(),
    )
    summarizer = _write(
        root / "benchmark/summarize.py",
        Path(floor.benchmark_summarize.__file__).read_bytes(),
    )

    models: dict[int, dict] = {}
    for profile in floor.PROFILES:
        infer = _write(
            root / f"models/person/{profile}/config_infer_primary.txt",
            f"model-engine-file=/models/person/{profile}/fixture.engine\n",
        )
        engine = _write(
            root / f"models/person/{profile}/fixture.engine",
            f"engine-{profile}\n".encode(),
        )
        models[profile] = {
            "size": profile,
            "infer_config": _relative(root, infer),
            "engine": _relative(root, engine),
            "infer_config_sha256": _sha(infer),
            "engine_size_bytes": engine.stat().st_size,
            "engine_sha256": _sha(engine),
            "person_only_classes": [0],
        }

    scenes = []
    video_paths: dict[str, Path] = {}
    for index in range(floor.SCENES):
        scene_id = f"scene_{index:02d}"
        video = _write(root / f"data/videos/{scene_id}.mp4", f"video-{scene_id}\n")
        video_paths[scene_id] = video
        scenes.append(
            {
                "id": scene_id,
                "source_manifest_id": scene_id,
                "video_path": _relative(root, video),
                "benchmark_type": f"camera_type_{index:02d}",
                "security_camera_relevance": "primary",
                "notes": "fixture",
            }
        )
    manifest = {
        "schema_version": floor.SCENE_MANIFEST_SCHEMA,
        "campaign": {
            "streams": 12,
            "model_input_sizes": [640, 960],
            "duration_seconds": 300,
            "warmup_seconds": 15,
            "perf_interval_seconds": 5,
            "startup_timeout_seconds": 60,
            "gpu_sample_interval_seconds": 1,
        },
        "scenes": scenes,
        "accuracy_policy": {},
        "source_catalog": {},
    }
    manifest_path = _json(root / "validation/scene_benchmark/scenes.json", manifest)

    runs = []
    result_root = root / "validation/results/scene-benchmark"
    for index, scene in enumerate(scenes):
        video = video_paths[scene["id"]]
        video_stat = video.stat()
        for profile in floor.PROFILES:
            run_root = result_root / scene["id"] / str(profile)
            config = _write(run_root / "deepstream.txt", f"profile={profile}\nscene={scene['id']}\n")
            per_stream_fps = (8.34 if profile == 640 else 5.84) + index * 0.04
            perf_header = "**PERF:  " + "\t".join(
                f"FPS {stream_id} (Avg)" for stream_id in range(floor.STREAMS)
            )
            perf_line = "**PERF:  " + "\t".join(
                f"{per_stream_fps:.2f} ({per_stream_fps:.2f})"
                for _ in range(floor.STREAMS)
            )
            log = _write(
                run_root / "deepstream.log",
                "\n".join([perf_header, *([perf_line] * floor.EXPECTED_PERF_INTERVALS)])
                + "\n",
            )
            gpu_csv = run_root / "gpu.csv"
            gpu_csv.parent.mkdir(parents=True, exist_ok=True)
            with gpu_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "timestamp",
                        "gpu_index",
                        "gpu_name",
                        "gpu_utilization_percent",
                        "memory_utilization_percent",
                        "memory_used_mib",
                        "memory_total_mib",
                        "temperature_c",
                        "power_draw_w",
                        "sm_clock_mhz",
                        "memory_clock_mhz",
                        "power_requested_limit_w",
                        "power_current_limit_w",
                        "power_default_limit_w",
                        "pstate",
                        "clock_event_reasons_active_mask",
                        "clock_event_sw_power_cap",
                        "clock_event_sw_thermal_slowdown",
                        "clock_event_hw_slowdown",
                        "clock_event_hw_thermal_slowdown",
                        "clock_event_hw_power_brake_slowdown",
                    ]
                )
                for sample in range(floor.DURATION_SECONDS):
                    writer.writerow(
                        [
                            f"2026/07/16 08:{sample // 60:02d}:{sample % 60:02d}.000",
                            "0",
                            "fixture GPU",
                            "60",
                            "20",
                            "1200",
                            "16384",
                            "60",
                            "85",
                            "1500",
                            "7000",
                            "115",
                            "115",
                            "115",
                            "P0",
                            "0x0",
                            "Not Active",
                            "Not Active",
                            "Not Active",
                            "Not Active",
                            "Not Active",
                        ]
                    )
            raw_throughput = floor.benchmark_summarize.parse_perf(
                log,
                floor.STREAMS,
                floor.DURATION_SECONDS,
                floor.PERF_INTERVAL_SECONDS,
                active_row_start=0,
                active_row_end=floor.EXPECTED_PERF_INTERVALS,
            )
            raw_gpu = floor.benchmark_summarize.parse_gpu_metrics(
                gpu_csv,
                floor.DURATION_SECONDS,
                sample_start=0,
                sample_end=floor.DURATION_SECONDS,
            )
            p05 = raw_throughput["aggregate_current_fps"]["p05"]
            fingerprint_input = {
                "run_schema_version": floor.RUN_SCHEMA,
                "benchmark_code": {
                    "runner_sha256": _sha(runner),
                    "summarizer_sha256": _sha(summarizer),
                },
                "scene_id": scene["id"],
                "video_path": scene["video_path"],
                "video_size_bytes": video_stat.st_size,
                "video_mtime_ns": video_stat.st_mtime_ns,
                "video_sha256": _sha(video),
                "model_profile": models[profile],
                "config_sha256": _sha(config),
                "duration_seconds": 300,
                "warmup_seconds": 15,
                "perf_interval_seconds": 5,
                "startup_timeout_seconds": 60,
                "streams": 12,
                "image": "deepsafe-deepstream:9.0",
                "image_id": "sha256:" + "a" * 64,
                "gpu_index": 0,
                "gpu_identity": _gpu_identity(),
                "power_profile": {"available": True, "value": "performance"},
                "max_temperature_c": 86.0,
                "power_safety_policy": _scene_power_policy(),
                "preflight_power_limits_w": {"current": 115.0, "default": 115.0},
                "platform_thermal_sources": {"available": False},
            }
            fingerprint = floor._canonical_sha256(fingerprint_input)
            status_path = run_root / "status.json"
            image_id = "sha256:" + "a" * 64
            container_name = f"deepsafe-scene-{profile}-{scene['id']}"
            docker_command = [
                "docker",
                "run",
                "--rm",
                "--pull=never",
                "--name",
                container_name,
                "--gpus",
                "device=0",
                "--ipc=host",
                "--env",
                "GST_DEBUG=1",
                "--env",
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
                "--volume",
                f"{root.resolve()}:/workspace:ro",
                "--volume",
                f"{(root / 'models').resolve()}:/models:ro",
                "--workdir",
                "/workspace",
                image_id,
                "deepstream-app",
                "-c",
                f"/workspace/{_relative(root, config)}",
            ]
            runtime_container_id = hashlib.sha256(
                f"{scene['id']}/{profile}".encode("utf-8")
            ).hexdigest()
            runtime_cmd = docker_command[docker_command.index(image_id) + 1 :]
            runtime_attestation = {
                "schema_version": "deepsafe.scene-runtime-container-attestation/v1",
                "status": "verified_running",
                "captured_at_utc": "2026-07-16T08:00:01Z",
                "launch_command_sha256": floor._launch_command_sha256(
                    docker_command
                ),
                "container_id": runtime_container_id,
                "container_name": container_name,
                "image_id": image_id,
                "config_image": image_id,
                "entrypoint": ["/opt/nvidia/nvidia_entrypoint.sh"],
                "cmd": runtime_cmd,
                "actual_process": {
                    "path": "/opt/nvidia/nvidia_entrypoint.sh",
                    "args": runtime_cmd,
                },
                "working_dir": "/workspace",
                "required_environment": sorted(
                    [
                        "GST_DEBUG=1",
                        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
                    ]
                ),
                "gpu_device_request": {
                    "driver": "nvidia",
                    "count": 0,
                    "device_ids": ["0"],
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
                    "started_at_utc": "2026-07-16T08:00:00.500000000Z",
                },
            }
            runtime_container = _json(
                run_root / "runtime-container.json", runtime_attestation
            )
            status = {
                "schema_version": floor.RUN_SCHEMA,
                "status": "complete",
                "fingerprint": fingerprint,
                "fingerprint_input": fingerprint_input,
                "scene": {
                    "id": scene["id"],
                    "benchmark_type": scene["benchmark_type"],
                },
                "video": {
                    "path": scene["video_path"],
                    "size_bytes": video_stat.st_size,
                    "mtime_ns": video_stat.st_mtime_ns,
                    "sha256": _sha(video),
                },
                "model": models[profile],
                "simulation": {
                    "streams": 12,
                    "num_sources": 12,
                    "file_loop": True,
                    "requested_duration_seconds": 300,
                    "warmup_seconds": 15,
                    "perf_interval_seconds": 5,
                    "startup_timeout_seconds": 60,
                },
                "timing": {
                    "started_at_utc": "2026-07-16T08:00:00Z",
                    "finished_at_utc": "2026-07-16T08:05:01Z",
                    "measurement_complete": True,
                    "measurement_wall_time_seconds": 300.1,
                    "measurement_perf_row_bounds": [
                        0,
                        floor.EXPECTED_PERF_INTERVALS,
                    ],
                    "measurement_gpu_sample_bounds": [0, floor.DURATION_SECONDS],
                },
                "process": {
                    "container_name": container_name,
                    "requested_image": "deepsafe-deepstream:9.0",
                    "resolved_image_id": image_id,
                    "docker_command": docker_command,
                    "launch_command_sha256": floor._launch_command_sha256(
                        docker_command
                    ),
                    "runtime_container_id": runtime_container_id,
                    "exit_code": 130,
                    "premature_exit": False,
                },
                "runtime_container_attestation": runtime_attestation,
                "runtime_container_attestation_error": None,
                "throughput": raw_throughput,
                "gpu": raw_gpu,
                "safety": {
                    "preflight": {
                        "status": "ok",
                        "image": "deepsafe-deepstream:9.0",
                        "image_id": "sha256:" + "a" * 64,
                        "gpu_identity": _gpu_identity(),
                        "power_safety_policy": _scene_power_policy(),
                        "safety_events": [],
                        "diagnostic_events": [],
                        "gpu_safety_assessments": [
                            {
                                "power_limit_telemetry_complete": True,
                                "power_limit_drop_detected": False,
                                "dangerous_slowdown_active": False,
                            },
                            {
                                "power_limit_telemetry_complete": True,
                                "power_limit_drop_detected": False,
                                "dangerous_slowdown_active": False,
                            },
                        ],
                        "xid": {"available": True, "count": 0, "lines": []},
                        "performance_profile_required": True,
                        "power_profile": {"available": True, "value": "performance"},
                    },
                    "run_xid_before": {"available": True, "count": 0, "lines": []},
                    "post_run_xid": {"available": True, "count": 0, "lines": []},
                    "post_run_power_profile": {"available": True, "value": "performance"},
                    "new_xid_lines": [],
                    "monitor_abort_reason_code": None,
                    "monitor_abort_reason": None,
                    "monitor_safety_event": None,
                    "monitor_query_errors": [],
                    "monitor_samples": 334,
                    "gpu_identity_checks": 12,
                    "last_observed_gpu_identity": _gpu_identity(),
                    "power_safety_policy": _scene_power_policy(),
                    "monitor_diagnostic_events_first_50": [],
                    "monitor_diagnostic_event_counts": {},
                    "temperature_threshold_samples": 0,
                    "maximum_temperature_c": 60.0,
                    "power_limit_drop_samples": 0,
                    "slowdown_active_samples": 0,
                    "max_consecutive_slowdown_samples": 0,
                },
                "deepstream_log_fatal_matches": {"cuda_error": 0, "gst_error": 0},
                "failure_reasons": [],
                "artifacts": {
                    "config": _relative(root, config),
                    "deepstream_log": _relative(root, log),
                    "gpu_csv": _relative(root, gpu_csv),
                    "runtime_container": _relative(root, runtime_container),
                    "status": _relative(root, status_path),
                },
            }
            _json(status_path, status)
            runs.append(
                {
                    "scene_id": scene["id"],
                    "benchmark_type": scene["benchmark_type"],
                    "model_input_size": profile,
                    "status": "complete",
                    "fingerprint": fingerprint,
                    "status_path": _relative(root, status_path),
                }
            )
    summary = {
        "schema_version": floor.SUMMARY_SCHEMA,
        "generated_at_utc": "2026-07-16T09:00:00Z",
        "streams": 12,
        "duration_seconds_per_run": 300,
        "warmup_seconds_per_run": 15,
        "selected_scenes": 12,
        "selected_sizes": [640, 960],
        "expected_runs": 24,
        "status_counts": {"complete": 24},
        "runs": runs,
    }
    summary_path = _json(result_root / "matrix-summary.json", summary)
    return {
        "root": root,
        "summary": summary_path,
        "manifest": manifest_path,
        "result_root": result_root,
        "tool": tool_copy,
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _status(fixture: dict[str, Path], scene: str = "scene_00", profile: int = 640) -> Path:
    return fixture["result_root"] / scene / str(profile) / "status.json"


def _rewrite_status_fingerprint(
    fixture: dict[str, Path], scene: str = "scene_00", profile: int = 640
) -> None:
    path = _status(fixture, scene, profile)
    status = _load(path)
    status["fingerprint"] = floor._canonical_sha256(status["fingerprint_input"])
    _json(path, status)
    summary = _load(fixture["summary"])
    for row in summary["runs"]:
        if row["scene_id"] == scene and row["model_input_size"] == profile:
            row["fingerprint"] = status["fingerprint"]
    _json(fixture["summary"], summary)


def _rewrite_all_scene_policies(
    fixture: dict[str, Path], mode: str
) -> None:
    summary = _load(fixture["summary"])
    policy = _scene_power_policy(mode)
    for row in summary["runs"]:
        path = _status(
            fixture,
            row["scene_id"],
            row["model_input_size"],
        )
        status = _load(path)
        status["fingerprint_input"]["power_safety_policy"] = copy.deepcopy(policy)
        status["safety"]["power_safety_policy"] = copy.deepcopy(policy)
        status["safety"]["preflight"]["power_safety_policy"] = copy.deepcopy(
            policy
        )
        status["fingerprint"] = floor._canonical_sha256(
            status["fingerprint_input"]
        )
        row["fingerprint"] = status["fingerprint"]
        _json(path, status)
    _json(fixture["summary"], summary)


@pytest.fixture
def campaign(tmp_path: Path) -> dict[str, Path]:
    return _campaign(tmp_path)


def _derive(campaign: dict[str, Path]) -> dict:
    return floor.derive_floor(
        summary_path=campaign["summary"],
        scene_manifest_path=campaign["manifest"],
        project_root=campaign["root"],
    )


def test_happy_path_freezes_exact_conservative_floors(campaign, monkeypatch):
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    artifact = _derive(campaign)

    assert artifact["status"] == "frozen"
    assert artifact["contract"]["observed_safe_runs"] == 24
    assert len(artifact["inputs"]["statuses"]) == 24
    assert artifact["profiles"]["640"]["aggregate_fps_floor"] == 80.0
    assert artifact["profiles"]["640"]["per_stream_fps_floor"] == 6.6
    assert artifact["profiles"]["960"]["aggregate_fps_floor"] == 56.0
    assert artifact["profiles"]["960"]["per_stream_fps_floor"] == 4.6
    assert artifact["profiles"]["640"]["minimum_source_scene_id"] == "scene_00"
    assert artifact["generated_at_utc"] == "2026-07-16T09:00:00Z"
    assert artifact["gpu_or_docker_executed_by_tool"] is False
    assert artifact["fingerprint"] == floor._canonical_sha256(artifact["fingerprint_input"])
    assert artifact["contract"]["observed_distinct_live_video_paths"] == 12
    assert artifact["contract"]["observed_distinct_live_video_sha256"] == 12
    assert artifact["contract"]["observed_distinct_benchmark_types"] == 12
    first_raw = artifact["inputs"]["statuses"][0]["raw_output_evidence"]
    assert first_raw["measurement_bounds"] == {
        "perf_active_rows": [0, 60],
        "gpu_samples": [0, 300],
    }
    assert first_raw["deepstream_log"]["sha256"] == _sha(
        campaign["result_root"] / "scene_00/640/deepstream.log"
    )
    assert first_raw["runtime_container"]["sha256"] == _sha(
        campaign["result_root"] / "scene_00/640/runtime-container.json"
    )
    assert len(first_raw["runtime_container_projection_sha256"]) == 64


def test_intrinsic_generation_ignores_future_validity_boundaries(monkeypatch):
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)

    generated_at, generation = floor._generation_choice(
        [
            {
                "generated_at_utc": "2026-07-16T09:00:00Z",
                "gpu_smoke": {
                    "expires_at_utc": "2099-07-17T09:00:00Z",
                    "valid_until_utc": "2099-07-18T09:00:00Z",
                },
            }
        ]
    )

    assert generated_at == "2026-07-16T09:00:00Z"
    assert generation == {"mode": "intrinsic_latest", "epoch_seconds": None}


def test_workstation_managed_diagnostics_do_not_block_floor_freeze(campaign):
    path = _status(campaign)
    status = _load(path)
    preflight = status["safety"]["preflight"]
    preflight["gpu_safety_assessments"][0]["power_limit_drop_detected"] = True
    preflight["gpu_safety_assessments"][1]["dangerous_slowdown_active"] = True
    preflight["diagnostic_events"] = [
        _static_diagnostic_event("power_limit_below_default", sample_number=1),
        _static_diagnostic_event("sustained_clock_slowdown", sample_number=2),
    ]
    safety = status["safety"]
    safety["monitor_diagnostic_events_first_50"] = [
        _static_diagnostic_event("temperature_threshold", sample_number=3),
        _static_diagnostic_event("power_limit_below_default", sample_number=4),
        _static_diagnostic_event("sustained_clock_slowdown", sample_number=5),
    ]
    safety["monitor_diagnostic_event_counts"] = {
        "temperature_threshold": 3,
        "power_limit_below_default": 2,
        "sustained_clock_slowdown": 1,
    }
    safety["temperature_threshold_samples"] = 3
    safety["maximum_temperature_c"] = 91.0
    safety["power_limit_drop_samples"] = 2
    safety["slowdown_active_samples"] = 4
    safety["max_consecutive_slowdown_samples"] = 2
    _json(path, status)

    artifact = _derive(campaign)
    assert artifact["status"] == "frozen"
    assert artifact["contract"]["observed_safe_runs"] == 24


def test_legacy_strict_zero_diagnostic_matrix_can_freeze(campaign):
    _rewrite_all_scene_policies(
        campaign, matrix.LEGACY_STRICT_GPU_OPERATING_POLICY_MODE
    )
    assert _derive(campaign)["status"] == "frozen"


def test_legacy_strict_diagnostic_counter_still_fails_closed(campaign):
    _rewrite_all_scene_policies(
        campaign, matrix.LEGACY_STRICT_GPU_OPERATING_POLICY_MODE
    )
    path = _status(campaign)
    status = _load(path)
    status["safety"]["slowdown_active_samples"] = 1
    status["safety"]["max_consecutive_slowdown_samples"] = 1
    _json(path, status)

    with pytest.raises(floor.ThroughputFloorError, match="legacy-strict"):
        _derive(campaign)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_diagnostic_counts", "safety contract fields are incomplete"),
        ("preflight_power_telemetry_missing", "power telemetry is incomplete"),
        ("runtime_driver_drift", "runtime GPU identity/driver binding differs"),
    ],
)
def test_workstation_policy_keeps_required_telemetry_and_identity_fail_closed(
    campaign, mutation, match
):
    path = _status(campaign)
    status = _load(path)
    if mutation == "missing_diagnostic_counts":
        status["safety"].pop("monitor_diagnostic_event_counts")
    elif mutation == "preflight_power_telemetry_missing":
        status["safety"]["preflight"]["gpu_safety_assessments"][0][
            "power_limit_telemetry_complete"
        ] = False
    else:
        status["safety"]["last_observed_gpu_identity"][
            "driver_version"
        ] = "changed"
    _json(path, status)

    with pytest.raises(floor.ThroughputFloorError, match=match):
        _derive(campaign)


def test_missing_status_fails_closed(campaign):
    _status(campaign).unlink()
    with pytest.raises(floor.ThroughputFloorError, match="missing|unreadable"):
        _derive(campaign)


def test_missing_runtime_container_attestation_fails_closed(campaign):
    path = campaign["result_root"] / "scene_00/640/runtime-container.json"
    path.unlink()

    with pytest.raises(floor.ThroughputFloorError, match="missing|unreadable"):
        _derive(campaign)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("image", "runtime container image/name identity differs"),
        ("argv", "Entrypoint/Cmd/Path/Args"),
        ("gpu", "GPU device request differs"),
        ("mount", "mount/IPC/lifecycle"),
        ("launch_hash", "launch command hash differs"),
        ("timeline", "timeline differs"),
    ],
)
def test_coherently_rewritten_runtime_attestation_is_rejected(
    campaign, mutation, match
):
    runtime_path = campaign["result_root"] / "scene_00/640/runtime-container.json"
    status_path = _status(campaign)
    attestation = _load(runtime_path)
    status = _load(status_path)
    if mutation == "image":
        forged = "sha256:" + "f" * 64
        attestation["image_id"] = forged
        attestation["config_image"] = forged
    elif mutation == "argv":
        attestation["cmd"][-1] = "/workspace/forged.txt"
        attestation["actual_process"]["args"][-1] = "/workspace/forged.txt"
    elif mutation == "gpu":
        attestation["gpu_device_request"]["device_ids"] = ["7"]
    elif mutation == "mount":
        attestation["mounts"][0]["read_write"] = True
    elif mutation == "launch_hash":
        attestation["launch_command_sha256"] = "f" * 64
        status["process"]["launch_command_sha256"] = "f" * 64
    else:
        attestation["captured_at_utc"] = "2026-07-17T08:00:00Z"
    # The attacker coherently rewrites both the standalone receipt and its status
    # projection. Independent reconstruction must still reject the semantics.
    status["runtime_container_attestation"] = copy.deepcopy(attestation)
    _json(runtime_path, attestation)
    _json(status_path, status)

    with pytest.raises(floor.ThroughputFloorError, match=match):
        _derive(campaign)


def test_v1_is_estimate_only_and_never_freezes(campaign):
    path = _status(campaign)
    status = _load(path)
    status["schema_version"] = "deepsafe.scene-benchmark-run/v1"
    status["fingerprint_input"]["run_schema_version"] = status["schema_version"]
    _json(path, status)
    _rewrite_status_fingerprint(campaign)

    with pytest.raises(floor.ThroughputFloorError, match="estimate-only"):
        _derive(campaign)


def test_duplicate_summary_identity_fails_closed(campaign):
    summary = _load(campaign["summary"])
    summary["runs"][1] = copy.deepcopy(summary["runs"][0])
    _json(campaign["summary"], summary)
    with pytest.raises(floor.ThroughputFloorError, match="duplicate"):
        _derive(campaign)


def test_summary_status_fingerprint_mismatch_fails_closed(campaign):
    summary = _load(campaign["summary"])
    summary["runs"][0]["fingerprint"] = "0" * 64
    _json(campaign["summary"], summary)
    with pytest.raises(floor.ThroughputFloorError, match="summary/status fingerprint"):
        _derive(campaign)


def test_status_fingerprint_tamper_fails_closed(campaign):
    path = _status(campaign)
    status = _load(path)
    status["fingerprint"] = "f" * 64
    _json(path, status)
    with pytest.raises(floor.ThroughputFloorError, match="fingerprint_input SHA256"):
        _derive(campaign)


def test_fingerprint_input_tamper_fails_closed(campaign):
    path = _status(campaign)
    status = _load(path)
    status["fingerprint_input"]["streams"] = 11
    _json(path, status)
    with pytest.raises(floor.ThroughputFloorError, match="fingerprint_input SHA256"):
        _derive(campaign)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("requested_image", "process requested image binding differs"),
        ("resolved_image", "process resolved image binding differs"),
        ("mutable_tag_command", "canonical offline command"),
        ("online_pull_command", "canonical offline command"),
    ],
)
def test_process_command_and_image_binding_fail_closed(campaign, mutation, match):
    path = _status(campaign)
    status = _load(path)
    process = status["process"]
    if mutation == "requested_image":
        process["requested_image"] = "attacker/deepstream:latest"
    elif mutation == "resolved_image":
        process["resolved_image_id"] = "sha256:" + "b" * 64
    elif mutation == "mutable_tag_command":
        command = process["docker_command"]
        command[command.index(process["resolved_image_id"])] = process[
            "requested_image"
        ]
    else:
        process["docker_command"].remove("--pull=never")
    _json(path, status)

    with pytest.raises(floor.ThroughputFloorError, match=match):
        _derive(campaign)


def test_process_resolved_image_must_be_full_immutable_docker_id(campaign):
    path = _status(campaign)
    status = _load(path)
    previous = status["fingerprint_input"]["image_id"]
    malformed = "sha256:fixture"
    status["fingerprint_input"]["image_id"] = malformed
    status["safety"]["preflight"]["image_id"] = malformed
    status["process"]["resolved_image_id"] = malformed
    status["process"]["docker_command"] = [
        malformed if value == previous else value
        for value in status["process"]["docker_command"]
    ]
    _json(path, status)
    _rewrite_status_fingerprint(campaign)

    with pytest.raises(floor.ThroughputFloorError, match="immutable Docker image ID"):
        _derive(campaign)


def test_coherently_rewritten_status_p05_cannot_override_raw_perf(campaign):
    path = _status(campaign)
    status = _load(path)
    # Output metrics are deliberately not part of the legacy scene input
    # fingerprint.  The raw replay must still reject this internally coherent
    # status/summary fingerprint chain.
    status["throughput"]["aggregate_current_fps"]["p05"] = 9999.0
    _json(path, status)

    with pytest.raises(floor.ThroughputFloorError, match="differs from raw DeepStream"):
        _derive(campaign)


def test_raw_deepstream_log_rewrite_is_recomputed_and_rejected(campaign):
    path = campaign["result_root"] / "scene_00/640/deepstream.log"
    content = path.read_text(encoding="utf-8")
    path.write_text(
        content.replace("8.34 (8.34)", "98.34 (98.34)", 1),
        encoding="utf-8",
    )

    with pytest.raises(floor.ThroughputFloorError, match="differs from raw DeepStream"):
        _derive(campaign)


def test_raw_gpu_csv_rewrite_is_recomputed_and_rejected(campaign):
    path = campaign["result_root"] / "scene_00/640/gpu.csv"
    content = path.read_text(encoding="utf-8")
    path.write_text(content.replace(",60,20,1200,", ",90,20,1200,", 1), encoding="utf-8")

    with pytest.raises(floor.ThroughputFloorError, match="differs from raw GPU CSV"):
        _derive(campaign)


def test_duplicate_requested_video_path_is_rejected(campaign):
    manifest = _load(campaign["manifest"])
    manifest["scenes"][1]["video_path"] = manifest["scenes"][0]["video_path"]
    _json(campaign["manifest"], manifest)

    with pytest.raises(floor.ThroughputFloorError, match="distinct requested video paths"):
        _derive(campaign)


def test_distinct_paths_with_duplicate_live_video_hash_are_rejected(campaign):
    first = campaign["root"] / "data/videos/scene_00.mp4"
    second = campaign["root"] / "data/videos/scene_01.mp4"
    second.write_bytes(first.read_bytes())

    with pytest.raises(floor.ThroughputFloorError, match="distinct live video SHA256"):
        _derive(campaign)


def test_fewer_than_ten_distinct_benchmark_types_are_rejected(campaign):
    manifest = _load(campaign["manifest"])
    for index in (1, 2, 3):
        manifest["scenes"][index]["benchmark_type"] = manifest["scenes"][0][
            "benchmark_type"
        ]
    _json(campaign["manifest"], manifest)

    with pytest.raises(floor.ThroughputFloorError, match="at least 10 distinct benchmark types"):
        _derive(campaign)


@pytest.mark.parametrize(
    "relative",
    [
        "data/videos/scene_00.mp4",
        "validation/results/scene-benchmark/scene_00/640/deepstream.txt",
        "models/person/640/config_infer_primary.txt",
        "models/person/640/fixture.engine",
        "validation/scene_benchmark/run_matrix.py",
    ],
)
def test_live_video_config_model_and_control_tamper_fails_closed(campaign, relative):
    path = campaign["root"] / relative
    path.write_bytes(path.read_bytes() + b"tampered\n")
    with pytest.raises(floor.ThroughputFloorError, match="sha256|byte count|executing module"):
        _derive(campaign)


def test_project_parser_must_equal_the_actually_imported_parser(campaign):
    path = campaign["root"] / "benchmark/summarize.py"
    path.write_bytes(path.read_bytes() + b"# substituted parser\n")

    with pytest.raises(floor.ThroughputFloorError, match="imported raw parser"):
        _derive(campaign)


@pytest.mark.parametrize("mutation,match", [
    ("gpu_inactive", "GPU utilization mean"),
    ("safety_event", "new NVIDIA Xid"),
    ("failure", "failure reasons"),
])
def test_inactive_gpu_or_unsafe_run_fails_closed(campaign, mutation, match):
    path = _status(campaign)
    status = _load(path)
    if mutation == "gpu_inactive":
        status["gpu"]["metrics"]["gpu_utilization_percent"] = {"mean": 0.0, "max": 0.0}
    elif mutation == "safety_event":
        status["safety"]["new_xid_lines"] = ["NVRM: Xid 79"]
    else:
        status["failure_reasons"] = ["fatal_patterns_in_deepstream_log"]
    _json(path, status)
    with pytest.raises(floor.ThroughputFloorError, match=match):
        _derive(campaign)


def test_deterministic_source_date_epoch_and_live_rederive(campaign, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1784188800")
    first = _derive(campaign)
    second = _derive(campaign)
    assert first == second
    artifact_path = _json(campaign["root"] / "validation/results/endurance/floor.json", first)

    # Verification remains reproducible after the environment override is gone:
    # the recorded epoch is validated and applied only to deterministic metadata.
    monkeypatch.delenv("SOURCE_DATE_EPOCH")
    verification = floor.verify_floor(
        artifact_path,
        summary_path=campaign["summary"],
        scene_manifest_path=campaign["manifest"],
        project_root=campaign["root"],
    )
    assert verification["status"] == "verified"
    assert verification["live_rederived"] is True
    assert verification["verified_safe_runs"] == 24


def test_verify_rederives_instead_of_trusting_rehashed_floor(campaign):
    artifact = _derive(campaign)
    artifact["profiles"]["640"]["aggregate_fps_floor"] = 9999.0
    artifact["fingerprint_input"] = floor._artifact_projection(artifact)
    artifact["fingerprint"] = floor._canonical_sha256(artifact["fingerprint_input"])
    with pytest.raises(floor.ThroughputFloorError, match="live re-derivation"):
        floor.verify_floor(
            artifact,
            summary_path=campaign["summary"],
            scene_manifest_path=campaign["manifest"],
            project_root=campaign["root"],
        )


def test_verify_detects_live_source_change_after_freeze(campaign):
    artifact = _derive(campaign)
    path = campaign["root"] / "data/videos/scene_00.mp4"
    path.write_bytes(path.read_bytes() + b"changed-after-freeze\n")
    with pytest.raises(floor.ThroughputFloorError, match="sha256|byte count"):
        floor.verify_floor(
            artifact,
            summary_path=campaign["summary"],
            scene_manifest_path=campaign["manifest"],
            project_root=campaign["root"],
        )


def test_verify_detects_runtime_attestation_bytes_changed_after_freeze(campaign):
    artifact = _derive(campaign)
    path = campaign["result_root"] / "scene_00/640/runtime-container.json"
    path.write_bytes(path.read_bytes() + b" \n")

    with pytest.raises(floor.ThroughputFloorError, match="live re-derivation"):
        floor.verify_floor(
            artifact,
            summary_path=campaign["summary"],
            scene_manifest_path=campaign["manifest"],
            project_root=campaign["root"],
        )


def test_cli_derive_and_verify(campaign, monkeypatch, capsys):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1784188800")
    output = campaign["root"] / "validation/results/endurance/throughput-floor.json"
    assert floor.main(
        [
            "derive",
            "--project-root",
            str(campaign["root"]),
            "--summary",
            str(campaign["summary"]),
            "--scene-manifest",
            str(campaign["manifest"]),
            "--output",
            str(output),
        ]
    ) == 0
    assert output.is_file()
    assert floor.main(
        [
            "verify",
            "--project-root",
            str(campaign["root"]),
            "--artifact",
            str(output),
            "--summary",
            str(campaign["summary"]),
            "--scene-manifest",
            str(campaign["manifest"]),
        ]
    ) == 0
    assert '"status": "verified"' in capsys.readouterr().out
