import copy
import json
import csv
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.summarize import parse_gpu_metrics, parse_perf
from validation.scene_benchmark.run_matrix import (
    DEFAULT_MANIFEST,
    EXPECTED_PERSON_FILTER,
    GPU_CSV_HEADER,
    LEGACY_GPU_CSV_HEADER,
    GpuMonitor,
    active_area_dimensions,
    assess_gpu_safety,
    count_active_perf_rows,
    discover_platform_thermal_sources,
    load_manifest,
    read_platform_thermal_row,
    render_deepstream_config,
    should_skip,
    validate_person_profile,
)
import validation.scene_benchmark.run_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]


def test_matrix_reentry_evidence_blocks_before_gpu_preflight(tmp_path, monkeypatch):
    from validation.gpu_reentry_evidence import ReentryEvidenceError

    manifest = {
        "campaign": {
            "streams": 12,
            "model_input_sizes": [640, 960],
            "duration_seconds": 300,
            "warmup_seconds": 15,
            "perf_interval_seconds": 5,
            "startup_timeout_seconds": 30,
        },
        "scenes": [{"id": "scene", "video_path": "video.mp4"}],
    }

    @contextlib.contextmanager
    def fake_lock(index):
        yield {"path": "/tmp/fake.lock", "pid": 1}

    monkeypatch.setattr(matrix, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(
        matrix,
        "probe_video",
        lambda path: {"width": 640, "height": 360, "frames": 10},
    )
    monkeypatch.setattr(matrix, "validate_person_profile", lambda size: {})
    monkeypatch.setattr(matrix, "project_relative", lambda path: str(path))
    monkeypatch.setattr(matrix, "gpu_lock", fake_lock)
    monkeypatch.setattr(
        matrix,
        "prevalidate_runtime_compatibility",
        lambda *args, **kwargs: {"status": "production_ready"},
    )
    monkeypatch.setattr(matrix, "write_matrix_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        matrix,
        "preflight",
        lambda **kwargs: pytest.fail("GPU preflight must not run before re-entry passes"),
    )
    monkeypatch.setattr(
        "validation.gpu_reentry_evidence.require_reentry_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ReentryEvidenceError("physical evidence incomplete")
        ),
    )

    result = matrix.main(
        [
            "--manifest",
            "synthetic.json",
            "--output",
            str(tmp_path),
            "--scenes",
            "scene",
            "--sizes",
            "640",
            "--duration",
            "1",
            "--warmup",
            "0",
            "--perf-interval",
            "1",
            "--startup-timeout",
            "1",
        ]
    )
    assert result == 3
    artifact = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert artifact["status"] == "reentry_blocked"
    assert artifact["gpu_process_started"] is False


def test_scene_manifest_has_twelve_distinct_licensed_local_types():
    manifest = load_manifest(DEFAULT_MANIFEST)
    scenes = manifest["scenes"]

    assert len(scenes) == 12
    assert len({scene["benchmark_type"] for scene in scenes}) == 12
    assert manifest["campaign"]["streams"] == 12
    assert manifest["campaign"]["model_input_sizes"] == [640, 960]
    assert manifest["campaign"]["duration_seconds"] == 300
    assert manifest["campaign"]["warmup_seconds"] == 15
    assert all((ROOT / scene["video_path"]).is_file() for scene in scenes)
    assert all(scene["source_metadata"]["license"]["spdx"] for scene in scenes)
    assert "Repeated frames" in manifest["accuracy_policy"]["performance_runs"]
    assert "every source frame once" in manifest["accuracy_policy"]["accuracy_runs"]


@pytest.mark.parametrize("size", [640, 960])
def test_deploy_person_profiles_filter_every_non_person_coco_class(size):
    profile = validate_person_profile(size)
    assert profile["person_only_classes"] == [0]
    infer = (ROOT / profile["infer_config"]).read_text(encoding="utf-8")
    line = next(line for line in infer.splitlines() if line.startswith("filter-out-class-ids="))
    assert [int(value) for value in line.partition("=")[2].split(";")] == EXPECTED_PERSON_FILTER
    assert len(profile["infer_config_sha256"]) == 64
    assert len(profile["engine_sha256"]) == 64


def test_generated_config_is_looped_twelve_source_fake_sink_person_only(tmp_path):
    destination = tmp_path / "deepstream.txt"
    source = ROOT / "data/derived/open-h264/fr_paris_snow_umbrellas.mp4"
    text = render_deepstream_config(
        destination,
        video_path=source,
        video_metadata={"width": 1920, "height": 1080},
        size=640,
        streams=12,
        perf_interval_seconds=5,
    )

    assert destination.read_text(encoding="utf-8") == text
    assert "[source0]\nenable=1\ntype=3\n" in text
    assert "num-sources=12" in text
    assert "[sink0]\nenable=1\ntype=1\nsync=0\nqos=0" in text
    assert "config-file=/models/person/640/config_infer_primary.txt" in text
    assert "width=640\nheight=360\nenable-padding=0" in text
    assert "[tests]\nfile-loop=1" in text

    with pytest.raises(ValueError, match=r"\.txt extension"):
        render_deepstream_config(
            tmp_path / "deepstream.ini",
            video_path=source,
            video_metadata={"width": 1920, "height": 1080},
            size=640,
            streams=12,
            perf_interval_seconds=5,
        )


@pytest.mark.parametrize(
    ("source_width", "source_height", "size", "expected"),
    [
        (1920, 1080, 640, (640, 360)),
        (1920, 1080, 960, (960, 540)),
        (1080, 1920, 640, (360, 640)),
        (384, 288, 960, (960, 720)),
        (478, 270, 960, (960, 542)),
    ],
)
def test_active_area_dimensions_preserve_source_aspect_without_nvinfer_upscale(
    source_width, source_height, size, expected
):
    assert active_area_dimensions(source_width, source_height, size) == expected


def _perf_line(values):
    return "**PERF:  " + "\t".join(f"{value:.2f} ({value:.2f})" for value in values)


def test_perf_parser_uses_only_bounded_steady_state_and_requires_every_stream(tmp_path):
    log = tmp_path / "deepstream.log"
    header = "**PERF:  " + "\t".join(f"FPS {index} (Avg)" for index in range(12))
    startup = [_perf_line([1.0] * 12), _perf_line([2.0] * 12)]
    steady = [_perf_line([50.0] * 12), _perf_line([51.0] * 12)]
    teardown = [_perf_line([3.0] * 12)]
    log.write_text("\n".join([header, *startup, *steady, *teardown]) + "\n")

    result = parse_perf(
        log,
        expected_streams=12,
        duration_seconds=10,
        perf_interval_seconds=5,
        active_row_start=2,
        active_row_end=4,
    )
    assert result["status"] == "ok"
    assert result["perf_intervals_analyzed"] == 2
    assert result["aggregate_current_fps"]["mean"] == 606.0
    assert count_active_perf_rows(log, 12) == 5

    broken = tmp_path / "broken.log"
    inactive = [50.0] * 11 + [0.0]
    broken.write_text("\n".join([header, _perf_line(inactive), _perf_line(inactive)]))
    failed = parse_perf(broken, 12, 10, 5)
    assert failed["status"] == "inactive_streams"
    assert failed["inactive_stream_ids"] == [11]


def test_gpu_parser_excludes_startup_and_teardown_samples(tmp_path):
    metrics = tmp_path / "gpu.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # This deliberately uses the pre-power-guard schema: old campaign CSVs
        # must remain parseable after adding power/throttle columns.
        writer.writerow(LEGACY_GPU_CSV_HEADER)
        for index in range(8):
            writer.writerow(
                [
                    f"2026/07/16 00:00:0{index}.000",
                    "0",
                    "GPU",
                    str(index * 10),
                    "1",
                    "100",
                    "1000",
                    "50",
                    "80",
                    "1500",
                    "7000",
                ]
            )

    result = parse_gpu_metrics(
        metrics, duration_seconds=3, sample_start=2, sample_end=6
    )
    assert result["status"] == "ok"
    assert result["samples_analyzed"] == 4
    assert result["metrics"]["gpu_utilization_percent"]["mean"] == 35.0
    assert result["analysis_first_timestamp"].endswith("02.000")
    assert result["analysis_last_timestamp"].endswith("05.000")
    assert result["power_throttle_telemetry_available"] == {
        "power_limits": False,
        "pstate": False,
        "clock_event_reasons": False,
    }


def _gpu_row(**overrides):
    values = {
        "timestamp": "2026/07/16 00:00:00.000",
        "gpu_index": "0",
        "gpu_name": "GPU",
        "gpu_utilization_percent": "90",
        "memory_utilization_percent": "50",
        "memory_used_mib": "100",
        "memory_total_mib": "1000",
        "temperature_c": "81",
        "power_draw_w": "110",
        "sm_clock_mhz": "1400",
        "memory_clock_mhz": "7000",
        "power_requested_limit_w": "[N/A]",
        "power_current_limit_w": "115",
        "power_default_limit_w": "115",
        "pstate": "P0",
        "clock_event_reasons_active_mask": "0x4",
        "clock_event_sw_power_cap": "Active",
        "clock_event_sw_thermal_slowdown": "Not Active",
        "clock_event_hw_slowdown": "Not Active",
        "clock_event_hw_thermal_slowdown": "Not Active",
        "clock_event_hw_power_brake_slowdown": "Not Active",
    }
    values.update({key: str(value) for key, value in overrides.items()})
    return [values[column] for column in GPU_CSV_HEADER]


def test_power_guard_detects_effective_limit_drop_with_requested_na():
    row = dict(zip(GPU_CSV_HEADER, _gpu_row(power_current_limit_w="55")))
    result = assess_gpu_safety(row, power_limit_drop_tolerance_w=5.0)

    assert result["power_limit_telemetry_complete"] is True
    assert result["power_limit_drop_detected"] is True
    assert result["power_limit_drop_w_by_field"] == {
        "power_current_limit_w": 60.0
    }
    assert result["clock_event_sw_power_cap"] is True


def test_gpu_monitor_persists_safety_event_on_power_limit_drop(tmp_path, monkeypatch):
    rows = iter(
        [
            _gpu_row(),
            _gpu_row(
                timestamp="2026/07/16 00:00:01.000",
                power_current_limit_w="55",
                pstate="P3",
            ),
        ]
    )
    monkeypatch.setattr(
        "validation.scene_benchmark.run_matrix.query_gpu_row", lambda _gpu: next(rows)
    )
    monitor = GpuMonitor(
        tmp_path / "gpu.csv",
        tmp_path / "gpu-safety-event.json",
        0,
        86.0,
        5.0,
        2,
        operating_policy_mode="legacy_strict",
    )
    monkeypatch.setattr(monitor.stop_event, "wait", lambda _timeout: False)
    monitor.run()

    event = json.loads((tmp_path / "gpu-safety-event.json").read_text())
    assert monitor.safety_reason_code == "power_limit_below_default"
    assert event["code"] == "power_limit_below_default"
    assert event["sample_number"] == 2
    assert event["gpu_csv_line_number"] == 3
    assert event["snapshot"]["pstate"] == "P3"


def test_gpu_monitor_requires_consecutive_slowdown_samples(tmp_path, monkeypatch):
    rows = iter(
        [
            _gpu_row(clock_event_sw_thermal_slowdown="Active"),
            _gpu_row(clock_event_sw_thermal_slowdown="Not Active"),
            _gpu_row(clock_event_hw_slowdown="Active"),
            _gpu_row(clock_event_hw_slowdown="Active"),
        ]
    )
    monkeypatch.setattr(
        "validation.scene_benchmark.run_matrix.query_gpu_row", lambda _gpu: next(rows)
    )
    monitor = GpuMonitor(
        tmp_path / "gpu.csv",
        tmp_path / "gpu-safety-event.json",
        0,
        86.0,
        5.0,
        2,
        operating_policy_mode="legacy_strict",
    )
    monkeypatch.setattr(monitor.stop_event, "wait", lambda _timeout: False)
    monitor.run()

    assert monitor.safety_reason_code == "sustained_clock_slowdown"
    assert monitor.safety_event["sample_number"] == 4
    assert monitor.max_consecutive_slowdown_samples == 2


def test_gpu_monitor_workstation_managed_records_static_signals_without_abort(
    tmp_path, monkeypatch
):
    rows = [
        _gpu_row(
            temperature_c="90",
            power_current_limit_w="55",
            clock_event_sw_thermal_slowdown="Active",
        ),
        _gpu_row(
            timestamp="2026/07/16 00:00:01.000",
            temperature_c="91",
            power_current_limit_w="55",
            clock_event_hw_slowdown="Active",
        ),
    ]
    monitor = GpuMonitor(
        tmp_path / "gpu.csv",
        tmp_path / "gpu-safety-event.json",
        0,
        86.0,
        5.0,
        2,
    )

    def query(_gpu):
        row = rows.pop(0)
        if not rows:
            monitor.stop_event.set()
        return row

    monkeypatch.setattr(matrix, "query_gpu_row", query)
    monitor.run()

    assert monitor.operating_policy_mode == "workstation_managed"
    assert monitor.safety_reason is None
    assert monitor.safety_event is None
    assert not (tmp_path / "gpu-safety-event.json").exists()
    assert monitor.temperature_threshold_samples == 2
    assert monitor.maximum_temperature_c == 91.0
    assert monitor.power_limit_drop_samples == 2
    assert monitor.slowdown_active_samples == 2
    assert monitor.diagnostic_event_counts == {
        "temperature_threshold": 2,
        "power_limit_below_default": 2,
        "sustained_clock_slowdown": 1,
    }
    assert all(
        event["disposition"] == "record_only_workstation_hardware_managed"
        for event in monitor.diagnostic_events
    )


def test_gpu_monitor_missing_required_temperature_telemetry_remains_fail_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        matrix, "query_gpu_row", lambda _gpu: _gpu_row(temperature_c="[N/A]")
    )
    monitor = GpuMonitor(
        tmp_path / "gpu.csv",
        tmp_path / "gpu-safety-event.json",
        0,
        86.0,
        5.0,
        2,
    )

    monitor.run()

    assert monitor.safety_reason_code == "temperature_telemetry_unavailable"
    assert monitor.safety_event["code"] == "temperature_telemetry_unavailable"


def test_gpu_monitor_identity_driver_drift_remains_fail_closed(
    tmp_path, monkeypatch
):
    expected = {
        "index": "0",
        "uuid": "GPU-fixture",
        "name": "GPU",
        "driver_version": "590.48.01",
        "memory.total": "16384",
        "pci.bus_id": "00000000:01:00.0",
    }
    observed = {**expected, "driver_version": "591.00.00"}
    monkeypatch.setattr(matrix, "query_gpu_row", lambda _gpu: _gpu_row())
    monkeypatch.setattr(matrix, "query_gpu_identity", lambda _gpu: observed)
    monitor = GpuMonitor(
        tmp_path / "gpu.csv",
        tmp_path / "gpu-safety-event.json",
        0,
        86.0,
        5.0,
        2,
        expected_gpu_identity=expected,
    )

    monitor.run()

    assert monitor.safety_reason_code == "gpu_identity_drift"
    assert monitor.gpu_identity_checks == 1
    assert monitor.safety_event["observed_gpu_identity"]["driver_version"] == (
        "591.00.00"
    )


def test_gpu_monitor_malformed_slowdown_telemetry_remains_fail_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        matrix,
        "query_gpu_row",
        lambda _gpu: _gpu_row(clock_event_hw_slowdown="[N/A]"),
    )
    monitor = GpuMonitor(
        tmp_path / "gpu.csv",
        tmp_path / "gpu-safety-event.json",
        0,
        86.0,
        5.0,
        2,
    )

    monitor.run()

    assert monitor.safety_reason_code == "slowdown_telemetry_unavailable"


def test_preflight_workstation_managed_records_static_signals_but_stays_ok(
    monkeypatch,
):
    monkeypatch.setattr(matrix.shutil, "which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(
        matrix,
        "command",
        lambda args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=("sha256:" + "a" * 64 if args[:3] == ["docker", "image", "inspect"] else ""),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        matrix,
        "query_gpu_row",
        lambda _gpu: _gpu_row(
            temperature_c="90",
            power_current_limit_w="55",
            clock_event_hw_thermal_slowdown="Active",
        ),
    )
    monkeypatch.setattr(matrix.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        matrix, "read_power_profile", lambda: {"available": True, "value": "performance"}
    )
    monkeypatch.setattr(
        matrix, "read_xid_log", lambda: {"available": True, "count": 0, "lines": []}
    )
    monkeypatch.setattr(
        matrix,
        "discover_platform_thermal_sources",
        lambda: {"available": False, "sources": [], "columns": ["timestamp"]},
    )
    monkeypatch.setattr(
        matrix,
        "query_gpu_identity",
        lambda _gpu: {
            "index": "0",
            "uuid": "GPU-fixture",
            "name": "GPU",
            "driver_version": "fixture",
            "memory.total": "16384",
            "pci.bus_id": "00000000:01:00.0",
        },
    )

    report = matrix.preflight(
        image="deepsafe-deepstream:9.0",
        gpu_index=0,
        max_temperature_c=86.0,
        allow_non_performance_profile=False,
        power_limit_drop_tolerance_w=5.0,
        slowdown_consecutive_samples=2,
    )

    assert report["status"] == "ok"
    assert report["safety_events"] == []
    assert {event["code"] for event in report["diagnostic_events"]} == {
        "temperature_threshold",
        "power_limit_below_default",
        "sustained_clock_slowdown",
    }
    assert report["power_safety_policy"]["operating_policy_mode"] == (
        "workstation_managed"
    )
    assert report["power_safety_policy"]["abort_slowdown_flags"] == []


def test_preflight_workstation_managed_does_not_relax_required_power_telemetry(
    monkeypatch,
):
    monkeypatch.setattr(matrix.shutil, "which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(
        matrix,
        "command",
        lambda args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=("sha256:" + "a" * 64 if args[:3] == ["docker", "image", "inspect"] else ""),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        matrix,
        "query_gpu_row",
        lambda _gpu: _gpu_row(power_current_limit_w="[N/A]"),
    )
    monkeypatch.setattr(matrix.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        matrix, "read_power_profile", lambda: {"available": True, "value": "performance"}
    )
    monkeypatch.setattr(
        matrix, "read_xid_log", lambda: {"available": True, "count": 0, "lines": []}
    )
    monkeypatch.setattr(
        matrix,
        "discover_platform_thermal_sources",
        lambda: {"available": False, "sources": [], "columns": ["timestamp"]},
    )
    monkeypatch.setattr(
        matrix,
        "query_gpu_identity",
        lambda _gpu: {
            "index": "0",
            "uuid": "GPU-fixture",
            "name": "GPU",
            "driver_version": "fixture",
            "memory.total": "16384",
            "pci.bus_id": "00000000:01:00.0",
        },
    )

    report = matrix.preflight(
        image="deepsafe-deepstream:9.0",
        gpu_index=0,
        max_temperature_c=86.0,
        allow_non_performance_profile=False,
        power_limit_drop_tolerance_w=5.0,
        slowdown_consecutive_samples=2,
    )

    assert report["status"] == "safety_abort"
    assert report["safety_events"][0]["code"] == (
        "power_limit_telemetry_unavailable"
    )


def test_gpu_parser_summarizes_new_power_throttle_columns(tmp_path):
    metrics = tmp_path / "gpu.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(GPU_CSV_HEADER)
        writer.writerow(_gpu_row())
        writer.writerow(
            _gpu_row(
                timestamp="2026/07/16 00:00:01.000",
                power_current_limit_w="55",
                pstate="P3",
                clock_event_sw_thermal_slowdown="Active",
            )
        )

    result = parse_gpu_metrics(metrics, duration_seconds=2)
    assert result["status"] == "ok"
    assert result["metrics"]["power_current_limit_w"]["min"] == 55.0
    assert result["pstate_counts"] == {"P0": 1, "P3": 1}
    assert result["active_clock_event_samples"][
        "clock_event_sw_thermal_slowdown"
    ] == 1


def test_platform_thermal_discovery_and_read_are_manifested_best_effort(tmp_path):
    hwmon_root = tmp_path / "hwmon"
    dell = hwmon_root / "hwmon6"
    dell.mkdir(parents=True)
    (dell / "name").write_text("dell_smm\n")
    (dell / "fan1_input").write_text("2000\n")
    (dell / "temp1_input").write_text("55000\n")
    thermal_root = tmp_path / "thermal"
    zone = thermal_root / "thermal_zone5"
    zone.mkdir(parents=True)
    (zone / "type").write_text("TVGA\n")
    (zone / "temp").write_text("50\n")

    manifest = discover_platform_thermal_sources(hwmon_root, thermal_root)
    row, errors = read_platform_thermal_row(manifest, "timestamp")

    assert manifest["available"] is True
    assert manifest["failure_policy"].endswith("never abort benchmark")
    assert manifest["missing_thermal_zone_types"] == ["TCPU", "TSKN"]
    assert manifest["columns"] == [
        "timestamp",
        "dell_fan1_rpm",
        "dell_temp1_c",
        "thermal_tvga_c",
    ]
    assert row == ["timestamp", "2000.000", "55.000", "50.000"]
    assert errors == []

    (dell / "temp1_input").write_text("nan\n")
    malformed_row, malformed_errors = read_platform_thermal_row(
        manifest, "timestamp-2"
    )
    assert malformed_row == ["timestamp-2", "2000.000", "", "50.000"]
    assert len(malformed_errors) == 1
    assert "non-finite" in malformed_errors[0]


def test_resume_only_skips_matching_complete_fingerprint(tmp_path):
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps({"status": "complete", "fingerprint": "same"}),
        encoding="utf-8",
    )
    assert should_skip(status, "same") is True
    assert should_skip(status, "different") is False
    assert should_skip(status, "same", force=True) is False

    status.write_text(
        json.dumps({"status": "interrupted", "fingerprint": "same"}),
        encoding="utf-8",
    )
    assert should_skip(status, "same") is False


def test_scene_docker_command_is_offline_and_uses_exact_resolved_image_id():
    image_id = "sha256:" + "a" * 64
    config_path = ROOT / "validation/results/fixture/deepstream.txt"
    command = matrix.docker_command(
        image=image_id,
        gpu_index=0,
        config_path=config_path,
        container_name="deepsafe-scene-test",
    )

    assert command[:4] == ["docker", "run", "--rm", "--pull=never"]
    assert command.count(image_id) == 1
    assert command[command.index(image_id) + 1 :] == [
        "deepstream-app",
        "-c",
        matrix.container_path(config_path),
    ]
    assert "device=0" in command


def _scene_runtime_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "PROJECT_ROOT", tmp_path)
    (tmp_path / "models").mkdir()
    config_path = tmp_path / "attempt/deepstream.txt"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("[application]\n", encoding="utf-8")
    image_id = "sha256:" + "a" * 64
    container_id = "b" * 64
    container_name = "deepsafe-scene-640-fixture"
    launch = matrix.docker_command(
        image=image_id,
        gpu_index=0,
        config_path=config_path,
        container_name=container_name,
    )
    cmd = launch[launch.index(image_id) + 1 :]
    entrypoint = ["/opt/nvidia/nvidia_entrypoint.sh"]
    inspect = {
        "Id": container_id,
        "Name": f"/{container_name}",
        "Image": image_id,
        "Path": entrypoint[0],
        "Args": list(cmd),
        "Config": {
            "Image": image_id,
            "Cmd": list(cmd),
            "Entrypoint": entrypoint,
            "WorkingDir": "/workspace",
            "Env": [
                "PATH=/usr/local/bin:/usr/bin",
                "GST_DEBUG=1",
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
            ],
        },
        "HostConfig": {
            "AutoRemove": True,
            "IpcMode": "host",
            "DeviceRequests": [
                {
                    "Driver": "nvidia",
                    "Count": 0,
                    "DeviceIDs": ["0"],
                    "Capabilities": [["gpu"]],
                    "Options": {},
                }
            ],
        },
        "State": {
            "Running": True,
            "StartedAt": "2026-07-16T08:00:00.000000000Z",
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": str(tmp_path),
                "Destination": "/workspace",
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": str(tmp_path / "models"),
                "Destination": "/models",
                "RW": False,
            },
        ],
    }
    return launch, inspect, container_id, container_name, image_id


def _install_scene_runtime_docker(
    monkeypatch, launch, inspect, container_id, container_name
):
    del launch
    responses = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=f"{container_id}\t{container_name}\n",
                stderr="",
            ),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps([inspect]),
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr(matrix, "command", lambda *_args, **_kwargs: next(responses))


def test_scene_runtime_attestation_is_atomically_written_from_exact_inspect(
    tmp_path, monkeypatch
):
    launch, inspect, container_id, container_name, image_id = _scene_runtime_fixture(
        tmp_path, monkeypatch
    )
    _install_scene_runtime_docker(
        monkeypatch, launch, inspect, container_id, container_name
    )
    destination = tmp_path / "attempt/runtime-container.json"

    result = matrix.capture_and_write_runtime_container_attestation(
        destination,
        launch_command=launch,
        container_name=container_name,
        resolved_image_id=image_id,
        gpu_index=0,
    )

    assert result["schema_version"] == matrix.RUNTIME_CONTAINER_ATTESTATION_SCHEMA
    assert result["container_id"] == container_id
    assert result["image_id"] == image_id
    assert result["cmd"] == launch[launch.index(image_id) + 1 :]
    assert result["gpu_device_request"]["device_ids"] == ["0"]
    assert result["launch_command_sha256"] == (
        matrix.canonical_launch_command_sha256(launch)
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == result
    assert not destination.with_suffix(".json.tmp").exists()


def test_scene_runtime_attestation_waits_until_exact_name_is_visible(
    tmp_path, monkeypatch
):
    launch, _inspect, _container_id, container_name, image_id = (
        _scene_runtime_fixture(tmp_path, monkeypatch)
    )
    monkeypatch.setattr(
        matrix,
        "command",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    destination = tmp_path / "attempt/runtime-container.json"

    result = matrix.capture_and_write_runtime_container_attestation(
        destination,
        launch_command=launch,
        container_name=container_name,
        resolved_image_id=image_id,
        gpu_index=0,
    )

    assert result is None
    assert not destination.exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("image", "image/name identity"),
        ("cmd", "Config.Cmd"),
        ("actual_process", "Path/Args"),
        ("gpu", "GPU device ID"),
        ("mount", "read-only mount"),
        ("ipc", "lifecycle/IPC"),
        ("workdir", "working directory"),
        ("environment", "required environment"),
        ("running", "not running"),
    ],
)
def test_scene_runtime_attestation_rejects_inspect_drift(
    tmp_path, monkeypatch, mutation, match
):
    launch, inspect, container_id, container_name, image_id = _scene_runtime_fixture(
        tmp_path, monkeypatch
    )
    inspect = copy.deepcopy(inspect)
    if mutation == "image":
        inspect["Image"] = "sha256:" + "c" * 64
    elif mutation == "cmd":
        inspect["Config"]["Cmd"][-1] = "/workspace/forged.txt"
    elif mutation == "actual_process":
        inspect["Args"][-1] = "/workspace/forged.txt"
    elif mutation == "gpu":
        inspect["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["7"]
    elif mutation == "mount":
        inspect["Mounts"][0]["RW"] = True
    elif mutation == "ipc":
        inspect["HostConfig"]["IpcMode"] = "private"
    elif mutation == "workdir":
        inspect["Config"]["WorkingDir"] = "/tmp"
    elif mutation == "environment":
        inspect["Config"]["Env"].remove("GST_DEBUG=1")
    else:
        inspect["State"]["Running"] = False
    _install_scene_runtime_docker(
        monkeypatch, launch, inspect, container_id, container_name
    )

    with pytest.raises(RuntimeError, match=match):
        matrix.capture_runtime_container_attestation(
            launch_command=launch,
            container_name=container_name,
            resolved_image_id=image_id,
            gpu_index=0,
        )


def test_scene_runtime_attestation_missing_or_error_is_fail_closed(tmp_path):
    destination = tmp_path / "runtime-container.json"
    assert matrix.runtime_container_attestation_failure_reasons(
        None, None, destination
    ) == ["runtime_container_attestation_missing"]
    assert matrix.runtime_container_attestation_failure_reasons(
        None, "RuntimeError: inspect drift", destination
    ) == ["runtime_container_attestation_error=RuntimeError: inspect drift"]
    assert matrix.runtime_container_attestation_failure_reasons(
        {"status": "verified_running"}, None, destination
    ) == ["runtime_container_attestation_artifact_missing"]


def test_scene_run_cannot_complete_without_live_runtime_attestation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(matrix, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        matrix,
        "require_runtime_compatibility",
        lambda *args, **kwargs: {
            "status": "production_ready",
            "resolved_image_id": "sha256:" + "a" * 64,
            "receipt": {
                "path": "validation/results/ds9-runtime-compatibility/current/receipt.json",
                "bytes": 1,
                "sha256": "b" * 64,
            },
        },
    )
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture-video")
    video_stat = video.stat()
    monkeypatch.setattr(
        matrix,
        "probe_video",
        lambda _path: {
            "path": "video.mp4",
            "size_bytes": video_stat.st_size,
            "mtime_ns": video_stat.st_mtime_ns,
            "sha256": "1" * 64,
            "width": 640,
            "height": 360,
        },
    )
    monkeypatch.setattr(matrix, "validate_person_profile", lambda _size: {})
    monkeypatch.setattr(
        matrix,
        "read_xid_log",
        lambda: {"available": True, "count": 0, "lines": []},
    )
    monkeypatch.setattr(
        matrix,
        "read_power_profile",
        lambda: {"available": True, "value": "performance"},
    )
    monkeypatch.setattr(
        matrix,
        "capture_and_write_runtime_container_attestation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        matrix,
        "command",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    monkeypatch.setattr(
        matrix,
        "parse_perf",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "header_streams": 12,
            "inactive_stream_ids": [],
            "streams_with_nonpositive_fps": [],
            "aggregate_current_fps": {},
        },
    )
    monkeypatch.setattr(
        matrix,
        "parse_gpu_metrics",
        lambda *_args, **_kwargs: {"status": "ok"},
    )

    class FakeMonitor:
        def __init__(self, *_args, **_kwargs):
            self.safety_reason = None
            self.safety_reason_code = None
            self.safety_event = None
            self.query_errors = []
            self.samples = 0
            self.power_limit_drop_samples = 0
            self.slowdown_active_samples = 0
            self.max_consecutive_slowdown_samples = 0
            self.platform_thermal_samples = 0
            self.platform_thermal_error_count = 0
            self.platform_thermal_errors = []
            self.ident = None

        def start(self):
            return None

        def stop(self):
            return None

        def join(self, *_args, **_kwargs):
            return None

    class FakeProcess:
        returncode = 0
        pid = 12345

        def poll(self):
            return 0

    monkeypatch.setattr(matrix, "GpuMonitor", FakeMonitor)
    monkeypatch.setattr(
        matrix.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    image_id = "sha256:" + "a" * 64
    preflight = {
        "image": "deepsafe-deepstream:9.0",
        "image_id": image_id,
        "gpu_identity": {"name": "fixture GPU"},
        "power_profile": {"available": True, "value": "performance"},
        "performance_profile_required": True,
        "power_safety_policy": {"fixture": True},
        "gpu_safety_assessments": [{"power_limits_w": {}}],
        "platform_thermal_sources": {
            "schema_version": "fixture/v1",
            "available": False,
            "columns": [],
            "sources": [],
        },
    }
    scene = {
        "id": "scene-fixture",
        "video_path": "video.mp4",
        "benchmark_type": "fixture",
        "security_camera_relevance": "primary",
        "source_manifest_id": "fixture",
        "source_metadata": {},
        "notes": "fixture",
    }

    status = matrix.run_one(
        scene=scene,
        size=640,
        output_root=tmp_path / "results",
        duration=1,
        warmup=0,
        perf_interval=1,
        startup_timeout=1,
        streams=12,
        image="deepsafe-deepstream:9.0",
        gpu_index=0,
        kill_grace=1,
        max_temperature_c=86.0,
        power_limit_drop_tolerance_w=5.0,
        slowdown_consecutive_samples=2,
        campaign_manifest={"accuracy_policy": {}},
        preflight_report=preflight,
        force=True,
    )

    assert status["status"] == "failed"
    assert "runtime_container_attestation_missing" in status["failure_reasons"]
    assert status["runtime_container_attestation"] is None
    assert status["process"]["runtime_container_id"] is None


def test_archive_previous_scene_attempt_includes_runtime_attestation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(matrix, "PROJECT_ROOT", tmp_path)
    run_dir = tmp_path / "scene/640"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text("{}", encoding="utf-8")
    (run_dir / "runtime-container.json").write_text("{}", encoding="utf-8")

    archived = matrix.archive_previous_attempt(run_dir)

    assert archived is not None
    destination = tmp_path / archived
    assert (destination / "status.json").is_file()
    assert (destination / "runtime-container.json").is_file()
    assert not (run_dir / "runtime-container.json").exists()
