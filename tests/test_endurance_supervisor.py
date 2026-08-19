import csv
import contextlib
import fcntl
import io
import json
import os
import re
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from admin.app import app
from validation.endurance import supervisor as endurance
from validation.endurance.supervisor import (
    FilteredLogCollector,
    LatencyBuckets,
    apply_perf_window_contract,
    build_plan,
    campaign_config_fingerprint,
    create_gpu_monitor,
    create_session_receipt,
    gpu_safety_health_gate,
    finalize_attempt_receipt,
    load_campaign,
    main,
    normalize_power_safety_policy,
    parsed_telemetry_contract_gates,
    process_tree_rss_mib,
    require_compatible_session_history,
    reconcile_active,
    render_deepstream_config,
    summarize_latency,
    trend,
    validate_resume_checkpoint,
    verify_attempt_receipt,
    verify_session_binding,
)


def _verified_floor_binding(
    *,
    image="deepsafe-deepstream:9.0",
    image_id="sha256:test",
    gpu_identity=None,
):
    gpu_identity = gpu_identity or {
        "index": "0",
        "uuid": "GPU-test",
        "name": "Fake GPU",
        "driver_version": "test",
        "memory.total": "1 MiB",
        "pci.bus_id": "00000000:01:00.0",
    }
    return {
        "schema_version": endurance.THROUGHPUT_FLOOR_BINDING_SCHEMA,
        "status": "verified",
        "artifact_schema": endurance.THROUGHPUT_FLOOR_ARTIFACT_SCHEMA,
        "artifact_fingerprint": "f" * 64,
        "artifact_pin": {
            "path": "validation/results/endurance/throughput-floor.json",
            "size_bytes": 1,
            "sha256": "e" * 64,
        },
        "profiles": {
            "640": {
                "aggregate_fps_floor": 100.0,
                "per_stream_fps_floor": 8.3,
            },
            "960": {
                "aggregate_fps_floor": 70.0,
                "per_stream_fps_floor": 5.8,
            },
        },
        "verification": {
            "status": "verified",
            "live_rederived": True,
            "verified_safe_runs": 24,
        },
        "source_runtime_identity": {
            "image": image,
            "image_id": image_id,
            "gpu_index": 0,
            "gpu_identity": gpu_identity,
            "power_profile": {"available": True, "value": "performance"},
            "max_temperature_c": 86.0,
            "power_safety_policy": {
                "operating_policy_mode": "workstation_managed",
                "hardware_protection_owner": (
                    "workstation_bios_ec_nvidia_driver"
                ),
                "static_signal_action": (
                    "record_measurement_quality_diagnostic"
                ),
                "power_limit_drop_tolerance_w": 5.0,
                "slowdown_consecutive_samples": 2,
                "preflight_samples": 2,
                "preflight_sample_interval_seconds": 1.0,
                "power_limit_fields": endurance.POWER_LIMIT_FIELDS,
                "diagnostic_slowdown_flags": endurance.ABORT_SLOWDOWN_FLAGS,
                "abort_slowdown_flags": [],
                "required_telemetry_failure_action": "safety_abort",
            },
        },
        "source_inputs": {
            "summary": {"path": "summary.json", "bytes": 1, "sha256": "a" * 64},
            "scene_manifest": {
                "path": "scenes.json",
                "bytes": 1,
                "sha256": "b" * 64,
            },
        },
    }


def _floor_evidence(profile=640):
    binding = _verified_floor_binding()
    item = binding["profiles"][str(profile)]
    return {
        "schema_version": "deepsafe.endurance-throughput-floor-evidence/v1",
        "status": "verified",
        "artifact_fingerprint": binding["artifact_fingerprint"],
        "profile": profile,
        **item,
    }


def _mock_verified_floor(monkeypatch):
    @contextlib.contextmanager
    def fake_gpu_lease_session(**_kwargs):
        yield SimpleNamespace()

    monkeypatch.setattr(endurance, "gpu_lease_session", fake_gpu_lease_session)
    monkeypatch.setattr(
        endurance,
        "resolve_throughput_floor_binding",
        lambda *args, **kwargs: deepcopy(_verified_floor_binding()),
    )
    monkeypatch.setattr(
        endurance,
        "verify_live_throughput_floor",
        lambda campaign: campaign["throughput_floor"],
    )
    monkeypatch.setattr(
        endurance,
        "validate_throughput_floor_preflight_runtime",
        lambda campaign, report: campaign["throughput_floor"][
            "source_runtime_identity"
        ],
    )


def test_production_supervisor_holds_repository_lease_around_entire_body(monkeypatch):
    events = []
    args = SimpleNamespace(
        dry_run=False,
        plan_only=False,
        acknowledge_seven_day_run=True,
        gpu_index=3,
    )

    @contextlib.contextmanager
    def fake_session(**kwargs):
        events.append(("enter", kwargs))
        yield
        events.append(("exit", kwargs))

    monkeypatch.setattr(endurance, "gpu_lease_session", fake_session)
    monkeypatch.setattr(
        endurance, "_run_supervisor_body", lambda received: events.append(("body", received)) or 7
    )
    command = ["python3", "supervisor.py", "--acknowledge-seven-day-run"]

    assert endurance.run_supervisor(args, lease_command=command) == 7
    assert [item[0] for item in events] == ["enter", "body", "exit"]
    assert events[0][1] == {
        "owner_kind": "endurance_7day",
        "gpu_index": 3,
        "command": command,
        "ttl_seconds": 30,
        "terminate_on_heartbeat_failure": True,
    }


def test_seven_day_plan_is_balanced_and_uses_six_hour_segments():
    campaign = load_campaign()
    plan = build_plan(campaign)

    assert len(plan) == 28
    assert [item["profile"] for item in plan[:4]] == [640, 960, 640, 960]
    assert {item["duration_seconds"] for item in plan} == {6 * 60 * 60}
    assert sum(item["duration_seconds"] for item in plan) == 7 * 24 * 60 * 60
    assert sum(item["duration_seconds"] for item in plan if item["profile"] == 640) == 84 * 60 * 60
    assert sum(item["duration_seconds"] for item in plan if item["profile"] == 960) == 84 * 60 * 60
    assert {item["campaign_day"] for item in plan} == set(range(1, 8))
    assert campaign["power_safety"]["power_limit_drop_tolerance_w"] == 5.0
    assert campaign["power_safety"]["slowdown_consecutive_samples"] == 2
    assert campaign["power_safety"]["operating_policy_mode"] == "workstation_managed"
    assert campaign["power_safety"]["power_limit_drop_action"] == (
        "record_measurement_quality_diagnostic"
    )
    assert campaign["power_safety"]["abort_slowdown_flags"] == []
    assert "clock_event_reasons_active_mask" in campaign["power_safety"][
        "clock_event_telemetry_fields"
    ]
    assert "clock_event_sw_power_cap" in campaign["power_safety"][
        "clock_event_telemetry_fields"
    ]


def test_endurance_power_policy_is_backward_compatible_and_validated():
    defaults = normalize_power_safety_policy({})
    assert defaults["power_limit_drop_tolerance_w"] == 5.0
    assert defaults["slowdown_consecutive_samples"] == 2
    assert defaults["operating_policy_mode"] == "workstation_managed"
    assert defaults["abort_slowdown_flags"] == []

    legacy = normalize_power_safety_policy(
        {"power_safety": {"operating_policy_mode": "legacy_strict"}}
    )
    assert legacy["abort_slowdown_flags"] == endurance.ABORT_SLOWDOWN_FLAGS
    assert legacy["power_limit_drop_action"] == "immediate_safety_abort"

    with pytest.raises(ValueError, match="power_limit_drop_tolerance_w"):
        normalize_power_safety_policy(
            {"power_safety": {"power_limit_drop_tolerance_w": 0}}
        )
    with pytest.raises(ValueError, match="slowdown_consecutive_samples"):
        normalize_power_safety_policy(
            {"power_safety": {"slowdown_consecutive_samples": 0}}
        )
    with pytest.raises(ValueError, match="positive integer"):
        normalize_power_safety_policy(
            {"power_safety": {"slowdown_consecutive_samples": 1.5}}
        )

    baseline = {"name": "campaign", "power_safety": defaults}
    stricter = {
        "name": "campaign",
        "power_safety": {
            **defaults,
            "power_limit_drop_tolerance_w": 4.0,
        },
    }
    assert campaign_config_fingerprint(baseline, []) != campaign_config_fingerprint(
        stricter, []
    )


def test_endurance_monitor_receives_fingerprinted_power_policy(tmp_path, monkeypatch):
    captured = {}

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr("validation.endurance.supervisor.GpuMonitor", FakeMonitor)
    campaign = {
        "max_temperature_c": 86.0,
        "power_safety": normalize_power_safety_policy(
            {
                "power_safety": {
                    "power_limit_drop_tolerance_w": 5.0,
                    "slowdown_consecutive_samples": 2,
                }
            }
        ),
    }
    event_path = tmp_path / "gpu-safety-event.json"
    platform_path = tmp_path / "platform-thermal.csv"
    platform_manifest = {
        "available": True,
        "best_effort": True,
        "columns": ["timestamp", "thermal_tvga_c"],
        "sources": [],
    }
    expected_gpu_identity = _verified_floor_binding()["source_runtime_identity"][
        "gpu_identity"
    ]
    create_gpu_monitor(
        campaign,
        path=tmp_path / "gpu.csv",
        event_path=event_path,
        gpu_index=0,
        platform_thermal_path=platform_path,
        platform_thermal_manifest=platform_manifest,
        expected_gpu_identity=expected_gpu_identity,
    )

    assert captured["args"] == (
        tmp_path / "gpu.csv",
        event_path,
        0,
        86.0,
        5.0,
        2,
    )
    assert captured["kwargs"] == {
        "platform_thermal_path": platform_path,
        "platform_thermal_manifest": platform_manifest,
        "operating_policy_mode": "workstation_managed",
        "expected_gpu_identity": expected_gpu_identity,
    }


def test_legacy_endurance_gpu_safety_abort_is_a_non_retriable_health_gate():
    gate = gpu_safety_health_gate(
        SimpleNamespace(
            safety_reason_code="power_limit_below_default",
            safety_reason="GPU current power limit is 55 W below default",
        ),
        normalize_power_safety_policy(
            {"power_safety": {"operating_policy_mode": "legacy_strict"}}
        ),
    )

    assert gate == {
        "name": "gpu_safety_abort",
        "detail": (
            "code=power_limit_below_default; "
            "GPU current power limit is 55 W below default"
        ),
        "retriable": False,
    }


def test_workstation_diagnostic_counters_do_not_gate_endurance():
    monitor = SimpleNamespace(
        safety_reason=None,
        safety_reason_code=None,
        diagnostic_event_counts={
            "temperature_threshold": 11,
            "power_limit_below_default": 4,
            "sustained_clock_slowdown": 2,
        },
        temperature_threshold_samples=11,
        power_limit_drop_samples=4,
        slowdown_active_samples=9,
    )
    policy = normalize_power_safety_policy({})

    assert gpu_safety_health_gate(monitor, policy) is None


def test_legacy_strict_diagnostic_counters_remain_a_health_gate():
    monitor = SimpleNamespace(
        safety_reason=None,
        safety_reason_code=None,
        diagnostic_event_counts={"temperature_threshold": 1},
        temperature_threshold_samples=1,
        power_limit_drop_samples=0,
        slowdown_active_samples=0,
    )
    policy = normalize_power_safety_policy(
        {"power_safety": {"operating_policy_mode": "legacy_strict"}}
    )

    gate = gpu_safety_health_gate(monitor, policy)
    assert gate is not None
    assert gate["name"] == "gpu_safety_abort"
    assert "legacy_strict" in gate["detail"]


def test_preflight_required_telemetry_abort_pauses_campaign_without_starting_gpu(
    tmp_path, monkeypatch
):
    _mock_verified_floor(monkeypatch)
    captured = {}

    def fake_preflight(**kwargs):
        captured.update(kwargs)
        return {
            "status": "safety_abort",
            "safety_events": [
                {
                    "code": "power_limit_telemetry_unavailable",
                    "reason": "Required current/default power telemetry is unavailable",
                }
            ],
            "xid": {"available": True},
            "power_profile": {"available": True, "value": "performance"},
        }

    monkeypatch.setattr("validation.endurance.supervisor.preflight", fake_preflight)
    monkeypatch.setattr(
        "validation.gpu_reentry_evidence.require_reentry_evidence",
        lambda *args, **kwargs: {
            "report_sha256": "a" * 64,
            "verification": {"status": "ready_for_operator_review"},
        },
    )
    monkeypatch.setattr(
        "validation.endurance.supervisor.read_kernel_oom_log",
        lambda: {"available": True, "lines": [], "count": 0},
    )
    monkeypatch.setattr(
        "validation.endurance.supervisor.project_relative", lambda path: str(path)
    )

    assert (
        main(
            [
                "--output",
                str(tmp_path),
                "--acknowledge-seven-day-run",
            ]
        )
        == 2
    )
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    preflight = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert checkpoint["state"] == "paused_health_gate"
    assert "power_limit_telemetry_unavailable" in checkpoint[
        "campaign_health_gates"
    ][-1]["detail"]
    assert preflight["safety_events"][0]["code"] == (
        "power_limit_telemetry_unavailable"
    )
    assert captured["power_limit_drop_tolerance_w"] == 5.0
    assert captured["slowdown_consecutive_samples"] == 2


def test_reentry_evidence_blocks_production_before_gpu_preflight(tmp_path, monkeypatch):
    from validation.gpu_reentry_evidence import ReentryEvidenceError

    _mock_verified_floor(monkeypatch)

    monkeypatch.setattr(
        "validation.gpu_reentry_evidence.require_reentry_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ReentryEvidenceError("operator physical evidence is incomplete")
        ),
    )
    monkeypatch.setattr(
        "validation.endurance.supervisor.preflight",
        lambda **kwargs: pytest.fail("GPU preflight must not run before re-entry passes"),
    )
    monkeypatch.setattr(
        "validation.endurance.supervisor.project_relative", lambda path: str(path)
    )

    assert (
        main(
            [
                "--output",
                str(tmp_path),
                "--acknowledge-seven-day-run",
            ]
        )
        == 2
    )
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    preflight = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert checkpoint["state"] == "paused_health_gate"
    assert "ReentryEvidenceError" in checkpoint["campaign_health_gates"][-1]["detail"]
    assert preflight["status"] == "reentry_blocked"
    assert preflight["gpu_process_started"] is False


def test_endurance_config_uses_twelve_distinct_cameras(tmp_path):
    campaign = load_campaign()
    text = render_deepstream_config(campaign, 640, tmp_path / "deepstream.txt")

    assert text == endurance.render_deepstream_config_text(campaign, 640)
    assert len(re.findall(r"^\[source\d+\]$", text, re.MULTILINE)) == 12
    assert len(re.findall(r"^\[sink\d+\]$", text, re.MULTILINE)) == 12
    assert len(set(re.findall(r"^uri=(.+)$", text, re.MULTILINE))) == 12
    assert len(re.findall(r"^type=2$", text, re.MULTILINE)) == 12
    assert "type=3" not in text
    assert re.findall(r"^camera-id=(\d+)$", text, re.MULTILINE) == [
        str(value) for value in range(12)
    ]
    assert re.findall(r"^source-id=(\d+)$", text, re.MULTILINE) == [
        str(value) for value in range(12)
    ]
    assert "num-sources=" not in text
    assert "batch-size=12" in text
    assert "config-file=/models/person/640/config_infer_primary.txt" in text
    assert "[tests]\nfile-loop=1" in text


def test_production_command_contract_binds_resolved_runtime_and_config(
    tmp_path, monkeypatch
):
    campaign = load_campaign()
    requested_image = "nvcr.io/nvidia/deepstream:9.0"
    resolved_image_id = "sha256:" + "1" * 64
    gpu_identity = {
        "index": "0",
        "uuid": "GPU-test",
        "name": "GPU Test",
        "driver_version": "590.0",
        "memory.total": "16384",
        "pci.bus_id": "00000000:01:00.0",
    }
    campaign["execution_request"] = {
        "image": requested_image,
        "gpu_index": 0,
    }
    campaign["throughput_floor"] = _verified_floor_binding(
        image=requested_image,
        image_id=resolved_image_id,
        gpu_identity=gpu_identity,
    )
    monkeypatch.setattr(
        endurance,
        "verify_live_throughput_floor",
        lambda value: value["throughput_floor"],
    )
    config_path = tmp_path / "deepstream.txt"
    rendered = render_deepstream_config(campaign, 640, config_path)
    segment = build_plan(campaign)[0]
    captured = {}

    def fake_command(**kwargs):
        captured.update(kwargs)
        return ["docker", "run", kwargs["image"]]

    monkeypatch.setattr(endurance, "production_docker_command", fake_command)
    monkeypatch.setattr(
        endurance,
        "verify_session_binding",
        lambda binding, value: {
            "runtime_identity": {
                "resolved_image_id": resolved_image_id,
                "gpu_index": 0,
                "gpu_identity": gpu_identity,
            }
        },
    )
    contract = endurance.production_command_contract(
        campaign=campaign,
        segment=segment,
        attempt=1,
        output=tmp_path,
        requested_image=requested_image,
        resolved_image_id=resolved_image_id,
        gpu_index=0,
        gpu_identity=gpu_identity,
        session_binding={"session_id": "session-test", "receipt": {}},
        config_path=config_path,
    )

    assert captured["image"] == resolved_image_id
    assert contract["rendered_config_sha256"] == endurance.hashlib.sha256(
        rendered.encode("utf-8")
    ).hexdigest()
    assert contract["container_name"] == "deepsafe-endurance-000-1"
    assert contract["container_labels"][
        "io.deepsafe.endurance.gpu-uuid"
    ] == "GPU-test"
    assert contract["container_labels"][
        "io.deepsafe.endurance.rendered-config-sha256"
    ] == contract["rendered_config_sha256"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("execution_request", "image"), "deepstream:other"),
        (("execution_request", "gpu_index"), 1),
        (("max_temperature_c",), 85.0),
        (("power_safety", "power_limit_drop_tolerance_w"), 4.0),
        (("power_safety", "slowdown_consecutive_samples"), 3),
        (("power_safety", "operating_policy_mode"), "legacy_strict"),
        (("power_safety", "static_signal_action"), "safety_abort"),
        (("power_safety", "power_limit_fields"), ["wrong"]),
        (("power_safety", "diagnostic_slowdown_flags"), ["wrong"]),
        (("power_safety", "abort_slowdown_flags"), ["wrong"]),
    ],
)
def test_floor_runtime_request_rejects_every_campaign_identity_mismatch(
    path, replacement
):
    campaign = load_campaign()
    campaign["execution_request"] = {
        "image": "deepsafe-deepstream:9.0",
        "gpu_index": 0,
    }
    campaign["throughput_floor"] = _verified_floor_binding()
    assert endurance.validate_throughput_floor_runtime_request(campaign) == (
        campaign["throughput_floor"]["source_runtime_identity"]
    )

    changed = deepcopy(campaign)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(RuntimeError, match="differs"):
        endurance.validate_throughput_floor_runtime_request(changed)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("image",), "deepstream:other"),
        (("image_id",), "sha256:other"),
        (("gpu_index",), 1),
        (("gpu_identity", "uuid"), "GPU-other"),
        (("gpu_identity", "driver_version"), "other-driver"),
        (("power_profile", "value"), "balanced"),
        (("max_temperature_c",), 85.0),
        (("power_safety_policy", "preflight_samples"), 3),
        (("power_safety_policy", "preflight_sample_interval_seconds"), 2.0),
        (("power_safety_policy", "hardware_protection_owner"), "other"),
        (("power_safety_policy", "required_telemetry_failure_action"), "ignore"),
    ],
)
def test_floor_preflight_requires_exact_image_gpu_power_identity(path, replacement):
    campaign = load_campaign()
    campaign["execution_request"] = {
        "image": "deepsafe-deepstream:9.0",
        "gpu_index": 0,
    }
    campaign["throughput_floor"] = _verified_floor_binding()
    source = campaign["throughput_floor"]["source_runtime_identity"]
    report = {"status": "ok", **deepcopy(source)}
    assert endurance.validate_throughput_floor_preflight_runtime(
        campaign, report
    ) == source

    changed = deepcopy(report)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(RuntimeError):
        endurance.validate_throughput_floor_preflight_runtime(campaign, changed)


def test_legacy_strict_campaign_and_floor_policy_can_cross_bind_exactly():
    campaign = load_campaign()
    campaign["execution_request"] = {
        "image": "deepsafe-deepstream:9.0",
        "gpu_index": 0,
    }
    campaign["power_safety"] = normalize_power_safety_policy(
        {"power_safety": {"operating_policy_mode": "legacy_strict"}}
    )
    binding = _verified_floor_binding()
    binding["source_runtime_identity"]["power_safety_policy"] = {
        key: deepcopy(campaign["power_safety"][key])
        for key in endurance.SCENE_POWER_SAFETY_POLICY_FIELDS
    }
    campaign["throughput_floor"] = binding

    assert endurance.validate_throughput_floor_runtime_request(campaign) == (
        binding["source_runtime_identity"]
    )


def test_floor_live_verification_failure_prevents_docker_command_creation(
    tmp_path, monkeypatch
):
    campaign = load_campaign()
    campaign["execution_request"] = {
        "image": "deepsafe-deepstream:9.0",
        "gpu_index": 0,
    }
    campaign["throughput_floor"] = _verified_floor_binding()
    config_path = tmp_path / "deepstream.txt"
    render_deepstream_config(campaign, 640, config_path)
    command_calls = []

    def reject_changed_floor(_campaign):
        raise RuntimeError("frozen matrix changed")

    monkeypatch.setattr(
        endurance, "verify_live_throughput_floor", reject_changed_floor
    )
    monkeypatch.setattr(
        endurance,
        "production_docker_command",
        lambda **kwargs: command_calls.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="frozen matrix changed"):
        endurance.production_command_contract(
            campaign=campaign,
            segment=build_plan(campaign)[0],
            attempt=1,
            output=tmp_path,
            requested_image="deepsafe-deepstream:9.0",
            resolved_image_id="sha256:test",
            gpu_index=0,
            gpu_identity=campaign["throughput_floor"]["source_runtime_identity"][
                "gpu_identity"
            ],
            session_binding={"session_id": "unreachable", "receipt": {}},
            config_path=config_path,
        )

    assert command_calls == []


def test_frozen_p05_floor_failure_is_non_retriable():
    campaign = {"throughput_floor": _verified_floor_binding()}

    failed, gate = endurance.throughput_floor_health_evaluation(
        {"aggregate_current_fps": {"p05": 99.9}}, campaign, 640
    )
    assert failed["status"] == "failed_below_floor"
    assert gate["name"] == "throughput_floor"
    assert gate["retriable"] is False

    passed, gate = endurance.throughput_floor_health_evaluation(
        {"aggregate_current_fps": {"p05": 100.0}}, campaign, 640
    )
    assert passed["status"] == "passed"
    assert gate is None


def test_latency_output_is_aggregated_and_raw_lines_never_reach_log(tmp_path):
    pairs = " ".join(f"{20 + index}.0 ({19 + index}.0)" for index in range(12))
    source = io.StringIO(
        "\n"
        "************BATCH-NUM = 1**************\n"
        "Source id = 0 Frame_num = 10 Frame latency = 12.500000 (ms)\n"
        "Source id = 1 Frame_num = 10 Frame latency = 17.500000 (ms)\n"
        f"**PERF: {pairs}\n"
        "(deepstream-app:76): GStreamer-WARNING **: 18:00:00.000: "
        "../gst/gstpad.c:4463:gst_pad_chain_data_unchecked:"
        "<nvv4l2decoder11:sink> Got data flow before segment event\n"
        "ordinary diagnostic\n"
    )
    collector = FilteredLogCollector(
        source,
        log_path=tmp_path / "deepstream.log",
        perf_path=tmp_path / "perf.csv",
        latency_path=tmp_path / "latency.csv",
        start_monotonic=time.monotonic(),
        expected_streams=12,
        latency_bucket_seconds=60,
        max_log_bytes=1024,
    )
    collector.run()

    log = (tmp_path / "deepstream.log").read_text(encoding="utf-8")
    assert "Frame latency" not in log
    assert "BATCH-NUM" not in log
    assert "Got data flow before segment event" not in log
    assert "ordinary diagnostic" in log
    assert collector.latency_lines_suppressed == 2
    assert collector.loop_segment_warnings_suppressed == 1
    assert collector.blank_lines_suppressed == 1
    assert collector.perf_rows[0]["aggregate_fps"] == sum(20 + index for index in range(12))
    latency = (tmp_path / "latency.csv").read_text(encoding="utf-8")
    assert ",-1,2,15.0,15.0,17.25,17.5" in latency


def test_perf_collector_requires_all_twelve_streams_to_be_positive(tmp_path):
    current = [0.0, *(20.0 + index for index in range(11))]
    pairs = " ".join(f"{value} ({value})" for value in current)
    collector = FilteredLogCollector(
        io.StringIO(f"**PERF: {pairs}\n"),
        log_path=tmp_path / "deepstream.log",
        perf_path=tmp_path / "perf.csv",
        latency_path=tmp_path / "latency.csv",
        start_monotonic=time.monotonic(),
        expected_streams=12,
        latency_bucket_seconds=60,
        max_log_bytes=1024,
    )

    collector.run()

    assert collector.perf_rows == []
    assert collector.last_perf_monotonic is None
    assert collector.perf_rows_rejected_nonpositive == 1


def _perf_window_fixture(
    tmp_path: Path,
    elapsed_values: list[float],
    *,
    duration_seconds: int = 21600,
    malformed_log_rows: int = 0,
) -> tuple[dict, Path]:
    header = "**PERF:  " + "\t".join(
        f"FPS {stream_id} (Avg)" for stream_id in range(12)
    )
    valid_line = "**PERF:  " + "\t".join(
        "50.00 (50.00)" for _ in range(12)
    )
    # Concurrent stdout writes can tear the 12-stream table at any stream
    # boundary; exercise 0..11 recovered pairs rather than one fixed shape.
    malformed = [
        "**PERF:  "
        + "\t".join(
            "50.00 (50.00)" for _ in range(malformed_index % 12)
        )
        for malformed_index in range(malformed_log_rows)
    ]
    log_rows = [header]
    malformed_iter = iter(malformed)
    malformed_stride = max(1, len(elapsed_values) // max(1, malformed_log_rows))
    for row_index, _elapsed in enumerate(elapsed_values, start=1):
        log_rows.append(valid_line)
        if malformed_log_rows and row_index % malformed_stride == 0:
            torn = next(malformed_iter, None)
            if torn is not None:
                log_rows.append(torn)
    log_rows.extend(malformed_iter)
    log_path = tmp_path / "deepstream.log"
    log_path.write_text(
        "\n".join(log_rows) + "\n",
        encoding="utf-8",
    )
    perf_path = tmp_path / "perf.csv"
    with perf_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["elapsed_seconds", "aggregate_fps", "per_stream_mean_fps"]
        )
        for elapsed in elapsed_values:
            writer.writerow([f"{elapsed:.3f}", "600.0", "50.0"])
    return endurance.parse_perf(log_path, 12, duration_seconds, 5), perf_path


def _apply_perf_window(
    parsed: dict,
    perf_path: Path,
    *,
    duration_seconds: int = 21600,
) -> dict:
    return apply_perf_window_contract(
        parsed,
        perf_path,
        expected_streams=12,
        duration_seconds=duration_seconds,
        perf_interval_seconds=5,
        minimum_coverage_fraction=0.95,
        startup_grace_seconds=120,
        perf_stall_timeout_seconds=180,
    )


def test_perf_window_accepts_actual_attempt_01_coverage_and_torn_stdout_pattern(
    tmp_path,
):
    # attempt-01 had 4,254 complete rows out of 4,320 requested and 52
    # structurally torn PERF lines.  Distribute the 66 missing ticks across the
    # full six hours, including two adjacent missing ticks (a 15 second gap).
    removed_ticks = {65 * value for value in range(1, 65)} | {1000, 1001}
    elapsed = [
        tick * 5 + 1.095 + (0.002 if tick == 1002 else 0.0)
        for tick in range(1, 4321)
        if tick not in removed_ticks
    ]
    elapsed[-1] = 21608.523
    parsed, perf_path = _perf_window_fixture(
        tmp_path,
        elapsed,
        malformed_log_rows=52,
    )

    result = _apply_perf_window(parsed, perf_path)

    assert len(elapsed) == 4254
    assert parsed["status"] == "insufficient_perf_window"
    assert result["status"] == "ok"
    assert result["raw_parser_status"] == "insufficient_perf_window"
    contract = result["window_contract"]
    assert contract["status"] == "ok"
    assert contract["valid_complete_rows"] == 4254
    assert contract["malformed_log_rows"] == 52
    assert contract["row_coverage_fraction"] == 0.984722
    assert contract["temporal_slot_coverage_fraction"] >= 0.95
    assert contract["first_elapsed_seconds"] == 6.095
    assert contract["last_elapsed_seconds"] == 21608.523
    assert contract["maximum_observed_gap_seconds"] == 15.002
    assert contract["minimum_required_coverage_fraction"] == 0.95
    assert contract["failed_checks"] == []
    assert all(contract["checks"].values())


def test_perf_window_rejects_sparse_samples_despite_full_elapsed_span(tmp_path):
    # One fewer than the 95% quota, spread across the entire interval so the
    # endpoint and gap checks cannot mask insufficient measurement coverage.
    elapsed = [
        5 + index * (21595 / 4102)
        for index in range(4103)
    ]
    parsed, perf_path = _perf_window_fixture(tmp_path, elapsed)

    result = _apply_perf_window(parsed, perf_path)

    assert result["status"] == "invalid_perf_window"
    contract = result["window_contract"]
    assert contract["row_coverage_fraction"] == 0.949769
    assert "row_coverage" in contract["failed_checks"]


def test_perf_window_rejects_clustered_rows_even_when_row_quota_and_endpoints_pass(
    tmp_path,
):
    # 4,103 rows are tightly clustered near startup and a final row supplies a
    # superficially valid tail endpoint.  Row count alone is exactly 95%.
    clustered = [1 + index * (99 / 4102) for index in range(4103)]
    elapsed = [*clustered, 21600.0]
    parsed, perf_path = _perf_window_fixture(tmp_path, elapsed)

    result = _apply_perf_window(parsed, perf_path)

    assert result["window_contract"]["row_coverage_fraction"] == 0.95
    assert result["window_contract"]["checks"]["startup_endpoint"] is True
    assert result["window_contract"]["checks"]["tail_endpoint"] is True
    assert result["status"] == "invalid_perf_window"
    assert {
        "temporal_slot_coverage",
        "maximum_gap",
    }.issubset(result["window_contract"]["failed_checks"])


def test_perf_window_rejects_missing_tail_at_exact_row_coverage_floor(tmp_path):
    elapsed = [tick * 5.0 for tick in range(1, 4105)]
    parsed, perf_path = _perf_window_fixture(tmp_path, elapsed)

    result = _apply_perf_window(parsed, perf_path)

    assert result["window_contract"]["row_coverage_fraction"] == 0.95
    assert result["window_contract"]["checks"]["maximum_gap"] is True
    assert result["status"] == "invalid_perf_window"
    assert {"tail_endpoint", "elapsed_span"}.issubset(
        result["window_contract"]["failed_checks"]
    )


def test_perf_window_rejects_log_csv_count_or_aggregate_crossbind(tmp_path):
    elapsed = [tick * 5.0 for tick in range(1, 4321)]
    parsed, perf_path = _perf_window_fixture(tmp_path, elapsed)
    rows = perf_path.read_text(encoding="utf-8").splitlines()
    # Keep the CSV row internally consistent while changing its aggregate away
    # from the raw DeepStream PERF line.
    rows[100] = rows[100].split(",", 1)[0] + ",601.0,50.083333"
    perf_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    aggregate_mismatch = _apply_perf_window(parsed, perf_path)
    assert aggregate_mismatch["status"] == "invalid_perf_window"
    assert "log_csv_aggregate_crosscheck" in aggregate_mismatch[
        "window_contract"
    ]["failed_checks"]

    perf_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    count_mismatch = _apply_perf_window(parsed, perf_path)
    assert count_mismatch["status"] == "invalid_perf_window"
    assert "log_csv_row_count_crosscheck" in count_mismatch[
        "window_contract"
    ]["failed_checks"]


def test_perf_window_does_not_relax_non_window_parser_failures(tmp_path):
    elapsed = [tick * 5.0 for tick in range(1, 4321)]
    parsed, perf_path = _perf_window_fixture(tmp_path, elapsed)
    parsed["status"] = "inactive_streams"
    parsed["inactive_stream_ids"] = [11]
    parsed["active_streams"] = 11

    result = _apply_perf_window(parsed, perf_path)

    assert result["status"] == "invalid_perf_window"
    assert result["raw_parser_status"] == "inactive_streams"
    assert {
        "raw_parser_status_eligible",
        "parser_stream_contract",
    }.issubset(result["window_contract"]["failed_checks"])


@pytest.mark.parametrize(
    "malformed_row",
    [
        "NaN,600.0,50.0",
        "10.000,Infinity,Infinity",
        "10.000,600.0",
        "10.000,601.0,50.0",
    ],
)
def test_perf_window_malformed_csv_fails_closed_without_exception(
    tmp_path,
    malformed_row,
):
    elapsed = [tick * 5.0 for tick in range(1, 4321)]
    parsed, perf_path = _perf_window_fixture(tmp_path, elapsed)
    rows = perf_path.read_text(encoding="utf-8").splitlines()
    rows[2] = malformed_row
    perf_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = _apply_perf_window(parsed, perf_path)

    assert result["status"] == "invalid_perf_window"
    assert result["window_contract"]["checks"]["csv_well_formed"] is False
    assert result["window_contract"]["csv_errors_first_20"]


def test_latency_contract_requires_per_source_coverage_for_all_twelve_streams(tmp_path):
    path = tmp_path / "latency.csv"
    buckets = LatencyBuckets(path, 60)
    buckets.add(0.0, 0, 10.0)
    buckets.add(61.0, 0, 11.0)
    buckets.close()

    summary = summarize_latency(
        path,
        60,
        expected_streams=12,
        duration_seconds=120,
    )

    assert summary["status"] == "missing_sources"
    assert summary["observed_source_ids"] == [0]
    assert summary["missing_source_ids"] == list(range(1, 12))
    assert summary["per_source"]["0"]["coverage_fraction"] == 1.0
    assert summary["per_source"]["1"]["coverage_fraction"] == 0.0
    gates = parsed_telemetry_contract_gates(
        {"status": "ok"},
        {
            "status": "ok",
            "gpu_names": ["GPU Test"],
            "metrics": {
                "gpu_utilization_percent": {"max": 1.0},
                "memory_used_mib": {"max": 1.0},
            },
        },
        summary,
    )
    assert [gate["name"] for gate in gates] == ["latency_source_contract"]


def test_parsed_perf_and_gpu_contract_failures_are_hard_segment_gates():
    gates = parsed_telemetry_contract_gates(
        {"status": "inactive_streams"},
        {
            "status": "incomplete_gpu_metrics",
            "gpu_names": ["GPU Test"],
            "metrics": {
                "gpu_utilization_percent": {"max": 1.0},
                "memory_used_mib": {"max": 1.0},
            },
        },
        {
            "status": "ok",
            "expected_streams": 12,
            "expected_source_ids": list(range(12)),
            "missing_source_ids": [],
            "unexpected_source_ids": [],
            "per_source": {str(source_id): {} for source_id in range(12)},
        },
    )

    assert [gate["name"] for gate in gates] == [
        "perf_contract",
        "gpu_metrics_contract",
    ]
    assert all(gate["retriable"] for gate in gates)


def test_zero_utilization_or_wrong_gpu_name_cannot_prove_active_load():
    latency = {
        "status": "ok",
        "expected_streams": 12,
        "expected_source_ids": list(range(12)),
        "missing_source_ids": [],
        "unexpected_source_ids": [],
        "per_source": {str(source_id): {} for source_id in range(12)},
    }
    gates = parsed_telemetry_contract_gates(
        {"status": "ok"},
        {
            "status": "ok",
            "gpu_names": ["Wrong GPU"],
            "metrics": {
                "gpu_utilization_percent": {"max": 0.0},
                "memory_used_mib": {"max": 2048.0},
            },
        },
        latency,
        expected_gpu_name="GPU Test",
    )

    assert [gate["name"] for gate in gates] == ["gpu_active_load_contract"]


def test_trend_reports_growth_per_hour_after_warmup():
    result = trend([0, 1, 2, 3, 4, 5, 6, 7], 1800)
    assert result["status"] == "ok"
    assert result["growth_per_hour"] > 0


def test_rss_sampler_includes_the_process_tree():
    rss_mib, process_count = process_tree_rss_mib(os.getpid())
    assert rss_mib > 0
    assert process_count >= 1


def _reconcile_fixture(tmp_path, monkeypatch, status=None):
    output = tmp_path / "campaign"
    attempt_dir = output / "segments/segment-000-640/attempt-01"
    attempt_dir.mkdir(parents=True)
    if status is None:
        status = {"status": "running", "health_gates": []}
    (attempt_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    session_binding = {
        "session_id": "session-test",
        "receipt": {"path": "session.json", "size_bytes": 1, "sha256": "a" * 64},
    }
    throughput_floor = _verified_floor_binding()
    checkpoint = {
        "dry_run": False,
        "config_fingerprint": "c" * 64,
        "static_input_fingerprint": "s" * 64,
        "throughput_floor": throughput_floor,
        "state": "running",
        "validated_seconds": 0,
        "unexpected_restarts": 0,
        "orphan_recoveries": 0,
        "campaign_health_gates": [],
        "sessions": [session_binding],
        "active": {
            "segment_id": "segment-000-640",
            "profile": 640,
            "attempt": 1,
            "attempt_dir": str(attempt_dir),
            "container_name": "deepsafe-endurance-000-1",
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "session": session_binding,
            "throughput_floor_fingerprint": throughput_floor[
                "artifact_fingerprint"
            ],
        },
        "segments": [
            {
                "index": 0,
                "segment_id": "segment-000-640",
                "profile": 640,
                "duration_seconds": 10,
                "campaign_day": 1,
                "status": "pending",
                "validated_seconds": 0,
                "attempts": [],
                "attempt_receipts": [],
            }
        ],
    }
    monkeypatch.setattr(endurance, "project_relative", lambda path: str(path))
    return output, attempt_dir, checkpoint, session_binding


def _ok_input_sweep():
    return {
        "schema_version": "deepsafe.endurance-input-pin-sweep/v1",
        "status": "ok",
        "checked_at_utc": "test",
        "input_pins_sha256": "a" * 64,
        "live_pins_fingerprint": "b" * 64,
        "actual": {},
        "mismatches": [],
    }


def test_reconcile_records_an_uncommitted_attempt_for_resume(tmp_path, monkeypatch):
    output, _attempt_dir, checkpoint, _session = _reconcile_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        "validation.endurance.supervisor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    reconcile_active(checkpoint, output)

    assert checkpoint["active"] is None
    assert checkpoint["state"] == "paused"
    assert checkpoint["unexpected_restarts"] == 1
    assert checkpoint["segments"][0]["status"] == "failed"
    assert checkpoint["segments"][0]["attempts"][0]["health_gates"][-1]["name"] == "supervisor_orphan_recovery"
    assert len(checkpoint["segments"][0]["attempt_receipts"]) == 1
    assert checkpoint["segments"][0]["attempts"][0]["attempt_finalization"][
        "method"
    ] == "orphan_recovery_fail_closed"


def test_reconcile_fails_closed_when_orphan_container_cannot_be_removed(
    tmp_path, monkeypatch
):
    output, _attempt_dir, checkpoint, session = _reconcile_fixture(
        tmp_path, monkeypatch
    )
    labels = endurance.container_binding_labels(
        config_fingerprint=checkpoint["config_fingerprint"],
        session_binding=session,
        segment=checkpoint["segments"][0],
        attempt=1,
        output=output,
        throughput_floor_fingerprint=checkpoint["throughput_floor"][
            "artifact_fingerprint"
        ],
    )
    container_id = "a" * 64
    inspect = json.dumps(
        [{"Id": container_id, "Name": "/deepsafe-endurance-000-1", "Config": {"Labels": labels}}]
    )
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout=f"{container_id}\tdeepsafe-endurance-000-1\n", stderr=""),
            SimpleNamespace(returncode=0, stdout=inspect, stderr=""),
            SimpleNamespace(returncode=1, stdout="", stderr="remove failed"),
        ]
    )
    monkeypatch.setattr(
        "validation.endurance.supervisor.subprocess.run",
        lambda *args, **kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match="Cannot remove bound endurance"):
        reconcile_active(checkpoint, output)

    assert checkpoint["active"] is not None
    assert checkpoint["orphan_recoveries"] == 0
    assert checkpoint["state"] == "paused_health_gate"
    assert checkpoint["campaign_health_gates"][-1]["name"] == "orphan_recovery"


def test_reconcile_fails_closed_when_orphan_removal_cannot_be_verified(
    tmp_path, monkeypatch
):
    output, _attempt_dir, checkpoint, session = _reconcile_fixture(
        tmp_path, monkeypatch
    )
    labels = endurance.container_binding_labels(
        config_fingerprint=checkpoint["config_fingerprint"],
        session_binding=session,
        segment=checkpoint["segments"][0],
        attempt=1,
        output=output,
        throughput_floor_fingerprint=checkpoint["throughput_floor"][
            "artifact_fingerprint"
        ],
    )
    container_id = "b" * 64
    present = SimpleNamespace(
        returncode=0,
        stdout=f"{container_id}\tdeepsafe-endurance-000-1\n",
        stderr="",
    )
    responses = iter(
        [
            present,
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [{"Id": container_id, "Name": "/deepsafe-endurance-000-1", "Config": {"Labels": labels}}]
                ),
                stderr="",
            ),
            SimpleNamespace(returncode=0, stdout=container_id, stderr=""),
            present,
        ]
    )
    monkeypatch.setattr(
        "validation.endurance.supervisor.subprocess.run",
        lambda *args, **kwargs: next(responses),
    )
    with pytest.raises(RuntimeError, match="removal was not verified"):
        reconcile_active(checkpoint, output)

    assert checkpoint["active"] is not None
    assert checkpoint["orphan_recoveries"] == 0
    assert checkpoint["state"] == "paused_health_gate"
    assert checkpoint["campaign_health_gates"][-1]["name"] == "orphan_recovery"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("segment_id", "segment-999-640"),
        ("profile", 960),
        ("attempt", 2),
        ("attempt_dir", "/tmp/not-the-canonical-attempt"),
        ("container_name", "attacker-selected-container"),
        ("session", {"session_id": "unknown", "receipt": {}}),
    ],
)
def test_reconcile_rejects_active_binding_mismatch_before_any_docker_command(
    tmp_path, monkeypatch, field, bad_value
):
    output, _attempt_dir, checkpoint, _session = _reconcile_fixture(
        tmp_path, monkeypatch
    )
    checkpoint["active"][field] = bad_value
    active_before = deepcopy(checkpoint["active"])
    docker_calls = []

    def forbidden_docker(*args, **kwargs):
        docker_calls.append((args, kwargs))
        raise AssertionError("Docker must not run for an untrusted active binding")

    monkeypatch.setattr(endurance.subprocess, "run", forbidden_docker)

    with pytest.raises(RuntimeError, match="Active checkpoint binding rejected"):
        reconcile_active(checkpoint, output)

    assert docker_calls == []
    assert checkpoint["active"] == active_before
    assert checkpoint["state"] == "paused_health_gate"
    assert checkpoint["campaign_health_gates"][-1]["name"] == "active_binding_integrity"
    persisted = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert persisted["active"] == active_before


def test_reconcile_docker_ps_rc1_is_not_container_absence(tmp_path, monkeypatch):
    output, _attempt_dir, checkpoint, _session = _reconcile_fixture(
        tmp_path, monkeypatch
    )
    active_before = deepcopy(checkpoint["active"])
    calls = []

    def daemon_failure(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon",
        )

    monkeypatch.setattr(endurance.subprocess, "run", daemon_failure)

    with pytest.raises(RuntimeError, match="docker ps returncode=1"):
        reconcile_active(checkpoint, output)

    assert len(calls) == 1
    assert calls[0][:3] == ["docker", "ps", "-a"]
    assert checkpoint["active"] == active_before
    assert checkpoint["segments"][0]["attempts"] == []
    assert checkpoint["state"] == "paused_health_gate"
    assert "Cannot connect" in checkpoint["campaign_health_gates"][-1]["detail"]


def test_reconcile_never_removes_exact_name_container_with_foreign_labels(
    tmp_path, monkeypatch
):
    output, _attempt_dir, checkpoint, _session = _reconcile_fixture(
        tmp_path, monkeypatch
    )
    active_before = deepcopy(checkpoint["active"])
    container_id = "c" * 64
    responses = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=f"{container_id}\tdeepsafe-endurance-000-1\n",
                stderr="",
            ),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "Id": container_id,
                            "Name": "/deepsafe-endurance-000-1",
                            "Config": {"Labels": {"owner": "someone-else"}},
                        }
                    ]
                ),
                stderr="",
            ),
        ]
    )
    calls = []

    def fake_docker(command, **kwargs):
        calls.append(command)
        return next(responses)

    monkeypatch.setattr(endurance.subprocess, "run", fake_docker)

    with pytest.raises(RuntimeError, match="labels do not match"):
        reconcile_active(checkpoint, output)

    assert [command[1] for command in calls] == ["ps", "inspect"]
    assert checkpoint["active"] == active_before
    assert checkpoint["orphan_recoveries"] == 0
    assert checkpoint["state"] == "paused_health_gate"


def _runtime_container_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(endurance, "PROJECT_ROOT", tmp_path)
    image_id = "sha256:" + "1" * 64
    container_id = "2" * 64
    container_name = "deepsafe-endurance-000-1"
    labels = {
        "io.deepsafe.endurance.segment-id": "segment-000-640",
        "io.deepsafe.endurance.gpu-uuid": "GPU-test",
    }
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--gpus",
        "device=0",
        image_id,
        "deepstream-app",
        "-c",
        "/workspace/attempt/deepstream.txt",
    ]
    contract = {
        "schema_version": "deepsafe.endurance-production-command/v1",
        "resolved_image_id": image_id,
        "gpu_index": 0,
        "container_name": container_name,
        "container_labels": labels,
        "command": command,
    }
    cmd = ["deepstream-app", "-c", "/workspace/attempt/deepstream.txt"]
    entrypoint = ["/opt/nvidia/nvidia_entrypoint.sh"]
    inspect = {
        "Id": container_id,
        "Name": f"/{container_name}",
        "Image": image_id,
        "Path": entrypoint[0],
        "Args": cmd,
        "Config": {
            "Image": image_id,
            "Labels": {**labels, "org.opencontainers.image.title": "DeepStream"},
            "Cmd": cmd,
            "Entrypoint": entrypoint,
            "WorkingDir": "/workspace",
            "Env": [
                "PATH=/usr/local/bin:/usr/bin",
                "GST_DEBUG=1",
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
                "NVDS_ENABLE_LATENCY_MEASUREMENT=1",
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
            "StartedAt": "2026-07-16T06:00:00.000000000Z",
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
    return contract, inspect, container_id


def _runtime_docker_responses(contract, inspect, container_id):
    return iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=f"{container_id}\t{contract['container_name']}\n",
                stderr="",
            ),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps([inspect]),
                stderr="",
            ),
        ]
    )


def test_runtime_container_attestation_binds_actual_image_argv_gpu_and_ro_mounts(
    tmp_path, monkeypatch
):
    contract, inspect, container_id = _runtime_container_fixture(tmp_path, monkeypatch)
    responses = _runtime_docker_responses(contract, inspect, container_id)
    monkeypatch.setattr(
        endurance,
        "_run_recovery_docker",
        lambda *args, **kwargs: next(responses),
    )

    result = endurance.capture_runtime_container_attestation(contract)

    assert result["schema_version"] == endurance.RUNTIME_CONTAINER_ATTESTATION_SCHEMA
    assert result["status"] == "verified_running"
    assert result["container_id"] == container_id
    assert result["image_id"] == contract["resolved_image_id"]
    assert result["cmd"] == [
        "deepstream-app",
        "-c",
        "/workspace/attempt/deepstream.txt",
    ]
    assert result["gpu_device_request"]["device_ids"] == ["0"]
    assert all(mount["read_write"] is False for mount in result["mounts"])
    assert result["command_contract_sha256"] == endurance._sha256(contract)


def test_runtime_container_attestation_returns_none_before_exact_name_exists(
    tmp_path, monkeypatch
):
    contract, _inspect, _container_id = _runtime_container_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        endurance,
        "_run_recovery_docker",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    assert endurance.capture_runtime_container_attestation(contract) is None


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("image", "image/name identity"),
        ("cmd", "Config.Cmd"),
        ("path", "Path/Args"),
        ("gpu", "GPU device ID"),
        ("mount", "mount contract"),
        ("label", "labels differ"),
    ],
)
def test_runtime_container_attestation_rejects_inspect_drift(
    tmp_path, monkeypatch, mutation, match
):
    contract, inspect, container_id = _runtime_container_fixture(tmp_path, monkeypatch)
    if mutation == "image":
        inspect["Image"] = "sha256:" + "9" * 64
    elif mutation == "cmd":
        inspect["Config"]["Cmd"][-1] = "/workspace/attacker.txt"
    elif mutation == "path":
        inspect["Args"] = ["sh", "-c", "true"]
    elif mutation == "gpu":
        inspect["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["1"]
    elif mutation == "mount":
        inspect["Mounts"][0]["RW"] = True
    else:
        inspect["Config"]["Labels"][
            "io.deepsafe.endurance.segment-id"
        ] = "segment-attacker"
    responses = _runtime_docker_responses(contract, inspect, container_id)
    monkeypatch.setattr(
        endurance,
        "_run_recovery_docker",
        lambda *args, **kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match=match):
        endurance.capture_runtime_container_attestation(contract)


def test_pending_drift_candidate_crash_can_never_reconcile_as_healthy(
    tmp_path, monkeypatch
):
    output, attempt_dir, checkpoint, session = _reconcile_fixture(
        tmp_path, monkeypatch
    )
    candidate_segment = {
        key: checkpoint["segments"][0][key]
        for key in ("index", "segment_id", "profile", "duration_seconds", "campaign_day")
    }
    candidate = {
        "schema_version": endurance.SEGMENT_SCHEMA,
        "status": "healthy",
        "dry_run": False,
        "config_fingerprint": checkpoint["config_fingerprint"],
        "static_input_fingerprint": checkpoint["static_input_fingerprint"],
        "throughput_floor": _floor_evidence(640),
        "session": session,
        "segment": candidate_segment,
        "attempt": 1,
        "attempt_finalization": endurance._pending_attempt_finalization(),
        "validated_seconds": 10,
        "timing": {"synthetic": False},
        "health_gates": [],
        "retriable": False,
        "artifact_pins": {},
        "throughput": {"aggregate_current_fps": {"mean": 10.0}},
        "latency": {"p95_ms_across_buckets": {"mean": 10.0}},
    }
    endurance.atomic_write_json(attempt_dir / "status.json", candidate)
    for index in range(1, 4):
        prior = {
            "throughput": {"aggregate_current_fps": {"mean": 100.0}},
            "latency": {"p95_ms_across_buckets": {"mean": 10.0}},
        }
        checkpoint["segments"].append(
            {
                "index": index,
                "segment_id": f"segment-{index:03d}-640",
                "profile": 640,
                "duration_seconds": 10,
                "campaign_day": 1,
                "status": "healthy",
                "validated_seconds": 10,
                "attempts": [prior],
                "attempt_receipts": [None],
            }
        )
    checkpoint["validated_seconds"] = 30
    campaign = {
        "drift": {
            "baseline_segments_per_profile": 3,
            "max_fps_drop_fraction": 0.2,
            "max_latency_p95_increase_fraction": 0.5,
            "max_latency_p95_increase_ms": 50.0,
        }
    }
    drift_gates = endurance.evaluate_cross_segment_drift(
        campaign, checkpoint, candidate
    )
    assert [gate["name"] for gate in drift_gates] == ["cross_segment_fps_drift"]
    monkeypatch.setattr(
        endurance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    reconcile_active(checkpoint, output)

    recovered = checkpoint["segments"][0]["attempts"][0]
    assert recovered["status"] == "failed"
    assert recovered["validated_seconds"] == 0
    assert recovered["attempt_finalization"]["method"] == "orphan_recovery_fail_closed"
    assert recovered["attempt_finalization"]["recovered_prior_state"]["state"] == (
        "awaiting_cross_segment_drift"
    )
    assert checkpoint["segments"][0]["status"] == "failed"
    assert checkpoint["validated_seconds"] == 30


def test_mid_segment_stop_event_is_interrupted_not_health_failure(tmp_path, monkeypatch):
    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO("")
            self.returncode = None

        def poll(self):
            return self.returncode

    class FakeGpuMonitor:
        ident = None
        safety_reason = None
        safety_reason_code = None
        safety_event = None
        samples = 0
        platform_thermal_samples = 0
        platform_thermal_error_count = 0
        platform_thermal_errors = []
        query_errors = []
        power_limit_drop_samples = 0
        slowdown_active_samples = 0
        max_consecutive_slowdown_samples = 0

        def start(self):
            return None

        def stop(self):
            return None

    class FakeRssMonitor:
        def __init__(self, *args, **kwargs):
            self.ident = None
            self.samples = []
            self.errors = []

        def start(self):
            return None

        def stop(self):
            return None

    campaign = load_campaign()
    gpu_identity = {
        "index": "0",
        "uuid": "GPU-test",
        "name": "Fake GPU",
        "driver_version": "test",
        "memory.total": "1 MiB",
        "pci.bus_id": "00000000:01:00.0",
    }
    campaign["throughput_floor"] = _verified_floor_binding(
        image="test-image",
        image_id="test-image",
        gpu_identity=gpu_identity,
    )
    monkeypatch.setattr(
        endurance,
        "verify_live_throughput_floor",
        lambda value: value["throughput_floor"],
    )
    campaign["execution_request"] = {"image": "test-image", "gpu_index": 0}
    campaign["config_fingerprint"] = campaign_config_fingerprint(campaign, [])
    segment = {
        "index": 0,
        "segment_id": "segment-000-640",
        "profile": 640,
        "duration_seconds": 60,
        "campaign_day": 1,
    }
    stop_event = threading.Event()
    stop_event.set()
    kernel_log = {"available": True, "source": "test", "lines": [], "count": 0}

    monkeypatch.setattr(endurance.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(endurance, "create_gpu_monitor", lambda *args, **kwargs: FakeGpuMonitor())
    monkeypatch.setattr(endurance, "RssMonitor", FakeRssMonitor)
    monkeypatch.setattr(endurance, "production_docker_command", lambda **kwargs: ["fake"])
    monkeypatch.setattr(
        endurance,
        "verify_session_binding",
        lambda binding, value: {
            "runtime_identity": {
                "resolved_image_id": "test-image",
                "gpu_index": 0,
                "gpu_identity": gpu_identity,
            }
        },
    )
    monkeypatch.setattr(endurance, "read_xid_log", lambda: dict(kernel_log))
    monkeypatch.setattr(endurance, "read_kernel_oom_log", lambda: dict(kernel_log))
    monkeypatch.setattr(
        endurance,
        "read_power_profile",
        lambda: {"available": True, "value": "performance"},
    )
    monkeypatch.setattr(endurance, "project_relative", lambda path: str(path))
    monkeypatch.setattr(endurance, "live_input_pin_sweep", lambda campaign: _ok_input_sweep())

    def stop_process(process, *args, **kwargs):
        process.returncode = 130
        return (
            130,
            "sigterm",
            {
                "verified_absent": True,
                "removed": True,
                "container_name": "deepsafe-endurance-000-1",
                "checked_at_utc": "test",
            },
        )

    monkeypatch.setattr(endurance, "terminate_production_container", stop_process)

    result = endurance.run_production_segment(
        campaign=campaign,
        segment=segment,
        attempt=1,
        attempt_dir=tmp_path / "attempt-01",
        campaign_root=tmp_path,
        image="test-image",
        gpu_index=0,
        kill_grace=1,
        stop_event=stop_event,
        session_binding={"session_id": "test", "receipt": {}},
        expected_gpu_identity=gpu_identity,
    )

    assert result["status"] == "interrupted"
    assert result["validated_seconds"] == 0
    assert result["health_gates"] == []
    assert result["interruption"]["kind"] == "operator_stop"
    assert result["retriable"] is False
    assert result["attempt_finalization"]["state"] == "awaiting_cross_segment_drift"
    assert result["container_cleanup"]["verified_absent"] is True
    assert endurance.failed_health_attempt_count({"attempts": [result]}) == 0


def test_interrupted_attempt_pauses_and_resumes_without_health_ack(tmp_path, monkeypatch):
    _mock_verified_floor(monkeypatch)
    campaign_template = load_campaign()
    campaign_template["duration_seconds"] = 20
    campaign_template["segment_seconds"] = 10
    campaign_template["min_free_disk_bytes"] = 0
    run_calls = 0

    monkeypatch.setattr(
        endurance,
        "load_campaign",
        lambda path=endurance.DEFAULT_CAMPAIGN: json.loads(
            json.dumps(campaign_template)
        ),
    )
    monkeypatch.setattr(endurance, "project_relative", lambda path: str(path))
    monkeypatch.setattr(endurance, "live_input_pin_sweep", lambda campaign: _ok_input_sweep())

    def fake_write_plan(output, campaign, plan):
        endurance.atomic_write_json(output / "campaign-resolved.json", campaign)
        endurance.atomic_write_json(output / "plan.json", {"segments": plan})

    monkeypatch.setattr(endurance, "write_plan_artifacts", fake_write_plan)
    real_validate = endurance.validate_resume_checkpoint
    monkeypatch.setattr(
        endurance,
        "validate_resume_checkpoint",
        lambda checkpoint, campaign, plan, **kwargs: checkpoint,
    )
    monkeypatch.setattr(
        "validation.gpu_reentry_evidence.require_reentry_evidence",
        lambda *args, **kwargs: {
            "verification": {"status": "ready_for_operator_review"}
        },
    )
    gpu_identity = {
        "index": "0",
        "uuid": "GPU-test",
        "name": "Fake GPU",
        "driver_version": "test",
        "memory.total": "1 MiB",
        "pci.bus_id": "00000000:01:00.0",
    }
    monkeypatch.setattr(
        endurance,
        "preflight",
        lambda **kwargs: {
            "status": "ok",
            "image": kwargs["image"],
            "image_id": "sha256:test",
            "gpu_index": kwargs["gpu_index"],
            "gpu_identity": gpu_identity,
            "xid": {"available": True, "lines": []},
            "power_profile": {"available": True, "value": "performance"},
            "platform_thermal_sources": {},
            "safety_events": [],
        },
    )
    monkeypatch.setattr(
        endurance,
        "read_kernel_oom_log",
        lambda: {"available": True, "lines": [], "count": 0},
    )
    monkeypatch.setattr(
        endurance,
        "create_session_receipt",
        lambda **kwargs: {
            "session_id": kwargs["session_id"],
            "receipt": {
                "path": f"{kwargs['session_id']}.json",
                "size_bytes": 1,
                "sha256": "a" * 64,
            },
        },
    )
    monkeypatch.setattr(
        endurance, "require_compatible_session_history", lambda *args: None
    )
    monkeypatch.setattr(
        endurance,
        "finalize_attempt_receipt",
        lambda result, attempt_dir, session: {
            "path": str(attempt_dir / "attempt-receipt.json"),
            "size_bytes": 1,
            "sha256": "b" * 64,
        },
    )

    def fake_segment(**kwargs):
        nonlocal run_calls
        run_calls += 1
        segment = {
            key: kwargs["segment"][key]
            for key in (
                "index",
                "segment_id",
                "profile",
                "duration_seconds",
                "campaign_day",
            )
        }
        interrupted = run_calls == 1
        if interrupted:
            kwargs["stop_event"].set()
        result = {
            "schema_version": endurance.SEGMENT_SCHEMA,
            "status": "interrupted" if interrupted else "healthy",
            "dry_run": False,
            "config_fingerprint": kwargs["campaign"]["config_fingerprint"],
            "static_input_fingerprint": kwargs["campaign"][
                "static_input_fingerprint"
            ],
            "session": kwargs["session_binding"],
            "segment": segment,
            "attempt": kwargs["attempt"],
            "attempt_finalization": endurance._pending_attempt_finalization(),
            "validated_seconds": 0 if interrupted else segment["duration_seconds"],
            "timing": {"synthetic": False},
            "health_gates": [],
            "interruption": (
                {"kind": "operator_stop", "detail": "test", "at_utc": "test"}
                if interrupted
                else None
            ),
            "retriable": False,
            "artifact_pins": {},
            "container_cleanup": {"verified_absent": True},
        }
        kwargs["attempt_dir"].mkdir(parents=True, exist_ok=True)
        endurance.atomic_write_json(kwargs["attempt_dir"] / "status.json", result)
        return result

    monkeypatch.setattr(endurance, "run_production_segment", fake_segment)
    args = [
        "--output",
        str(tmp_path),
        "--acknowledge-seven-day-run",
    ]

    assert main(args) == 130
    paused = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert paused["state"] == "paused"
    assert paused["campaign_health_gates"] == []
    assert paused["unexpected_restarts"] == 0
    assert paused["segments"][0]["status"] == "pending"
    assert paused["segments"][0]["attempts"][0]["status"] == "interrupted"
    assert endurance.failed_health_attempt_count(paused["segments"][0]) == 0

    # No --acknowledge-health-gate is needed for normal maintenance resume.
    assert main(args) == 0
    completed = json.loads(
        (tmp_path / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert completed["state"] == "complete"
    assert [
        attempt["status"] for attempt in completed["segments"][0]["attempts"]
    ] == ["interrupted", "healthy"]
    assert completed["unexpected_restarts"] == 0

    monkeypatch.setattr(endurance, "validate_resume_checkpoint", real_validate)


def test_normal_run_cleanup_failure_preserves_active_and_never_retries(
    tmp_path, monkeypatch
):
    _mock_verified_floor(monkeypatch)
    campaign_template = load_campaign()
    campaign_template["duration_seconds"] = 20
    campaign_template["segment_seconds"] = 10
    campaign_template["min_free_disk_bytes"] = 0
    run_calls = 0

    monkeypatch.setattr(
        endurance,
        "load_campaign",
        lambda path=endurance.DEFAULT_CAMPAIGN: json.loads(
            json.dumps(campaign_template)
        ),
    )
    monkeypatch.setattr(endurance, "project_relative", lambda path: str(path))
    monkeypatch.setattr(endurance, "live_input_pin_sweep", lambda campaign: _ok_input_sweep())

    def fake_write_plan(output, campaign, plan):
        endurance.atomic_write_json(output / "campaign-resolved.json", campaign)
        endurance.atomic_write_json(output / "plan.json", {"segments": plan})

    monkeypatch.setattr(endurance, "write_plan_artifacts", fake_write_plan)
    monkeypatch.setattr(
        "validation.gpu_reentry_evidence.require_reentry_evidence",
        lambda *args, **kwargs: {
            "verification": {"status": "ready_for_operator_review"}
        },
    )
    gpu_identity = {
        "index": "0",
        "uuid": "GPU-test",
        "name": "Fake GPU",
        "driver_version": "test",
        "memory.total": "1 MiB",
        "pci.bus_id": "00000000:01:00.0",
    }
    monkeypatch.setattr(
        endurance,
        "preflight",
        lambda **kwargs: {
            "status": "ok",
            "image": kwargs["image"],
            "image_id": "sha256:test",
            "gpu_index": kwargs["gpu_index"],
            "gpu_identity": gpu_identity,
            "xid": {"available": True, "lines": []},
            "power_profile": {"available": True, "value": "performance"},
            "platform_thermal_sources": {},
            "safety_events": [],
        },
    )
    monkeypatch.setattr(
        endurance,
        "read_kernel_oom_log",
        lambda: {"available": True, "lines": [], "count": 0},
    )
    monkeypatch.setattr(
        endurance,
        "create_session_receipt",
        lambda **kwargs: {
            "session_id": kwargs["session_id"],
            "receipt": {
                "path": f"{kwargs['session_id']}.json",
                "size_bytes": 1,
                "sha256": "a" * 64,
            },
        },
    )
    monkeypatch.setattr(
        endurance, "require_compatible_session_history", lambda *args: None
    )
    monkeypatch.setattr(
        endurance,
        "finalize_attempt_receipt",
        lambda *args, **kwargs: pytest.fail(
            "A cleanup-failed active attempt must not receive a receipt"
        ),
    )

    def cleanup_failed_segment(**kwargs):
        nonlocal run_calls
        run_calls += 1
        segment = {
            key: kwargs["segment"][key]
            for key in (
                "index",
                "segment_id",
                "profile",
                "duration_seconds",
                "campaign_day",
            )
        }
        result = {
            "schema_version": endurance.SEGMENT_SCHEMA,
            "status": "failed",
            "dry_run": False,
            "config_fingerprint": kwargs["campaign"]["config_fingerprint"],
            "static_input_fingerprint": kwargs["campaign"][
                "static_input_fingerprint"
            ],
            "session": kwargs["session_binding"],
            "segment": segment,
            "attempt": kwargs["attempt"],
            "attempt_finalization": endurance._pending_attempt_finalization(),
            "validated_seconds": 0,
            "timing": {"synthetic": False},
            "health_gates": [
                {
                    "name": "container_cleanup_integrity",
                    "detail": "docker rm failed",
                    "retriable": False,
                }
            ],
            "retriable": False,
            "artifact_pins": {},
            "container_cleanup": {
                "verified_absent": False,
                "removed": False,
                "error": "docker rm failed",
            },
        }
        kwargs["attempt_dir"].mkdir(parents=True, exist_ok=True)
        endurance.atomic_write_json(kwargs["attempt_dir"] / "status.json", result)
        return result

    monkeypatch.setattr(endurance, "run_production_segment", cleanup_failed_segment)

    assert main(
        ["--output", str(tmp_path), "--acknowledge-seven-day-run"]
    ) == 2

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert run_calls == 1
    assert checkpoint["state"] == "paused_health_gate"
    assert checkpoint["active"] is not None
    assert checkpoint["active"]["attempt"] == 1
    assert checkpoint["segments"][0]["attempts"] == []
    assert checkpoint["segments"][0]["attempt_receipts"] == []
    persisted = json.loads(
        (
            tmp_path
            / "segments/segment-000-640/attempt-01/status.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted["attempt_finalization"]["state"] == (
        "awaiting_cross_segment_drift"
    )


def _pin_paths_to_tmp_root(monkeypatch, root):
    root = root.resolve()

    def relative(path):
        return Path(path).resolve().relative_to(root).as_posix()

    def resolve(path):
        candidate = Path(path)
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    monkeypatch.setattr(endurance, "project_relative", relative)
    monkeypatch.setattr(endurance, "resolve_project_path", resolve)


def _session_campaign(image_id="sha256:image-a"):
    gpu_identity = {
        "index": "0",
        "uuid": "GPU-test",
        "name": "GPU Test",
        "driver_version": "590.0",
        "memory.total": "16384",
        "pci.bus_id": "00000000:01:00.0",
    }
    return {
        "name": "test-campaign",
        "config_fingerprint": "c" * 64,
        "static_input_fingerprint": "s" * 64,
        "input_pins": {
            "schema_version": "deepsafe.endurance-input-pins/v1",
            "source_media": [],
            "model_profiles": {},
            "control_files": [],
        },
        "throughput_floor": _verified_floor_binding(
            image="deepstream:test",
            image_id=image_id,
            gpu_identity=gpu_identity,
        ),
        "execution_request": {"image": "deepstream:test", "gpu_index": 0},
        "telemetry_interval_seconds": 1,
        "max_temperature_c": 86.0,
        "power_safety": normalize_power_safety_policy({}),
    }


def _create_test_session(tmp_path, monkeypatch, session_id, image_id):
    _pin_paths_to_tmp_root(monkeypatch, tmp_path)
    output = tmp_path / "campaign"
    session_dir = output / "sessions" / session_id
    session_dir.mkdir(parents=True)
    (output / "campaign-resolved.json").write_text("{}\n", encoding="utf-8")
    (output / "plan.json").write_text("{}\n", encoding="utf-8")
    preflight = {
        "status": "ok",
        "image": "deepstream:test",
        "image_id": image_id,
        "gpu_index": 0,
        "gpu_identity": {
            "index": "0",
            "uuid": "GPU-test",
            "name": "GPU Test",
            "driver_version": "590.0",
            "memory.total": "16384",
            "pci.bus_id": "00000000:01:00.0",
        },
        "power_profile": {"available": True, "value": "performance"},
        "max_temperature_c": 86.0,
        "power_safety_policy": _session_campaign(image_id)["throughput_floor"][
            "source_runtime_identity"
        ]["power_safety_policy"],
    }
    (session_dir / "preflight.json").write_text(
        json.dumps(preflight), encoding="utf-8"
    )
    reentry_path = tmp_path / "reentry.json"
    reentry_path.write_text('{"status":"ready"}\n', encoding="utf-8")
    reentry_pin = endurance._file_pin(reentry_path)
    reentry = {
        "report_path": str(reentry_path),
        "report_sha256": reentry_pin["sha256"],
        "verification": {"status": "ready_for_operator_review"},
    }
    binding = create_session_receipt(
        session_id=session_id,
        session_dir=session_dir,
        output=output,
        campaign=_session_campaign(image_id),
        preflight_report=preflight,
        reentry_receipt=reentry,
    )
    return output, session_dir, binding


def test_live_input_pin_sweep_detects_source_model_and_control_toctou(
    tmp_path, monkeypatch
):
    _pin_paths_to_tmp_root(monkeypatch, tmp_path)
    source = tmp_path / "source.mp4"
    infer = tmp_path / "config_infer_primary.txt"
    engine = tmp_path / "model.engine"
    control = tmp_path / "campaign.json"
    for path, content in (
        (source, "source-a\n"),
        (infer, "infer-a\n"),
        (engine, "engine-a\n"),
        (control, "control-a\n"),
    ):
        path.write_text(content, encoding="utf-8")
    source_pin = endurance._file_pin(source)
    infer_pin = endurance._file_pin(infer)
    engine_pin = endurance._file_pin(engine)
    control_pin = endurance._file_pin(control)
    campaign = {
        "input_pins": {
            "schema_version": "deepsafe.endurance-input-pins/v1",
            "source_media": [
                {
                    "camera_id": 0,
                    "scene_id": "source",
                    **source_pin,
                }
            ],
            "model_profiles": {
                "640": {
                    "infer_config": infer_pin["path"],
                    "infer_config_sha256": infer_pin["sha256"],
                    "engine": engine_pin["path"],
                    "engine_size_bytes": engine_pin["size_bytes"],
                    "engine_sha256": engine_pin["sha256"],
                }
            },
            "control_files": [control_pin],
        }
    }

    before = endurance.live_input_pin_sweep(campaign)
    assert before["status"] == "ok"
    assert before["mismatches"] == []

    source.write_text("source-b\n", encoding="utf-8")
    engine.write_text("engine-b\n", encoding="utf-8")
    control.write_text("control-b\n", encoding="utf-8")
    after = endurance.live_input_pin_sweep(campaign)

    assert after["status"] == "mismatch"
    assert {item["kind"] for item in after["mismatches"]} >= {
        "source_media",
        "model_engine",
        "control_file",
    }
    assert after["live_pins_fingerprint"] != before["live_pins_fingerprint"]


def test_session_receipt_pins_image_gpu_reentry_and_campaign_inputs(tmp_path, monkeypatch):
    _output, session_dir, binding = _create_test_session(
        tmp_path, monkeypatch, "session-one", "sha256:image-a"
    )

    receipt = verify_session_binding(binding, _session_campaign())

    assert set(binding) == {"session_id", "receipt"}
    assert set(binding["receipt"]) == {"path", "size_bytes", "sha256"}
    assert receipt["runtime_identity"]["resolved_image_id"] == "sha256:image-a"
    assert receipt["runtime_identity"]["gpu_identity"]["uuid"] == "GPU-test"
    assert receipt["input_pins_sha256"] == endurance._sha256(
        _session_campaign()["input_pins"]
    )
    assert receipt["campaign_artifacts"]["reentry_evidence"]["path"].startswith(
        "campaign/sessions/session-one/"
    )
    assert (session_dir / "reentry-evidence.json").is_file()


def test_session_creation_rejects_image_identity_changed_from_floor(tmp_path, monkeypatch):
    _output, _first_dir, _first = _create_test_session(
        tmp_path, monkeypatch, "session-one", "sha256:image-a"
    )
    output = tmp_path / "campaign"
    second_dir = output / "sessions/session-two"
    second_dir.mkdir(parents=True)
    preflight = {
        "status": "ok",
        "image": "deepstream:test",
        "image_id": "sha256:image-b",
        "gpu_index": 0,
        "gpu_identity": {
            "index": "0",
            "uuid": "GPU-test",
            "name": "GPU Test",
            "driver_version": "590.0",
            "memory.total": "16384",
            "pci.bus_id": "00000000:01:00.0",
        },
        "power_profile": {"available": True, "value": "performance"},
        "max_temperature_c": 86.0,
        "power_safety_policy": _session_campaign()["throughput_floor"][
            "source_runtime_identity"
        ]["power_safety_policy"],
    }
    (second_dir / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    reentry_path = tmp_path / "reentry-two.json"
    reentry_path.write_text('{"status":"ready"}\n', encoding="utf-8")
    reentry_pin = endurance._file_pin(reentry_path)
    with pytest.raises(RuntimeError, match="preflight runtime identity differs"):
        create_session_receipt(
            session_id="session-two",
            session_dir=second_dir,
            output=output,
            campaign=_session_campaign(),
            preflight_report=preflight,
            reentry_receipt={
                "report_path": str(reentry_path),
                "report_sha256": reentry_pin["sha256"],
                "verification": {"status": "ready_for_operator_review"},
            },
        )


def test_attempt_receipt_pins_exact_status_and_attempt_artifacts(tmp_path, monkeypatch):
    _output, _session_dir, binding = _create_test_session(
        tmp_path, monkeypatch, "session-one", "sha256:image-a"
    )
    attempt_dir = tmp_path / "campaign/segments/segment-000-640/attempt-01"
    attempt_dir.mkdir(parents=True)
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
    for filename in artifact_files.values():
        (attempt_dir / filename).write_text(f"{filename}\n", encoding="utf-8")
    status = {
        "schema_version": "deepsafe.endurance-segment/v1",
        "status": "healthy",
        "dry_run": False,
        "segment": {
            "index": 0,
            "segment_id": "segment-000-640",
            "profile": 640,
            "duration_seconds": 21600,
            "campaign_day": 1,
        },
        "attempt": 1,
        "attempt_finalization": {
            "state": "finalized",
            "method": "cross_segment_drift",
            "cross_segment_drift_gates": [],
            "finalized_at_utc": "2026-01-01T00:00:00+00:00",
        },
        "input_pin_sweeps": {
            "pre": {"status": "ok", "live_pins_fingerprint": "d" * 64},
            "post": {"status": "ok", "live_pins_fingerprint": "e" * 64},
        },
        "validated_seconds": 21600,
        "config_fingerprint": "c" * 64,
        "static_input_fingerprint": "s" * 64,
        "throughput_floor": _floor_evidence(640),
        "session": binding,
        "container_cleanup": {"verified_absent": True},
        "artifact_pins": {
            name: endurance._file_pin(attempt_dir / filename)
            for name, filename in artifact_files.items()
        },
    }
    (attempt_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")

    receipt_pin = finalize_attempt_receipt(status, attempt_dir, binding)

    assert set(receipt_pin) == {"path", "size_bytes", "sha256"}
    assert verify_attempt_receipt(status, attempt_dir, binding) == receipt_pin
    receipt = json.loads((attempt_dir / "attempt-receipt.json").read_text())
    assert receipt["status_pin"] == endurance._file_pin(attempt_dir / "status.json")
    assert receipt["artifact_pins"] == status["artifact_pins"]
    assert receipt["input_pin_sweep_fingerprints"] == {
        "pre": "d" * 64,
        "post": "e" * 64,
    }


def test_interrupted_receipt_explicitly_allows_stable_zero_byte_artifact(
    tmp_path, monkeypatch
):
    _output, _session_dir, binding = _create_test_session(
        tmp_path, monkeypatch, "session-one", "sha256:image-a"
    )
    attempt_dir = tmp_path / "campaign/segments/segment-000-640/attempt-01"
    attempt_dir.mkdir(parents=True)
    empty_log = attempt_dir / "deepstream.log"
    empty_log.write_bytes(b"")
    empty_pin = endurance._attempt_artifact_pin(empty_log, allow_empty=True)
    status = {
        "schema_version": endurance.SEGMENT_SCHEMA,
        "status": "interrupted",
        "dry_run": False,
        "segment": {
            "index": 0,
            "segment_id": "segment-000-640",
            "profile": 640,
            "duration_seconds": 21600,
            "campaign_day": 1,
        },
        "attempt": 1,
        "attempt_finalization": {
            "state": "finalized",
            "method": "cross_segment_drift",
            "cross_segment_drift_gates": [],
            "finalized_at_utc": "2026-01-01T00:00:00+00:00",
        },
        "validated_seconds": 0,
        "config_fingerprint": "c" * 64,
        "static_input_fingerprint": "s" * 64,
        "throughput_floor": _floor_evidence(640),
        "session": binding,
        "container_cleanup": {"verified_absent": True},
        "timing": {"synthetic": False},
        "health_gates": [],
        "interruption": {
            "kind": "operator_stop",
            "detail": "test",
            "at_utc": "2026-01-01T00:00:00+00:00",
        },
        "retriable": False,
        "artifact_pins": {"log": empty_pin},
    }
    endurance.atomic_write_json(attempt_dir / "status.json", status)

    receipt_pin = finalize_attempt_receipt(status, attempt_dir, binding)

    assert empty_pin["size_bytes"] == 0
    assert empty_pin["allow_empty"] is True
    assert verify_attempt_receipt(status, attempt_dir, binding) == receipt_pin


def test_resume_rejects_tampered_attempt_artifact_before_plan_rewrite(tmp_path, monkeypatch):
    output, _session_dir, binding = _create_test_session(
        tmp_path, monkeypatch, "session-one", "sha256:image-a"
    )
    plan = [
        {
            "index": 0,
            "segment_id": "segment-000-640",
            "profile": 640,
            "duration_seconds": 21600,
            "campaign_day": 1,
        }
    ]
    attempt_dir = output / "segments/segment-000-640/attempt-01"
    attempt_dir.mkdir(parents=True)
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
    for filename in artifact_files.values():
        (attempt_dir / filename).write_text(f"{filename}\n", encoding="utf-8")
    config = attempt_dir / "deepstream.txt"
    status = {
        "schema_version": "deepsafe.endurance-segment/v1",
        "status": "healthy",
        "dry_run": False,
        "segment": plan[0],
        "attempt": 1,
        "attempt_finalization": {
            "state": "finalized",
            "method": "cross_segment_drift",
            "cross_segment_drift_gates": [],
            "finalized_at_utc": "2026-01-01T00:00:00+00:00",
        },
        "input_pin_sweeps": {
            "pre": {"status": "ok", "live_pins_fingerprint": "d" * 64},
            "post": {"status": "ok", "live_pins_fingerprint": "e" * 64},
        },
        "validated_seconds": 21600,
        "config_fingerprint": "c" * 64,
        "static_input_fingerprint": "s" * 64,
        "throughput_floor": _floor_evidence(640),
        "session": binding,
        "container_cleanup": {"verified_absent": True},
        "timing": {"synthetic": False},
        "health_gates": [],
        "artifact_pins": {
            name: endurance._file_pin(attempt_dir / filename)
            for name, filename in artifact_files.items()
        },
    }
    (attempt_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    receipt_pin = finalize_attempt_receipt(status, attempt_dir, binding)
    campaign = _session_campaign()
    checkpoint = {
        "schema_version": "deepsafe.endurance-checkpoint/v1",
        "campaign_name": campaign["name"],
        "config_fingerprint": campaign["config_fingerprint"],
        "static_input_fingerprint": campaign["static_input_fingerprint"],
        "input_pins": campaign["input_pins"],
        "throughput_floor": campaign["throughput_floor"],
        "dry_run": False,
        "target_validated_seconds": 21600,
        "validated_seconds": 21600,
        "sessions": [binding],
        "active": None,
        "segments": [
            {
                **plan[0],
                "status": "healthy",
                "validated_seconds": 21600,
                "attempts": [status],
                "attempt_receipts": [receipt_pin],
            }
        ],
    }
    validate_resume_checkpoint(
        checkpoint, campaign, plan, dry_run=False, output=output
    )
    plan_before = (output / "plan.json").read_bytes()

    config.write_text("tampered config\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="immutable evidence verification failed"):
        validate_resume_checkpoint(
            checkpoint, campaign, plan, dry_run=False, output=output
        )
    assert (output / "plan.json").read_bytes() == plan_before


def test_no_gpu_dry_run_is_resumable_and_writes_daily_reports(tmp_path):
    args = [
        "--dry-run",
        "--campaign-seconds",
        "40",
        "--segment-seconds",
        "10",
        "--output",
        str(tmp_path),
        "--dry-run-fail-once",
        "segment-001-960",
    ]
    assert main(args) == 0
    first = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert first["state"] == "complete"
    assert first["validated_seconds"] == 40
    assert len(first["segments"]) == 4
    assert [len(segment["attempts"]) for segment in first["segments"]] == [1, 2, 1, 1]
    assert first["unexpected_restarts"] == 1
    assert first["power_safety_policy"]["power_limit_drop_tolerance_w"] == 5.0
    assert first["segments"][1]["attempts"][0]["health_gates"][0]["name"] == "synthetic_premature_exit"
    resolved = json.loads((tmp_path / "campaign-resolved.json").read_text(encoding="utf-8"))
    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert resolved["power_safety"] == first["power_safety_policy"]
    assert plan["config_fingerprint"] == first["config_fingerprint"]
    assert plan["power_safety_policy"] == first["power_safety_policy"]
    assert (tmp_path / "reports/day-01.json").is_file()
    assert (tmp_path / "reports/day-01.md").is_file()

    terminal_artifacts = {
        relative: (tmp_path / relative).read_bytes()
        for relative in (
            "checkpoint.json",
            "status.json",
            "reports/day-01.json",
            "reports/day-01.md",
        )
    }
    assert main(args) == 0
    resumed = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert [len(segment["attempts"]) for segment in resumed["segments"]] == [1, 2, 1, 1]
    assert {
        relative: (tmp_path / relative).read_bytes()
        for relative in terminal_artifacts
    } == terminal_artifacts

    stale_status = json.loads(
        (tmp_path / "status.json").read_text(encoding="utf-8")
    )
    stale_status["state"] = "running"
    endurance.atomic_write_json(tmp_path / "status.json", stale_status)

    assert main(args) == 0
    repaired_status = json.loads(
        (tmp_path / "status.json").read_text(encoding="utf-8")
    )
    repaired = json.loads(
        (tmp_path / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert repaired_status["state"] == "complete"
    assert repaired_status["validated_seconds"] == 40
    assert [len(segment["attempts"]) for segment in repaired["segments"]] == [
        1,
        2,
        1,
        1,
    ]


def test_terminal_projection_seal_commits_reports_before_complete_checkpoint(
    tmp_path,
    monkeypatch,
):
    checkpoint = {
        "state": "complete",
        "target_validated_seconds": 10,
        "validated_seconds": 10,
        "active": None,
        "campaign_health_gates": [],
        "segments": [
            {
                "status": "healthy",
                "duration_seconds": 10,
                "validated_seconds": 10,
            }
        ],
    }
    events = []
    monkeypatch.setattr(
        endurance,
        "write_reports",
        lambda *_args: events.append("terminal_reports"),
    )
    monkeypatch.setattr(
        endurance,
        "atomic_write_json",
        lambda path, _payload: events.append(("checkpoint", path)),
    )
    checkpoint_path = tmp_path / "checkpoint.json"

    endurance.seal_terminal_projections(
        checkpoint,
        {},
        tmp_path,
        checkpoint_path,
    )

    assert events == ["terminal_reports", ("checkpoint", checkpoint_path)]


def test_changed_resume_fingerprint_cannot_overwrite_canonical_plan(tmp_path):
    original_args = [
        "--dry-run",
        "--campaign-seconds",
        "40",
        "--segment-seconds",
        "10",
        "--output",
        str(tmp_path),
    ]
    assert main(original_args) == 0
    protected = {
        relative: (tmp_path / relative).read_bytes()
        for relative in (
            "campaign-resolved.json",
            "plan.json",
            "generated/deepstream-12x-mixed-640.txt",
            "generated/deepstream-12x-mixed-960.txt",
        )
    }

    with pytest.raises(SystemExit, match="fingerprint changed"):
        main(
            [
                "--dry-run",
                "--campaign-seconds",
                "60",
                "--segment-seconds",
                "10",
                "--output",
                str(tmp_path),
            ]
        )

    assert {
        relative: (tmp_path / relative).read_bytes() for relative in protected
    } == protected


def test_plan_only_refuses_checkpoint_output_without_overwriting_it(tmp_path):
    run_args = [
        "--dry-run",
        "--campaign-seconds",
        "40",
        "--segment-seconds",
        "10",
        "--output",
        str(tmp_path),
    ]
    assert main(run_args) == 0
    plan_before = (tmp_path / "plan.json").read_bytes()
    checkpoint_before = (tmp_path / "checkpoint.json").read_bytes()

    with pytest.raises(SystemExit, match="Plan-only refuses"):
        main([*run_args, "--plan-only"])

    assert (tmp_path / "plan.json").read_bytes() == plan_before
    assert (tmp_path / "checkpoint.json").read_bytes() == checkpoint_before


def test_plan_only_cannot_write_while_supervisor_lock_is_held(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    lock_path = tmp_path / "supervisor.lock"
    with lock_path.open("a+", encoding="utf-8") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="Another endurance supervisor owns"):
            main(
                [
                    "--dry-run",
                    "--plan-only",
                    "--campaign-seconds",
                    "40",
                    "--segment-seconds",
                    "10",
                    "--output",
                    str(tmp_path),
                ]
            )

    assert not (tmp_path / "plan.json").exists()
    assert not (tmp_path / "campaign-resolved.json").exists()


def test_admin_status_projects_supervisor_status_and_live_heartbeat(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(
        json.dumps({"available": True, "state": "running", "validated_seconds": 21600}),
        encoding="utf-8",
    )
    (tmp_path / "live.json").write_text(
        json.dumps({"state": "running", "profile": 960, "latest_aggregate_fps": 333.0}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSAFE_ENDURANCE_STATUS", str(tmp_path / "status.json"))

    with TestClient(app) as client:
        direct = client.get("/api/endurance")
        assert direct.status_code == 200
        assert direct.json()["live"]["profile"] == 960
        assert client.get("/api/status").json()["endurance_campaign"]["validated_seconds"] == 21600
