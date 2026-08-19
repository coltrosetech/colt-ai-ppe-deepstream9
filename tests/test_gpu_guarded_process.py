import json
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from validation import gpu_guarded_process as guarded


GPU_IDENTITY = {
    "index": "0",
    "uuid": "GPU-synthetic",
    "name": "synthetic",
    "driver_version": "590.48.01",
    "memory.total": "16384",
    "pci.bus_id": "00000000:01:00.0",
}


class FakeMonitor:
    def __init__(
        self,
        *args,
        safety_reason=None,
        safety_code="synthetic_abort",
        row_overrides=None,
        query_errors=None,
        **kwargs,
    ):
        self.path = Path(args[0])
        self.platform_path = kwargs.get("platform_thermal_path")
        self.ident = None
        self.safety_reason = safety_reason
        self.safety_reason_code = safety_code if safety_reason else None
        self.safety_event = (
            {"code": safety_code, "reason": safety_reason}
            if safety_reason
            else None
        )
        self.row_overrides = row_overrides or {}
        self.query_errors = list(query_errors or [])
        self.samples = 1
        self.power_limit_drop_samples = 0
        self.slowdown_active_samples = 0
        self.max_consecutive_slowdown_samples = 0
        self.platform_thermal_samples = 1
        self.platform_thermal_error_count = 0

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(guarded.GPU_CSV_HEADER)
            row = {
                "timestamp": "2026-07-16T00:00:01+00:00",
                "gpu_index": "0",
                "gpu_name": "synthetic",
                "gpu_utilization_percent": "80",
                "memory_utilization_percent": "20",
                "memory_used_mib": "1024",
                "memory_total_mib": "16384",
                "temperature_c": "70",
                "power_draw_w": "80",
                "sm_clock_mhz": "1200",
                "memory_clock_mhz": "1500",
                "power_requested_limit_w": "[N/A]",
                "power_current_limit_w": "115",
                "power_default_limit_w": "115",
                "pstate": "P0",
                "clock_event_reasons_active_mask": "0x0",
                "clock_event_sw_power_cap": "Not Active",
                "clock_event_sw_thermal_slowdown": "Not Active",
                "clock_event_hw_slowdown": "Not Active",
                "clock_event_hw_thermal_slowdown": "Not Active",
                "clock_event_hw_power_brake_slowdown": "Not Active",
            }
            row.update(self.row_overrides)
            writer.writerow([row[field] for field in guarded.GPU_CSV_HEADER])
        if self.platform_path is not None:
            Path(self.platform_path).write_text(
                "timestamp\n2026-07-16T00:00:01+00:00\n", encoding="utf-8"
            )
        self.ident = 1

    def stop(self):
        return None

    def join(self, timeout=None):
        return None


class FakeProcess:
    def __init__(self, *, initially_running=True):
        self.pid = 12345
        self.returncode = None if initially_running else 0
        self.poll_calls = 0

    def poll(self):
        self.poll_calls += 1
        if self.returncode is None and self.poll_calls >= 2:
            self.returncode = 0
        return self.returncode


def _paths(tmp_path: Path):
    root = tmp_path.resolve()
    artifact_root = root / "run/safety"
    return root, artifact_root, root / "run/deepstream.log"


@pytest.mark.parametrize(
    ("duration_seconds", "expected_samples"),
    (
        (0.0, 1),
        (0.5, 1),
        (1.5, 1),
        (1.500001, 2),
        (3.000165, 3),
        (3.5, 3),
        (3.500001, 4),
    ),
)
def test_one_hz_coverage_allows_only_the_half_interval_scheduler_boundary(
    duration_seconds, expected_samples
):
    assert guarded._expected_telemetry_samples(duration_seconds) == expected_samples


@pytest.mark.parametrize("duration_seconds", (-0.000001, float("inf"), float("nan")))
def test_one_hz_coverage_rejects_invalid_durations(duration_seconds):
    with pytest.raises(ValueError, match="finite and non-negative"):
        guarded._expected_telemetry_samples(duration_seconds)


def test_container_inspect_classifies_only_missing_name_as_transient(monkeypatch):
    monkeypatch.setattr(
        guarded.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Error response from daemon: No such container: just-created",
        ),
    )

    with pytest.raises(guarded.ContainerNotVisibleError, match="not published"):
        guarded.inspect_running_container_image("just-created")


def test_container_inspect_daemon_error_is_not_treated_as_startup_race(monkeypatch):
    monkeypatch.setattr(
        guarded.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon",
        ),
    )

    with pytest.raises(RuntimeError, match="inspection failed"):
        guarded.inspect_running_container_image("just-created")


def test_container_image_handshake_timeout_remains_bounded_and_fail_closed():
    process = FakeProcess()
    clock = iter((100.0, 110.0))
    attempts = []

    def never_visible(name):
        attempts.append(name)
        raise guarded.ContainerNotVisibleError("not published")

    with pytest.raises(TimeoutError, match="10.000 seconds"):
        guarded.wait_for_running_container_image(
            "synthetic",
            process,
            inspector=never_visible,
            sleeper=lambda _seconds: pytest.fail("deadline has already elapsed"),
            monotonic=lambda: next(clock),
        )

    assert attempts == ["synthetic"]


def _preflight_payload(
    image="image",
    *,
    status="ok",
    safety_events=None,
    gpu_identity=None,
):
    return {
        "status": status,
        "checked_at_utc": "2026-07-16T00:00:00+00:00",
        "image": image,
        "image_id": "sha256:" + "1" * 64,
        "gpu_identity": dict(gpu_identity or GPU_IDENTITY),
        "power_profile": {"available": True, "value": "performance"},
        "power_safety_policy": {},
        "safety_events": list(safety_events or []),
        "platform_thermal_sources": {"available": False, "columns": ["timestamp"]},
    }


def _patch_safe_environment(
    monkeypatch,
    *,
    monitor_factory=FakeMonitor,
    operating_policy_id=guarded.WORKSTATION_MANAGED_POLICY_ID,
):
    operating_policy = guarded.operating_policy_contract(operating_policy_id)
    monkeypatch.setattr(
        guarded,
        "inspect_running_container_image",
        lambda _name: "sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        guarded,
        "require_reentry_evidence",
        lambda *args, **kwargs: {
            "report_path": "validation/results/gpu-reentry/evidence.json",
            "report_sha256": "a" * 64,
            "gpu_identity": {"available": True, "fields": dict(GPU_IDENTITY)},
            "operating_policy": operating_policy,
            "verification": {
                "status": "ready_for_operator_review",
                "operating_policy": operating_policy,
            },
        },
    )
    monkeypatch.setattr(
        guarded,
        "require_runtime_compatibility",
        lambda *args, **kwargs: {
            "status": "production_ready",
            "receipt": {
                "path": "validation/results/ds9-runtime-compatibility/current/receipt.json",
                "bytes": 123,
                "sha256": "b" * 64,
            },
            "resolved_image_id": "sha256:" + "1" * 64,
            "runtime_control_manifest_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        guarded,
        "preflight",
        lambda **kwargs: _preflight_payload(kwargs["image"]),
    )
    monkeypatch.setattr(
        guarded, "read_xid_log", lambda: {"available": True, "lines": [], "count": 0}
    )
    monkeypatch.setattr(
        guarded, "query_gpu_identity", lambda _index: dict(GPU_IDENTITY)
    )
    monkeypatch.setattr(
        guarded,
        "read_power_profile",
        lambda: {"available": True, "value": "performance"},
    )
    monkeypatch.setattr(guarded, "new_xid_lines", lambda before, after: [])
    monkeypatch.setattr(
        guarded,
        "scan_fatal_log",
        lambda path: {"gstreamer_error": 0, "cuda_error": 0},
    )
    monkeypatch.setattr(guarded, "GpuMonitor", monitor_factory)


def test_reentry_failure_is_persisted_before_popen(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    called = False

    def blocked(*args, **kwargs):
        raise RuntimeError("operator declaration missing")

    def forbidden_popen(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Popen must not be called")

    monkeypatch.setattr(guarded, "require_reentry_evidence", blocked)
    with pytest.raises(guarded.GpuGuardError, match="re-entry evidence"):
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=forbidden_popen,
        )
    assert called is False
    report = json.loads((artifact_root / "gpu-guard-report.json").read_text())
    assert report["status"] == "blocked_before_start"
    assert report["process"]["started"] is False
    assert report["failure_reasons"] == ["gpu_reentry_evidence_blocked"]


@pytest.mark.parametrize("case", ("unknown", "contract_tamper", "receipt_mismatch"))
def test_reentry_operating_policy_tamper_blocks_before_preflight_and_popen(
    tmp_path, monkeypatch, case
):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    policy = guarded.operating_policy_contract(
        guarded.WORKSTATION_MANAGED_POLICY_ID
    )
    verified_policy = dict(policy)
    if case == "unknown":
        policy = {"id": "unknown"}
        verified_policy = dict(policy)
    elif case == "contract_tamper":
        policy = {**policy, "hardware_protection_owner": "caller_override"}
        verified_policy = dict(policy)
    else:
        verified_policy = guarded.operating_policy_contract(
            guarded.LEGACY_STRICT_PHYSICAL_POLICY_ID
        )
    monkeypatch.setattr(
        guarded,
        "require_reentry_evidence",
        lambda *args, **kwargs: {
            "gpu_identity": {"available": True, "fields": dict(GPU_IDENTITY)},
            "operating_policy": policy,
            "verification": {
                "status": "ready_for_operator_review",
                "operating_policy": verified_policy,
            },
        },
    )
    monkeypatch.setattr(
        guarded,
        "preflight",
        lambda **kwargs: pytest.fail("invalid policy must block before preflight"),
    )

    with pytest.raises(guarded.GpuGuardError, match="operating policy"):
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: pytest.fail("must not launch"),
        )
    report = json.loads((artifact_root / "gpu-guard-report.json").read_text())
    assert report["failure_reasons"] == ["gpu_reentry_operating_policy_invalid"]


def test_preflight_safety_abort_never_starts_process(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(
        monkeypatch,
        operating_policy_id=guarded.LEGACY_STRICT_PHYSICAL_POLICY_ID,
    )
    monkeypatch.setattr(
        guarded,
        "preflight",
        lambda **kwargs: {
            "status": "safety_abort",
            "safety_events": [{"code": "power_limit_below_default"}],
            "platform_thermal_sources": {},
        },
    )

    with pytest.raises(guarded.GpuGuardError, match="preflight requested"):
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: pytest.fail("must not launch"),
        )
    report = json.loads((artifact_root / "gpu-guard-report.json").read_text())
    assert report["status"] == "blocked_before_start"
    assert report["failure_reasons"] == ["gpu_preflight_safety_abort"]


def test_workstation_preflight_static_signal_is_pinned_diagnostic_and_launches(
    tmp_path, monkeypatch
):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    diagnostic = {
        "code": "temperature_threshold",
        "reason": "synthetic static threshold",
        "operating_policy_mode": guarded.DEFAULT_GPU_OPERATING_POLICY_MODE,
        "measurement_quality_signal": True,
        "disposition": "record_only_workstation_hardware_managed",
    }

    def diagnostic_preflight(**kwargs):
        payload = _preflight_payload(kwargs["image"])
        payload["diagnostic_events"] = [diagnostic]
        return payload

    monkeypatch.setattr(
        guarded,
        "preflight",
        diagnostic_preflight,
    )
    process = FakeProcess()

    report = guarded.run_guarded_docker(
        ["docker", "run", "image"],
        project_root=root,
        artifact_root=artifact_root,
        log_path=log_path,
        container_name="synthetic",
        image="image",
        gpu_index=0,
        reentry_evidence_path=root / "evidence.json",
        popen_factory=lambda *args, **kwargs: process,
        sleeper=lambda _seconds: None,
    )

    assert report["status"] == "complete"
    assert report["policy"]["operating_policy_id"] == "workstation_managed"
    assert report["policy"]["temperature_power_slowdown_action"] == "record_only"
    assert report["diagnostics"]["preflight_static_signals"] == [diagnostic]
    persisted_preflight = json.loads(
        (artifact_root / "gpu-preflight.json").read_text()
    )
    assert persisted_preflight["collector_status"] == "ok"
    assert persisted_preflight["status"] == "ok"
    assert persisted_preflight["safety_events"] == []
    assert persisted_preflight["diagnostic_events"] == [diagnostic]


def test_workstation_legacy_preflight_static_abort_is_normalized_to_diagnostic():
    allowed, diagnostics, disposition = guarded._preflight_disposition(
        {
            "status": "safety_abort",
            "safety_events": [
                {
                    "code": "temperature_threshold",
                    "reason": "legacy static threshold",
                    "disposition": "safety_abort",
                }
            ],
        },
        guarded.WORKSTATION_MANAGED_POLICY_ID,
    )

    assert allowed is True
    assert disposition == "record_only_workstation_static_signals"
    assert diagnostics == [
        {
            "code": "temperature_threshold",
            "reason": "legacy static threshold",
            "operating_policy_mode": guarded.DEFAULT_GPU_OPERATING_POLICY_MODE,
            "measurement_quality_signal": True,
            "disposition": "record_only_workstation_hardware_managed",
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("operating_policy_mode", "legacy_strict"),
        ("measurement_quality_signal", False),
        ("disposition", "safety_abort"),
    ),
)
def test_workstation_native_preflight_diagnostic_requires_exact_disposition(
    field, value
):
    diagnostic = {
        "code": "temperature_threshold",
        "operating_policy_mode": guarded.DEFAULT_GPU_OPERATING_POLICY_MODE,
        "measurement_quality_signal": True,
        "disposition": "record_only_workstation_hardware_managed",
    }
    diagnostic[field] = value

    allowed, events, disposition = guarded._preflight_disposition(
        {
            "status": "ok",
            "safety_events": [],
            "diagnostic_events": [diagnostic],
        },
        guarded.WORKSTATION_MANAGED_POLICY_ID,
    )

    assert allowed is False
    assert events == []
    assert disposition == "unknown or policy-mismatched preflight diagnostics"


@pytest.mark.parametrize(
    "code",
    ("temperature_telemetry_unavailable", "power_limit_telemetry_unavailable"),
)
def test_workstation_preflight_unreadable_telemetry_remains_fail_closed(
    tmp_path, monkeypatch, code
):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    monkeypatch.setattr(
        guarded,
        "preflight",
        lambda **kwargs: _preflight_payload(
            kwargs["image"],
            status="safety_abort",
            safety_events=[{"code": code, "reason": "unreadable"}],
        ),
    )

    with pytest.raises(guarded.GpuGuardError, match="preflight requested"):
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: pytest.fail("must not launch"),
        )
    report = json.loads((artifact_root / "gpu-guard-report.json").read_text())
    assert report["failure_reasons"] == ["gpu_preflight_safety_abort"]


def test_safe_process_writes_complete_guard_report(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    process = FakeProcess()
    launched = []

    def popen(command, **kwargs):
        launched.append(command)
        return process

    report = guarded.run_guarded_docker(
        ["docker", "run", "--name", "synthetic", "image"],
        project_root=root,
        artifact_root=artifact_root,
        log_path=log_path,
        container_name="synthetic",
        image="image",
        gpu_index=0,
        reentry_evidence_path=root / "evidence.json",
        popen_factory=popen,
        sleeper=lambda seconds: None,
    )
    assert report["status"] == "complete"
    assert report["failure_reasons"] == []
    assert report["process"]["exit_code"] == 0
    assert report["process"]["container_image_id"] == "sha256:" + "1" * 64
    assert report["process"]["container_image_inspection"]["status"] == "verified"
    assert report["process"]["container_image_inspection"]["attempts"] == 1
    assert report["telemetry"]["samples"] == 1
    assert report["preflight"]["requested_image"] == "image"
    assert report["preflight"]["resolved_image_id"] == "sha256:" + "1" * 64
    assert report["ds9_runtime_compatibility"]["status"] == "production_ready"
    assert launched[0][-1] == "sha256:" + "1" * 64
    assert report["command"] == launched[0]
    assert report["requested_command"][-1] == "image"
    assert report["telemetry"]["coverage"]["coverage_satisfied"] is True
    assert report["telemetry"]["coverage"]["endpoint_tolerance_seconds"] == 1.5
    assert report["telemetry"]["coverage"]["sample_count_grace_seconds"] == 0.5
    assert report["operating_policy_id"] == "workstation_managed"
    assert report["operating_policy"]["id"] == "workstation_managed"
    assert report["artifact_receipt"]["path"].endswith(
        "gpu-guard-artifact-receipt.json"
    )
    receipt = json.loads(
        (artifact_root / "gpu-guard-artifact-receipt.json").read_text()
    )
    assert receipt["operating_policy_id"] == "workstation_managed"
    assert receipt["operating_policy"] == report["operating_policy"]
    assert receipt["safety_event"]["present"] is False
    assert receipt["record_only_diagnostic_event"]["present"] is False
    persisted = json.loads((artifact_root / "gpu-guard-report.json").read_text())
    assert persisted["schema_version"] == guarded.SCHEMA_VERSION
    assert persisted["status"] == "complete"


def test_guard_retries_missing_container_until_live_image_is_inspectable(
    tmp_path, monkeypatch
):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    process = FakeProcess()
    inspections = []

    def delayed_inspector(name):
        inspections.append(name)
        if len(inspections) == 1:
            raise guarded.ContainerNotVisibleError("create is not published yet")
        return "sha256:" + "1" * 64

    report = guarded.run_guarded_docker(
        ["docker", "run", "--rm", "--name", "synthetic", "image"],
        project_root=root,
        artifact_root=artifact_root,
        log_path=log_path,
        container_name="synthetic",
        image="image",
        gpu_index=0,
        reentry_evidence_path=root / "evidence.json",
        popen_factory=lambda *args, **kwargs: process,
        container_image_inspector=delayed_inspector,
        sleeper=lambda _seconds: None,
    )

    assert report["status"] == "complete"
    assert inspections == ["synthetic", "synthetic"]
    assert report["process"]["container_image_id"] == "sha256:" + "1" * 64
    assert report["process"]["container_image_inspection"]["status"] == "verified"
    assert report["process"]["container_image_inspection"]["attempts"] == 2


def test_guard_fails_closed_when_docker_exits_before_container_is_inspectable(
    tmp_path, monkeypatch
):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    process = FakeProcess(initially_running=False)
    process.returncode = 125

    def never_visible(_name):
        raise guarded.ContainerNotVisibleError("not published")

    with pytest.raises(guarded.GpuGuardError, match="acceptance-safely") as raised:
        guarded.run_guarded_docker(
            ["docker", "run", "--rm", "--name", "synthetic", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: process,
            container_image_inspector=never_visible,
            sleeper=lambda _seconds: None,
        )

    assert any(
        "before the named container became inspectable" in reason
        for reason in raised.value.report["failure_reasons"]
    )
    assert "docker_exit_code=125" in raised.value.report["failure_reasons"]
    assert (
        raised.value.report["process"]["container_image_inspection"]["status"]
        == "failed"
    )


def test_guard_passes_exact_native_workstation_mode_and_identity_to_live_probes(
    tmp_path, monkeypatch
):
    root, artifact_root, log_path = _paths(tmp_path)
    captured = {}

    def monitor_factory(*args, **kwargs):
        captured["monitor_mode"] = kwargs.get("operating_policy_mode")
        captured["monitor_identity"] = kwargs.get("expected_gpu_identity")
        return FakeMonitor(*args, **kwargs)

    _patch_safe_environment(monkeypatch, monitor_factory=monitor_factory)

    def preflight_probe(**kwargs):
        captured["preflight_mode"] = kwargs.get("operating_policy_mode")
        return _preflight_payload(kwargs["image"])

    monkeypatch.setattr(guarded, "preflight", preflight_probe)
    process = FakeProcess()
    report = guarded.run_guarded_docker(
        ["docker", "run", "image"],
        project_root=root,
        artifact_root=artifact_root,
        log_path=log_path,
        container_name="synthetic",
        image="image",
        gpu_index=0,
        reentry_evidence_path=root / "evidence.json",
        popen_factory=lambda *args, **kwargs: process,
        sleeper=lambda _seconds: None,
    )

    assert report["status"] == "complete"
    assert captured == {
        "preflight_mode": guarded.DEFAULT_GPU_OPERATING_POLICY_MODE,
        "monitor_mode": guarded.DEFAULT_GPU_OPERATING_POLICY_MODE,
        "monitor_identity": GPU_IDENTITY,
    }


def test_ds9_compatibility_failure_is_persisted_before_popen(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    called = False

    def blocked(*args, **kwargs):
        raise RuntimeError("pending_gpu_smoke")

    def forbidden_popen(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Popen must not be called")

    monkeypatch.setattr(guarded, "require_runtime_compatibility", blocked)
    with pytest.raises(guarded.GpuGuardError, match="compatibility blocked"):
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=forbidden_popen,
        )
    assert called is False
    report = json.loads((artifact_root / "gpu-guard-report.json").read_text())
    assert report["failure_reasons"] == ["ds9_runtime_compatibility_blocked"]
    assert report["process"]["started"] is False


def _static_smoke_command(image_id="sha256:" + "1" * 64):
    return [
        "docker",
        "run",
        "--pull=never",
        "--network=none",
        "--read-only",
        image_id,
        "python3",
        guarded.STATIC_CANDIDATE_SMOKE_WORKER,
        "--inside-container",
    ]


def test_static_candidate_smoke_uses_exact_id_and_candidate_verifier(
    tmp_path, monkeypatch
):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    verified = []
    monkeypatch.setattr(
        guarded,
        "require_runtime_compatibility",
        lambda *args, **kwargs: pytest.fail("production verifier is circular here"),
    )
    monkeypatch.setattr(
        guarded,
        "require_static_candidate_compatibility",
        lambda *args, **kwargs: verified.append(kwargs["resolved_image_id"])
        or {
            "status": "static_candidate_ready_for_guarded_gpu_smoke",
            "production_ready": False,
            "receipt": {"path": "candidate.json", "bytes": 1, "sha256": "a" * 64},
            "resolved_image_id": "sha256:" + "1" * 64,
        },
    )
    process = FakeProcess()
    report = guarded.run_guarded_docker(
        _static_smoke_command(),
        project_root=root,
        artifact_root=artifact_root,
        log_path=log_path,
        container_name="synthetic",
        image="deepsafe-deepstream:9.0",
        gpu_index=0,
        reentry_evidence_path=root / "evidence.json",
        ds9_compatibility_receipt_path=root / "candidate.json",
        compatibility_mode="static_candidate_smoke",
        popen_factory=lambda *args, **kwargs: process,
        sleeper=lambda seconds: None,
    )
    assert verified == ["sha256:" + "1" * 64]
    assert report["requested_command"] == _static_smoke_command()
    assert report["command"] == _static_smoke_command()
    assert report["compatibility_mode"] == "static_candidate_smoke"


def test_static_candidate_tag_retarget_blocks_before_popen(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    monkeypatch.setattr(
        guarded,
        "preflight",
        lambda **kwargs: {
            "status": "ok",
            "checked_at_utc": "2026-07-16T00:00:00+00:00",
            "image": kwargs["image"],
            "image_id": "sha256:" + "2" * 64,
            "gpu_identity": dict(GPU_IDENTITY),
            "power_profile": {"available": True, "value": "performance"},
            "power_safety_policy": {},
            "safety_events": [],
            "platform_thermal_sources": {"available": False, "columns": ["timestamp"]},
        },
    )
    monkeypatch.setattr(
        guarded,
        "require_static_candidate_compatibility",
        lambda *args, **kwargs: {
            "status": "static_candidate_ready_for_guarded_gpu_smoke"
        },
    )
    launched = False

    def forbidden(*args, **kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("retargeted tag must not launch")

    with pytest.raises(guarded.GpuGuardError, match="immutable Docker image binding"):
        guarded.run_guarded_docker(
            _static_smoke_command("sha256:" + "1" * 64),
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="deepsafe-deepstream:9.0",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            ds9_compatibility_receipt_path=root / "candidate.json",
            compatibility_mode="static_candidate_smoke",
            popen_factory=forbidden,
        )
    assert launched is False


def test_active_monitor_abort_stops_container_and_fails(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(
        monkeypatch,
        monitor_factory=lambda *args, **kwargs: FakeMonitor(
            *args, safety_reason="current power limit fell", **kwargs
        ),
    )
    process = FakeProcess()

    def stop(candidate, name, grace):
        assert candidate is process
        assert name == "synthetic"
        process.returncode = 130
        return 130, "sigint"

    monkeypatch.setattr(guarded, "stop_attached_container", stop)
    with pytest.raises(guarded.GpuGuardError, match="acceptance-safely") as raised:
        guarded.run_guarded_docker(
            ["docker", "run", "--name", "synthetic", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: process,
            sleeper=lambda seconds: None,
        )
    report = raised.value.report
    assert report["status"] == "safety_abort"
    assert report["process"]["termination_method"] == "sigint"
    assert any("gpu_safety_abort=synthetic_abort" in item for item in report["failure_reasons"])


def test_workstation_runtime_static_signals_are_recorded_without_stopping_process(
    tmp_path, monkeypatch
):
    root, artifact_root, log_path = _paths(tmp_path)
    row_overrides = {
        "temperature_c": "90",
        "power_current_limit_w": "55",
        "clock_event_hw_thermal_slowdown": "Active",
    }
    _patch_safe_environment(
        monkeypatch,
        monitor_factory=lambda *args, **kwargs: FakeMonitor(
            *args,
            safety_reason="synthetic static signal",
            safety_code="temperature_threshold",
            row_overrides=row_overrides,
            **kwargs,
        ),
    )
    monkeypatch.setattr(
        guarded,
        "stop_attached_container",
        lambda *args, **kwargs: pytest.fail("record-only signal must not stop process"),
    )
    process = FakeProcess()

    report = guarded.run_guarded_docker(
        ["docker", "run", "image"],
        project_root=root,
        artifact_root=artifact_root,
        log_path=log_path,
        container_name="synthetic",
        image="image",
        gpu_index=0,
        reentry_evidence_path=root / "evidence.json",
        popen_factory=lambda *args, **kwargs: process,
        sleeper=lambda _seconds: None,
    )

    diagnostics = report["diagnostics"]["runtime_static_signals"]
    assert report["status"] == "complete"
    assert report["failure_reasons"] == []
    assert report["process"]["termination_method"] is None
    assert report["telemetry"]["safety_event"] is None
    assert report["telemetry"]["power_limit_drop_samples"] == 1
    assert report["telemetry"]["slowdown_active_samples"] == 1
    assert diagnostics["temperature_threshold_samples"] == 1
    assert diagnostics["power_limit_drop_samples"] == 1
    assert diagnostics["slowdown_active_samples"] == 1
    assert report["diagnostics"]["monitor_static_signal_event"]["code"] == (
        "temperature_threshold"
    )
    receipt = json.loads(
        (artifact_root / "gpu-guard-artifact-receipt.json").read_text()
    )
    assert receipt["operating_policy_id"] == "workstation_managed"
    assert receipt["safety_event"]["present"] is False
    assert receipt["diagnostics"] == report["diagnostics"]


def test_legacy_runtime_static_signal_still_aborts(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(
        monkeypatch,
        operating_policy_id=guarded.LEGACY_STRICT_PHYSICAL_POLICY_ID,
        monitor_factory=lambda *args, **kwargs: FakeMonitor(
            *args,
            safety_reason="legacy thermal threshold",
            safety_code="temperature_threshold",
            **kwargs,
        ),
    )
    process = FakeProcess()

    def stop(candidate, _name, _grace):
        candidate.returncode = 130
        return 130, "sigint"

    monkeypatch.setattr(guarded, "stop_attached_container", stop)
    with pytest.raises(guarded.GpuGuardError, match="acceptance-safely") as raised:
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: process,
            sleeper=lambda _seconds: None,
        )
    assert raised.value.report["status"] == "safety_abort"
    assert raised.value.report["policy"]["temperature_power_slowdown_action"] == (
        "software_abort"
    )


def test_workstation_malformed_runtime_telemetry_fails_closed(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(
        monkeypatch,
        monitor_factory=lambda *args, **kwargs: FakeMonitor(
            *args, row_overrides={"temperature_c": "[N/A]"}, **kwargs
        ),
    )
    process = FakeProcess()

    with pytest.raises(guarded.GpuGuardError, match="acceptance-safely") as raised:
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: process,
            sleeper=lambda _seconds: None,
        )
    assert "gpu_telemetry_malformed_or_unreadable" in raised.value.report[
        "failure_reasons"
    ]


def test_workstation_runtime_query_error_remains_fail_closed(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(
        monkeypatch,
        monitor_factory=lambda *args, **kwargs: FakeMonitor(
            *args, query_errors=["synthetic nvidia-smi read failure"], **kwargs
        ),
    )
    process = FakeProcess()

    with pytest.raises(guarded.GpuGuardError, match="acceptance-safely") as raised:
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: process,
            sleeper=lambda _seconds: None,
        )
    assert "gpu_telemetry_query_errors=1" in raised.value.report["failure_reasons"]


def test_preflight_gpu_identity_drift_blocks_before_popen(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    changed = {**GPU_IDENTITY, "uuid": "GPU-retargeted"}
    monkeypatch.setattr(
        guarded,
        "preflight",
        lambda **kwargs: _preflight_payload(
            kwargs["image"], gpu_identity=changed
        ),
    )

    with pytest.raises(guarded.GpuGuardError, match="identity drift"):
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: pytest.fail("must not launch"),
        )
    report = json.loads((artifact_root / "gpu-guard-report.json").read_text())
    assert report["failure_reasons"] == ["gpu_identity_drift_before_start"]


def test_runtime_gpu_identity_drift_in_csv_fails_closed(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(
        monkeypatch,
        monitor_factory=lambda *args, **kwargs: FakeMonitor(
            *args, row_overrides={"gpu_name": "different-gpu"}, **kwargs
        ),
    )
    process = FakeProcess()

    with pytest.raises(guarded.GpuGuardError, match="acceptance-safely") as raised:
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: process,
            sleeper=lambda _seconds: None,
        )
    assert "gpu_identity_drift_in_runtime_telemetry" in raised.value.report[
        "failure_reasons"
    ]


def test_postflight_gpu_identity_drift_fails_closed(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    monkeypatch.setattr(
        guarded,
        "query_gpu_identity",
        lambda _index: {**GPU_IDENTITY, "driver_version": "changed-driver"},
    )
    process = FakeProcess()

    with pytest.raises(guarded.GpuGuardError, match="acceptance-safely") as raised:
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: process,
            sleeper=lambda _seconds: None,
        )
    assert "gpu_identity_drift_after_run" in raised.value.report["failure_reasons"]


def test_new_xid_driver_fault_remains_fail_closed(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    monkeypatch.setattr(
        guarded,
        "new_xid_lines",
        lambda _before, _after: ["NVRM: Xid (PCI:0000:01:00): 79"],
    )
    process = FakeProcess()

    with pytest.raises(guarded.GpuGuardError, match="acceptance-safely") as raised:
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: process,
            sleeper=lambda _seconds: None,
        )
    assert "new_nvidia_xid_events=1" in raised.value.report["failure_reasons"]


def test_nonzero_process_exit_remains_fail_closed(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    process = FakeProcess(initially_running=False)
    process.returncode = 42

    with pytest.raises(guarded.GpuGuardError, match="acceptance-safely") as raised:
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: process,
            sleeper=lambda _seconds: None,
        )
    assert "docker_exit_code=42" in raised.value.report["failure_reasons"]


def test_guard_artifact_pin_failure_remains_fail_closed(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    process = FakeProcess()
    monkeypatch.setattr(
        guarded,
        "_artifact_pin",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic pin failure")
        ),
    )

    with pytest.raises(guarded.GpuGuardError, match="acceptance-safely") as raised:
        guarded.run_guarded_docker(
            ["docker", "run", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: process,
            sleeper=lambda _seconds: None,
        )
    assert any(
        reason.startswith("guard_artifact_receipt_failed=RuntimeError")
        for reason in raised.value.report["failure_reasons"]
    )


def test_running_container_image_mismatch_fails_closed(tmp_path, monkeypatch):
    root, artifact_root, log_path = _paths(tmp_path)
    _patch_safe_environment(monkeypatch)
    process = FakeProcess()
    monkeypatch.setattr(
        guarded,
        "inspect_running_container_image",
        lambda _name: "sha256:" + "2" * 64,
    )
    monkeypatch.setattr(
        guarded,
        "stop_attached_container",
        lambda candidate, _name, _grace: (
            setattr(candidate, "returncode", 130) or (130, "sigint")
        ),
    )
    with pytest.raises(guarded.GpuGuardError, match="acceptance-safely") as raised:
        guarded.run_guarded_docker(
            ["docker", "run", "--name", "synthetic", "image"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
            popen_factory=lambda *args, **kwargs: process,
            sleeper=lambda seconds: None,
        )
    assert any(
        "running container image differs" in reason
        for reason in raised.value.report["failure_reasons"]
    )


def test_rejects_non_docker_command_without_writing(tmp_path):
    root, artifact_root, log_path = _paths(tmp_path)
    with pytest.raises(ValueError, match="Docker"):
        guarded.run_guarded_docker(
            ["python", "unsafe.py"],
            project_root=root,
            artifact_root=artifact_root,
            log_path=log_path,
            container_name="synthetic",
            image="image",
            gpu_index=0,
            reentry_evidence_path=root / "evidence.json",
        )
    assert not artifact_root.exists()
