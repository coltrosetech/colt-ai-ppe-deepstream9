import copy
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from validation import ds9_gpu_smoke as smoke
from validation import ds9_runtime_compatibility as gate


NONCE = "a" * 64
IMAGE_ID = "sha256:" + "1" * 64
GPU_UUID = "GPU-8cbaba1c-2629-a732-f528-66f459089ef6"


def _create_inputs(root: Path):
    video = root / smoke.DEFAULT_VIDEO
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"fixture smoke video")
    for profile, paths in smoke._model_paths(root).items():
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{profile}-{name}\n".encode())
    return video


def test_rendered_configs_are_engine_only_and_pin_both_parsers():
    for profile in smoke.PROFILES:
        cuda = smoke.render_infer_config(profile, "cuda")
        cpu = smoke.render_infer_config(profile, "cpu")
        assert "onnx-file=" not in cuda
        assert "onnx-file=" not in cpu
        assert f"model-engine-file=/models/{profile}.engine" in cuda
        assert "parse-bbox-func-name=NvDsInferParseYoloCuda" in cuda
        assert "parse-bbox-func-name=NvDsInferParseYolo\n" in cpu
        assert f"width={profile}" in smoke.render_deepstream_config(profile, "cuda")


def test_dry_plan_never_calls_subprocess_and_stays_blocked_for_auth(
    tmp_path, monkeypatch
):
    root = tmp_path.resolve()
    video = _create_inputs(root)
    session = root / smoke.SESSION_PREFIX / NONCE
    static_pin = {"path": "static.json", "bytes": 1, "sha256": "2" * 64}
    build_pin = {"path": "build.json", "bytes": 1, "sha256": "4" * 64}
    static = {
        "image": {
            "resolved_image_id": IMAGE_ID,
            "labels": {"com.deepsafe.deepstream-yolo.parser-sha256": "3" * 64},
        },
        "runtime_controls": {"pin": {"sha256": "5" * 64}},
    }
    monkeypatch.setattr(
        smoke, "_offline_static_candidate", lambda *args, **kwargs: (static, static_pin)
    )
    monkeypatch.setattr(
        smoke,
        "validate_production_receipt",
        lambda *args, **kwargs: (
            {"schema_version": smoke.PARSER_BUILD_SCHEMA_VERSION},
            build_pin,
        ),
    )
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry plan cannot call subprocess"),
    )
    monkeypatch.setattr(
        smoke.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("dry plan cannot launch subprocess"),
    )
    plan = smoke.build_plan(
        session_root=session,
        image=smoke.DEFAULT_IMAGE,
        gpu_index=0,
        gpu_uuid=GPU_UUID,
        static_candidate_receipt=root / "static.json",
        parser_build_receipt=root / "build.json",
        authorization=None,
        video=video,
        project_root=root,
    )
    assert plan["status"] == "blocked"
    assert plan["blockers"] == ["operator_authorization_missing"]
    assert plan["dry_run"] == {
        "docker_called": False,
        "gpu_process_started": False,
        "gpu_telemetry_queried": False,
    }
    assert plan["contract"]["path"].endswith("inputs/probe-contract.json")
    contract_path = root / plan["contract"]["path"]
    contract = json.loads(contract_path.read_text())
    definition = smoke.validate_contract_semantics(contract, project_root=root)
    assert definition == plan["definition"]
    runs = {item["run_id"]: item for item in contract["runs"]}
    for run_id in ("640-cpu", "960-cpu"):
        assert runs[run_id]["environment"] == {}
        assert runs[run_id]["kernel_marker"] is None
    for run_id in ("640-cuda", "960-cuda"):
        assert runs[run_id]["environment"] == smoke._kernel_proof_environment(
            NONCE, run_id
        )
        assert runs[run_id]["kernel_marker"] == (
            f"{run_id}/{smoke.KERNEL_PROOF_MARKER_NAME}"
        )
    assert contract["gpu_fd_policy"] == smoke._gpu_fd_policy()
    forged = copy.deepcopy(contract)
    forged["parity_thresholds"]["bbox_abs_tolerance_px"] = 999.0
    forged_definition = dict(forged)
    forged_definition.pop("definition_sha256")
    forged["definition_sha256"] = smoke.canonical_sha256(forged_definition)
    with pytest.raises(smoke.Ds9GpuSmokeError, match="closed definition"):
        smoke.validate_contract_semantics(forged, project_root=root)

    forged_fd_policy = copy.deepcopy(contract)
    forged_fd_policy["gpu_fd_policy"]["terminal_teardown"][
        "exit_grace_seconds"
    ] = 0.6
    forged_definition = dict(forged_fd_policy)
    forged_definition.pop("definition_sha256")
    forged_fd_policy["definition_sha256"] = smoke.canonical_sha256(
        forged_definition
    )
    with pytest.raises(smoke.Ds9GpuSmokeError, match="closed definition"):
        smoke.validate_contract_semantics(forged_fd_policy, project_root=root)

    forged_identity = copy.deepcopy(contract)
    forged_identity["container_process_identity"]["uid"] += 1
    forged_definition = dict(forged_identity)
    forged_definition.pop("definition_sha256")
    forged_identity["definition_sha256"] = smoke.canonical_sha256(
        forged_definition
    )
    with pytest.raises(smoke.Ds9GpuSmokeError, match="container process identity"):
        smoke.validate_contract_semantics(forged_identity, project_root=root)


def _authorization(root: Path, *, issued: datetime, expires: datetime) -> tuple[Path, dict]:
    relative = (smoke.SESSION_PREFIX / NONCE).as_posix()
    payload = {
        "schema_version": smoke.AUTH_SCHEMA_VERSION,
        "status": "approved",
        "operator_identity": "test-operator",
        "campaign_nonce": NONCE,
        "session_id": f"ds9-gpu-smoke-{NONCE}",
        "authorized_session_root": relative,
        "issued_at_utc": issued.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": expires.isoformat().replace("+00:00", "Z"),
        "resolved_image_id": IMAGE_ID,
        "smoke_definition_sha256": "2" * 64,
        "static_candidate_receipt_sha256": "3" * 64,
        "parser_production_build_receipt_sha256": "4" * 64,
        "gpu_index": 0,
        "gpu_uuid": GPU_UUID,
        "approved_checks": list(smoke.REQUIRED_CHECKS),
        "single_use": True,
    }
    path = root / "authorization.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.chmod(path, 0o440)
    return path, payload


def _validate_auth(path: Path, root: Path, now: datetime):
    return smoke.validate_authorization(
        path,
        project_root=root,
        expected_nonce=NONCE,
        expected_session_root=(smoke.SESSION_PREFIX / NONCE).as_posix(),
        expected_image_id=IMAGE_ID,
        expected_definition_sha256="2" * 64,
        expected_static_receipt_sha256="3" * 64,
        expected_parser_build_receipt_sha256="4" * 64,
        expected_gpu_index=0,
        expected_gpu_uuid=GPU_UUID,
        now=now,
    )


def test_authorization_is_exact_immutable_and_max_24h(tmp_path):
    root = tmp_path.resolve()
    now = datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc)
    path, _ = _authorization(root, issued=now, expires=now + timedelta(hours=1))
    payload, pin = _validate_auth(path, root, now + timedelta(minutes=1))
    assert payload["single_use"] is True
    assert pin["path"] == "authorization.json"
    os.chmod(path, 0o644)
    with pytest.raises(smoke.Ds9GpuSmokeError, match="mode 0440"):
        _validate_auth(path, root, now + timedelta(minutes=1))


def test_expired_or_overlong_authorization_is_rejected(tmp_path):
    root = tmp_path.resolve()
    now = datetime(2026, 7, 16, 10, 0, tzinfo=timezone.utc)
    expired, _ = _authorization(
        root, issued=now - timedelta(hours=2), expires=now - timedelta(hours=1)
    )
    with pytest.raises(smoke.Ds9GpuSmokeError, match="not current"):
        _validate_auth(expired, root, now)
    expired.unlink()
    overlong, _ = _authorization(
        root, issued=now, expires=now + timedelta(hours=24, seconds=1)
    )
    with pytest.raises(smoke.Ds9GpuSmokeError, match="exceeds 24h"):
        _validate_auth(overlong, root, now + timedelta(minutes=1))


def _kitti_line(*, left=10.0, score=0.9):
    values = [
        0, 0, 0, left, 10, 20, 30, 0, 0, 0, 0, 0, 0, 0, score
    ]
    return "person " + " ".join(str(value) for value in values) + "\n"


def _kernel_marker(run_id: str, pid: int) -> dict:
    return {
        "binary_version": 86,
        "campaign_nonce": NONCE,
        "cuda_device_synchronize": 0,
        "cuda_func_get_attributes": 0,
        "cuda_get_last_error": 0,
        "kernel": smoke.KERNEL_PROOF_KERNEL_NAME,
        "launch_count_at_marker": 1,
        "marker_path": f"/evidence/{run_id}/{smoke.KERNEL_PROOF_MARKER_NAME}",
        "marker_write_count": 1,
        "number_of_blocks": 2,
        "output_size": 1024,
        "pid": pid,
        "ptx_version": 86,
        "run_id": run_id,
        "schema_version": smoke.KERNEL_PROOF_SCHEMA_VERSION,
        "threads_per_block": 1024,
    }


def _fd_sample(
    pid: int,
    sample_index: int,
    *,
    observed: list[str] | None = None,
    read_errors: list[dict] | None = None,
    started_at: int | None = None,
    completed_at: int | None = None,
) -> dict:
    started = started_at if started_at is not None else sample_index * 1_000_000
    completed = completed_at if completed_at is not None else started + 100_000
    return {
        "sample_index": sample_index,
        "pid": pid,
        "proc_fd_root": f"/proc/{pid}/fd",
        "started_at_monotonic_ns": started,
        "completed_at_monotonic_ns": completed,
        "observed_nvidia_device_fds": sorted(observed or []),
        "read_errors": read_errors or [],
    }


def _fd_error(
    pid: int,
    error_number: int,
    observed_at: int,
    *,
    scope: str = "proc_fd_root",
    path: str | None = None,
) -> dict:
    return {
        "scope": scope,
        "path": path or f"/proc/{pid}/fd",
        "errno": error_number,
        "errno_name": smoke.errno.errorcode[error_number],
        "exception_type": (
            smoke.GPU_FD_TERMINAL_EXCEPTION_TYPES.get(error_number, "OSError")
        ),
        "observed_at_monotonic_ns": observed_at,
    }


def _set_terminal_fd_teardown(
    run_data: dict, *, error_number: int, elapsed_ns: int = 100_000_000
) -> None:
    pid = run_data["pid"]
    successful = _fd_sample(
        pid,
        1,
        observed=["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"],
        started_at=1_000_000,
        completed_at=1_100_000,
    )
    error_at = 1_250_000
    root_error = _fd_error(pid, error_number, error_at)
    terminal_sample = _fd_sample(
        pid,
        2,
        read_errors=[root_error],
        started_at=1_200_000,
        completed_at=1_300_000,
    )
    exit_at = error_at + elapsed_ns
    run_data["gpu_fd_sample_count"] = 2
    run_data["gpu_fd_samples"] = [successful, terminal_sample]
    run_data["gpu_fd_read_errors"] = []
    run_data["gpu_fd_terminal_evidence"] = smoke._terminal_fd_evidence(
        sample=terminal_sample,
        root_error=root_error,
        observed_before_error=set(successful["observed_nvidia_device_fds"]),
        exit_observed_at_monotonic_ns=exit_at,
        exit_returncode=0,
    )


def _raw_fixture(root: Path):
    session = root / smoke.SESSION_PREFIX / NONCE
    raw = session / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    engines = {str(profile): str(profile)[0] * 64 for profile in smoke.PROFILES}
    runs = []
    inner_runs = {}
    for run_id in smoke.RUN_IDS:
        profile_text, parser = run_id.split("-", 1)
        profile = int(profile_text)
        runs.append(
            {
                "run_id": run_id,
                "profile": profile,
                "parser": parser,
                "parser_function": smoke.PARSER_FUNCTIONS[parser],
                "argv": [
                    "deepstream-app",
                    "-c",
                    f"/contract/deepstream-{run_id}.txt",
                ],
            }
        )
        run_root = raw / run_id
        kitti = run_root / "kitti"
        kitti.mkdir(parents=True)
        (kitti / "00_000_000000.txt").write_text(_kitti_line(), encoding="utf-8")
        engine_path = f"/models/{profile}.engine"
        (run_root / "deepstream.log").write_text(
            "NvDsInferContext: deserializeEngineAndBackend() ok "
            f"deserialized trt engine from :{engine_path}\n"
            "NvDsInferContext: generateBackendContext() ok "
            f"Use deserialized engine model: {engine_path}\n",
            encoding="utf-8",
        )
        pid = 4200 + len(inner_runs)
        marker = (
            f"{run_id}/{smoke.KERNEL_PROOF_MARKER_NAME}"
            if parser == "cuda"
            else None
        )
        environment = smoke._kernel_proof_environment(NONCE, run_id)
        if parser == "cuda":
            marker_path = run_root / smoke.KERNEL_PROOF_MARKER_NAME
            marker_path.write_text(
                json.dumps(_kernel_marker(run_id, pid), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.chmod(marker_path, 0o440)
        inner_runs[run_id] = {
            "argv": ["deepstream-app", "-c", f"/contract/deepstream-{run_id}.txt"],
            "returncode": 0,
            "pid": pid,
            "log": f"{run_id}/deepstream.log",
            "kitti": f"{run_id}/kitti",
            "nvidia_device_fds_observed": sorted(
                ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"]
            ),
            "gpu_fd_sample_count": 2,
            "gpu_fd_samples": [
                _fd_sample(
                    pid,
                    1,
                    observed=[
                        "/dev/nvidia0",
                        "/dev/nvidiactl",
                        "/dev/nvidia-uvm",
                    ],
                ),
                _fd_sample(pid, 2),
            ],
            "gpu_fd_read_errors": [],
            "gpu_fd_terminal_evidence": None,
            "kernel_marker": marker,
            "kernel_proof_environment": environment,
        }
    contract = {
        "campaign_nonce": NONCE,
        "session_id": f"ds9-gpu-smoke-{NONCE}",
        "resolved_image_id": IMAGE_ID,
        "definition_sha256": "d" * 64,
        "parser": {"sha256": "3" * 64, "static_sm86_cubin_required": True},
        "gpu": {"index": 0, "uuid": GPU_UUID, "compute_capability": "8.6"},
        "inputs": {
            "models": {
                str(profile): {"engine": {"sha256": engines[str(profile)]}}
                for profile in smoke.PROFILES
            }
        },
        "runs": runs,
        "gpu_fd_policy": smoke._gpu_fd_policy(),
        "kernel_proof_policy": smoke._kernel_proof_policy(),
    }
    (raw / "gpu-identity.log").write_text(
        f"0, {GPU_UUID}, RTX A5000 Laptop GPU, 8.6\n", encoding="utf-8"
    )
    inner = {
        "schema_version": smoke.INNER_SCHEMA_VERSION,
        "status": "completed_for_host_replay",
        "created_at_utc": "2026-07-16T10:00:00Z",
        "campaign_nonce": NONCE,
        "session_id": contract["session_id"],
        "resolved_image_id": IMAGE_ID,
        "definition_sha256": contract["definition_sha256"],
        "parser_sha256": "3" * 64,
        "gpu_identity_command": [],
        "gpu_identity_log": "gpu-identity.log",
        "engine_sha256_before": engines,
        "engine_sha256_after": engines,
        "runs": inner_runs,
    }
    (raw / "inner-run.json").write_text(json.dumps(inner) + "\n", encoding="utf-8")
    return session, contract


def test_raw_replay_proves_all_five_checks(tmp_path):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    checks, metrics, artifacts = smoke.replay_raw_session(
        session_root=session, contract=contract, project_root=root
    )
    assert checks == {name: "pass" for name in smoke.REQUIRED_CHECKS}
    assert metrics["cuda_parser"]["gpu_fd_all_deepstream_runs"] is True
    assert metrics["cuda_parser"]["ptx_jit_disabled_for_kernel_proof"] is True
    assert set(artifacts["runs"]) == set(smoke.RUN_IDS)


def test_raw_replay_rejects_engine_fallback_even_with_positive_lines(tmp_path):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    log = session / "raw/640-cuda/deepstream.log"
    log.write_text(log.read_text() + "failed to deserialize; fallback\n", encoding="utf-8")
    checks, _, _ = smoke.replay_raw_session(
        session_root=session, contract=contract, project_root=root
    )
    assert checks["deepstream_640_engine_deserialize_no_fallback"] == "fail"


def test_raw_replay_rejects_cpu_cuda_bbox_drift(tmp_path):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    (session / "raw/960-cpu/kitti/00_000_000000.txt").write_text(
        _kitti_line(left=11.0), encoding="utf-8"
    )
    checks, metrics, _ = smoke.replay_raw_session(
        session_root=session, contract=contract, project_root=root
    )
    assert checks["cpu_cuda_parser_parity_960"] == "fail"
    assert metrics["parity"]["960"]["matched_detections"] == 0


def test_raw_replay_requires_nvidia_fds_for_every_deepstream_process(tmp_path):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    inner_path = session / "raw/inner-run.json"
    inner = json.loads(inner_path.read_text())
    inner["runs"]["640-cpu"]["nvidia_device_fds_observed"] = ["/dev/nvidiactl"]
    inner_path.write_text(json.dumps(inner) + "\n", encoding="utf-8")
    checks, metrics, _ = smoke.replay_raw_session(
        session_root=session, contract=contract, project_root=root
    )
    assert checks["cuda_parser_kernel_launch_sm86"] == "pass"
    assert checks["deepstream_640_engine_deserialize_no_fallback"] == "fail"
    assert metrics["cuda_parser"]["gpu_fd_all_deepstream_runs"] is False


@pytest.mark.parametrize("error_number", [smoke.errno.EACCES, smoke.errno.ENOENT])
def test_raw_replay_accepts_only_proven_terminal_root_fd_teardown(
    tmp_path, error_number
):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    inner_path = session / "raw/inner-run.json"
    inner = json.loads(inner_path.read_text())
    _set_terminal_fd_teardown(
        inner["runs"]["640-cpu"], error_number=error_number
    )
    inner_path.write_text(json.dumps(inner) + "\n", encoding="utf-8")
    checks, metrics, artifacts = smoke.replay_raw_session(
        session_root=session, contract=contract, project_root=root
    )
    assert checks == {name: "pass" for name in smoke.REQUIRED_CHECKS}
    fd_evidence = artifacts["runs"]["640-cpu"]["gpu_fd"]
    assert fd_evidence["status"] == "pass"
    assert fd_evidence["read_errors"] == []
    assert fd_evidence["validation_failures"] == []
    assert fd_evidence["terminal_teardown"]["root_error"]["errno"] == error_number
    assert metrics["cuda_parser"]["gpu_fd_all_deepstream_runs"] is True


@pytest.mark.parametrize(
    "case",
    [
        "required_fd_missing_before_error",
        "prior_read_error",
        "different_pid_root",
        "exit_after_grace",
        "nonzero_exit",
        "root_errno_not_allowed",
    ],
)
def test_raw_replay_fails_closed_for_unproven_terminal_fd_error(tmp_path, case):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    inner_path = session / "raw/inner-run.json"
    inner = json.loads(inner_path.read_text())
    run_data = inner["runs"]["640-cpu"]
    _set_terminal_fd_teardown(run_data, error_number=smoke.errno.EACCES)

    if case == "required_fd_missing_before_error":
        run_data["gpu_fd_samples"][0]["observed_nvidia_device_fds"] = [
            "/dev/nvidia0",
            "/dev/nvidiactl",
        ]
    elif case == "prior_read_error":
        prior_sample = run_data["gpu_fd_samples"][0]
        prior_sample["read_errors"] = [
            _fd_error(
                run_data["pid"],
                smoke.errno.EACCES,
                prior_sample["started_at_monotonic_ns"] + 1,
                scope="fd_entry",
                path=f"/proc/{run_data['pid']}/fd/99",
            )
        ]
        run_data["gpu_fd_read_errors"] = list(prior_sample["read_errors"])
    elif case == "different_pid_root":
        different_root = f"/proc/{run_data['pid'] + 1}/fd"
        terminal_sample = run_data["gpu_fd_samples"][-1]
        terminal_sample["proc_fd_root"] = different_root
        terminal_sample["read_errors"][0]["path"] = different_root
        run_data["gpu_fd_terminal_evidence"]["proc_fd_root"] = different_root
        run_data["gpu_fd_terminal_evidence"]["root_error"]["path"] = different_root
    elif case == "exit_after_grace":
        _set_terminal_fd_teardown(
            run_data,
            error_number=smoke.errno.EACCES,
            elapsed_ns=smoke.GPU_FD_TERMINAL_EXIT_GRACE_NS + 1,
        )
    elif case == "nonzero_exit":
        run_data["gpu_fd_terminal_evidence"]["exit_returncode"] = 1
    elif case == "root_errno_not_allowed":
        _set_terminal_fd_teardown(run_data, error_number=smoke.errno.EPERM)

    inner_path.write_text(json.dumps(inner) + "\n", encoding="utf-8")
    checks, metrics, artifacts = smoke.replay_raw_session(
        session_root=session, contract=contract, project_root=root
    )
    assert checks["deepstream_640_engine_deserialize_no_fallback"] == "fail"
    assert metrics["cuda_parser"]["gpu_fd_all_deepstream_runs"] is False
    fd_evidence = artifacts["runs"]["640-cpu"]["gpu_fd"]
    assert fd_evidence["status"] == "fail"
    assert fd_evidence["validation_failures"]


def test_individual_fd_disappearance_remains_a_benign_snapshot_race(
    monkeypatch,
):
    def disappearing_fd(_path):
        raise FileNotFoundError(smoke.errno.ENOENT, "descriptor closed")

    monkeypatch.setattr(smoke.Path, "iterdir", lambda path: [path / "17"])
    monkeypatch.setattr(smoke.os, "readlink", disappearing_fd)
    sample = smoke._nvidia_fd_sample(4242, 1)
    assert sample["observed_nvidia_device_fds"] == []
    assert sample["read_errors"] == []


@pytest.mark.parametrize("run_id", ["640-cuda", "960-cuda"])
def test_raw_replay_rejects_missing_cuda_kernel_marker(tmp_path, run_id):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    (session / "raw" / run_id / smoke.KERNEL_PROOF_MARKER_NAME).unlink()
    with pytest.raises(smoke.Ds9GpuSmokeError, match="missing/extra artifacts"):
        smoke.replay_raw_session(
            session_root=session, contract=contract, project_root=root
        )


def test_raw_replay_rejects_forged_kernel_marker_binding(tmp_path):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    marker_path = session / "raw/640-cuda" / smoke.KERNEL_PROOF_MARKER_NAME
    os.chmod(marker_path, 0o600)
    marker = json.loads(marker_path.read_text())
    marker["pid"] += 1
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    os.chmod(marker_path, 0o440)
    with pytest.raises(smoke.Ds9GpuSmokeError, match="binding differs"):
        smoke.replay_raw_session(
            session_root=session, contract=contract, project_root=root
        )


def test_raw_replay_requires_compute86_ptx_attribute_with_ptx_jit_disabled(tmp_path):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    marker_path = session / "raw/640-cuda" / smoke.KERNEL_PROOF_MARKER_NAME
    os.chmod(marker_path, 0o600)
    marker = json.loads(marker_path.read_text())
    marker["ptx_version"] = 75
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    os.chmod(marker_path, 0o440)
    with pytest.raises(smoke.Ds9GpuSmokeError, match="launch facts differ"):
        smoke.replay_raw_session(
            session_root=session, contract=contract, project_root=root
        )


def test_raw_replay_rejects_duplicate_marker_json_keys(tmp_path):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    marker_path = session / "raw/640-cuda" / smoke.KERNEL_PROOF_MARKER_NAME
    os.chmod(marker_path, 0o600)
    payload = json.dumps(_kernel_marker("640-cuda", 4200))
    marker_path.write_text(payload[:-1] + ', "pid": 4200}\n', encoding="utf-8")
    os.chmod(marker_path, 0o440)
    with pytest.raises(smoke.Ds9GpuSmokeError, match="duplicate JSON key"):
        smoke.replay_raw_session(
            session_root=session, contract=contract, project_root=root
        )


def test_raw_replay_rejects_cpu_marker_and_extra_marker(tmp_path):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    extra = session / "raw/640-cpu" / smoke.KERNEL_PROOF_MARKER_NAME
    extra.write_text(json.dumps(_kernel_marker("640-cuda", 4201)) + "\n")
    os.chmod(extra, 0o440)
    with pytest.raises(smoke.Ds9GpuSmokeError, match="missing/extra artifacts"):
        smoke.replay_raw_session(
            session_root=session, contract=contract, project_root=root
        )


def test_generic_gpu_signals_cannot_replace_kernel_marker(tmp_path):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    marker_path = session / "raw/960-cuda" / smoke.KERNEL_PROOF_MARKER_NAME
    marker_path.unlink()
    # Detections, parser function, clean CUDA logs and all NVIDIA FDs remain.
    with pytest.raises(smoke.Ds9GpuSmokeError, match="missing/extra artifacts"):
        smoke.replay_raw_session(
            session_root=session, contract=contract, project_root=root
        )


def test_raw_replay_rejects_oversized_deepstream_log(tmp_path):
    root = tmp_path.resolve()
    session, contract = _raw_fixture(root)
    log_path = session / "raw/640-cpu/deepstream.log"
    log_path.write_bytes(b"x" * (smoke.MAX_RAW_LOG_BYTES + 1))
    with pytest.raises(smoke.Ds9GpuSmokeError, match="exceeds its byte limit"):
        smoke.replay_raw_session(
            session_root=session, contract=contract, project_root=root
        )


def test_cpu_runs_have_no_kernel_proof_environment_in_closed_contract():
    for run_id in ("640-cpu", "960-cpu"):
        assert smoke._kernel_proof_environment(NONCE, run_id) == {}
    for run_id in ("640-cuda", "960-cuda"):
        environment = smoke._kernel_proof_environment(NONCE, run_id)
        assert environment[smoke.KERNEL_PROOF_ENV["enable"]] == "1"
        assert environment[smoke.KERNEL_PROOF_ENV["run_id"]] == run_id
        assert environment[smoke.KERNEL_PROOF_ENV["disable_ptx_jit"]] == "1"


def test_kernel_marker_matches_published_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (
            smoke.PROJECT_ROOT
            / "validation/schemas/ds9-cuda-kernel-proof-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(
        _kernel_marker("640-cuda", 4242)
    )


def test_smoke_docker_command_uses_exact_id_never_mutable_tag(tmp_path):
    root = tmp_path.resolve()
    video = _create_inputs(root)
    session = root / smoke.SESSION_PREFIX / NONCE
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    (session / "raw").mkdir()
    for run_id in smoke.RUN_IDS:
        (inputs / f"infer-{run_id}.txt").write_text("fixture\n", encoding="utf-8")
        (inputs / f"deepstream-{run_id}.txt").write_text("fixture\n", encoding="utf-8")
    (inputs / "probe-contract.json").write_text("{}\n", encoding="utf-8")
    plan = {
        "campaign_nonce": NONCE,
        "session_root": (smoke.SESSION_PREFIX / NONCE).as_posix(),
        "requested_image": smoke.DEFAULT_IMAGE,
        "resolved_image_id": IMAGE_ID,
        "gpu": {"index": 0},
        "container_process_identity": smoke.container_process_identity(),
    }
    for item in inputs.iterdir():
        os.chmod(item, 0o440)
    command, _ = smoke.build_docker_command(
        plan=plan, project_root=root, video=video
    )
    assert command.count(IMAGE_ID) == 1
    assert smoke.DEFAULT_IMAGE not in command
    assert "--pull=never" in command
    user_index = command.index("--user")
    assert command[user_index + 1] == f"{os.geteuid()}:{os.getegid()}"


def test_smoke_docker_command_rejects_sealed_input_owner_mismatch(tmp_path):
    root = tmp_path.resolve()
    video = _create_inputs(root)
    session = root / smoke.SESSION_PREFIX / NONCE
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    (session / "raw").mkdir()
    for run_id in smoke.RUN_IDS:
        (inputs / f"infer-{run_id}.txt").write_text("fixture\n", encoding="utf-8")
        (inputs / f"deepstream-{run_id}.txt").write_text("fixture\n", encoding="utf-8")
    (inputs / "probe-contract.json").write_text("{}\n", encoding="utf-8")
    for item in inputs.iterdir():
        os.chmod(item, 0o440)
    os.chmod(inputs / "probe-contract.json", 0o444)
    plan = {
        "campaign_nonce": NONCE,
        "session_root": (smoke.SESSION_PREFIX / NONCE).as_posix(),
        "requested_image": smoke.DEFAULT_IMAGE,
        "resolved_image_id": IMAGE_ID,
        "gpu": {"index": 0},
        "container_process_identity": smoke.container_process_identity(),
    }
    with pytest.raises(smoke.Ds9GpuSmokeError, match="ownership/mode differs"):
        smoke.build_docker_command(plan=plan, project_root=root, video=video)


def test_five_pass_strings_without_raw_evidence_are_rejected(tmp_path):
    root = tmp_path.resolve()
    for name, relative in gate.RUNTIME_CONTROL_PATHS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"control={name}\n", encoding="utf-8")
    pins = gate.runtime_control_pins(root)
    manifest_path = root / gate.RUNTIME_CONTROL_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": gate.CONTROL_MANIFEST_SCHEMA_VERSION,
                "artifacts": pins,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    controls = gate.validate_runtime_control_manifest(root)
    evidence_path = root / smoke.SESSION_PREFIX / NONCE / "gpu-smoke-evidence.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": smoke.EVIDENCE_SCHEMA_VERSION,
                "status": "pass",
                "resolved_image_id": IMAGE_ID,
                "runtime_control_manifest_sha256": controls["pin"]["sha256"],
                "checks": {name: "pass" for name in smoke.REQUIRED_CHECKS},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(evidence_path, 0o440)
    with pytest.raises(gate.Ds9CompatibilityError, match="raw replay failed"):
        gate._load_gpu_smoke_evidence(
            evidence_path,
            project_root=root,
            resolved_image_id=IMAGE_ID,
            controls=controls,
            expected_parser_sha256="3" * 64,
        )
