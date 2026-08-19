from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

from validation import gpu_lease as v1
from validation import gpu_lease_v2 as v2
from validation import gpu_lease_v3 as lease


ROOT = Path(__file__).resolve().parents[1]
BOOT_ID = v1._read_boot_id()
GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"


def handshake(
    run_id: str = "1" * 64,
    scope: str = "2" * 64,
    contract: str = "3" * 64,
    requested: str = "4" * 64,
    label: str = "5" * 64,
) -> dict[str, str]:
    return {
        "DEEPSAFE_GPU_LEASE_V3_SCHEMA": lease.CONTAINER_HANDSHAKE_SCHEMA,
        "DEEPSAFE_GPU_LEASE_V3_RUN_ID": run_id,
        "DEEPSAFE_GPU_LEASE_V3_CONTRACT_SHA256": contract,
        "DEEPSAFE_GPU_LEASE_V3_REQUESTED_ARGV_SHA256": requested,
        "DEEPSAFE_GPU_LEASE_V3_SCOPE_IDENTITY_SHA256": scope,
        "DEEPSAFE_GPU_LEASE_V3_LABEL_VALUE_SHA256": label,
    }


def pending_state(journal: lease.V3Journal, handle: lease.WatchdogHandle, gpu: int = 3) -> dict[str, Any]:
    requested = [str(lease.TRUSTED_DOCKER_CLI), "run", "image:test"]
    return lease._state_core(
        contract_fingerprint=journal.contract_fingerprint,
        phase="pending",
        gpu_index=gpu,
        run_id="1" * 64,
        owner=lease._owner_identity(),
        scope=handle.scope,
        requested=requested,
        effective=requested,
        inner_root=journal.root / "inner-v2",
        label_hash="5" * 64,
        inner_lease_id=None,
        daemon=None,
        container_handshake=handshake(
            scope=handle.scope["identity_sha256"],
            contract=journal.contract_fingerprint,
            requested=lease.command_argv_sha256(requested),
        ),
        created_at=journal.wall_now(),
        created_ns=journal.monotonic_ns(),
        last_event=None,
    )


def test_exact_direct_command_is_normalized() -> None:
    assert lease.normalize_direct_docker_run(["docker", "run", "--gpus=all", "image:x"]) == [
        "/usr/bin/docker", "run", "--gpus=all", "image:x"
    ]
    exact = ["/usr/bin/docker", "run", "image:x"]
    assert lease.normalize_direct_docker_run(exact) == exact


@pytest.mark.parametrize(
    "argv",
    [
        ["sh", "-c", "docker run image:x"],
        [sys.executable, "wrapper.py", "docker", "run", "image:x"],
        ["docker", "--host=unix:///tmp/x", "run", "image:x"],
        ["docker", "--context", "evil", "run", "image:x"],
        ["docker", "run"],
        ["docker", "run", "--rm", "image:x"],
        ["docker", "run", "--rm=true", "image:x"],
        ["docker", "run", "-d", "image:x"],
        ["docker", "run", "-itd", "image:x"],
        ["docker", "run", "--detach", "image:x"],
        ["docker", "run", "--restart=always", "image:x"],
        ["docker", "run", "--label-file=x", "image:x"],
        ["docker", "run", "--env-file=x", "image:x"],
        ["docker", "run", "--env", "DEEPSAFE_GPU_LEASE_V3_RUN_ID=bad", "image:x"],
        ["docker", "run", "-eDEEPSAFE_GPU_LEASE_V3_RUN_ID=bad", "image:x"],
        ["docker", "run", "--cidfile=/tmp/x", "image:x"],
        ["docker", "run", "--managed-docker-cidfile=/tmp/x", "image:x"],
        ["docker", "run", f"--label={lease.DOCKER_LABEL_KEY}=bad", "image:x"],
        ["docker", "run", f"-l{lease.DOCKER_LABEL_KEY}=bad", "image:x"],
        ["podman", "run", "image:x"],
    ],
)
def test_wrapper_global_nested_and_unsafe_forms_are_rejected(argv: list[str]) -> None:
    with pytest.raises(lease.LeaseIntegrityError):
        lease.normalize_direct_docker_run(argv)


def test_public_parser_has_no_managed_cidfile_compatibility_lane() -> None:
    with pytest.raises(SystemExit):
        lease.build_parser().parse_args([
            "run", "--managed-docker-cidfile", "/tmp/x",
            "--gpu-index", "0", "--owner-kind", "legacy_validation",
        ])


def test_effective_command_has_one_owned_label_cidfile_and_handshake(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    requested, effective, intent = lease._command_with_v2_authority(
        ["/usr/bin/docker", "run", "image:x"],
        cidfile=private / "docker.cid",
        label_value="a" * 64,
        container_handshake=handshake(),
    )
    assert requested[:3] == ["/usr/bin/docker", "run", f"--cidfile={private / 'docker.cid'}"]
    assert sum(item.startswith("--env=DEEPSAFE_GPU_LEASE_V3_") for item in requested) == 6
    assert effective[:3] == [
        "/usr/bin/docker", "run", f"--label={lease.DOCKER_LABEL_KEY}={'a' * 64}"
    ]
    assert intent.mode == "direct"


def test_gate_closed_parent_eof_exits_without_workload() -> None:
    handle = lease._spawn_test_watchdog()
    handle.close_parent_channel()
    handle.process.wait(timeout=3)
    assert handle.process.returncode == 0


def test_parent_eof_after_gate_cancels_action_beyond_daemon_quiet_window(tmp_path: Path) -> None:
    marker = tmp_path / "late-action"
    handle = lease._spawn_test_watchdog()
    handle.open_gate(
        [
            sys.executable,
            "-c",
            f"import pathlib,time;time.sleep(2.5);pathlib.Path({str(marker)!r}).write_text('bad')",
        ],
        test_only=True,
    )
    handle.close_parent_channel()
    handle.process.wait(timeout=5)
    time.sleep(2.7)
    assert not marker.exists()


def test_normal_completion_and_scope_absence() -> None:
    handle = lease._spawn_test_watchdog()
    handle.open_gate(["/bin/true"], test_only=True)
    assert handle.wait_result(5)["returncode"] == 0
    evidence = lease.terminate_and_prove_scope_absent(handle.scope)
    assert evidence["verified_absent"] is True
    assert evidence["empty_queries"] == 20


def test_parent_stop_terminates_long_running_workload(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    handle = lease._spawn_test_watchdog()
    handle.open_gate(
        [sys.executable, "-c", f"import time,pathlib;time.sleep(2);pathlib.Path({str(marker)!r}).touch()"],
        test_only=True,
    )
    handle.stop(signal.SIGTERM)
    result = handle.wait_result(5)
    assert result["reason"] == "parent_stop"
    time.sleep(2.1)
    assert not marker.exists()


def test_watchdog_normal_signal_terminates_scope(tmp_path: Path) -> None:
    marker = tmp_path / "signal-late"
    handle = lease._spawn_test_watchdog()
    handle.open_gate(
        [sys.executable, "-c", f"import pathlib,time;time.sleep(2);pathlib.Path({str(marker)!r}).touch()"],
        test_only=True,
    )
    started = lease._recv_control(handle.control, 2)
    assert started is not None and started["op"] == "STARTED"
    os.kill(handle.process.pid, signal.SIGTERM)
    result = handle.wait_result(5)
    assert result["reason"] == "watchdog_signal"
    time.sleep(2.1)
    assert not marker.exists()


@pytest.mark.parametrize(
    "stage",
    ["after_popen", "after_identity", "before_seal", "after_seal_before_gate", "after_gate"],
)
def test_real_parent_sigkill_at_launch_stages_never_allows_delayed_action(
    stage: str, tmp_path: Path
) -> None:
    marker = tmp_path / f"late-{stage}"
    script = f"""
import os,signal,sys,time
from pathlib import Path
from validation import gpu_lease_v3 as v3
stage={stage!r}; marker={str(marker)!r}
def kill(_name=None): os.kill(os.getpid(),signal.SIGKILL)
if stage=='after_popen':
    v3._spawn_test_watchdog(hook=lambda name: kill() if name=='after_popen' else None)
h=v3._spawn_test_watchdog()
if stage=='after_identity': kill()
if stage=='before_seal':
    p=Path({str(tmp_path / 'before-seal')!r}); p.write_text('pending'); fd=os.open(p,os.O_RDONLY); os.fsync(fd); os.close(fd); kill()
if stage=='after_seal_before_gate':
    p=Path({str(tmp_path / 'sealed')!r}); p.write_text('sealed'); fd=os.open(p,os.O_RDONLY); os.fsync(fd); os.close(fd); kill()
h.open_gate([sys.executable,'-c',f'import pathlib,time;time.sleep(2.5);pathlib.Path({{marker!r}}).write_text("bad")'],test_only=True)
if stage=='after_gate': kill()
"""
    child = subprocess.Popen([sys.executable, "-c", script], cwd=ROOT)
    child.wait(timeout=5)
    assert child.returncode == -signal.SIGKILL
    time.sleep(2.8)
    assert not marker.exists()


def test_leader_gone_descendant_live_is_found_and_killed(tmp_path: Path) -> None:
    token = "9" * 64
    child_pid_file = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time,pathlib;"
        f"p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid));"
        "time.sleep(.5)"
    )
    environment = dict(os.environ)
    environment[lease.SCOPE_ENV] = token
    leader = subprocess.Popen(
        [sys.executable, "-c", script], env=environment, start_new_session=True
    )
    scope = lease._capture_scope_identity(leader.pid, token)
    leader.wait(timeout=3)
    child_pid = int(child_pid_file.read_text())
    assert lease._read_proc_stat(child_pid) is not None
    evidence = lease.terminate_and_prove_scope_absent(scope)
    assert evidence["leader_status"] == "gone"
    assert child_pid in evidence["observed_pids"]
    assert evidence["verified_absent"] is True


def test_real_pid_pgid_reuse_collision_fails_closed() -> None:
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
    try:
        process = v1._read_process_identity(unrelated.pid)
        record = lease._read_proc_stat(unrelated.pid)
        assert process is not None and record is not None
        cgroup, ns_dev, ns_ino = lease._read_cgroup_identity(unrelated.pid)
        base = {
            "scope_id": "8" * 64,
            "pid": unrelated.pid,
            "uid": process["uid"],
            "boot_id": process["boot_id"],
            "start_ticks": process["start_ticks"],
            "session_id": unrelated.pid,
            "process_group": unrelated.pid,
            "cgroup_path": cgroup,
            "cgroup_ns_dev": ns_dev,
            "cgroup_ns_ino": ns_ino,
        }
        scope = {**base, "identity_sha256": lease.canonical_sha256(base)}
        with pytest.raises(lease.LeaseIntegrityError, match="reused"):
            lease.terminate_and_prove_scope_absent(scope)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_journal_pending_state_and_recovery_receipt_are_fsynced_and_replayed(tmp_path: Path) -> None:
    handle = lease._spawn_test_watchdog()
    journal = lease.V3Journal(tmp_path / "v3", contract_fingerprint="a" * 64)
    try:
        with journal.lock(3) as paths:
            state = journal.replace_state(paths, pending_state(journal, handle))
            assert journal.read_state(paths, 3) == state
        handle.close_parent_channel()
        handle.process.wait(timeout=3)
        cleanup = lease.terminate_and_prove_scope_absent(handle.scope)
        with journal.lock(3) as paths:
            event = lease._event(
                journal=journal,
                event_type="pending_recovery",
                state_core_sha256=state["state_fingerprint_sha256"],
                state=state,
                previous_fingerprint=state["state_fingerprint_sha256"],
                host_cleanup=cleanup,
                docker_evidence=None,
            )
            pin = journal.write_receipt(paths, event)
            assert pin["path"].startswith("receipts/")
            journal.remove_state(paths, state)
            assert journal.read_state(paths, 3) is None
    finally:
        handle.close_parent_channel()
        if handle.process.poll() is None:
            handle.process.terminate(); handle.process.wait(timeout=3)


def test_stale_recovery_orders_host_absence_before_daemon_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = lease._spawn_test_watchdog()
    journal = lease.V3Journal(tmp_path / "v3", contract_fingerprint="a" * 64)
    with journal.lock(4) as paths:
        state = journal.replace_state(paths, pending_state(journal, handle, gpu=4))
    handle.close_parent_channel(); handle.process.wait(timeout=3)
    order: list[str] = []
    real_terminate = lease.terminate_and_prove_scope_absent

    def host(*args: Any, **kwargs: Any) -> dict[str, Any]:
        order.append("host")
        return real_terminate(*args, **kwargs)

    class Inner:
        def recover_stale_supervised(self, **_kwargs: Any) -> dict[str, Any]:
            order.append("daemon")
            return {"status": "fake-v2-recovered"}

    monkeypatch.setattr(lease, "terminate_and_prove_scope_absent", host)
    monkeypatch.setattr(lease, "_inner_has_state", lambda *_args: True)
    result = lease.recover_stale_v3(
        gpu_index=4,
        journal=journal,
        inner_manager=Inner(),  # type: ignore[arg-type]
        backend=object(),  # type: ignore[arg-type]
        force_stale_for_test=True,
    )
    assert order == ["host", "daemon"]
    assert result["host_scope_cleanup"]["verified_absent"] is True
    with journal.lock(4) as paths:
        assert journal.read_state(paths, 4) is None


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds

    def wall_now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=self.value)


class FakeBackend:
    def __init__(self, late_on_call: int | None = 7) -> None:
        self.daemon = {
            "daemon_id": "daemon-v3-test",
            "name": "v3-test-daemon",
            "docker_root_dir": "/var/lib/docker",
            "server_version": "test",
            "docker_cli": {"path": "/usr/bin/docker", "bytes": 1, "sha256": "0" * 64},
            "environment_sha256": v2._docker_environment_sha256(v2._sanitized_runtime_environment()),
            "context_sha256": "d" * 64,
        }
        self.containers: dict[str, dict[str, Any]] = {}
        self.calls = 0
        self.removed: list[str] = []
        self.late_on_call = late_on_call

    def add(self, cid: str, token: str) -> None:
        self.containers[cid] = {
            "Id": cid,
            "Config": {"Labels": {v2.DOCKER_LABEL_KEY: token}},
        }

    def identity(self) -> dict[str, Any]:
        return copy.deepcopy(self.daemon)

    def list_ids(self, key: str, value: str) -> Sequence[str]:
        self.calls += 1
        if self.late_on_call is not None and self.calls == self.late_on_call:
            cid = "b" * 64
            self.add(cid, value)
        return [cid for cid, item in self.containers.items() if item["Config"]["Labels"].get(key) == value]

    def list_ids_by_label_key(self, key: str) -> Sequence[str]:
        return [cid for cid, item in self.containers.items() if key in item["Config"]["Labels"]]

    def inspect(self, cid: str) -> dict[str, Any]:
        return copy.deepcopy(self.containers[cid])

    def remove(self, cid: str) -> v2.DockerResult:
        self.removed.append(cid); self.containers.pop(cid, None)
        return v2.DockerResult(0, cid.encode(), b"")


def test_fake_backend_late_id_after_initial_empty_queries_is_removed(tmp_path: Path) -> None:
    private = tmp_path / "private"; private.mkdir(mode=0o700)
    specification = v1._validate_docker_cidfile_parent(
        v1._DockerCidfile("docker", private / "docker.cid")
    )
    intent = v2.DockerSupervisionIntent(
        "direct", "1" * 64, specification,
        docker_cli_sha256="0" * 64,
        docker_environment_sha256=v2._docker_environment_sha256(v2._sanitized_runtime_environment()),
    )
    backend = FakeBackend(); clock = Clock()
    result = v2.cleanup_docker_supervision(
        backend=backend,
        intent=intent,
        requested_command=["/usr/bin/docker", "run", f"--cidfile={specification.path}", "image:x"],
        effective_command=[
            "/usr/bin/docker", "run", f"--label={v2.DOCKER_LABEL_KEY}={'1' * 64}",
            f"--cidfile={specification.path}", "image:x",
        ],
        daemon=backend.daemon,
        prelaunch_ids=[],
        child={"outcome": "success", "returncode": 0, "stop_reason": None},
        monotonic=clock.monotonic,
        wall_now=clock.wall_now,
        sleeper=clock.sleep,
    )
    assert backend.removed == ["b" * 64]
    assert result["verified_absent"] is True
    assert result["quiet_window"]["resets"] >= 1


@pytest.mark.parametrize("mode", ["success", "timeout"])
def test_full_cpu_fake_run_seals_gate_proves_scope_then_releases_v2_and_v3(
    mode: str, tmp_path: Path
) -> None:
    contract = lease.load_contract()
    journal = lease.V3Journal(
        tmp_path / "v3", contract_fingerprint=contract["contract_fingerprint_sha256"]
    )
    manager = lease.InnerV2Manager(
        root=journal.root / "inner-v2",
        gpu_uuid_resolver=lambda _index: GPU_UUID,
        nvidia_smi_sha256="0" * 64,
    )
    canonical_paths = manager._paths

    def isolated(gpu_index: int) -> dict[str, Path]:
        paths = canonical_paths(gpu_index)
        paths["legacy"] = tmp_path / f"legacy-{gpu_index}.lock"
        return paths

    manager._paths = isolated  # type: ignore[method-assign]
    backend = FakeBackend(late_on_call=None)

    class Proxy:
        def __init__(self) -> None:
            self.inner = lease._spawn_test_watchdog()
            self.process = self.inner.process
            self.control = self.inner.control
            self.scope = self.inner.scope

        def open_gate(
            self, argv: Sequence[str], *, child_environment: dict[str, str] | None = None
        ) -> None:
            label = next(item for item in argv if item.startswith(f"--label={v2.DOCKER_LABEL_KEY}="))
            token = label.rsplit("=", 1)[1]
            cidfile = Path(next(item for item in argv if item.startswith("--cidfile=")).split("=", 1)[1])
            cid = "c" * 64
            backend.add(cid, token)
            cidfile.write_text(cid + "\n", encoding="ascii")
            command = ["/bin/true"] if mode == "success" else ["/bin/sleep", "30"]
            self.inner.open_gate(command, test_only=True)

        def stop(self, signum: int = signal.SIGTERM) -> None:
            self.inner.stop(signum)

        def close_parent_channel(self) -> None:
            self.inner.close_parent_channel()

    args = argparse.Namespace(
        command_argv=["--", "/usr/bin/docker", "run", "image:test"],
        managed_docker_cidfile=None,
        timeout_seconds=5.0 if mode == "success" else 0.15,
        expected_docker_cli_sha256="0" * 64,
        gpu_index=251,
        owner_kind="legacy_validation",
        ttl_seconds=5,
    )
    order: list[str] = []
    call = lambda: lease._run_v3(
        args, _journal=journal, _manager=manager, _backend=backend,
        _watchdog_factory=Proxy,  # type: ignore[arg-type]
        _hooks={
            "after_seal_before_gate": lambda: order.append("sealed"),
            "after_gate": lambda: order.append("gate"),
            "host_scope_absent_before_docker_cleanup": lambda: order.append("host_absent"),
            "inner_released": lambda: order.append("v2_released"),
        },
    )
    if mode == "timeout":
        with pytest.raises(lease.LeaseIntegrityError, match="exceeded timeout"):
            call()
    else:
        assert call() == 0
    assert order == ["sealed", "gate", "host_absent", "v2_released"]
    assert backend.removed == ["c" * 64]
    with journal.lock(251) as paths:
        assert journal.read_state(paths, 251) is None
    assert not lease._inner_has_state(manager, 251)


@pytest.mark.parametrize(
    "stage",
    [
        "after_pending_state",
        "after_inner_acquire_before_seal",
        "after_seal_before_gate",
        "after_gate",
    ],
)
def test_actual_v3_fake_runner_parent_sigkill_at_sealed_gate_boundaries(
    stage: str, tmp_path: Path
) -> None:
    marker = tmp_path / f"actual-{stage}-late"
    pid = os.fork()
    if pid == 0:
        try:
            contract = lease.load_contract()
            journal = lease.V3Journal(
                tmp_path / f"run-{stage}",
                contract_fingerprint=contract["contract_fingerprint_sha256"],
            )
            manager = lease.InnerV2Manager(
                root=journal.root / "inner-v2",
                gpu_uuid_resolver=lambda _index: GPU_UUID,
                nvidia_smi_sha256="0" * 64,
            )
            canonical_paths = manager._paths

            def isolated(gpu_index: int) -> dict[str, Path]:
                paths = canonical_paths(gpu_index)
                paths["legacy"] = tmp_path / f"actual-legacy-{stage}-{gpu_index}.lock"
                return paths

            manager._paths = isolated  # type: ignore[method-assign]
            backend = FakeBackend(late_on_call=None)

            class Proxy:
                def __init__(self) -> None:
                    self.inner = lease._spawn_test_watchdog()
                    self.process = self.inner.process; self.control = self.inner.control; self.scope = self.inner.scope

                def open_gate(self, argv: Sequence[str], **_kwargs: Any) -> None:
                    label = next(x for x in argv if x.startswith(f"--label={v2.DOCKER_LABEL_KEY}="))
                    cidfile = Path(next(x for x in argv if x.startswith("--cidfile=")).split("=", 1)[1])
                    cid = "d" * 64; backend.add(cid, label.rsplit("=", 1)[1]); cidfile.write_text(cid + "\n")
                    self.inner.open_gate(
                        [sys.executable, "-c", f"import pathlib,time;time.sleep(2.5);pathlib.Path({str(marker)!r}).touch()"],
                        test_only=True,
                    )

                def stop(self, signum: int = signal.SIGTERM) -> None: self.inner.stop(signum)
                def close_parent_channel(self) -> None: self.inner.close_parent_channel()

            args = argparse.Namespace(
                command_argv=["--", "/usr/bin/docker", "run", "image:test"],
                managed_docker_cidfile=None,
                timeout_seconds=10.0,
                expected_docker_cli_sha256="0" * 64,
                gpu_index=250,
                owner_kind="legacy_validation",
                ttl_seconds=5,
            )
            lease._run_v3(
                args,
                _journal=journal,
                _manager=manager,
                _backend=backend,
                _watchdog_factory=Proxy,  # type: ignore[arg-type]
                _hooks={stage: lambda: os.kill(os.getpid(), signal.SIGKILL)},
            )
        finally:
            os._exit(99)
    waited, status = os.waitpid(pid, 0)
    assert waited == pid and os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
    time.sleep(2.9)
    assert not marker.exists()


def test_contract_and_held_status_require_frozen_artifacts() -> None:
    contract = lease.load_contract()
    assert lease.GPU_LEASE_V3_API_READY is False
    assert lease.LIVE_PLAN_AUTHORIZED is False
    assert contract["command_policy"]["unsupervised_lane"] is False
    assert contract["command_policy"]["nested_lane"] is False
    assert contract["implementation"] == lease._artifact_pin(ROOT / "validation/gpu_lease_v3.py")


def test_held_bundle_status_and_cpu_lifecycle_are_pathless_and_gpu_free(tmp_path: Path) -> None:
    contract = lease.load_contract()
    with lease.open_held_execution_bundle(
        expected_contract_fingerprint=contract["contract_fingerprint_sha256"],
        expected_source_sha256=contract["implementation"]["sha256"],
        expected_python_sha256=contract["python_interpreter"]["sha256"],
    ) as bundle:
        status = bundle.run(["held-status"], timeout=10, check=True)
        value = json.loads(status.stdout)
        assert value["direct_docker_only"] is True
        assert value["unsupervised_lane"] is False
        assert value["gpu_lease_v3_api_ready"] is False
        assert value["live_plan_authorized"] is False
        lifecycle = bundle.run(
            ["held-cpu-lifecycle", "--state-root", str(tmp_path / "held-state"), "--gpu-index", "254"],
            timeout=20,
            check=True,
        )
        result = json.loads(lifecycle.stdout)
        assert result["status"] == "held_cpu_lifecycle_passed"
        assert result["watchdog_returncode"] == 0
        assert result["host_scope_verified_absent"] is True
