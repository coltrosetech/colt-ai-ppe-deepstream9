from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import pytest

from validation import gpu_lease as v1
from validation import gpu_lease_v2 as v2
from validation import gpu_lease_v4 as lease


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"


class FakeBackend:
    def __init__(self) -> None:
        self.daemon = {
            "daemon_id": "daemon-v4-test",
            "name": "v4-test-daemon",
            "docker_root_dir": "/var/lib/docker",
            "server_version": "test",
            "docker_cli": {
                "path": "/usr/bin/docker",
                "bytes": 1,
                "sha256": "0" * 64,
            },
            "environment_sha256": v2._docker_environment_sha256(
                v2._sanitized_runtime_environment()
            ),
            "context_sha256": "d" * 64,
        }
        self.containers: dict[str, dict[str, Any]] = {}
        self.removed: list[str] = []
        self.outage = False
        self.identity_drift = False

    def add(self, cid: str, token: str) -> None:
        self.containers[cid] = {
            "Id": cid,
            "Config": {"Labels": {v2.DOCKER_LABEL_KEY: token}},
        }

    def identity(self) -> dict[str, Any]:
        if self.outage:
            raise OSError("fake daemon outage")
        value = copy.deepcopy(self.daemon)
        if self.identity_drift:
            value["context_sha256"] = "e" * 64
        return value

    def list_ids(self, key: str, value: str) -> Sequence[str]:
        if self.outage:
            raise OSError("fake daemon outage")
        return [
            cid
            for cid, item in self.containers.items()
            if item["Config"]["Labels"].get(key) == value
        ]

    def list_ids_by_label_key(self, key: str) -> Sequence[str]:
        if self.outage:
            raise OSError("fake daemon outage")
        return [
            cid for cid, item in self.containers.items()
            if key in item["Config"]["Labels"]
        ]

    def inspect(self, cid: str) -> dict[str, Any]:
        if self.outage:
            raise OSError("fake daemon outage")
        return copy.deepcopy(self.containers[cid])

    def remove(self, cid: str) -> v2.DockerResult:
        if self.outage:
            raise OSError("fake daemon outage")
        self.removed.append(cid)
        self.containers.pop(cid, None)
        return v2.DockerResult(0, cid.encode(), b"")


def isolated_manager(tmp_path: Path, journal: lease.V4Journal) -> lease.InnerV2Manager:
    manager = lease.InnerV2Manager(
        root=journal.root / "inner-v2",
        gpu_uuid_resolver=lambda _index: GPU_UUID,
        nvidia_smi_sha256="0" * 64,
    )
    original = manager._paths

    def paths(gpu_index: int) -> dict[str, Path]:
        value = original(gpu_index)
        value["legacy"] = tmp_path / f"legacy-{gpu_index}.lock"
        return value

    manager._paths = paths  # type: ignore[method-assign]
    return manager


class Proxy:
    def __init__(self, backend: FakeBackend, mode: str) -> None:
        self.backend = backend
        self.mode = mode
        self.inner = lease._spawn_test_watchdog()
        self.process = self.inner.process
        self.control = self.inner.control
        self.scope = self.inner.scope

    def open_gate(
        self, argv: Sequence[str], *, child_environment: dict[str, str] | None = None
    ) -> None:
        label = next(
            item for item in argv
            if item.startswith(f"--label={v2.DOCKER_LABEL_KEY}=")
        )
        token = label.rsplit("=", 1)[1]
        cidfile = Path(
            next(item for item in argv if item.startswith("--cidfile=")).split("=", 1)[1]
        )
        cid = "c" * 64
        self.backend.add(cid, token)
        cidfile.write_text(cid + "\n", encoding="ascii")
        commands = {
            "success": ["/bin/true"],
            "nonzero": [sys.executable, "-c", "raise SystemExit(7)"],
            "timeout": ["/bin/sleep", "30"],
            "signal": ["/bin/sleep", "30"],
            "popen_failure": ["/definitely/not/a/real/executable"],
        }
        self.inner.open_gate(commands[self.mode], test_only=True)

    def stop(self, signum: int = signal.SIGTERM) -> None:
        self.inner.stop(signum)

    def close_parent_channel(self) -> None:
        self.inner.close_parent_channel()


def run_fixture(
    tmp_path: Path,
    mode: str,
    *,
    after_gate: Any = None,
    workload_started: Any = None,
    retry_budget: int | None = None,
) -> tuple[Any, lease.V4Journal, lease.InnerV2Manager, FakeBackend, dict[str, Any]]:
    contract = lease.load_contract()
    journal = lease.V4Journal(
        tmp_path / "v4", contract_fingerprint=contract["contract_fingerprint_sha256"]
    )
    manager = isolated_manager(tmp_path, journal)
    backend = FakeBackend()
    captured: dict[str, Any] = {}

    def factory() -> Proxy:
        return Proxy(backend, mode)

    hooks: dict[str, Any] = {
        "after_seal_before_gate": lambda: captured.update(
            state=journal.read_state(journal.prepare(251), 251)
        ),
    }
    if after_gate is not None:
        hooks["after_gate"] = lambda: after_gate(backend)
    if workload_started is not None:
        hooks["workload_started"] = workload_started
    args = argparse.Namespace(
        command_argv=["--", "/usr/bin/docker", "run", "image:test"],
        managed_docker_cidfile=None,
        timeout_seconds=0.12 if mode == "timeout" else 5.0,
        expected_docker_cli_sha256="0" * 64,
        gpu_index=251,
        owner_kind="legacy_validation",
        ttl_seconds=5,
    )

    def call() -> int:
        return lease._run_v4(
            args,
            _journal=journal,
            _manager=manager,
            _backend=backend,
            _watchdog_factory=factory,  # type: ignore[arg-type]
            _hooks=hooks,
            _max_daemon_retries=retry_budget,
        )

    return call, journal, manager, backend, captured


def test_short_parser_uses_only_option_key() -> None:
    for argv in (
        ["docker", "run", "-efoo=disabled", "image:x"],
        ["docker", "run", "-v/data:/data", "image:x"],
    ):
        assert lease.normalize_direct_docker_run(argv)[0] == "/usr/bin/docker"
    for argv in (
        ["docker", "run", "-d", "image:x"],
        ["docker", "run", "-dit", "image:x"],
        ["docker", "run", "--detach", "image:x"],
    ):
        with pytest.raises(lease.LeaseIntegrityError):
            lease.normalize_direct_docker_run(argv)


def test_exact_member_cgroup_drift_never_calls_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    member = lease._Member(42, os.getuid(), 100, 42, 42, "/x", 1, 2)
    scope = {
        "boot_id": v1._read_boot_id(),
        "scope_id": "a" * 64,
        "cgroup_path": "/x",
        "cgroup_ns_dev": 1,
        "cgroup_ns_ino": 2,
    }
    monkeypatch.setattr(
        v1,
        "_read_process_identity",
        lambda *_args: {"uid": os.getuid(), "start_ticks": 100},
    )
    monkeypatch.setattr(
        lease.v3,
        "_read_proc_stat",
        lambda *_args: {"state": "S", "process_group": 42, "session_id": 42},
    )
    monkeypatch.setattr(lease.v3, "_read_cgroup_identity", lambda *_args: ("/drift", 1, 2))
    monkeypatch.setattr(lease, "_read_scope_token", lambda *_args: "a" * 64)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(lease.LeaseIntegrityError, match="immediately"):
        lease._signal_exact_member(member, signal.SIGTERM, scope)
    assert killed == []


def test_watchdog_parent_eof_cancels_delayed_action(tmp_path: Path) -> None:
    marker = tmp_path / "late-action"
    handle = lease._spawn_test_watchdog()
    handle.open_gate(
        [
            sys.executable,
            "-c",
            f"import pathlib,time;time.sleep(2);pathlib.Path({str(marker)!r}).touch()",
        ],
        test_only=True,
    )
    handle.close_parent_channel()
    handle.process.wait(timeout=5)
    time.sleep(2.2)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("success", 0), ("nonzero", 7), ("signal", 128 + signal.SIGTERM)],
)
def test_full_run_success_nonzero_and_signal(
    mode: str, expected: int, tmp_path: Path
) -> None:
    signal_hook = (
        (lambda: os.kill(os.getpid(), signal.SIGTERM)) if mode == "signal" else None
    )
    call, journal, manager, backend, _captured = run_fixture(
        tmp_path, mode, workload_started=signal_hook
    )
    assert call() == expected
    with journal.lock(251) as paths:
        assert journal.read_state(paths, 251) is None
    assert lease._inner_has_state(manager, 251) is False
    assert backend.removed == ["c" * 64]


@pytest.mark.parametrize(
    ("mode", "message"),
    [("timeout", "exceeded timeout"), ("popen_failure", "Popen failed")],
)
def test_full_run_timeout_and_popen_failure_finalize(
    mode: str, message: str, tmp_path: Path
) -> None:
    call, journal, manager, backend, _captured = run_fixture(tmp_path, mode)
    with pytest.raises(lease.LeaseIntegrityError, match=message):
        call()
    with journal.lock(251) as paths:
        assert journal.read_state(paths, 251) is None
    assert lease._inner_has_state(manager, 251) is False
    assert backend.removed == ["c" * 64]


@pytest.mark.parametrize("failure", ["outage", "identity_drift"])
def test_daemon_failure_preserves_active_outer_and_inner_state(
    failure: str, tmp_path: Path
) -> None:
    def fail(backend: FakeBackend) -> None:
        setattr(backend, failure, True)

    call, journal, manager, _backend, _captured = run_fixture(
        tmp_path, "success", after_gate=fail, retry_budget=1
    )
    with pytest.raises(lease.LeaseIntegrityError, match="retry budget"):
        call()
    with journal.lock(251) as paths:
        state = journal.read_state(paths, 251)
        assert state is not None and state["phase"] == "active"
    assert lease._inner_has_state(manager, 251) is True


def test_active_missing_inner_requires_exact_durable_terminal_receipt(
    tmp_path: Path,
) -> None:
    call, journal, manager, backend, captured = run_fixture(tmp_path, "success")
    assert call() == 0
    active = captured["state"]
    assert active["phase"] == "active"
    with journal.lock(251) as paths:
        journal.replace_state(paths, active)
    result = lease.recover_stale_v4(
        gpu_index=251,
        journal=journal,
        inner_manager=manager,
        backend=backend,
        force_stale_for_test=True,
    )
    assert result["docker_evidence"]["kind"] == "verified_terminal_receipt"
    assert result["docker_evidence"] is not None
    with journal.lock(251) as paths:
        assert journal.read_state(paths, 251) is None
        stale = [
            journal._parse_receipt(path.read_bytes())
            for path in paths["receipts"].glob("*.json")
            if journal._parse_receipt(path.read_bytes())["event_type"] == "stale_recovery"
        ]
    assert len(stale) == 1 and stale[0]["docker_evidence"] is not None


@pytest.mark.parametrize("tamper", ["missing", "replaced"])
def test_missing_or_replaced_terminal_receipt_fails_closed_and_preserves_outer(
    tamper: str, tmp_path: Path,
) -> None:
    call, journal, manager, backend, captured = run_fixture(tmp_path, "success")
    assert call() == 0
    inner_paths = manager._prepare_paths(251)
    acquire_raw: bytes | None = None
    release_path: Path | None = None
    for path in inner_paths["receipts"].glob("*.json"):
        event, raw = manager._read_receipt(path)
        if event["event_type"] in {"acquire", "stale_recover_acquire"}:
            acquire_raw = raw
        if event["event_type"] == "release":
            release_path = path
    assert release_path is not None
    if tamper == "missing":
        release_path.unlink()
    else:
        assert acquire_raw is not None
        release_path.write_bytes(acquire_raw)
    active = captured["state"]
    with journal.lock(251) as paths:
        journal.replace_state(paths, active)
    with pytest.raises(lease.LeaseIntegrityError):
        lease.recover_stale_v4(
            gpu_index=251,
            journal=journal,
            inner_manager=manager,
            backend=backend,
            force_stale_for_test=True,
        )
    with journal.lock(251) as paths:
        assert journal.read_state(paths, 251) == active


def test_contract_held_bundle_concurrent_32_of_32(tmp_path: Path) -> None:
    contract = lease.load_contract()
    assert lease.GPU_LEASE_V4_API_READY is False
    assert lease.LIVE_PLAN_AUTHORIZED is False
    with lease.open_held_execution_bundle(
        expected_contract_fingerprint=contract["contract_fingerprint_sha256"],
        expected_source_sha256=contract["implementation"]["sha256"],
        expected_python_sha256=contract["python_interpreter"]["sha256"],
    ) as bundle:
        def one(_index: int) -> bool:
            completed = bundle.run(["held-status"], timeout=20, check=True)
            value = json.loads(completed.stdout)
            return (
                value["status"] == "held_runtime_verified"
                and value["per_child_open_file_descriptions"] is True
                and value["pread_replay"] is True
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(one, range(32))) == [True] * 32
        lifecycle = bundle.run(
            [
                "held-cpu-lifecycle",
                "--state-root", str(tmp_path / "held-state"),
                "--gpu-index", "254",
            ],
            timeout=20,
            check=True,
        )
        assert json.loads(lifecycle.stdout)["docker_evidence_kind"] == "verified_never_acquired"


def test_frozen_predecessor_hashes_and_modes() -> None:
    expected = {
        "validation/gpu_lease.py": (0o664, "ba1f47d6486b2a0badb5b2c108dd0ff4ea77d86011ffc5c01b6364f45f451884"),
        "validation/gpu_lease_v2.py": (0o664, "21a93645745c8331c98e2b3a7b086471cf2335ab8a5d82eacccd98eb014c8199"),
        "validation/gpu_lease_v3.py": (0o440, "e06ca80bd94adf8ea941366097013c97c697afdae956ad4d52c28ea3460292d7"),
        "validation/contracts/gpu-lease-v3.json": (0o440, "a297626d5c7fb5e838353be7a58c52279f26508795cdc1f11ed950491e144cc4"),
    }
    import hashlib

    for relative, (mode, digest) in expected.items():
        path = ROOT / relative
        assert path.stat().st_mode & 0o777 == mode
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
