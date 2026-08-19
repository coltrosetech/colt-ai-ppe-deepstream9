from __future__ import annotations

import fcntl
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import validation.gpu_lease as lease_module
from validation.gpu_lease import (
    LeaseAuthorizationError,
    LeaseBusyError,
    LeaseIntegrityError,
    LeaseManager,
    command_argv_sha256,
    contract_projection,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"
BOOT_A = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
BOOT_B = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"


class FakeClock:
    def __init__(self) -> None:
        self.ns = 1_000_000_000
        self.wall = datetime(2026, 7, 18, tzinfo=timezone.utc)

    def monotonic_ns(self) -> int:
        return self.ns

    def wall_now(self) -> datetime:
        return self.wall

    def advance(self, seconds: int) -> None:
        self.ns += seconds * 1_000_000_000
        self.wall += timedelta(seconds=seconds)


class FakeIdentity:
    def __init__(self, boot: str = BOOT_A) -> None:
        self.boot = boot
        self.start_ticks = 12345
        self.present = True
        self.missing_pids: set[int] = set()

    def read(self, pid: int, boot_id: str | None = None) -> dict[str, object] | None:
        if not self.present or pid in self.missing_pids:
            return None
        base: dict[str, object] = {
            "uid": os.getuid(),
            "pid": pid,
            "boot_id": boot_id or self.boot,
            "start_ticks": self.start_ticks,
        }
        from validation.gpu_lease import canonical_sha256

        return {**base, "identity_sha256": canonical_sha256(base)}


def manager(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    identity: FakeIdentity | None = None,
    boot: list[str] | None = None,
    before_state_replace=None,
) -> LeaseManager:
    clock = clock or FakeClock()
    identity = identity or FakeIdentity()
    boot = boot or [BOOT_A]
    value = LeaseManager(
        root=tmp_path / "leases",
        gpu_uuid_resolver=lambda _index: GPU_UUID,
        boot_id_reader=lambda: boot[0],
        process_identity_reader=identity.read,
        monotonic_ns=clock.monotonic_ns,
        wall_now=clock.wall_now,
        before_state_replace=before_state_replace,
    )
    canonical_paths = value._paths

    def isolated_paths(gpu_index: int):
        paths = canonical_paths(gpu_index)
        paths["legacy"] = tmp_path / f"legacy-gpu-{gpu_index}.lock"
        return paths

    value._paths = isolated_paths  # type: ignore[method-assign]
    return value


def acquire(value: LeaseManager, gpu_index: int = 231, owner_pid: int | None = None):
    return value.acquire(
        gpu_index=gpu_index,
        owner_kind="capacity_5min",
        command=["docker", "run", "image"],
        ttl_seconds=5,
        owner_pid=os.getpid() if owner_pid is None else owner_pid,
    )


def run_args(
    command: list[str],
    *,
    gpu_index: int,
    timeout_seconds: float | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        command_argv=["--", *command],
        gpu_index=gpu_index,
        owner_kind="legacy_validation",
        ttl_seconds=5,
        timeout_seconds=timeout_seconds,
    )


def run_with_devnull(
    args: SimpleNamespace,
    value: LeaseManager,
    **kwargs,
) -> int:
    descriptor = os.open(os.devnull, os.O_WRONLY)
    docker_cleanup = kwargs.pop(
        "_docker_cleanup", lambda _specification: {"applicable": False}
    )
    try:
        return lease_module._run_under_lease(
            args,
            _manager=value,
            _stdout_fd=descriptor,
            _stderr_fd=descriptor,
            _docker_cleanup=docker_cleanup,
            **kwargs,
        )
    finally:
        os.close(descriptor)


def wait_for_pid_file(path: Path) -> list[int]:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            values = [int(value) for value in path.read_text().split()]
        except (FileNotFoundError, ValueError):
            time.sleep(0.01)
            continue
        if values:
            return values
    raise AssertionError(f"managed child did not publish PID file: {path}")


def process_is_running(pid: int) -> bool:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return len(fields) > 2 and fields[2] != "Z"


def assert_processes_stopped(pids: list[int]) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and any(process_is_running(pid) for pid in pids):
        time.sleep(0.01)
    assert not [pid for pid in pids if process_is_running(pid)]


def assert_released_with_valid_receipts(
    value: LeaseManager,
    tmp_path: Path,
    *,
    gpu_index: int,
    command: list[str],
) -> None:
    assert value.inspect(gpu_index=gpu_index)["status"] == "free"
    receipt_paths = list((tmp_path / f"leases/gpu-{gpu_index}/receipts").glob("*.json"))
    receipts = [json.loads(path.read_text()) for path in receipt_paths]
    assert sorted(receipt["event_type"] for receipt in receipts) == ["acquire", "release"]
    assert {receipt["lease"]["command_argv_sha256"] for receipt in receipts} == {
        command_argv_sha256(command)
    }
    for path, receipt in zip(receipt_paths, receipts, strict=True):
        unsigned = {
            key: item for key, item in receipt.items() if key != "event_fingerprint_sha256"
        }
        assert receipt["event_fingerprint_sha256"] == lease_module.canonical_sha256(unsigned)
        metadata = path.lstat()
        assert metadata.st_nlink == 1
        assert metadata.st_mode & 0o777 == 0o600


def test_checked_in_contract_and_live_pins_replay() -> None:
    contract = load_contract()
    projection = contract_projection()

    assert contract["status"] == "locked"
    assert projection["contract_fingerprint_sha256"] == contract[
        "contract_fingerprint_sha256"
    ]
    assert projection["artifact"]["path"] == "validation/contracts/gpu-lease-v1.json"


def test_acquire_renew_release_receipts_never_store_plain_capability(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    value = manager(tmp_path, clock=clock)
    credentials = acquire(value)
    state_path = tmp_path / "leases/gpu-231/active.json"
    raw = state_path.read_text(encoding="utf-8")

    assert credentials.capability not in raw
    assert value.inspect(gpu_index=231)["status"] == "active"
    with pytest.raises(LeaseAuthorizationError):
        value.renew(
            gpu_index=231,
            lease_id=credentials.lease_id,
            capability="0" * 64,
        )

    clock.advance(2)
    renewed = value.renew(
        gpu_index=231,
        lease_id=credentials.lease_id,
        capability=credentials.capability,
    )
    assert renewed.expires_at_utc != credentials.expires_at_utc
    result = value.release(
        gpu_index=231,
        lease_id=credentials.lease_id,
        capability=credentials.capability,
    )
    assert result["status"] == "released"
    assert value.inspect(gpu_index=231)["status"] == "free"
    events = sorted((tmp_path / "leases/gpu-231/receipts").glob("*.json"))
    assert [json.loads(path.read_text())["event_type"] for path in events].count(
        "release"
    ) == 1


def test_expired_but_same_live_process_is_never_recovered(tmp_path: Path) -> None:
    clock = FakeClock()
    value = manager(tmp_path, clock=clock)
    first = acquire(value, 232)
    clock.advance(6)

    assert value.inspect(gpu_index=232)["status"] == "active_expired_owner_alive"
    with pytest.raises(LeaseBusyError, match="expired_live_owner"):
        acquire(value, 232)

    value.release(
        gpu_index=232, lease_id=first.lease_id, capability=first.capability
    )


def test_dead_owner_requires_expiry_then_receipt_bound_recovery(tmp_path: Path) -> None:
    clock = FakeClock()
    identity = FakeIdentity()
    value = manager(tmp_path, clock=clock, identity=identity)
    acquire(value, 233, owner_pid=41001)
    identity.missing_pids.add(41001)

    assert value.inspect(gpu_index=233)["status"] == "active_orphan_grace"
    with pytest.raises(LeaseBusyError, match="active_or_grace"):
        acquire(value, 233, owner_pid=41002)

    clock.advance(6)
    identity.start_ticks += 1
    recovered = acquire(value, 233, owner_pid=41002)
    state = value.inspect(gpu_index=233)["state"]
    receipt_path = tmp_path / "leases/gpu-233" / state["last_event"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["event_type"] == "stale_recover_acquire"
    assert receipt["decision"]["stale_recovery_reason"] == "expired_owner_missing"
    value.release(
        gpu_index=233,
        lease_id=recovered.lease_id,
        capability=recovered.capability,
    )


def test_boot_change_allows_immediate_receipted_recovery(tmp_path: Path) -> None:
    boot = [BOOT_A]
    identity = FakeIdentity()
    value = manager(tmp_path, identity=identity, boot=boot)
    acquire(value, 234)
    boot[0] = BOOT_B

    recovered = acquire(value, 234)
    state = value.inspect(gpu_index=234)["state"]
    receipt = json.loads(
        (tmp_path / "leases/gpu-234" / state["last_event"]["path"]).read_text()
    )
    assert receipt["decision"]["stale_recovery_reason"] == "reboot"
    value.release(
        gpu_index=234,
        lease_id=recovered.lease_id,
        capability=recovered.capability,
    )


def test_pid_reuse_is_detected_by_start_ticks(tmp_path: Path) -> None:
    clock = FakeClock()
    identity = FakeIdentity()
    value = manager(tmp_path, clock=clock, identity=identity)
    acquire(value, 235)
    identity.start_ticks += 99
    clock.advance(6)

    recovered = acquire(value, 235)
    assert value.inspect(gpu_index=235)["state"]["lease"]["owner"][
        "start_ticks"
    ] == identity.start_ticks
    value.release(
        gpu_index=235,
        lease_id=recovered.lease_id,
        capability=recovered.capability,
    )


def test_dangling_state_symlink_and_hardlinked_receipt_fail_closed(
    tmp_path: Path,
) -> None:
    value = manager(tmp_path)
    paths = value._prepare_paths(236)
    paths["state"].symlink_to(tmp_path / "missing-target")
    with pytest.raises(LeaseIntegrityError):
        acquire(value, 236)
    paths["state"].unlink()

    credentials = acquire(value, 236)
    state = json.loads(paths["state"].read_text())
    receipt = paths["directory"] / state["last_event"]["path"]
    os.link(receipt, tmp_path / "receipt-hardlink.json")
    with pytest.raises(LeaseIntegrityError, match="single-link"):
        value.inspect(gpu_index=236)
    os.unlink(tmp_path / "receipt-hardlink.json")
    value.release(
        gpu_index=236,
        lease_id=credentials.lease_id,
        capability=credentials.capability,
    )


def test_state_temp_name_swap_is_detected_before_atomic_publish(tmp_path: Path) -> None:
    attacked = False

    def swap(temporary: Path, _state: Path) -> None:
        nonlocal attacked
        attacked = True
        temporary.unlink()
        temporary.symlink_to(tmp_path / "attacker-content")

    value = manager(tmp_path, before_state_replace=swap)
    with pytest.raises(LeaseIntegrityError):
        acquire(value, 237)
    assert attacked is True
    assert not (tmp_path / "leases/gpu-237/active.json").exists()


def test_concurrent_acquire_has_exactly_one_winner(tmp_path: Path) -> None:
    value = manager(tmp_path)
    barrier = threading.Barrier(3)
    outcomes: list[str] = []
    credentials = []

    def worker() -> None:
        barrier.wait()
        try:
            credentials.append(acquire(value, 238))
            outcomes.append("won")
        except LeaseBusyError:
            outcomes.append("busy")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["busy", "won"]
    value.release(
        gpu_index=238,
        lease_id=credentials[0].lease_id,
        capability=credentials[0].capability,
    )


def test_session_holds_legacy_flock_and_asserts_parent_identity(tmp_path: Path) -> None:
    value = manager(tmp_path)
    command = ["bash", "job.sh"]
    with value.session(
        gpu_index=239,
        owner_kind="endurance_7day",
        command=command,
        ttl_seconds=5,
    ) as session:
        assert session.credentials is not None
        held = value.assert_held(
            gpu_index=239,
            lease_id=session.credentials.lease_id,
            capability=session.credentials.capability,
            owner_pid=os.getpid(),
            command_digest=command_argv_sha256(command),
        )
        assert held["status"] == "held"
        descriptor = os.open(tmp_path / "legacy-gpu-239.lock", os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
    assert value.inspect(gpu_index=239)["status"] == "free"


@pytest.mark.parametrize(
    "contender",
    ["site_distance_25m", "person_training", "capacity_5min"],
)
def test_endurance_session_excludes_every_other_heavy_lane(
    tmp_path: Path, contender: str
) -> None:
    value = manager(tmp_path)
    with value.session(
        gpu_index=241,
        owner_kind="endurance_7day",
        command=["endurance-supervisor"],
        ttl_seconds=5,
    ):
        with pytest.raises(LeaseBusyError, match="legacy GPU lock"):
            value.acquire(
                gpu_index=241,
                owner_kind=contender,
                command=[contender, "workload"],
                ttl_seconds=5,
                owner_pid=os.getpid(),
            )


def test_gpu_entrypoint_wrappers_gate_before_gpu_or_docker() -> None:
    training = (ROOT / "models/person/run_yolo26_training.sh").read_text()
    capacity = (ROOT / "benchmark/run_5min.sh").read_text()

    assert training.index("validation.gpu_lease run") < training.index("exec docker run")
    assert training.index("validation.gpu_lease assert-held") < training.index(
        "exec docker run"
    )
    assert '--gpus "device=$gpu_index"' in training
    assert capacity.index("validation.gpu_lease run") < capacity.index(
        'nvidia-smi -i "$GPU_INDEX"'
    )
    assert capacity.index("validation.gpu_lease assert-held") < capacity.index(
        'nvidia-smi -i "$GPU_INDEX"'
    )


def test_duplicate_key_state_is_rejected(tmp_path: Path) -> None:
    value = manager(tmp_path)
    paths = value._prepare_paths(240)
    paths["state"].write_text(
        '{"schema_version":"deepsafe.gpu-lease-state/v1",'
        '"schema_version":"deepsafe.gpu-lease-state/v1"}',
        encoding="utf-8",
    )
    os.chmod(paths["state"], 0o600)
    with pytest.raises(LeaseIntegrityError, match="strict JSON"):
        value.inspect(gpu_index=240)


def test_run_drains_large_stdout_and_stderr_and_releases_receipts(tmp_path: Path) -> None:
    value = manager(tmp_path)
    command = [
        sys.executable,
        "-c",
        "import os; os.write(1, b'o' * 2097152); os.write(2, b'e' * 2097152)",
    ]

    assert run_with_devnull(run_args(command, gpu_index=242), value) == 0
    assert_released_with_valid_receipts(
        value, tmp_path, gpu_index=242, command=command
    )


@pytest.mark.parametrize(
    ("hook_name", "signum", "gpu_index", "lease_was_acquired"),
    [
        ("handlers_installed_before_session", signal.SIGTERM, 243, False),
        ("before_popen", signal.SIGINT, 244, True),
    ],
)
def test_run_signal_before_popen_never_launches_and_restores_handlers(
    tmp_path: Path,
    hook_name: str,
    signum: int,
    gpu_index: int,
    lease_was_acquired: bool,
) -> None:
    value = manager(tmp_path)
    marker = tmp_path / f"unexpected-child-{gpu_index}"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('bad')",
        str(marker),
    ]
    prior_handlers = {
        managed: signal.getsignal(managed) for managed in lease_module.RUN_MANAGED_SIGNALS
    }
    prior_mask = set(signal.pthread_sigmask(signal.SIG_BLOCK, set()))

    result = run_with_devnull(
        run_args(command, gpu_index=gpu_index),
        value,
        _hooks={hook_name: lambda: os.kill(os.getpid(), signum)},
    )

    assert result == 128 + signum
    assert not marker.exists()
    assert {
        managed: signal.getsignal(managed) for managed in lease_module.RUN_MANAGED_SIGNALS
    } == prior_handlers
    assert set(signal.pthread_sigmask(signal.SIG_BLOCK, set())) == prior_mask
    if lease_was_acquired:
        assert_released_with_valid_receipts(
            value, tmp_path, gpu_index=gpu_index, command=command
        )
    else:
        assert value.inspect(gpu_index=gpu_index)["status"] == "free"
        assert not list(
            (tmp_path / f"leases/gpu-{gpu_index}/receipts").glob("*.json")
        )


@pytest.mark.parametrize(
    ("hook_name", "signum", "gpu_index"),
    [
        ("after_popen_before_publish", signal.SIGHUP, 245),
        ("child_published", signal.SIGTERM, 246),
    ],
)
def test_run_signal_after_popen_kills_reaps_and_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hook_name: str,
    signum: int,
    gpu_index: int,
) -> None:
    monkeypatch.setattr(lease_module, "RUN_CHILD_TERM_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(lease_module, "RUN_CHILD_POLL_SECONDS", 0.01)
    value = manager(tmp_path)
    pid_file = tmp_path / f"child-{gpu_index}.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import os, pathlib, sys, time; "
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
            "time.sleep(30)"
        ),
        str(pid_file),
    ]
    observed_pids: list[int] = []

    def inject_signal() -> None:
        observed_pids.extend(wait_for_pid_file(pid_file))
        os.kill(os.getpid(), signum)

    result = run_with_devnull(
        run_args(command, gpu_index=gpu_index),
        value,
        _hooks={hook_name: inject_signal},
    )

    assert result == 128 + signum
    assert_processes_stopped(observed_pids)
    assert_released_with_valid_receipts(
        value, tmp_path, gpu_index=gpu_index, command=command
    )


def test_run_signal_after_release_is_caught_before_handler_restore(tmp_path: Path) -> None:
    value = manager(tmp_path)
    command = [sys.executable, "-c", "pass"]
    prior = signal.getsignal(signal.SIGTERM)

    result = run_with_devnull(
        run_args(command, gpu_index=247),
        value,
        _hooks={
            "lease_released_before_handler_restore": lambda: os.kill(
                os.getpid(), signal.SIGTERM
            )
        },
    )

    assert result == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == prior
    assert_released_with_valid_receipts(
        value, tmp_path, gpu_index=247, command=command
    )


def test_run_sigterm_ignoring_process_tree_is_killed_without_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lease_module, "RUN_CHILD_TERM_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(lease_module, "RUN_CHILD_POLL_SECONDS", 0.01)
    value = manager(tmp_path)
    pid_file = tmp_path / "ignoring-tree.pid"
    grandchild = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    leader = (
        "import os,pathlib,signal,subprocess,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"grand=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
        "pathlib.Path(sys.argv[1]).write_text(f'{os.getpid()} {grand.pid}'); "
        "time.sleep(30)"
    )
    command = [sys.executable, "-c", leader, str(pid_file)]
    observed_pids: list[int] = []

    def terminate_tree() -> None:
        observed_pids.extend(wait_for_pid_file(pid_file))
        os.kill(os.getpid(), signal.SIGTERM)

    result = run_with_devnull(
        run_args(command, gpu_index=248),
        value,
        _hooks={"child_published": terminate_tree},
    )

    assert result == 128 + signal.SIGTERM
    assert_processes_stopped(observed_pids)
    assert_released_with_valid_receipts(
        value, tmp_path, gpu_index=248, command=command
    )


def test_run_fast_leader_exit_with_detached_pipe_descendant_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lease_module, "RUN_CHILD_TERM_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(lease_module, "RUN_CHILD_POLL_SECONDS", 0.01)
    value = manager(tmp_path)
    pid_file = tmp_path / "orphan-descendant.pid"
    descendant = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
    leader = (
        "import os,pathlib,subprocess,sys; "
        f"child=subprocess.Popen([sys.executable,'-c',{descendant!r}],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))"
    )
    command = [sys.executable, "-c", leader, str(pid_file)]

    with pytest.raises(LeaseIntegrityError, match="orphan descendant"):
        run_with_devnull(run_args(command, gpu_index=230), value)

    pids = wait_for_pid_file(pid_file)
    assert_processes_stopped(pids)
    assert_released_with_valid_receipts(
        value, tmp_path, gpu_index=230, command=command
    )


def test_run_timeout_kills_sigterm_ignoring_child_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lease_module, "RUN_CHILD_TERM_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(lease_module, "RUN_CHILD_POLL_SECONDS", 0.01)
    value = manager(tmp_path)
    pid_file = tmp_path / "timeout-child.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import os,pathlib,signal,sys,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
            "time.sleep(30)"
        ),
        str(pid_file),
    ]

    with pytest.raises(LeaseIntegrityError, match="exceeded run timeout"):
        run_with_devnull(
            run_args(command, gpu_index=249, timeout_seconds=0.1), value
        )

    pids = wait_for_pid_file(pid_file)
    assert_processes_stopped(pids)
    assert_released_with_valid_receipts(
        value, tmp_path, gpu_index=249, command=command
    )


def test_run_log_write_exception_drains_to_eof_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lease_module, "RUN_CHILD_TERM_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(lease_module, "RUN_CHILD_POLL_SECONDS", 0.01)
    value = manager(tmp_path)
    command = [
        sys.executable,
        "-c",
        (
            "import os,time; "
            "os.write(1,b'o'*2097152); os.write(2,b'e'*2097152); time.sleep(30)"
        ),
    ]
    calls = 0
    calls_lock = threading.Lock()

    def broken_writer(_descriptor: int, _payload: bytes) -> None:
        nonlocal calls
        with calls_lock:
            calls += 1
        raise OSError("deterministic log sink failure")

    started = time.monotonic()
    with pytest.raises(LeaseIntegrityError, match="output drain failed closed"):
        run_with_devnull(
            run_args(command, gpu_index=250), value, _writer=broken_writer
        )

    assert time.monotonic() - started < 3.0
    assert 1 <= calls <= 2
    assert_released_with_valid_receipts(
        value, tmp_path, gpu_index=250, command=command
    )


def test_run_log_backpressure_has_bounded_failure_and_no_deadlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lease_module, "RUN_OUTPUT_STALL_SECONDS", 0.1)
    monkeypatch.setattr(lease_module, "RUN_CHILD_TERM_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(lease_module, "RUN_CHILD_POLL_SECONDS", 0.01)
    monkeypatch.setattr(lease_module, "RUN_DRAIN_JOIN_SECONDS", 1.0)
    value = manager(tmp_path)
    command = [
        sys.executable,
        "-c",
        "import os,time; os.write(1,b'x'*2097152); time.sleep(30)",
    ]
    read_fd, write_fd = os.pipe()
    stderr_fd = os.open(os.devnull, os.O_WRONLY)
    os.set_blocking(write_fd, False)
    try:
        while True:
            try:
                os.write(write_fd, b"p" * 65536)
            except BlockingIOError:
                break
        with pytest.raises(LeaseIntegrityError, match="output drain failed closed"):
            lease_module._run_under_lease(
                run_args(command, gpu_index=251),
                _manager=value,
                _stdout_fd=write_fd,
                _stderr_fd=stderr_fd,
                _docker_cleanup=lambda _specification: {"applicable": False},
            )
    finally:
        os.close(write_fd)
        os.close(read_fd)
        os.close(stderr_fd)

    assert_released_with_valid_receipts(
        value, tmp_path, gpu_index=251, command=command
    )


def test_run_child_inherits_unblocked_managed_signals_and_parent_mask_is_restored(
    tmp_path: Path,
) -> None:
    value = manager(tmp_path)
    output = tmp_path / "child-signal-mask.txt"
    output_fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    command = [
        sys.executable,
        "-c",
        (
            "import signal; "
            "blocked=signal.pthread_sigmask(signal.SIG_BLOCK,set()); "
            "print(int(any(value in blocked for value in "
            "(signal.SIGINT,signal.SIGTERM,signal.SIGHUP))))"
        ),
    ]
    original_mask = set(signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM}))
    try:
        assert lease_module._run_under_lease(
            run_args(command, gpu_index=252),
            _manager=value,
            _stdout_fd=output_fd,
            _stderr_fd=output_fd,
            _docker_cleanup=lambda _specification: {"applicable": False},
        ) == 0
        restored = set(signal.pthread_sigmask(signal.SIG_BLOCK, set()))
        assert signal.SIGTERM in restored
    finally:
        os.close(output_fd)
        signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)

    assert output.read_text().strip() == "0"
    assert_released_with_valid_receipts(
        value, tmp_path, gpu_index=252, command=command
    )


def test_managed_nested_docker_cid_cleanup_retries_while_lease_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lease_module, "RUN_CHILD_TERM_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(lease_module, "RUN_CHILD_POLL_SECONDS", 0.01)
    monkeypatch.setattr(lease_module, "RUN_DOCKER_CLEANUP_RETRY_SECONDS", 0.01)
    value = manager(tmp_path)
    cidfile = tmp_path / "nested.cid"
    pid_file = tmp_path / "nested.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import os,pathlib,signal,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "pathlib.Path(os.environ['DEEPSAFE_MANAGED_DOCKER_CIDFILE'])"
            ".write_text('a'*64); "
            f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
            "time.sleep(30)"
        ),
    ]
    args = run_args(command, gpu_index=253)
    args.managed_docker_cidfile = cidfile
    cleanup_calls = 0

    def exact_cleanup(specification) -> dict[str, object]:
        nonlocal cleanup_calls
        cleanup_calls += 1
        assert specification is not None
        assert specification.path == cidfile
        assert lease_module._read_docker_cidfile(specification) == "a" * 64
        assert value.inspect(gpu_index=253)["status"] == "active"
        descriptor = os.open(tmp_path / "legacy-gpu-253.lock", os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        if cleanup_calls == 1:
            raise OSError("simulated Docker daemon outage")
        return {
            "applicable": True,
            "container_id": "a" * 64,
            "verified_absent": True,
        }

    def terminate_nested_runner() -> None:
        wait_for_pid_file(pid_file)
        os.kill(os.getpid(), signal.SIGTERM)

    assert run_with_devnull(
        args,
        value,
        _hooks={"child_published": terminate_nested_runner},
        _docker_cleanup=exact_cleanup,
    ) == 128 + signal.SIGTERM

    assert cleanup_calls == 2
    assert_released_with_valid_receipts(
        value, tmp_path, gpu_index=253, command=command
    )


def test_managed_nested_docker_cid_is_verified_on_normal_child_exit(
    tmp_path: Path,
) -> None:
    value = manager(tmp_path)
    cidfile = tmp_path / "normal-nested.cid"
    command = [
        sys.executable,
        "-c",
        (
            "import os,pathlib; "
            "pathlib.Path(os.environ['DEEPSAFE_MANAGED_DOCKER_CIDFILE'])"
            ".write_text('b'*64)"
        ),
    ]
    args = run_args(command, gpu_index=255)
    args.managed_docker_cidfile = cidfile
    observed: list[str] = []

    def exact_cleanup(specification) -> dict[str, object]:
        assert specification is not None
        observed.append(lease_module._read_docker_cidfile(specification) or "")
        assert value.inspect(gpu_index=255)["status"] == "active"
        return {
            "applicable": True,
            "container_id": "b" * 64,
            "verified_absent": True,
        }

    assert run_with_devnull(
        args, value, _docker_cleanup=exact_cleanup
    ) == 0
    assert observed == ["b" * 64]
    assert_released_with_valid_receipts(
        value, tmp_path, gpu_index=255, command=command
    )


def test_managed_docker_cidfile_policy_rejects_mismatch_and_unsafe_parent(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    expected = private / "expected.cid"
    other = private / "other.cid"
    args = SimpleNamespace(managed_docker_cidfile=expected)

    with pytest.raises(LeaseIntegrityError, match="differs"):
        lease_module._managed_docker_cidfile(
            args, ["docker", "run", "--cidfile", str(other), "image"]
        )
    with pytest.raises(LeaseIntegrityError, match="must receive"):
        lease_module._managed_docker_cidfile(args, ["docker", "run", "image"])

    bound = lease_module._managed_docker_cidfile(
        args, [sys.executable, "runner.py"]
    )
    assert bound is not None
    original = tmp_path / "private-original"
    private.rename(original)
    private.mkdir(mode=0o700)
    expected.write_bytes(b"a" * 64)
    with pytest.raises(LeaseIntegrityError, match="parent identity differs"):
        lease_module._read_docker_cidfile(bound)

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o777)
    os.chmod(unsafe, 0o777)
    with pytest.raises(LeaseIntegrityError, match="owner-only"):
        lease_module._managed_docker_cidfile(
            SimpleNamespace(managed_docker_cidfile=unsafe / "container.cid"),
            [sys.executable, "runner.py"],
        )


def test_docker_cidfile_reader_requires_exact_single_link_hex_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cidfile = tmp_path / "exact.cid"
    specification = lease_module._validate_docker_cidfile_parent(
        lease_module._DockerCidfile("docker", cidfile)
    )
    cidfile.write_bytes(b"c" * 64 + b"\n")
    assert lease_module._read_docker_cidfile(specification) == "c" * 64

    cidfile.write_bytes(b" " + b"c" * 64)
    with pytest.raises(LeaseIntegrityError, match="content differs"):
        lease_module._read_docker_cidfile(specification)

    cidfile.write_bytes(b"d" * 64)
    hardlink = tmp_path / "exact-hardlink.cid"
    os.link(cidfile, hardlink)
    with pytest.raises(LeaseIntegrityError, match="identity differs"):
        lease_module._read_docker_cidfile(specification)
    hardlink.unlink()

    cidfile.write_bytes(b"f" * 64)
    real_read = os.read
    raced = False

    def mutate_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal raced
        value = real_read(descriptor, size)
        if size == 66 and value and not raced:
            raced = True
            cidfile.write_bytes(b"0" * 64)
        return value

    monkeypatch.setattr(lease_module.os, "read", mutate_after_first_read)
    with pytest.raises(LeaseIntegrityError, match="changed while reading"):
        lease_module._read_docker_cidfile(specification)
    monkeypatch.setattr(lease_module.os, "read", real_read)

    cidfile.unlink()
    target = tmp_path / "target.cid"
    target.write_bytes(b"e" * 64)
    cidfile.symlink_to(target)
    with pytest.raises(LeaseIntegrityError, match="securely open"):
        lease_module._read_docker_cidfile(specification)


def test_session_release_error_is_not_swallowed_by_active_exception(
    tmp_path: Path,
) -> None:
    value = manager(tmp_path)

    def broken_release(**_kwargs):
        raise LeaseIntegrityError("deterministic release failure")

    value.release = broken_release  # type: ignore[method-assign]
    with pytest.raises(LeaseIntegrityError, match="release failed closed"):
        with value.session(
            gpu_index=254,
            owner_kind="legacy_validation",
            command=[sys.executable, "-c", "pass"],
            ttl_seconds=5,
        ):
            raise RuntimeError("body failed first")
