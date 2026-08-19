from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import pytest

import validation.gpu_lease as v1
import validation.gpu_lease_v2 as lease


GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"
BOOT_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
CID_A = "a" * 64
CID_B = "b" * 64
CID_C = "c" * 64


class SweepClock:
    def __init__(self) -> None:
        self.seconds = 0.0
        self.base = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return self.seconds

    def monotonic_ns(self) -> int:
        return 1_000_000_000 + int(self.seconds * 1_000_000_000)

    def wall_now(self) -> datetime:
        return self.base + timedelta(seconds=self.seconds)

    def sleep(self, seconds: float) -> None:
        self.seconds += seconds
        time.sleep(0)


class FakeIdentity:
    def __init__(self) -> None:
        self.present = True

    def read(self, pid: int, boot_id: str | None = None) -> dict[str, Any]:
        if not self.present:
            return None  # type: ignore[return-value]
        base = {
            "uid": os.getuid(),
            "pid": pid,
            "boot_id": boot_id or BOOT_ID,
            "start_ticks": 12345,
        }
        return {**base, "identity_sha256": lease.canonical_sha256(base)}


class FakeBackend:
    def __init__(self) -> None:
        self.daemon = {
            "daemon_id": "daemon-test-001",
            "name": "deepsafe-test-daemon",
            "server_version": "28.0.0",
            "docker_root_dir": "/var/lib/docker-test",
            "docker_cli": {
                "path": str(lease.TRUSTED_DOCKER_CLI),
                "bytes": 1,
                "sha256": "0" * 64,
            },
            "environment_sha256": lease._docker_environment_sha256(
                lease._sanitized_runtime_environment()
            ),
            "context_sha256": "d" * 64,
        }
        self.containers: dict[str, dict[str, Any]] = {}
        self.list_script: list[Any] = []
        self.key_list_script: list[Any] = []
        self.identity_script: list[Any] = []
        self.remove_script: list[int] = []
        self.removed: list[str] = []
        self.list_calls = 0
        self.identity_calls = 0

    def add(self, container_id: str, token: str, *, inspected_id: str | None = None) -> None:
        self.containers[container_id] = {
            "Id": inspected_id or container_id,
            "Config": {"Labels": {lease.DOCKER_LABEL_KEY: token}},
        }

    def identity(self) -> dict[str, str]:
        self.identity_calls += 1
        if self.identity_script:
            item = self.identity_script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return dict(item)
        return dict(self.daemon)

    def list_ids(self, label_key: str, label_value: str) -> Sequence[str]:
        assert label_key == lease.DOCKER_LABEL_KEY
        self.list_calls += 1
        if self.list_script:
            item = self.list_script.pop(0)
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                return item(self, label_value)
            return list(item)
        return [
            container_id
            for container_id, value in self.containers.items()
            if value["Config"]["Labels"].get(label_key) == label_value
        ]

    def list_ids_by_label_key(self, label_key: str) -> Sequence[str]:
        assert label_key == lease.DOCKER_LABEL_KEY
        self.list_calls += 1
        if self.key_list_script:
            item = self.key_list_script.pop(0)
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                return item(self)
            return list(item)
        return [
            container_id
            for container_id, value in self.containers.items()
            if label_key in value["Config"]["Labels"]
        ]

    def inspect(self, container_id: str) -> dict[str, Any]:
        return copy.deepcopy(self.containers[container_id])

    def remove(self, container_id: str) -> lease.DockerResult:
        self.removed.append(container_id)
        returncode = self.remove_script.pop(0) if self.remove_script else 0
        if returncode == 0:
            self.containers.pop(container_id, None)
        return lease.DockerResult(returncode, container_id.encode(), b"")


def private_intent(
    tmp_path: Path,
    *,
    mode: str = "nested",
    token: str = "1" * 64,
) -> lease.DockerSupervisionIntent:
    private = tmp_path / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    specification = v1._validate_docker_cidfile_parent(
        v1._DockerCidfile("docker", private / "container.cid")
    )
    return lease.DockerSupervisionIntent(
        mode,
        token,
        specification,
        docker_cli_sha256="0" * 64,
        docker_environment_sha256=lease._docker_environment_sha256(
            lease._sanitized_runtime_environment()
        ),
    )


def cleanup(
    tmp_path: Path,
    backend: FakeBackend,
    *,
    intent: lease.DockerSupervisionIntent | None = None,
    outcome: str = "success",
    returncode: int | None = 0,
    prelaunch: Sequence[str] = (),
    clock: SweepClock | None = None,
    on_retry: Callable[[BaseException], None] | None = None,
) -> dict[str, Any]:
    intent = intent or private_intent(tmp_path)
    clock = clock or SweepClock()
    return lease.cleanup_docker_supervision(
        backend=backend,
        intent=intent,
        requested_command=[sys.executable, "nested.py"],
        effective_command=[sys.executable, "nested.py"],
        daemon=backend.daemon,
        prelaunch_ids=prelaunch,
        child={
            "outcome": outcome,
            "returncode": returncode,
            "stop_reason": None if outcome in {"success", "nonzero"} else outcome,
        },
        monotonic=clock.monotonic,
        wall_now=clock.wall_now,
        sleeper=clock.sleep,
        retry_seconds=0.1,
        on_retry=on_retry,
    )


def manager(
    tmp_path: Path,
    clock: SweepClock | None = None,
    *,
    identity: FakeIdentity | None = None,
    boot: list[str] | None = None,
) -> lease.LeaseManager:
    clock = clock or SweepClock()
    identity = identity or FakeIdentity()
    boot = boot or [BOOT_ID]
    value = lease.LeaseManager(
        root=tmp_path / "leases-v2",
        gpu_uuid_resolver=lambda _index: GPU_UUID,
        boot_id_reader=lambda: boot[0],
        process_identity_reader=identity.read,
        monotonic_ns=clock.monotonic_ns,
        wall_now=clock.wall_now,
    )
    canonical = value._paths

    def isolated(gpu_index: int) -> dict[str, Path]:
        paths = canonical(gpu_index)
        paths["legacy"] = tmp_path / f"legacy-v2-{gpu_index}.lock"
        return paths

    value._paths = isolated  # type: ignore[method-assign]
    return value


def crash_supervised_session(
    value: lease.LeaseManager,
    *,
    gpu_index: int,
    intent: lease.DockerSupervisionIntent,
    backend: FakeBackend,
    clock: SweepClock,
) -> lease.LeaseSession:
    session = value.session(
        gpu_index=gpu_index,
        owner_kind="legacy_validation",
        command=[sys.executable, "crashed-supervisor.py"],
        docker_supervision=intent,
        docker_backend=backend,
        docker_sleeper=clock.sleep,
        ttl_seconds=5,
    )
    session.__enter__()
    session._stop.set()
    assert session._thread is not None
    session._thread.join(timeout=2)
    assert not session._thread.is_alive()
    assert session._legacy_fd is not None
    os.close(session._legacy_fd)
    session._legacy_fd = None
    return session


def run_args(
    command: Sequence[str], cidfile: Path, *, gpu_index: int, timeout: float | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        command_argv=["--", *command],
        gpu_index=gpu_index,
        owner_kind="legacy_validation",
        ttl_seconds=5,
        timeout_seconds=timeout,
        managed_docker_cidfile=cidfile,
    )


def run_devnull(args: SimpleNamespace, value: lease.LeaseManager, **kwargs: Any) -> int:
    descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        return lease._run_under_lease(
            args,
            _manager=value,
            _stdout_fd=descriptor,
            _stderr_fd=descriptor,
            **kwargs,
        )
    finally:
        os.close(descriptor)


def mirror_runtime(tmp_path: Path, name: str = "mirror") -> tuple[Path, dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    mirror = tmp_path / name
    for relative in lease.HELD_RUNTIME_PATHS:
        destination = mirror / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes())
    lease_root = mirror / "validation/results/gpu-leases/v2"
    lease_root.mkdir(mode=0o700, parents=True)
    os.chmod(lease_root.parent, 0o700)
    os.chmod(lease_root, 0o700)
    contract_path = mirror / "validation/contracts/gpu-lease-v2.json"
    contract = json.loads(contract_path.read_text())
    for pin in (
        contract["implementation"],
        contract["schemas"]["state"],
        contract["schemas"]["event_receipt"],
    ):
        raw = (mirror / pin["path"]).read_bytes()
        pin["bytes"] = len(raw)
        pin["sha256"] = hashlib.sha256(raw).hexdigest()
    python_raw = lease.TRUSTED_PYTHON.read_bytes()
    contract["python_interpreter"] = {
        "path": str(lease.TRUSTED_PYTHON),
        "bytes": len(python_raw),
        "sha256": hashlib.sha256(python_raw).hexdigest(),
    }
    workspace_tree: dict[str, Any] = {}
    for label, relative in lease.WORKSPACE_TREE_PATHS.items():
        info = os.stat(mirror if relative == "." else mirror / relative)
        workspace_tree[label] = lease._workspace_directory_projection(
            label=label, relative=relative, info=info
        )
    contract["workspace_tree"] = workspace_tree
    unsigned = {
        key: value
        for key, value in contract.items()
        if key != "contract_fingerprint_sha256"
    }
    contract["contract_fingerprint_sha256"] = lease.canonical_sha256(unsigned)
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    return mirror, contract


def test_contract_and_v2_schemas_replay_without_touching_v1() -> None:
    contract = lease.load_contract()
    assert contract["schema_version"] == lease.CONTRACT_VERSION
    assert contract["docker_supervision_policy"]["container_id_file_authoritative"] is False
    assert lease.DEFAULT_LEASE_ROOT.as_posix().endswith("gpu-leases/v2")
    assert v1.DEFAULT_LEASE_ROOT.as_posix().endswith("gpu-leases/v1")


def test_normal_path_cli_cannot_authorize_run_or_held_status(capsys: pytest.CaptureFixture[str]) -> None:
    contract = lease.load_contract()
    common = [
        "--expected-contract-fingerprint",
        contract["contract_fingerprint_sha256"],
        "--expected-source-sha256",
        contract["implementation"]["sha256"],
    ]
    assert lease.main(["held-status", *common]) == 2
    assert "held-FD isolated bootstrap" in capsys.readouterr().err
    assert (
        lease.main(
            [
                "recover",
                *common,
                "--expected-docker-cli-sha256",
                "0" * 64,
                "--gpu-index",
                "0",
            ]
        )
        == 2
    )
    assert "held-FD isolated bootstrap" in capsys.readouterr().err


def test_launch_cli_requires_expected_contract_and_source_pins() -> None:
    parser = lease.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--gpu-index",
                "0",
                "--owner-kind",
                "legacy_validation",
                "--",
                sys.executable,
                "-c",
                "pass",
            ]
        )


def test_held_runtime_executes_open_fds_after_source_swap_and_ignores_python_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror, contract = mirror_runtime(tmp_path)
    marker = tmp_path / "sitecustomize-ran"
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "sitecustomize.py").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')\n"
    )
    (tmp_path / "startup.py").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')\n"
    )
    with lease.open_held_execution_bundle(
        expected_contract_fingerprint=contract["contract_fingerprint_sha256"],
        expected_source_sha256=contract["implementation"]["sha256"],
        expected_python_sha256=contract["python_interpreter"]["sha256"],
        workspace_root=mirror,
    ) as bundle:
        source = mirror / "validation/gpu_lease_v2.py"
        forged = source.with_name("forged-v2.py")
        forged.write_text("raise RuntimeError('pathname substitution executed')\n")
        os.replace(forged, source)
        monkeypatch.setenv("PYTHONPATH", str(poison))
        monkeypatch.setenv("PYTHONSTARTUP", str(tmp_path / "startup.py"))
        completed = bundle.run(["held-status"], timeout=20)
    assert completed.returncode == 0, completed.stderr.decode()
    status = json.loads(completed.stdout)
    assert status == {
        "contract_fingerprint_sha256": contract["contract_fingerprint_sha256"],
        "isolated": True,
        "site_disabled": True,
        "source_sha256": contract["implementation"]["sha256"],
        "status": "held_runtime_verified",
    }
    assert not marker.exists()


def test_held_bundle_rejects_workspace_inode_swap_after_resolve(tmp_path: Path) -> None:
    mirror, contract = mirror_runtime(tmp_path, "root-swap")
    moved = tmp_path / "root-swap-original"

    def swap_root() -> None:
        mirror.rename(moved)
        mirror.mkdir()

    with pytest.raises(lease.LeaseIntegrityError, match="changed after resolve"):
        lease.open_held_execution_bundle(
            expected_contract_fingerprint=contract["contract_fingerprint_sha256"],
            expected_source_sha256=contract["implementation"]["sha256"],
            expected_python_sha256=contract["python_interpreter"]["sha256"],
            workspace_root=mirror,
            _after_root_resolve=swap_root,
        )


def test_held_bundle_rejects_ancestor_swap_after_tree_open(tmp_path: Path) -> None:
    mirror, contract = mirror_runtime(tmp_path, "ancestor-swap")
    validation = mirror / "validation"
    moved = mirror / "validation-original"

    def swap_validation() -> None:
        validation.rename(moved)
        validation.mkdir()

    with pytest.raises(lease.LeaseIntegrityError, match="ancestor changed"):
        lease.open_held_execution_bundle(
            expected_contract_fingerprint=contract["contract_fingerprint_sha256"],
            expected_source_sha256=contract["implementation"]["sha256"],
            expected_python_sha256=contract["python_interpreter"]["sha256"],
            workspace_root=mirror,
            _after_tree_open=swap_validation,
        )


def test_held_bundle_rejects_intermediate_symlink(tmp_path: Path) -> None:
    mirror, contract = mirror_runtime(tmp_path, "symlink-tree")
    schemas = mirror / "validation/schemas"
    moved = mirror / "validation/schemas-real"
    schemas.rename(moved)
    schemas.symlink_to(moved, target_is_directory=True)
    with pytest.raises(lease.LeaseIntegrityError, match="cannot open held runtime directory"):
        lease.open_held_execution_bundle(
            expected_contract_fingerprint=contract["contract_fingerprint_sha256"],
            expected_source_sha256=contract["implementation"]["sha256"],
            expected_python_sha256=contract["python_interpreter"]["sha256"],
            workspace_root=mirror,
        )


def test_held_bundle_rejects_noncanonical_interpreter_substitution(
    tmp_path: Path,
) -> None:
    mirror, contract = mirror_runtime(tmp_path, "interpreter-substitution")
    substitute = tmp_path / "python3.12"
    substitute.write_bytes(lease.TRUSTED_PYTHON.read_bytes())
    substitute.chmod(0o755)
    with pytest.raises(lease.LeaseIntegrityError, match="interpreter path"):
        lease.open_held_execution_bundle(
            expected_contract_fingerprint=contract[
                "contract_fingerprint_sha256"
            ],
            expected_source_sha256=contract["implementation"]["sha256"],
            expected_python_sha256=contract["python_interpreter"]["sha256"],
            workspace_root=mirror,
            python_executable=substitute,
        )


def test_bundle_open_then_validation_rename_cannot_split_lease_domain(
    tmp_path: Path,
) -> None:
    mirror, contract = mirror_runtime(tmp_path, "post-open-ancestor-swap")
    validation = mirror / "validation"
    moved = mirror / "validation-held"
    with lease.open_held_execution_bundle(
        expected_contract_fingerprint=contract["contract_fingerprint_sha256"],
        expected_source_sha256=contract["implementation"]["sha256"],
        expected_python_sha256=contract["python_interpreter"]["sha256"],
        workspace_root=mirror,
    ) as bundle:
        validation.rename(moved)
        shutil.copytree(moved, validation)
        with pytest.raises(lease.LeaseIntegrityError, match="ancestor changed"):
            bundle.run(["held-status"], timeout=20)
        with pytest.raises(lease.LeaseIntegrityError, match="workspace tree identity"):
            lease.open_held_execution_bundle(
                expected_contract_fingerprint=contract[
                    "contract_fingerprint_sha256"
                ],
                expected_source_sha256=contract["implementation"]["sha256"],
                expected_python_sha256=contract["python_interpreter"]["sha256"],
                workspace_root=mirror,
            )


def test_held_isolated_runtime_executes_real_v2_state_and_release_validation(
    tmp_path: Path,
) -> None:
    mirror, contract = mirror_runtime(tmp_path, "held-lifecycle")
    state_root = tmp_path / "held-lifecycle-state"
    with lease.open_held_execution_bundle(
        expected_contract_fingerprint=contract["contract_fingerprint_sha256"],
        expected_source_sha256=contract["implementation"]["sha256"],
        expected_python_sha256=contract["python_interpreter"]["sha256"],
        workspace_root=mirror,
    ) as bundle:
        completed = bundle.run(
            [
                "held-cpu-lifecycle",
                "--state-root",
                str(state_root),
                "--gpu-index",
                "249",
            ],
            timeout=20,
        )
    assert completed.returncode == 0, completed.stderr.decode()
    result = json.loads(completed.stdout)
    assert result == {
        "event_schema": lease.EVENT_VERSION,
        "events": ["acquire", "release"],
        "state_schema": lease.STATE_VERSION,
        "status": "held_cpu_lifecycle_passed",
    }


@pytest.mark.parametrize(
    "unsafe",
    [
        ["--rm"],
        ["--rm=true"],
        ["--label-file=/tmp/labels"],
        ["--label", f"{lease.DOCKER_LABEL_KEY}=forged"],
        ["-l", f"{lease.DOCKER_LABEL_KEY}=forged"],
        ["-d"],
        ["-dit"],
        ["--detach=true"],
        ["--restart=always"],
    ],
)
def test_direct_label_is_supervisor_only_and_unsafe_options_rejected(
    tmp_path: Path, unsafe: list[str]
) -> None:
    intent = private_intent(tmp_path, mode="direct")
    with pytest.raises(lease.LeaseIntegrityError):
        lease.effective_docker_command(
            ["docker", "run", *unsafe, "sha256:" + "a" * 64], intent
        )


def test_direct_label_is_injected_exactly_once_and_nested_exports_exact_env(
    tmp_path: Path,
) -> None:
    direct = private_intent(tmp_path, mode="direct", token="2" * 64)
    requested = ["docker", "run", "--network=none", "image", "true"]
    effective = lease.effective_docker_command(requested, direct)
    assert effective[:3] == [
        str(lease.TRUSTED_DOCKER_CLI),
        "run",
        f"--label={lease.DOCKER_LABEL_KEY}={direct.label_value}",
    ]
    assert sum(item.startswith(f"--label={lease.DOCKER_LABEL_KEY}=") for item in effective) == 1
    nested = private_intent(tmp_path, mode="nested", token="3" * 64)
    assert nested.environment() == {
        lease.DOCKER_ENV_SCHEMA: lease.DOCKER_INTENT_VERSION,
        lease.DOCKER_ENV_MODE: "nested",
        lease.DOCKER_ENV_LABEL_KEY: lease.DOCKER_LABEL_KEY,
        lease.DOCKER_ENV_LABEL_VALUE: "3" * 64,
        lease.DOCKER_ENV_EXECUTABLE: str(lease.TRUSTED_DOCKER_CLI),
    }
    poisoned = {
        "PATH": "/tmp/evil",
        "PYTHONPATH": "/tmp/evil-python",
        "PYTHONSTARTUP": "/tmp/startup",
        "DOCKER_CONTEXT": "evil",
        "DOCKER_HOST": "tcp://evil:2375",
    }
    sanitized = lease._sanitized_runtime_environment(nested.environment())
    assert not (set(poisoned) - {"PATH", "DOCKER_HOST"}) & set(sanitized)
    assert sanitized["PATH"] == "/usr/bin:/bin"
    assert sanitized["DOCKER_HOST"] == lease.TRUSTED_DOCKER_HOST


def test_cid_a_to_b_swap_removes_only_daemon_a_and_never_victim_b(tmp_path: Path) -> None:
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    backend.add(CID_A, intent.label_value)
    intent.cidfile.path.write_text(CID_A, encoding="ascii")
    replacement = intent.cidfile.path.with_name("replacement.cid")
    replacement.write_text(CID_B, encoding="ascii")
    os.replace(replacement, intent.cidfile.path)
    result = cleanup(tmp_path, backend, intent=intent)
    assert backend.removed == [CID_A]
    assert CID_B not in backend.removed
    assert result["observed_ids"] == [CID_A]
    assert result["cidfile_diagnostic_only"]["container_id"] == CID_B
    assert result["cidfile_diagnostic_only"]["matches_authoritative"] is False
    assert result["verified_absent"] is True
    assert result["integrity_passed"] is False


def test_matching_cid_is_diagnostic_not_removal_authority(tmp_path: Path) -> None:
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    backend.add(CID_A, intent.label_value)
    intent.cidfile.path.write_text(CID_A, encoding="ascii")
    result = cleanup(tmp_path, backend, intent=intent)
    assert backend.removed == [CID_A]
    assert result["cidfile_diagnostic_only"]["matches_authoritative"] is True
    assert result["integrity_passed"] is True


def test_multiple_label_matches_are_all_removed_but_integrity_fails(tmp_path: Path) -> None:
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    backend.add(CID_A, intent.label_value)
    backend.add(CID_C, intent.label_value)
    intent.cidfile.path.write_text(CID_A, encoding="ascii")
    result = cleanup(tmp_path, backend, intent=intent)
    assert backend.removed == [CID_A, CID_C]
    assert result["observed_ids"] == [CID_A, CID_C]
    assert result["verified_absent"] is True
    assert result["integrity_passed"] is False


def test_daemon_outage_retries_and_callback_can_observe_held_lease(tmp_path: Path) -> None:
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    backend.add(CID_A, intent.label_value)
    intent.cidfile.path.write_text(CID_A, encoding="ascii")
    backend.identity_script = [OSError("down"), OSError("still down")]
    retries: list[str] = []
    result = cleanup(
        tmp_path,
        backend,
        intent=intent,
        on_retry=lambda exc: retries.append(str(exc)),
    )
    assert len(retries) == 2
    assert backend.removed == [CID_A]
    assert result["verified_absent"] is True


def test_daemon_retry_hook_runs_while_v2_lease_and_flock_are_held(tmp_path: Path) -> None:
    clock = SweepClock()
    value = manager(tmp_path, clock)
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    backend.identity_script = [OSError("daemon temporarily down")]
    args = run_args(
        [sys.executable, "-c", "raise SystemExit(9)"],
        intent.cidfile.path,
        gpu_index=247,
    )
    hook_calls = 0

    def held_retry() -> None:
        nonlocal hook_calls
        hook_calls += 1
        # The persisted state is published only after the daemon baseline, but
        # the repository-global legacy exclusion FD is already held.
        assert value.inspect(gpu_index=247)["status"] == "free"
        legacy = value._paths(247)["legacy"]
        descriptor = os.open(legacy, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        release_events = []
        receipt_root = tmp_path / "leases-v2/gpu-247/receipts"
        for path in receipt_root.glob("*.json"):
            event = json.loads(path.read_text())
            if event["event_type"] == "release":
                release_events.append(event)
        assert release_events == []

    assert run_devnull(
        args,
        value,
        _backend=backend,
        _hooks={"docker_cleanup_retry": held_retry},
        _token_factory=lambda _bytes: intent.label_value,
        _monotonic=clock.monotonic,
        _wall_now=clock.wall_now,
        _sleeper=clock.sleep,
    ) == 9
    assert hook_calls == 1
    assert value.inspect(gpu_index=247)["status"] == "free"


def test_late_cid_resets_quiet_window_and_is_removed(tmp_path: Path) -> None:
    intent = private_intent(tmp_path)
    backend = FakeBackend()

    def late(value: FakeBackend, token: str) -> list[str]:
        value.add(CID_A, token)
        return [CID_A]

    backend.list_script = [[], [], [], [], [], late]
    intent.cidfile.path.write_text(CID_A, encoding="ascii")
    result = cleanup(tmp_path, backend, intent=intent)
    assert backend.removed == [CID_A]
    assert result["quiet_window"]["resets"] == 1
    assert result["quiet_window"]["observed_duration_seconds"] >= 2.0
    assert result["quiet_window"]["empty_queries"] >= 20


def test_success_with_zero_daemon_ids_fails_but_nested_predocker_nonzero_can_release(
    tmp_path: Path,
) -> None:
    success = cleanup(tmp_path, FakeBackend())
    assert success["verified_absent"] is True
    assert success["integrity_passed"] is False
    failed = cleanup(
        tmp_path,
        FakeBackend(),
        outcome="nonzero",
        returncode=7,
    )
    assert failed["observed_ids"] == []
    assert failed["integrity_passed"] is True


@pytest.mark.parametrize(
    "raw",
    [
        b"short\n",
        (CID_A + "\n" + CID_A + "\n").encode(),
        ((CID_A + "\n") * (lease.MAX_DOCKER_IDS + 1)).encode(),
        b"\xff\n",
    ],
)
def test_malformed_duplicate_and_unbounded_docker_id_output_rejected(raw: bytes) -> None:
    with pytest.raises(lease.LeaseIntegrityError):
        lease._strict_id_lines(raw)


def test_inspect_mismatch_and_daemon_drift_retry_fail_closed(tmp_path: Path) -> None:
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    backend.add(CID_A, intent.label_value, inspected_id=CID_B)
    with pytest.raises(lease.LeaseIntegrityError, match="exact container ID"):
        lease._inspect_labeled_container(backend, intent, CID_A)
    drifted = dict(backend.daemon, daemon_id="other-daemon")
    backend.identity_script = [drifted, backend.daemon]
    retries: list[str] = []
    backend.containers.clear()
    result = cleanup(
        tmp_path,
        backend,
        intent=intent,
        outcome="nonzero",
        returncode=1,
        on_retry=lambda exc: retries.append(str(exc)),
    )
    assert any("drifted" in item for item in retries)
    assert result["verified_absent"] is True


def test_sigkill_stale_supervisor_is_recovered_from_sealed_label_hash(
    tmp_path: Path,
) -> None:
    clock = SweepClock()
    identity = FakeIdentity()
    value = manager(tmp_path, clock, identity=identity)
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    crash_supervised_session(
        value,
        gpu_index=250,
        intent=intent,
        backend=backend,
        clock=clock,
    )
    state = value.inspect(gpu_index=250)["state"]
    assert state["lease"]["docker_supervision"]["daemon"] == backend.daemon
    assert intent.label_value not in json.dumps(state)
    backend.add(CID_A, intent.label_value)
    identity.present = False
    clock.sleep(6)

    recovered = value.recover_stale_supervised(
        gpu_index=250,
        backend=backend,
        monotonic=clock.monotonic,
        wall_now=clock.wall_now,
        sleeper=clock.sleep,
        retry_seconds=0.1,
    )
    assert backend.removed == [CID_A]
    assert recovered["docker_recovery"]["matching_ids"] == [CID_A]
    assert recovered["docker_recovery"]["verified_absent"] is True
    assert value.inspect(gpu_index=250)["status"] == "free"
    receipts = [
        json.loads(path.read_text())
        for path in (tmp_path / "leases-v2/gpu-250/receipts").glob("*.json")
    ]
    event = next(
        item for item in receipts if item["event_type"] == "stale_supervised_recovery"
    )
    assert event["docker_recovery"] == recovered["docker_recovery"]
    assert event["next_lease_record_sha256"] is None


def test_stale_recovery_late_container_resets_quiet_window(tmp_path: Path) -> None:
    clock = SweepClock()
    identity = FakeIdentity()
    value = manager(tmp_path, clock, identity=identity)
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    crash_supervised_session(
        value,
        gpu_index=251,
        intent=intent,
        backend=backend,
        clock=clock,
    )
    identity.present = False
    clock.sleep(6)

    def late(candidate: FakeBackend) -> list[str]:
        candidate.add(CID_A, intent.label_value)
        return [CID_A]

    backend.key_list_script = [[], [], [], [], [], late]
    recovered = value.recover_stale_supervised(
        gpu_index=251,
        backend=backend,
        monotonic=clock.monotonic,
        wall_now=clock.wall_now,
        sleeper=clock.sleep,
        retry_seconds=0.1,
    )
    assert backend.removed == [CID_A]
    assert recovered["docker_recovery"]["quiet_window"]["resets"] == 1
    assert recovered["docker_recovery"]["quiet_window"]["empty_queries"] >= 20


def test_stale_recovery_daemon_outage_retries_with_global_lock_held(
    tmp_path: Path,
) -> None:
    clock = SweepClock()
    identity = FakeIdentity()
    value = manager(tmp_path, clock, identity=identity)
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    crash_supervised_session(
        value,
        gpu_index=252,
        intent=intent,
        backend=backend,
        clock=clock,
    )
    backend.add(CID_A, intent.label_value)
    identity.present = False
    clock.sleep(6)
    backend.identity_script = [OSError("daemon down"), OSError("daemon still down")]
    retries: list[str] = []

    def retry(error: BaseException) -> None:
        retries.append(str(error))
        legacy = value._paths(252)["legacy"]
        descriptor = os.open(legacy, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)

    recovered = value.recover_stale_supervised(
        gpu_index=252,
        backend=backend,
        monotonic=clock.monotonic,
        wall_now=clock.wall_now,
        sleeper=clock.sleep,
        retry_seconds=0.1,
        on_retry=retry,
    )
    assert len(retries) == 2
    assert recovered["docker_recovery"]["matching_ids"] == [CID_A]


def test_stale_recovery_hash_mismatch_is_never_a_removal_target(
    tmp_path: Path,
) -> None:
    clock = SweepClock()
    identity = FakeIdentity()
    value = manager(tmp_path, clock, identity=identity)
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    crash_supervised_session(
        value,
        gpu_index=253,
        intent=intent,
        backend=backend,
        clock=clock,
    )
    other_token = "9" * 64
    assert hashlib.sha256(other_token.encode()).hexdigest() != intent.label_value_sha256
    backend.add(CID_B, other_token)
    identity.present = False
    clock.sleep(6)
    recovered = value.recover_stale_supervised(
        gpu_index=253,
        backend=backend,
        monotonic=clock.monotonic,
        wall_now=clock.wall_now,
        sleeper=clock.sleep,
        retry_seconds=0.1,
    )
    evidence = recovered["docker_recovery"]
    assert backend.removed == []
    assert evidence["matching_ids"] == []
    assert evidence["mismatched_ids"] == [CID_B]
    assert CID_B in backend.containers


def test_stale_recovery_synthetic_hash_collision_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = SweepClock()
    identity = FakeIdentity()
    value = manager(tmp_path, clock, identity=identity)
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    crash_supervised_session(
        value,
        gpu_index=253,
        intent=intent,
        backend=backend,
        clock=clock,
    )
    backend.add(CID_A, intent.label_value)
    backend.add(CID_B, "9" * 64)
    identity.present = False
    clock.sleep(6)
    expected_label_sha = intent.label_value_sha256
    monkeypatch.setattr(
        lease,
        "_docker_label_sha256",
        lambda _value: expected_label_sha,
    )
    with pytest.raises(lease.DockerAttributionError, match="hash collision"):
        value.recover_stale_supervised(
            gpu_index=253,
            backend=backend,
            monotonic=clock.monotonic,
            wall_now=clock.wall_now,
            sleeper=clock.sleep,
            retry_seconds=0.1,
        )
    assert backend.removed == []
    assert value.inspect(gpu_index=253)["status"] == "stale_recoverable"


def test_stale_recovery_removes_all_exact_full_id_matches(tmp_path: Path) -> None:
    clock = SweepClock()
    identity = FakeIdentity()
    value = manager(tmp_path, clock, identity=identity)
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    crash_supervised_session(
        value,
        gpu_index=254,
        intent=intent,
        backend=backend,
        clock=clock,
    )
    backend.add(CID_A, intent.label_value)
    backend.add(CID_C, intent.label_value)
    identity.present = False
    clock.sleep(6)
    recovered = value.recover_stale_supervised(
        gpu_index=254,
        backend=backend,
        monotonic=clock.monotonic,
        wall_now=clock.wall_now,
        sleeper=clock.sleep,
        retry_seconds=0.1,
    )
    assert backend.removed == [CID_A, CID_C]
    assert recovered["docker_recovery"]["matching_ids"] == [CID_A, CID_C]


def test_reboot_with_daemon_identity_change_fails_closed(tmp_path: Path) -> None:
    clock = SweepClock()
    identity = FakeIdentity()
    boot = [BOOT_ID]
    value = manager(tmp_path, clock, identity=identity, boot=boot)
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    crash_supervised_session(
        value,
        gpu_index=255,
        intent=intent,
        backend=backend,
        clock=clock,
    )
    backend.add(CID_A, intent.label_value)
    boot[0] = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    backend.daemon = dict(backend.daemon, daemon_id="replacement-daemon")
    with pytest.raises(lease.DockerIdentityDriftError, match="exact original daemon"):
        value.recover_stale_supervised(
            gpu_index=255,
            backend=backend,
            monotonic=clock.monotonic,
            wall_now=clock.wall_now,
            sleeper=clock.sleep,
            retry_seconds=0.1,
        )
    assert backend.removed == []
    assert value.inspect(gpu_index=255)["status"] == "stale_recoverable"


def test_release_missing_or_forged_cleanup_is_rejected_and_state_remains(
    tmp_path: Path,
) -> None:
    clock = SweepClock()
    value = manager(tmp_path, clock)
    intent = private_intent(tmp_path)
    baseline_backend = FakeBackend()
    session = value.session(
        gpu_index=241,
        owner_kind="legacy_validation",
        command=[sys.executable, "nested.py"],
        docker_supervision=intent,
        docker_backend=baseline_backend,
        docker_sleeper=clock.sleep,
        ttl_seconds=5,
    )
    session.__enter__()
    with pytest.raises(lease.LeaseIntegrityError, match="release failed closed"):
        session.__exit__(None, None, None)
    assert value.inspect(gpu_index=241)["status"].startswith("active")

    # Use another GPU because the fail-closed state above is intentionally live.
    session = value.session(
        gpu_index=242,
        owner_kind="legacy_validation",
        command=[sys.executable, "nested.py"],
        docker_supervision=intent,
        docker_backend=FakeBackend(),
        docker_sleeper=clock.sleep,
        ttl_seconds=5,
    )
    session.__enter__()
    forged = cleanup(
        tmp_path,
        FakeBackend(),
        intent=intent,
        outcome="nonzero",
        returncode=1,
        clock=clock,
    )
    forged["label_value_sha256"] = "f" * 64
    session.set_docker_cleanup(forged)
    with pytest.raises(lease.LeaseIntegrityError, match="release failed closed"):
        session.__exit__(None, None, None)
    assert value.inspect(gpu_index=242)["status"].startswith("active")


def test_release_receipt_binds_exact_cleanup_and_occurs_after_absence(tmp_path: Path) -> None:
    clock = SweepClock()
    value = manager(tmp_path, clock)
    intent = private_intent(tmp_path)
    backend = FakeBackend()
    session = value.session(
        gpu_index=243,
        owner_kind="legacy_validation",
        command=[sys.executable, "nested.py"],
        docker_supervision=intent,
        docker_backend=backend,
        docker_sleeper=clock.sleep,
        ttl_seconds=5,
    )
    with session:
        backend.add(CID_A, intent.label_value)
        intent.cidfile.path.write_text(CID_A, encoding="ascii")
        attestation = cleanup(
            tmp_path, backend, intent=intent, clock=clock
        )
        session.set_docker_cleanup(attestation)
        assert value.inspect(gpu_index=243)["status"] == "active"
    assert value.inspect(gpu_index=243)["status"] == "free"
    receipts = sorted((tmp_path / "leases-v2/gpu-243/receipts").glob("*.json"))
    events = [json.loads(path.read_text()) for path in receipts]
    release_event = next(item for item in events if item["event_type"] == "release")
    assert release_event["docker_cleanup"] == attestation
    assert release_event["docker_cleanup"]["verified_absent"] is True
    assert release_event["created_at_utc"] >= attestation["verified_absent_at_utc"]


def test_nested_nonzero_without_container_finalizes_and_releases(tmp_path: Path) -> None:
    clock = SweepClock()
    value = manager(tmp_path, clock)
    intent = private_intent(tmp_path)
    args = run_args(
        [sys.executable, "-c", "raise SystemExit(7)"],
        intent.cidfile.path,
        gpu_index=244,
    )
    assert run_devnull(
        args,
        value,
        _backend=FakeBackend(),
        _token_factory=lambda _bytes: intent.label_value,
        _monotonic=clock.monotonic,
        _wall_now=clock.wall_now,
        _sleeper=clock.sleep,
    ) == 7
    assert value.inspect(gpu_index=244)["status"] == "free"


def test_popen_failure_still_finalizes_absence_before_release(tmp_path: Path) -> None:
    clock = SweepClock()
    value = manager(tmp_path, clock)
    intent = private_intent(tmp_path)
    args = run_args(
        ["/definitely/missing/deepsafe-v2-child"],
        intent.cidfile.path,
        gpu_index=248,
    )
    with pytest.raises(lease.LeaseIntegrityError, match="launch failed"):
        run_devnull(
            args,
            value,
            _backend=FakeBackend(),
            _token_factory=lambda _bytes: intent.label_value,
            _monotonic=clock.monotonic,
            _wall_now=clock.wall_now,
            _sleeper=clock.sleep,
        )
    assert value.inspect(gpu_index=248)["status"] == "free"
    events = [
        json.loads(path.read_text())
        for path in (tmp_path / "leases-v2/gpu-248/receipts").glob("*.json")
    ]
    released = next(item for item in events if item["event_type"] == "release")
    assert released["docker_cleanup"]["child"]["outcome"] == "popen_failure"
    assert released["docker_cleanup"]["verified_absent"] is True


def test_timeout_kills_group_then_cleans_daemon_before_release(tmp_path: Path) -> None:
    clock = SweepClock()
    value = manager(tmp_path, clock)
    intent = private_intent(tmp_path)
    backend = FakeBackend()

    def appear(value: FakeBackend, token: str) -> list[str]:
        value.add(CID_A, token)
        return [CID_A]

    backend.list_script = [[], appear]
    command = [
        sys.executable,
        "-c",
        (
            "import os,pathlib,time; "
            "pathlib.Path(os.environ['DEEPSAFE_MANAGED_DOCKER_CIDFILE'])"
            ".write_text('a'*64); time.sleep(30)"
        ),
    ]
    args = run_args(command, intent.cidfile.path, gpu_index=245, timeout=0.2)
    with pytest.raises(lease.LeaseIntegrityError, match="timeout"):
        run_devnull(
            args,
            value,
            _backend=backend,
            _token_factory=lambda _bytes: intent.label_value,
            _monotonic=clock.monotonic,
            _wall_now=clock.wall_now,
            _sleeper=clock.sleep,
        )
    assert backend.removed == [CID_A]
    assert value.inspect(gpu_index=245)["status"] == "free"


def test_signal_kills_group_then_cleans_daemon_before_release(tmp_path: Path) -> None:
    clock = SweepClock()
    value = manager(tmp_path, clock)
    intent = private_intent(tmp_path)
    backend = FakeBackend()

    def appear(value: FakeBackend, token: str) -> list[str]:
        value.add(CID_A, token)
        return [CID_A]

    backend.list_script = [[], appear]
    command = [
        sys.executable,
        "-c",
        (
            "import os,pathlib,time; "
            "pathlib.Path(os.environ['DEEPSAFE_MANAGED_DOCKER_CIDFILE'])"
            ".write_text('a'*64); time.sleep(30)"
        ),
    ]
    args = run_args(command, intent.cidfile.path, gpu_index=246)

    def interrupt() -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    result = run_devnull(
        args,
        value,
        _backend=backend,
        _hooks={"child_published": interrupt},
        _token_factory=lambda _bytes: intent.label_value,
        _monotonic=clock.monotonic,
        _wall_now=clock.wall_now,
        _sleeper=clock.sleep,
    )
    assert result == 128 + signal.SIGTERM
    assert backend.removed == [CID_A]
    assert value.inspect(gpu_index=246)["status"] == "free"
