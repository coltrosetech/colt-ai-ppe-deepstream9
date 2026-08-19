from __future__ import annotations

import copy
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from validation import gpu_lease as v1
from validation import gpu_lease_v4 as v4
from validation import gpu_lease_v6 as lease


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = lease.R7_GPU_UUID
IMAGE_DIGEST = "sha256:" + "a" * 64
IMAGE_REFERENCE = "registry.invalid/deepsafe/test@" + IMAGE_DIGEST


def _artifact_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    info = os.lstat(path)
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": f"{info.st_mode & 0o7777:04o}",
    }


def _materialize_plan(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    artifact = tmp_path / "model.engine"
    artifact.write_bytes(b"CPU-only fake model artifact\n")
    artifact.chmod(0o600)
    contract = lease.load_contract()
    argv = ["/usr/bin/docker", "run", "--network=none", IMAGE_REFERENCE]
    workload: dict[str, Any] = {
        "gpu": {"index": lease.R7_GPU_INDEX, "uuid": GPU_UUID},
        "owner_kind": "capacity_5min",
        "ttl_seconds": 30,
        "timeout_seconds": 300,
        "image": {"reference": IMAGE_REFERENCE, "digest": IMAGE_DIGEST},
        "image_argv_index": 3,
        "argv": argv,
        "argv_sha256": v1.command_argv_sha256(argv),
        "artifacts": [_artifact_pin(artifact)],
    }
    acceptance: dict[str, Any] = {
        "schema_version": "deepsafe.gpu-lease-v6-outer-acceptance/v1",
        "acceptance_id": "c" * 64,
        "created_at_utc": "2026-07-20T15:00:00Z",
        "user_notification": {
            "confirmed": True,
            "notification_id": "d" * 64,
            "notified_at_utc": "2026-07-20T14:59:00Z",
        },
        "workload_acceptance": {
            "accepted": True,
            "plan_id": "b" * 64,
            "gpu": workload["gpu"],
            "owner_kind": workload["owner_kind"],
            "argv_sha256": workload["argv_sha256"],
            "image": workload["image"],
        },
        "authority": {
            "decision": "ACCEPT",
            "scope": "exact-workload-plan-v6",
            "nontransferable": True,
        },
    }
    acceptance["acceptance_fingerprint_sha256"] = v1.canonical_sha256(acceptance)
    acceptance_path = tmp_path / "outer-acceptance.json"
    acceptance_raw = v1.canonical_bytes(acceptance) + b"\n"
    acceptance_path.write_bytes(acceptance_raw)
    acceptance_path.chmod(0o440)
    plan: dict[str, Any] = {
        "schema_version": lease.PLAN_VERSION,
        "plan_id": "b" * 64,
        "created_at": "2026-07-18T12:00:00Z",
        "activation_contract_fingerprint_sha256": contract[
            "contract_fingerprint_sha256"
        ],
        "execution_authorized": True,
        "v5_predecessor": {
            "contract": contract["base_v5"]["artifact"],
            "contract_fingerprint_sha256": lease.V5_CONTRACT_FINGERPRINT,
            "implementation": contract["base_v5"]["implementation"],
            "activation_eligible": False,
            "tool_drift_reason": contract["predecessor_tool_drift"]["reason"],
        },
        "v4_authority": {
            "contract": contract["execution_base_v4"]["artifact"],
            "contract_fingerprint_sha256": lease.V4_CONTRACT_FINGERPRINT,
            "implementation": contract["execution_base_v4"]["implementation"],
            "python_interpreter": contract["python_interpreter"],
        },
        "driver_prerequisite": contract["driver_prerequisite"],
        "outer_acceptance": {
            "receipt": _artifact_pin(acceptance_path),
            "acceptance_fingerprint_sha256": acceptance[
                "acceptance_fingerprint_sha256"
            ],
        },
        "tools": contract["tools"],
        "workload": workload,
    }
    plan["plan_fingerprint_sha256"] = v1.canonical_sha256(plan)
    path = tmp_path / "authorized-plan.json"
    raw = v1.canonical_bytes(plan) + b"\n"
    path.write_bytes(raw)
    path.chmod(0o600)
    expected = {
        "expected_plan_bytes": len(raw),
        "expected_plan_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_plan_fingerprint": plan["plan_fingerprint_sha256"],
        "expected_gpu_index": 0,
        "expected_gpu_uuid": GPU_UUID,
        "expected_owner_kind": "capacity_5min",
        "expected_argv_sha256": plan["workload"]["argv_sha256"],
        "expected_image_reference": IMAGE_REFERENCE,
        "expected_image_digest": IMAGE_DIGEST,
        "expected_outer_acceptance_path": str(acceptance_path),
        "expected_outer_acceptance_bytes": len(acceptance_raw),
        "expected_outer_acceptance_sha256": hashlib.sha256(
            acceptance_raw
        ).hexdigest(),
        "expected_outer_acceptance_fingerprint": acceptance[
            "acceptance_fingerprint_sha256"
        ],
    }
    return path, plan, expected


def _rewrite_plan(path: Path, plan: dict[str, Any], *, fingerprint: bool = True) -> dict[str, Any]:
    plan = copy.deepcopy(plan)
    if fingerprint:
        plan.pop("plan_fingerprint_sha256", None)
        plan["plan_fingerprint_sha256"] = v1.canonical_sha256(plan)
    raw = v1.canonical_bytes(plan) + b"\n"
    path.write_bytes(raw)
    path.chmod(0o600)
    return {
        "expected_plan_bytes": len(raw),
        "expected_plan_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _rewrite_outer_acceptance(
    path: Path, value: dict[str, Any], *, fingerprint: bool = True
) -> tuple[dict[str, Any], bytes]:
    value = copy.deepcopy(value)
    if fingerprint:
        value.pop("acceptance_fingerprint_sha256", None)
        value["acceptance_fingerprint_sha256"] = v1.canonical_sha256(value)
    path.chmod(0o600)
    raw = v1.canonical_bytes(value) + b"\n"
    path.write_bytes(raw)
    path.chmod(0o440)
    return value, raw


def _verify(path: Path, expected: dict[str, Any]) -> lease.VerifiedWorkloadPlan:
    return lease.verify_workload_plan(path, **expected)


def test_contract_is_closed_and_frozen_v4_authority_is_exact() -> None:
    contract = lease.load_contract()
    assert contract["status"] == "locked-no-live-plan"
    assert contract["execution_base_v4"]["contract_fingerprint_sha256"] == lease.V4_CONTRACT_FINGERPRINT
    assert contract["execution_base_v4"]["artifact"]["sha256"] == lease.V4_CONTRACT_SHA256
    assert contract["execution_base_v4"]["implementation"]["sha256"] == lease.V4_SOURCE_SHA256
    assert contract["base_v5"]["contract_fingerprint_sha256"] == lease.V5_CONTRACT_FINGERPRINT
    assert contract["predecessor_tool_drift"]["predecessor_activation_eligible_on_current_host"] is False
    assert contract["driver_prerequisite"]["boot_id"] == lease.R7_BOOT_ID
    assert hashlib.sha256((ROOT / "validation/gpu_lease_v4.py").read_bytes()).hexdigest() == lease.V4_SOURCE_SHA256
    assert hashlib.sha256((ROOT / "validation/contracts/gpu-lease-v4.json").read_bytes()).hexdigest() == lease.V4_CONTRACT_SHA256
    assert lease.GPU_LEASE_V6_API_READY is False
    assert lease.LIVE_PLAN_AUTHORIZED is False
    assert lease.USER_NOTIFICATION_ACCEPTED is False
    assert lease.OUTER_WORKLOAD_ACCEPTED is False
    assert lease.DEFAULT_WORKLOAD_PLAN is None
    assert lease.PUBLISHED_LIVE_PLAN is None
    assert v4.GPU_LEASE_V4_API_READY is False
    assert v4.LIVE_PLAN_AUTHORIZED is False


def test_cli_has_no_plan_default_and_is_closed(capsys: pytest.CaptureFixture[str]) -> None:
    assert lease.main(["status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "closed_no_live_plan"
    assert status["default_plan"] is None
    with pytest.raises(SystemExit):
        lease.build_parser().parse_args(["launch"])


def test_verify_is_separate_and_never_activates_parent(tmp_path: Path) -> None:
    path, _plan, expected = _materialize_plan(tmp_path)
    with _verify(path, expected) as verified:
        assert verified.projection["plan_fingerprint_sha256"] == expected[
            "expected_plan_fingerprint"
        ]
        assert lease.RESERVED_PLAN_LABEL in verified.projection["effective_argv"][2]
        assert v4.GPU_LEASE_V4_API_READY is False
        assert v4.LIVE_PLAN_AUTHORIZED is False


@pytest.mark.parametrize("bad", ["duplicate", "nan", "infinity"])
def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path: Path, bad: str) -> None:
    path, _plan, expected = _materialize_plan(tmp_path)
    if bad == "duplicate":
        raw = b'{"schema_version":"x","schema_version":"y"}'
    elif bad == "nan":
        raw = b'{"x":NaN}'
    else:
        raw = b'{"x":Infinity}'
    path.write_bytes(raw)
    path.chmod(0o600)
    expected.update(
        expected_plan_bytes=len(raw), expected_plan_sha256=hashlib.sha256(raw).hexdigest()
    )
    with pytest.raises(lease.LeaseIntegrityError, match="strict JSON"):
        _verify(path, expected)


def test_external_file_pin_and_self_fingerprint_are_both_required(tmp_path: Path) -> None:
    path, plan, expected = _materialize_plan(tmp_path)
    wrong = dict(expected, expected_plan_sha256="0" * 64)
    with pytest.raises(lease.LeaseIntegrityError, match="external plan file pin"):
        _verify(path, wrong)
    plan["owner_kind"] = "ignored-extra-root-field"
    updated = _rewrite_plan(path, plan, fingerprint=False)
    expected.update(updated)
    with pytest.raises(lease.LeaseIntegrityError):
        _verify(path, expected)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("expected_gpu_index", 1, "caller workload projection"),
        ("expected_gpu_uuid", "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "caller workload projection"),
        ("expected_owner_kind", "person_training", "caller workload projection"),
        ("expected_argv_sha256", "0" * 64, "caller workload projection"),
        ("expected_image_reference", "other.invalid/x@" + IMAGE_DIGEST, "caller workload projection"),
        ("expected_image_digest", "sha256:" + "c" * 64, "caller workload projection"),
    ],
)
def test_caller_gpu_owner_command_and_image_drift_fail_closed(
    tmp_path: Path, field: str, value: Any, match: str
) -> None:
    path, _plan, expected = _materialize_plan(tmp_path)
    expected[field] = value
    with pytest.raises(lease.LeaseAuthorizationError, match=match):
        _verify(path, expected)


def test_plan_schema_extra_field_and_artifact_tamper_fail_closed(tmp_path: Path) -> None:
    path, plan, expected = _materialize_plan(tmp_path)
    plan["unexpected"] = True
    updated = _rewrite_plan(path, plan)
    expected.update(updated, expected_plan_fingerprint=plan.get("plan_fingerprint_sha256", expected["expected_plan_fingerprint"]))
    # Use the rewritten plan's actual semantic fingerprint.
    decoded = json.loads(path.read_bytes())
    expected["expected_plan_fingerprint"] = decoded["plan_fingerprint_sha256"]
    with pytest.raises(lease.LeaseIntegrityError, match="additional properties"):
        _verify(path, expected)

    path, plan, expected = _materialize_plan(tmp_path / "second")
    artifact = Path(plan["workload"]["artifacts"][0]["path"])
    artifact.write_bytes(b"tampered")
    with pytest.raises(lease.LeaseIntegrityError, match="exact pin differs"):
        _verify(path, expected)


def test_symlink_plan_and_artifact_paths_are_rejected(tmp_path: Path) -> None:
    path, plan, expected = _materialize_plan(tmp_path)
    alias = tmp_path / "plan-link.json"
    alias.symlink_to(path)
    with pytest.raises((lease.LeaseIntegrityError, OSError)):
        _verify(alias, expected)
    artifact = Path(plan["workload"]["artifacts"][0]["path"])
    target = tmp_path / "target.engine"
    artifact.rename(target)
    artifact.symlink_to(target)
    with pytest.raises((lease.LeaseIntegrityError, OSError)):
        _verify(path, expected)


def test_double_leading_slash_is_not_literal_canonical(tmp_path: Path) -> None:
    path, _plan, expected = _materialize_plan(tmp_path)
    double = Path("/" + str(path))
    assert str(double).startswith("//")
    with pytest.raises(lease.LeaseIntegrityError, match="literal canonical"):
        _verify(double, expected)


def test_post_verification_plan_path_replacement_blocks_spawn(tmp_path: Path) -> None:
    path, _plan, expected = _materialize_plan(tmp_path)
    verified = _verify(path, expected)
    try:
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(path.read_bytes())
        replacement.chmod(0o600)
        os.replace(replacement, path)
        with pytest.raises(lease.LeaseIntegrityError, match="identity drifted"):
            lease.execute_workload_plan(verified, simulate=True)
    finally:
        verified.close()


def test_numeric_fd_reuse_is_not_closed_by_ambiguous_cleanup(tmp_path: Path) -> None:
    path, _plan, expected = _materialize_plan(tmp_path)
    verified = _verify(path, expected)
    stolen = verified.plan_file.fd
    os.close(stolen)
    replacement = os.open("/dev/null", os.O_RDONLY)
    assert replacement == stolen
    with pytest.raises(lease.LeaseIntegrityError, match="ownership lost"):
        verified.close()
    os.fstat(replacement)
    os.close(replacement)
    os.close(verified.plan_file.guard_fd)


def test_same_inode_numeric_fd_reuse_is_distinguished_by_ofd_guard(tmp_path: Path) -> None:
    path, _plan, expected = _materialize_plan(tmp_path)
    verified = _verify(path, expected)
    stolen = verified.plan_file.fd
    os.close(stolen)
    replacement = os.open(path, os.O_RDONLY)
    assert replacement == stolen
    with pytest.raises(lease.LeaseIntegrityError, match="ownership lost"):
        verified.close()
    os.fstat(replacement)
    os.close(replacement)
    os.close(verified.plan_file.guard_fd)


def test_reused_guard_fd_is_preserved_and_primary_is_not_closed(tmp_path: Path) -> None:
    path, _plan, expected = _materialize_plan(tmp_path)
    verified = _verify(path, expected)
    primary = verified.plan_file.fd
    guard = verified.plan_file.guard_fd
    os.close(guard)
    replacement = os.open(path, os.O_RDONLY)
    assert replacement == guard
    with pytest.raises(lease.LeaseIntegrityError, match="ownership lost"):
        verified.close()
    os.fstat(primary)
    os.fstat(replacement)
    os.close(primary)
    os.close(replacement)


def test_v4_short_value_canonicalization_drift_is_rejected(tmp_path: Path) -> None:
    path, plan, expected = _materialize_plan(tmp_path)
    argv = [
        "/usr/bin/docker", "run", "-eFOO=bar", "--network=none", IMAGE_REFERENCE
    ]
    plan["workload"]["argv"] = argv
    plan["workload"]["argv_sha256"] = v1.command_argv_sha256(argv)
    plan["workload"]["image_argv_index"] = 4
    updated = _rewrite_plan(path, plan)
    decoded = json.loads(path.read_bytes())
    expected.update(
        updated,
        expected_plan_fingerprint=decoded["plan_fingerprint_sha256"],
        expected_argv_sha256=decoded["workload"]["argv_sha256"],
    )
    with pytest.raises(lease.LeaseIntegrityError, match="canonical form"):
        _verify(path, expected)


def test_simulated_held_activation_ignores_parent_v4_gate_monkeypatch(tmp_path: Path) -> None:
    path, _plan, expected = _materialize_plan(tmp_path)
    original = (v4.GPU_LEASE_V4_API_READY, v4.LIVE_PLAN_AUTHORIZED)
    v4.GPU_LEASE_V4_API_READY = True
    v4.LIVE_PLAN_AUTHORIZED = True
    try:
        with _verify(path, expected) as verified:
            completed = lease.execute_workload_plan(verified, simulate=True)
        assert completed.returncode == 0, completed.stderr.decode()
        result = json.loads(completed.stdout)
        assert result["status"] == "simulated_held_activation_passed"
        assert result["gpu_probe"] == "simulated-no-device-call"
        assert result["v4_api_after_replay"] is True
        assert result["v4_live_after_replay"] is True
        assert v4.GPU_LEASE_V4_API_READY is True
        assert v4.LIVE_PLAN_AUTHORIZED is True
    finally:
        v4.GPU_LEASE_V4_API_READY, v4.LIVE_PLAN_AUTHORIZED = original


def test_per_child_fds_support_concurrent_simulated_activation(tmp_path: Path) -> None:
    path, _plan, expected = _materialize_plan(tmp_path)
    with _verify(path, expected) as verified:
        def run_once(_index: int) -> bool:
            completed = lease.execute_workload_plan(verified, simulate=True)
            return completed.returncode == 0 and json.loads(completed.stdout)["isolated"] is True

        with ThreadPoolExecutor(max_workers=4) as pool:
            assert all(pool.map(run_once, range(8)))


def test_activation_receipt_binds_plan_effective_command_and_v4_terminal_pin(
    tmp_path: Path,
) -> None:
    path, _plan, expected = _materialize_plan(tmp_path)
    with _verify(path, expected) as verified:
        owner = {
            "pid": 1234,
            "uid": os.getuid(),
            "boot_id": "11111111-2222-3333-4444-555555555555",
            "start_ticks": 99,
            "identity_sha256": "9" * 64,
        }
        scope = {
            "scope_id": "a" * 64,
            "pid": 1234,
            "uid": os.getuid(),
            "boot_id": "11111111-2222-3333-4444-555555555555",
            "start_ticks": 99,
            "session_id": 1234,
            "process_group": 1234,
            "cgroup_path": "/test",
            "cgroup_ns_dev": 1,
            "cgroup_ns_ino": 2,
            "identity_sha256": "0" * 64,
        }
        scope["identity_sha256"] = v1.canonical_sha256(
            {
                key: item for key, item in scope.items()
                if key != "identity_sha256"
            }
        )
        v2_requested = "3" * 64
        v4_effective = "4" * 64
        daemon = {
            "daemon_id": "daemon-test",
            "name": "test",
            "server_version": "test",
            "docker_root_dir": "/var/lib/docker",
            "docker_cli": verified.contract["tools"]["docker_cli"],
            "environment_sha256": "d" * 64,
            "context_sha256": "e" * 64,
        }
        active_state: dict[str, Any] = {
            "schema_version": v4.STATE_VERSION,
            "contract_fingerprint_sha256": lease.V4_CONTRACT_FINGERPRINT,
            "phase": "active",
            "gpu_index": 0,
            "run_id": "1" * 64,
            "owner": owner,
            "scope": scope,
            "command": {
                "requested_argv": verified.projection["effective_argv"],
                "requested_argv_sha256": verified.projection["effective_argv_sha256"],
                "v2_requested_argv_sha256": v2_requested,
                "effective_argv_sha256": v4_effective,
            },
            "inner_v2": {
                "root": "/held/inner-v2",
                "lease_id": "b" * 64,
                "command_argv_sha256": v2_requested,
                "label_key": "com.deepsafe.gpu-lease.supervision.v2",
                "label_value_sha256": "c" * 64,
                "daemon": daemon,
                "gpu_uuid": GPU_UUID,
            },
            "last_event": None,
            "created_at_utc": "2026-07-18T12:00:00Z",
            "created_monotonic_ns": 1,
            "state_fingerprint_sha256": "0" * 64,
        }
        active_pre_acquire = {
            key: item for key, item in active_state.items()
            if key != "state_fingerprint_sha256"
        }
        acquire_event: dict[str, Any] = {
            "schema_version": v4.EVENT_VERSION,
            "contract_fingerprint_sha256": lease.V4_CONTRACT_FINGERPRINT,
            "event_type": "acquire",
            "event_id": "6" * 64,
            "run_id": active_state["run_id"],
            "gpu_index": 0,
            "scope": scope,
            "state_core_sha256": v1.canonical_sha256(active_pre_acquire),
            "previous_state_fingerprint_sha256": "f" * 64,
            "host_scope_cleanup": None,
            "docker_evidence": {
                "kind": "inner_acquired",
                "inner_lease_id": "b" * 64,
                "terminal_event_type": None,
                "terminal_receipt": None,
                "absence_evidence_sha256": None,
                "receipt_scan_sha256": v1.canonical_sha256([]),
                "verified_absent": False,
            },
            "created_at_utc": "2026-07-18T12:00:01Z",
            "created_monotonic_ns": 2,
            "event_fingerprint_sha256": "0" * 64,
        }
        acquire_event["event_fingerprint_sha256"] = v1.canonical_sha256(
            {
                key: item for key, item in acquire_event.items()
                if key != "event_fingerprint_sha256"
            }
        )
        acquire_raw = v1.canonical_bytes(acquire_event) + b"\n"
        acquire_pin = {
            "path": "receipts/" + "6" * 64 + ".json",
            "bytes": len(acquire_raw),
            "sha256": hashlib.sha256(acquire_raw).hexdigest(),
            "event_fingerprint_sha256": acquire_event[
                "event_fingerprint_sha256"
            ],
        }
        active_state["last_event"] = acquire_pin
        active_state["state_fingerprint_sha256"] = v1.canonical_sha256(
            {
                key: item for key, item in active_state.items()
                if key != "state_fingerprint_sha256"
            }
        )
        active_raw = v1.canonical_bytes(active_state) + b"\n"
        state_binding = {
            "run_id": "1" * 64,
            "state_fingerprint_sha256": active_state["state_fingerprint_sha256"],
            "requested_argv_sha256": verified.projection["effective_argv_sha256"],
            "v2_requested_argv_sha256": v2_requested,
            "effective_argv_sha256": v4_effective,
            "scope_identity_sha256": scope["identity_sha256"],
            "active_state": active_state,
            "active_state_file": {
                "path": "validation/results/gpu-leases/v4/gpu-0/active.json",
                "bytes": len(active_raw),
                "sha256": hashlib.sha256(active_raw).hexdigest(),
            },
            "acquire_receipt": {
                "path": (
                    "validation/results/gpu-leases/v4/gpu-0/receipts/"
                    + "6" * 64 + ".json"
                ),
                "bytes": acquire_pin["bytes"],
                "sha256": acquire_pin["sha256"],
                "event_fingerprint_sha256": acquire_pin[
                    "event_fingerprint_sha256"
                ],
            },
            "acquire_event": acquire_event,
        }
        terminal_event: dict[str, Any] = {
            "schema_version": v4.EVENT_VERSION,
            "contract_fingerprint_sha256": lease.V4_CONTRACT_FINGERPRINT,
            "event_type": "release",
            "event_id": "d" * 64,
            "run_id": active_state["run_id"],
            "gpu_index": 0,
            "scope": scope,
            "state_core_sha256": active_state["state_fingerprint_sha256"],
            "previous_state_fingerprint_sha256": active_state[
                "state_fingerprint_sha256"
            ],
            "host_scope_cleanup": {
                "scope_identity_sha256": scope["identity_sha256"],
                "leader_status": "gone",
                "observed_pids": [],
                "term_pids": [],
                "kill_pids": [],
                "empty_queries": 20,
                "quiet_window_seconds": 0.1,
                "verified_absent": True,
            },
            "docker_evidence": {
                "kind": "verified_terminal_receipt",
                "inner_lease_id": "b" * 64,
                "terminal_event_type": "release",
                "terminal_receipt": {
                    "path": "receipts/" + "a" * 64 + ".json",
                    "bytes": 1,
                    "sha256": "a" * 64,
                    "event_fingerprint_sha256": "b" * 64,
                },
                "absence_evidence_sha256": "c" * 64,
                "receipt_scan_sha256": "d" * 64,
                "verified_absent": True,
            },
            "created_at_utc": "2026-07-18T12:00:02Z",
            "created_monotonic_ns": 3,
            "event_fingerprint_sha256": "0" * 64,
        }
        terminal_event["event_fingerprint_sha256"] = v1.canonical_sha256(
            {
                key: item for key, item in terminal_event.items()
                if key != "event_fingerprint_sha256"
            }
        )
        terminal_raw = v1.canonical_bytes(terminal_event) + b"\n"
        v4_pin = {
            "path": (
                "validation/results/gpu-leases/v4/gpu-0/receipts/"
                + "d" * 64 + ".json"
            ),
            "bytes": len(terminal_raw),
            "sha256": hashlib.sha256(terminal_raw).hexdigest(),
            "event_fingerprint_sha256": terminal_event[
                "event_fingerprint_sha256"
            ],
        }
        receipt = lease._build_activation_receipt(
            contract_fingerprint=verified.contract["contract_fingerprint_sha256"],
            plan_pin=verified.plan_file.pin,
            plan_fingerprint=verified.projection["plan_fingerprint_sha256"],
            authority_digest=verified.projection["authority_digest"],
            effective_argv_sha256=verified.projection["effective_argv_sha256"],
            gpu_index=0,
            gpu_uuid=GPU_UUID,
            owner_kind="capacity_5min",
            outer_acceptance=verified.plan["outer_acceptance"],
            driver_prerequisite=verified.contract["driver_prerequisite"],
            v4_state_binding=state_binding,
            outcome={"completion": "returned", "returncode": 0, "exception_type": None},
            v4_terminal_receipt=v4_pin,
            v4_terminal_event=terminal_event,
        )
        tampered = copy.deepcopy(receipt)
        tampered_event = tampered["v4_terminal_event"]
        tampered_event["run_id"] = "9" * 64
        tampered_event["event_fingerprint_sha256"] = v1.canonical_sha256(
            {
                key: item for key, item in tampered_event.items()
                if key != "event_fingerprint_sha256"
            }
        )
        tampered_raw = v1.canonical_bytes(tampered_event) + b"\n"
        tampered["v4_terminal_receipt"] = {
            "path": (
                "validation/results/gpu-leases/v4/gpu-0/receipts/"
                + "d" * 64 + ".json"
            ),
            "bytes": len(tampered_raw),
            "sha256": hashlib.sha256(tampered_raw).hexdigest(),
            "event_fingerprint_sha256": tampered_event[
                "event_fingerprint_sha256"
            ],
        }
        tampered["receipt_fingerprint_sha256"] = v1.canonical_sha256(
            {
                key: item for key, item in tampered.items()
                if key != "receipt_fingerprint_sha256"
            }
        )
        with pytest.raises(lease.LeaseIntegrityError, match="receipt lineage"):
            lease._validate_activation_receipt(tampered)
        lease_tampered = copy.deepcopy(receipt)
        lease_event = lease_tampered["v4_terminal_event"]
        lease_event["docker_evidence"]["inner_lease_id"] = "9" * 64
        lease_event["event_fingerprint_sha256"] = v1.canonical_sha256(
            {
                key: item for key, item in lease_event.items()
                if key != "event_fingerprint_sha256"
            }
        )
        lease_event_raw = v1.canonical_bytes(lease_event) + b"\n"
        lease_tampered["v4_terminal_receipt"].update(
            bytes=len(lease_event_raw),
            sha256=hashlib.sha256(lease_event_raw).hexdigest(),
            event_fingerprint_sha256=lease_event["event_fingerprint_sha256"],
        )
        lease_tampered["receipt_fingerprint_sha256"] = v1.canonical_sha256(
            {
                key: item for key, item in lease_tampered.items()
                if key != "receipt_fingerprint_sha256"
            }
        )
        with pytest.raises(lease.LeaseIntegrityError, match="receipt lineage"):
            lease._validate_activation_receipt(lease_tampered)
        published = lease._write_activation_receipt(receipt, root=tmp_path / "v6-receipts")
        assert published["receipt_fingerprint_sha256"] == receipt[
            "receipt_fingerprint_sha256"
        ]
        final = tmp_path / "v6-receipts/gpu-0/receipts" / f"{receipt['receipt_id']}.json"
        replay = json.loads(final.read_bytes())
        assert replay["plan_file"] == verified.plan_file.pin
        assert replay["plan_authority_digest"] == verified.projection["authority_digest"]
        assert replay["effective_argv_sha256"] == verified.projection["effective_argv_sha256"]
        assert replay["v4_state_binding"] == state_binding
        assert replay["v4_terminal_receipt"] == v4_pin
        assert replay["v4_terminal_event"] == terminal_event


def test_terminal_receipt_lineage_rejects_each_binding_mutation() -> None:
    binding = {
        "run_id": "1" * 64,
        "state_fingerprint_sha256": "2" * 64,
        "scope_identity_sha256": "3" * 64,
        "active_state": {"scope": {"identity_sha256": "3" * 64}},
    }
    event: dict[str, Any] = {
        "gpu_index": 0,
        "event_type": "release",
        "contract_fingerprint_sha256": lease.V4_CONTRACT_FINGERPRINT,
        "run_id": binding["run_id"],
        "scope": {"identity_sha256": binding["scope_identity_sha256"]},
        "state_core_sha256": binding["state_fingerprint_sha256"],
        "previous_state_fingerprint_sha256": binding["state_fingerprint_sha256"],
        "host_scope_cleanup": {"verified_absent": True},
        "docker_evidence": {"verified_absent": True},
    }
    assert lease._terminal_matches_state_binding(event, binding, 0)
    mutations = (
        ("contract_fingerprint_sha256", "9" * 64),
        ("run_id", "9" * 64),
        ("state_core_sha256", "9" * 64),
        ("previous_state_fingerprint_sha256", "9" * 64),
        ("event_type", "pending_recovery"),
        ("gpu_index", 1),
    )
    for field, value in mutations:
        changed = copy.deepcopy(event)
        changed[field] = value
        assert not lease._terminal_matches_state_binding(changed, binding, 0)
    changed = copy.deepcopy(event)
    changed["scope"]["identity_sha256"] = "9" * 64
    assert not lease._terminal_matches_state_binding(changed, binding, 0)
    changed = copy.deepcopy(event)
    changed["scope"]["cgroup_path"] = "/rehashed-drift"
    assert not lease._terminal_matches_state_binding(changed, binding, 0)
    changed = copy.deepcopy(event)
    changed["host_scope_cleanup"]["verified_absent"] = False
    assert not lease._terminal_matches_state_binding(changed, binding, 0)
    changed = copy.deepcopy(event)
    changed["docker_evidence"]["verified_absent"] = False
    assert not lease._terminal_matches_state_binding(changed, binding, 0)


def test_repository_publishes_no_plan_or_v6_receipt() -> None:
    root = ROOT / "validation/results/gpu-leases/v6"
    assert root.is_dir()
    assert list(root.iterdir()) == []
    assert lease.DEFAULT_WORKLOAD_PLAN is None
    assert lease.PUBLISHED_LIVE_PLAN is None


def test_current_tool_requalification_is_exact_and_v5_drift_is_visible() -> None:
    contract = lease.load_contract()
    for name, pin in contract["tools"].items():
        descriptor = os.open(pin["path"], os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            raw = os.pread(descriptor, info.st_size + 1, 0)
        finally:
            os.close(descriptor)
        assert len(raw) == pin["bytes"]
        assert hashlib.sha256(raw).hexdigest() == pin["sha256"]
        assert pin != contract["predecessor_tool_drift"]["v5_expected_tools"][name]
    assert contract["predecessor_tool_drift"] == {
        "predecessor_activation_eligible_on_current_host": False,
        "reason": "docker-and-nvidia-smi-exact-byte-pins-drifted",
        "v5_expected_tools": contract["predecessor_tool_drift"]["v5_expected_tools"],
        "v6_requalified_tools": contract["tools"],
    }


def test_tool_pin_self_resign_does_not_change_v6_contract_policy() -> None:
    contract = lease.load_contract()
    tampered = copy.deepcopy(contract)
    tampered["tools"]["docker_cli"]["sha256"] = "0" * 64
    tampered["predecessor_tool_drift"]["v6_requalified_tools"] = tampered["tools"]
    unsigned = {
        key: item for key, item in tampered.items()
        if key != "contract_fingerprint_sha256"
    }
    tampered["contract_fingerprint_sha256"] = v1.canonical_sha256(unsigned)
    artifacts = {
        relative: (ROOT / relative).read_bytes()
        for relative in lease.HELD_RUNTIME_PATHS
    }
    with pytest.raises(lease.LeaseIntegrityError, match="policy differs"):
        lease._validate_contract_value(tampered, artifacts)


def test_boot_drift_and_r7_receipt_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = lease.load_contract()
    raw = (ROOT / lease.R7_RECEIPT_PATH).read_bytes()
    assert lease._validate_r7_receipt(raw, contract)["boot_id"] == lease.R7_BOOT_ID
    tampered = raw.replace(b'"decision": "ACCEPT"', b'"decision": "REJECT"', 1)
    with pytest.raises(lease.LeaseIntegrityError, match="raw pin differs"):
        lease._validate_r7_receipt(tampered, contract)
    monkeypatch.setattr(
        lease, "_read_current_boot_id",
        lambda: "00000000-0000-0000-0000-000000000000",
    )
    with pytest.raises(lease.LeaseAuthorizationError, match="stale"):
        lease._require_current_r7_boot()


@pytest.mark.parametrize("kind", ["duplicate", "nan", "infinity"])
def test_outer_acceptance_strict_json_rejects_duplicate_and_nonfinite(
    tmp_path: Path, kind: str
) -> None:
    path, plan, expected = _materialize_plan(tmp_path)
    del path
    if kind == "duplicate":
        raw = b'{"schema_version":"x","schema_version":"y"}'
    elif kind == "nan":
        raw = b'{"x":NaN}'
    else:
        raw = b'{"x":Infinity}'
    schema_raw = (
        ROOT / "validation/schemas/gpu-lease-v6-outer-acceptance.schema.json"
    ).read_bytes()
    with pytest.raises(lease.LeaseIntegrityError, match="strict JSON"):
        lease._validate_outer_acceptance(
            raw,
            schema_raw=schema_raw,
            plan=plan,
            expected_fingerprint=expected[
                "expected_outer_acceptance_fingerprint"
            ],
        )


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_outer_acceptance_link_aliases_fail_closed(tmp_path: Path, kind: str) -> None:
    path, plan, expected = _materialize_plan(tmp_path)
    outer = Path(plan["outer_acceptance"]["receipt"]["path"])
    if kind == "symlink":
        target = tmp_path / "outer-target.json"
        outer.rename(target)
        outer.symlink_to(target)
    else:
        alias = tmp_path / "outer-hardlink.json"
        os.link(outer, alias)
    with pytest.raises((lease.LeaseIntegrityError, OSError), match="file policy|exact pin|Too many"):
        _verify(path, expected)


def test_post_verification_outer_acceptance_replacement_blocks_spawn(
    tmp_path: Path,
) -> None:
    path, plan, expected = _materialize_plan(tmp_path)
    verified = _verify(path, expected)
    outer = Path(plan["outer_acceptance"]["receipt"]["path"])
    try:
        replacement = tmp_path / "replacement-outer.json"
        replacement.write_bytes(outer.read_bytes())
        replacement.chmod(0o440)
        os.replace(replacement, outer)
        with pytest.raises(lease.LeaseIntegrityError, match="identity drifted"):
            lease.execute_workload_plan(verified, simulate=True)
    finally:
        verified.close()


@pytest.mark.parametrize(
    "reserved",
    [
        "--label=com.deepsafe.gpu-lease.plan.v6=" + "0" * 64,
        "--label=com.deepsafe.gpu-lease.plan.v5=" + "0" * 64,
    ],
)
def test_caller_reserved_authority_labels_are_rejected(
    tmp_path: Path, reserved: str
) -> None:
    path, plan, expected = _materialize_plan(tmp_path)
    argv = ["/usr/bin/docker", "run", reserved, "--network=none", IMAGE_REFERENCE]
    plan["workload"]["argv"] = argv
    plan["workload"]["argv_sha256"] = v1.command_argv_sha256(argv)
    plan["workload"]["image_argv_index"] = 4
    updated = _rewrite_plan(path, plan)
    decoded = json.loads(path.read_bytes())
    expected.update(
        updated,
        expected_plan_fingerprint=decoded["plan_fingerprint_sha256"],
        expected_argv_sha256=decoded["workload"]["argv_sha256"],
    )
    with pytest.raises(lease.LeaseIntegrityError, match="reserved plan label"):
        _verify(path, expected)


def test_self_resigned_plan_cannot_escape_r7_gpu_uuid(tmp_path: Path) -> None:
    path, plan, expected = _materialize_plan(tmp_path)
    wrong = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    plan["workload"]["gpu"]["uuid"] = wrong
    updated = _rewrite_plan(path, plan)
    decoded = json.loads(path.read_bytes())
    expected.update(
        updated,
        expected_plan_fingerprint=decoded["plan_fingerprint_sha256"],
        expected_gpu_uuid=wrong,
    )
    with pytest.raises(lease.LeaseAuthorizationError, match="caller workload projection"):
        _verify(path, expected)


def test_outer_acceptance_tamper_and_workload_reuse_fail_closed(tmp_path: Path) -> None:
    path, plan, expected = _materialize_plan(tmp_path)
    outer_path = Path(plan["outer_acceptance"]["receipt"]["path"])
    outer = json.loads(outer_path.read_bytes())
    outer["user_notification"]["confirmed"] = False
    outer, raw = _rewrite_outer_acceptance(outer_path, outer)
    plan["outer_acceptance"] = {
        "receipt": _artifact_pin(outer_path),
        "acceptance_fingerprint_sha256": outer["acceptance_fingerprint_sha256"],
    }
    updated = _rewrite_plan(path, plan)
    decoded = json.loads(path.read_bytes())
    expected.update(
        updated,
        expected_plan_fingerprint=decoded["plan_fingerprint_sha256"],
        expected_outer_acceptance_bytes=len(raw),
        expected_outer_acceptance_sha256=hashlib.sha256(raw).hexdigest(),
        expected_outer_acceptance_fingerprint=outer[
            "acceptance_fingerprint_sha256"
        ],
    )
    with pytest.raises(lease.LeaseIntegrityError, match="outer acceptance violates schema"):
        _verify(path, expected)

    path, plan, expected = _materialize_plan(tmp_path / "reused")
    plan["workload"]["owner_kind"] = "person_training"
    updated = _rewrite_plan(path, plan)
    decoded = json.loads(path.read_bytes())
    expected.update(
        updated,
        expected_plan_fingerprint=decoded["plan_fingerprint_sha256"],
        expected_owner_kind="person_training",
    )
    with pytest.raises(lease.LeaseAuthorizationError, match="outer acceptance semantics"):
        _verify(path, expected)
