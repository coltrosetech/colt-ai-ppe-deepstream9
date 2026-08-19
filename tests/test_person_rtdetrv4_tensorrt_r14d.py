from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from validation import person_rtdetrv4_tensorrt_r14d as lane
from validation import person_rtdetrv4_tensorrt_r14d_container as worker


R14C_IMMUTABLE = {
    "validation/person_rtdetrv4_tensorrt_r14c.py": (
        214_898, "0d2ceaa917204efa57b5e28d5b20a31b0fd2d414d8deef75415eb14395eed6af",
    ),
    "validation/person_rtdetrv4_tensorrt_r14c_container.py": (
        57_287, "2424de0f73e0a7b01a3037d7b6251c9cdd749a0425b8d1a0e46b49bccbf2cf47",
    ),
    "tests/test_person_rtdetrv4_tensorrt_r14c.py": (
        67_540, "f7468e85540c4119c8205761dbafbd80d12921eed5d6f9ac6cf5b346b0a27986",
    ),
    "docs/person-rtdetrv4-tensorrt-r14c.md": (
        11_164, "af8b935f807d36037be77388fc6f53bbaf030a49bad1a16330d6e9048f32e996",
    ),
    lane.R14C_PLAN_RELATIVE: (
        lane.R14C_PLAN_PIN["bytes"], lane.R14C_PLAN_PIN["sha256"],
    ),
}


def _write_immutable(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    path.chmod(0o440)
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_r14d_execution_and_plan_publication_are_explicitly_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert lane.GPU_LEASE_V2_API_READY is False
    assert lane.EXECUTION_GATE == (
        "blocked_until_gpu_lease_v2_cid_lifecycle_api_is_reviewed_and_pinned"
    )
    assert not lane.DEFAULT_PLAN.exists()
    assert not lane.RUNS_ROOT.exists()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("no process may start while the R14D gate is closed")

    monkeypatch.setattr(lane.subprocess, "Popen", forbidden)
    calls = (
        lambda: lane.build_plan(prepared_at_utc="2026-07-18T00:00:00+00:00"),
        lambda: lane.prepare_plan(),
        lambda: lane.verify_plan(lane.DEFAULT_PLAN, expected_fingerprint="0" * 64),
        lambda: lane.execute_stage(
            plan_path=lane.DEFAULT_PLAN,
            accepted_plan_fingerprint="0" * 64,
            stage="tensorrt_fp16_640",
            run_id="r14d-gate-test",
        ),
        lambda: lane.render_lease_command(
            [str(lane.DOCKER_CLI), "run"], stage="tensorrt_fp16_640",
        ),
    )
    for call in calls:
        with pytest.raises(lane.TensorRTR14DError, match="execution gate is closed"):
            call()
    assert not lane.DEFAULT_PLAN.exists()
    assert not lane.RUNS_ROOT.exists()


def test_repository_open_rejects_intermediate_symlink_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    (repository / "safe").mkdir(parents=True)
    outside.mkdir()
    (outside / "secret.bin").write_bytes(b"outside")
    (repository / "safe" / "jump").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(lane, "ROOT", repository)

    with pytest.raises((OSError, lane.TensorRTR14DError)):
        lane.open_held_source(repository / "safe" / "jump" / "secret.bin")
    with pytest.raises((OSError, lane.TensorRTR14DError)):
        lane.atomic_bytes(repository / "safe" / "jump" / "published.bin", b"bad")
    assert not (outside / "published.bin").exists()


def test_worker_output_bundle_requires_exact_inventory_and_retains_original_fds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    output = repository / "runs" / "one" / "output"
    output.mkdir(parents=True)
    monkeypatch.setattr(lane, "ROOT", repository)
    engine = _write_immutable(output / "engine.staging", b"engine-original")
    log = _write_immutable(output / "trtexec-build.log", b"build-log")
    _write_immutable(output / "worker-result.json", b"{}\n")
    extra = output / "undeclared.bin"
    _write_immutable(extra, b"extra")
    result = {"operation": "build", "outputs": [engine, log]}

    with pytest.raises(lane.TensorRTR14DError, match="missing or undeclared"):
        lane.hold_worker_output_bundle(result, output)
    extra.chmod(0o640)
    extra.unlink()

    descriptors, pins = lane.hold_worker_output_bundle(result, output)
    try:
        original = output / "engine.staging"
        swapped = repository / "engine.original"
        original.rename(swapped)
        _write_immutable(original, b"engine-original")
        size, digest = lane._hash_held_fd(descriptors["engine.staging"])
        assert (size, digest) == (engine["bytes"], engine["sha256"])
        with pytest.raises(lane.TensorRTR14DError, match="name/inode"):
            lane.replay_worker_output_bundle(
                operation="build", output_directory=output,
                descriptors=descriptors, pins=pins,
            )
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_execution_receipt_worker_outputs_are_an_exact_same_fd_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    output = repository / "runs" / "one" / "output"
    output.mkdir(parents=True)
    monkeypatch.setattr(lane, "ROOT", repository)
    local = {
        "trt-output.npz": {
            "path": "trt-output.npz", "bytes": 4,
            "sha256": hashlib.sha256(b"data").hexdigest(),
        },
    }
    expected = lane.worker_output_repository_pins(output, local)
    lane.validate_execution_worker_output_projection(
        {"outputs": expected}, output, local, context="unit receipt",
    )
    changed = json.loads(json.dumps(expected))
    changed[0]["sha256"] = "f" * 64
    with pytest.raises(lane.TensorRTR14DError, match="same-held-FD"):
        lane.validate_execution_worker_output_projection(
            {"outputs": changed}, output, local, context="unit receipt",
        )


@pytest.mark.parametrize("field", ["bytes", "sha256", "fingerprint_sha256"])
def test_recursive_predecessor_claim_must_equal_actual_observed_pin(field: str) -> None:
    predecessor = "tensorrt_fp16_640"
    stage = "tensorrt_fp16_960"
    observed = {
        "path": lane.repo_relative(lane._stage_commit_path(predecessor)),
        "bytes": 101,
        "sha256": "1" * 64,
        "fingerprint_sha256": "2" * 64,
    }
    assert lane.validate_claimed_predecessor_pin(
        stage=stage, predecessor=predecessor,
        claimed_pin=observed, observed_pin=observed,
    ) == observed
    claimed = dict(observed)
    claimed[field] = 102 if field == "bytes" else "3" * 64
    with pytest.raises(lane.TensorRTR14DError, match="claimed/observed"):
        lane.validate_claimed_predecessor_pin(
            stage=stage, predecessor=predecessor,
            claimed_pin=claimed, observed_pin=observed,
        )


def test_commit_execution_projection_includes_recursive_predecessor_claims() -> None:
    stage = "tensorrt_fp16_960"
    run_id = "r14d-projection-test"
    pin = {
        "path": lane.repo_relative(lane._stage_commit_path("tensorrt_fp16_640")),
        "bytes": 1, "sha256": "1" * 64, "fingerprint_sha256": "2" * 64,
    }
    commit = {
        "plan": {"id": "plan"},
        "r11_stage_receipt": {"id": "r11"},
        "worker_result": {"path": lane.repo_relative(lane._stage_worker_result_path(stage))},
        "execution_receipt": {
            "path": lane.repo_relative(lane._stage_receipt_paths(stage)[1]),
            "fingerprint_sha256": "4" * 64,
        },
    }
    execution = {
        "receipt_id": "rtdetrv4-s-r11-tensorrt-fp16-960-r14d",
        "stage": stage, "profile": 960, "status": "passed",
        "plan": commit["plan"], "r11_stage_receipt": commit["r11_stage_receipt"],
        "worker_result": commit["worker_result"], "fingerprint_sha256": "4" * 64,
        "prior_r14d_commits": [pin],
        "command": {
            "managed_docker_cidfile": str(
                lane.RUNS_ROOT / stage / run_id / "control/container.cid"
            ),
        },
    }
    lane._validate_commit_execution_projection(
        commit, execution, stage=stage, run_id=run_id,
        expected_prior_commit_pins=[pin],
    )
    execution["prior_r14d_commits"] = []
    with pytest.raises(lane.TensorRTR14DError, match="semantic projection"):
        lane._validate_commit_execution_projection(
            commit, execution, stage=stage, run_id=run_id,
            expected_prior_commit_pins=[pin],
        )


def test_worker_requires_sealed_contract_requested_and_effective_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = "a" * 64
    requested = "b" * 64
    effective = "c" * 64
    values = {
        "DEEPSAFE_GPU_LEASE_HELD": "1",
        "DEEPSAFE_GPU_LEASE_ID": "d" * 64,
        "DEEPSAFE_GPU_LEASE_GPU_INDEX": "0",
        "DEEPSAFE_GPU_LEASE_GPU_UUID": "GPU-00000000-0000-0000-0000-000000000000",
        "DEEPSAFE_GPU_LEASE_OWNER_KIND": "legacy_validation",
        "DEEPSAFE_GPU_LEASE_REQUESTED_COMMAND_SHA256": requested,
        "DEEPSAFE_GPU_LEASE_EFFECTIVE_COMMAND_SHA256": effective,
        "DEEPSAFE_GPU_LEASE_CONTRACT_SHA256": contract,
        "DEEPSAFE_MANAGED_DOCKER_CIDFILE": "/tmp/r14d/control/container.cid",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    observed = worker.lease_environment(
        expected_contract_sha256=contract,
        expected_requested_command_sha256=requested,
        expected_effective_command_sha256=effective,
    )
    assert observed["requested_command_sha256"] == requested
    assert observed["effective_command_sha256"] == effective

    monkeypatch.setenv("DEEPSAFE_GPU_LEASE_EFFECTIVE_COMMAND_SHA256", "e" * 64)
    with pytest.raises(worker.ContainerR14DError, match="effective command digest"):
        worker.lease_environment(
            expected_contract_sha256=contract,
            expected_requested_command_sha256=requested,
            expected_effective_command_sha256=effective,
        )


def test_lease_environment_is_scrubbed_and_launcher_is_isolated() -> None:
    hostile = {
        "PATH": "/attacker/bin", "PYTHONPATH": "/attacker/python",
        "PYTHONHOME": "/attacker/home", "DOCKER_CONTEXT": "attacker",
        "DOCKER_CONFIG": "/attacker/docker", "DOCKER_HOST": "tcp://attacker:2375",
        "LD_PRELOAD": "/attacker/lib.so", "BASH_ENV": "/attacker/bashrc",
    }
    environment = lane.isolated_lease_environment(hostile)
    assert environment == {
        "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/nonexistent", "DOCKER_HOST": "unix:///var/run/docker.sock",
    }
    assert not ({"PYTHONPATH", "PYTHONHOME", "DOCKER_CONTEXT", "DOCKER_CONFIG", "LD_PRELOAD", "BASH_ENV"} & environment.keys())
    launcher = lane.lease_launcher_contract()
    assert launcher["interpreter_flags"] == ["-I", "-S", "-B"]
    assert launcher["source_transport"] == "same_held_fd_sealed_memfd_pass_fds"
    assert launcher["contract_transport"] == "same_held_fd_sealed_memfd_pass_fds"
    assert launcher["module_mode"] is False
    assert launcher["pathname_import"] is False


def test_normalized_requested_and_effective_command_digests_are_exact() -> None:
    base = [
        str(lane.DOCKER_CLI), "run", "image", "worker.py",
        "--expected-lease-requested-command-sha256", "0" * 64,
        "--expected-lease-effective-command-sha256", "0" * 64,
    ]
    digest = lane.gpu_lease_command_sha256(base)
    sealed = list(base)
    sealed[5] = digest
    sealed[7] = digest
    assert lane.gpu_lease_command_sha256(sealed) == digest
    assert lane.lease_command_digests(sealed) == {
        "requested": digest, "effective": digest,
    }
    tampered_expectation = list(sealed)
    tampered_expectation[7] = "f" * 64
    with pytest.raises(lane.TensorRTR14DError, match="expectation argument"):
        lane.lease_command_digests(tampered_expectation)
    changed = list(sealed)
    changed.insert(2, "--network=none")
    assert lane.gpu_lease_command_sha256(changed) != digest


def test_pathname_cid_read_and_cleanup_are_unconditionally_forbidden() -> None:
    cidfile = lane.ROOT / "validation/results/fake/control/container.cid"
    with pytest.raises(lane.TensorRTR14DError, match="pathname-based CID reads"):
        lane._read_container_id(cidfile)
    with pytest.raises(lane.TensorRTR14DError, match="pathname-based Docker cleanup"):
        lane._force_remove_container(cidfile)


def test_all_r14d_schemas_are_draft_2020_12_valid() -> None:
    schemas = sorted((lane.ROOT / "validation/schemas").glob("*r14d.schema.json"))
    assert len(schemas) == 7
    for path in schemas:
        value = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(value)
        assert value["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert "r14d" in value["$id"]


def test_r14c_sources_tests_docs_and_plan_remain_byte_exact_and_read_only() -> None:
    for relative, (expected_bytes, expected_sha256) in R14C_IMMUTABLE.items():
        path = lane.ROOT / relative
        payload = path.read_bytes()
        assert len(payload) == expected_bytes
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        assert stat.S_IMODE(path.stat().st_mode) == 0o440
    plan = json.loads(lane.R14C_PLAN.read_text(encoding="utf-8"))
    assert plan["fingerprint_sha256"] == lane.R14C_PLAN_PIN["fingerprint_sha256"]
    assert lane.fingerprint(plan) == lane.R14C_PLAN_PIN["fingerprint_sha256"]
