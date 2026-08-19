from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from validation import person_rtdetrv4_tensorrt_r14e as lane
from validation import person_rtdetrv4_tensorrt_r14e_container as worker


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


def _r14d_pins() -> list[dict[str, object]]:
    pins: list[dict[str, object]] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"}.issubset(value):
                pins.append(value)
                return
            for child in value.values():
                visit(child)

    visit(lane.R14D_IMMUTABLE)
    return pins


def _write_immutable(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    path.chmod(0o440)
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _make_held_run(repository: Path) -> lane.HeldRunDirectories:
    run = repository / "runs" / "one"
    output = run / "output"
    control = run / "control"
    output.mkdir(parents=True, mode=0o700)
    control.mkdir(mode=0o700)
    run.chmod(0o700)
    output.chmod(0o700)
    control.chmod(0o700)
    return lane.HeldRunDirectories.open_existing(run)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(stream, format="PNG")
    return stream.getvalue()


def test_r14e_execution_and_plan_publication_are_explicitly_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert lane.GPU_LEASE_V2_API_READY is False
    assert lane.EXECUTION_GATE == (
        "blocked_until_gpu_lease_v2_cid_lifecycle_api_is_reviewed_and_pinned"
    )
    assert not lane.DEFAULT_PLAN.exists()
    assert not lane.RUNS_ROOT.exists()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("no process may start while the R14E gate is closed")

    monkeypatch.setattr(lane.subprocess, "Popen", forbidden)
    calls = (
        lambda: lane.build_plan(prepared_at_utc="2026-07-18T00:00:00+00:00"),
        lambda: lane.prepare_plan(),
        lambda: lane.verify_plan(lane.DEFAULT_PLAN, expected_fingerprint="0" * 64),
        lambda: lane.execute_stage(
            plan_path=lane.DEFAULT_PLAN,
            accepted_plan_fingerprint="0" * 64,
            stage="tensorrt_fp16_640",
            run_id="r14e-gate-test",
        ),
        lambda: lane._new_run_directory(
            "tensorrt_fp16_640", "r14e-direct-run-gate-test",
        ),
        lambda: lane.render_lease_command(
            [str(lane.DOCKER_CLI), "run"], stage="tensorrt_fp16_640",
        ),
    )
    for call in calls:
        with pytest.raises(lane.TensorRTR14EError, match="execution gate is closed"):
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

    with pytest.raises((OSError, lane.TensorRTR14EError)):
        lane.open_held_source(repository / "safe" / "jump" / "secret.bin")
    with pytest.raises((OSError, lane.TensorRTR14EError)):
        lane.atomic_bytes(repository / "safe" / "jump" / "published.bin", b"bad")
    assert not (outside / "published.bin").exists()


def test_worker_output_bundle_requires_exact_inventory_and_retains_original_fds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(lane, "ROOT", repository)
    run_directories = _make_held_run(repository)
    output = run_directories.output
    engine = _write_immutable(output / "engine.staging", b"engine-original")
    log = _write_immutable(output / "trtexec-build.log", b"build-log")
    _write_immutable(output / "worker-result.json", b"{}\n")
    extra = output / "undeclared.bin"
    _write_immutable(extra, b"extra")
    result = {"operation": "build", "outputs": [engine, log]}

    descriptors: dict[str, int] = {}
    try:
        with pytest.raises(lane.TensorRTR14EError, match="missing or undeclared"):
            lane.hold_worker_output_bundle(result, run_directories)
        extra.chmod(0o640)
        extra.unlink()

        descriptors, pins = lane.hold_worker_output_bundle(
            result, run_directories,
        )
        original = output / "engine.staging"
        swapped = repository / "engine.original"
        original.rename(swapped)
        _write_immutable(original, b"engine-original")
        size, digest = lane._hash_held_fd(descriptors["engine.staging"])
        assert (size, digest) == (engine["bytes"], engine["sha256"])
        with pytest.raises(lane.TensorRTR14EError, match="name/inode"):
            lane.replay_worker_output_bundle(
                operation="build", run_directories=run_directories,
                descriptors=descriptors, pins=pins,
            )
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
        run_directories.close()


def test_plan_source_bundle_uses_held_a_bytes_after_a_is_renamed_to_b(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.syspath_prepend(str(lane.CPU_RUNTIME_SITE_PACKAGES))
    monkeypatch.setattr(lane, "ROOT", repository)
    payloads = {
        "models/model.onnx": b"onnx-source",
        "models/model.engine": b"engine-source",
        "results/trt-output.npz": b"trt-source",
        "validation/parser.cpp": b"parser-source",
        "calibration/table.bin": b"calibration-source",
        "images/A.png": _png_bytes((255, 0, 0)),
    }
    pins: dict[str, dict[str, object]] = {}
    for relative, payload in payloads.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        pins[relative] = {
            "path": relative, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    plan: dict[str, object] = {
        "sources": {
            "onnx": pins["models/model.onnx"],
            "engine": pins["models/model.engine"],
            "tensorrt": pins["results/trt-output.npz"],
            "parser": pins["validation/parser.cpp"],
            "calibration": pins["calibration/table.bin"],
            "images": [pins["images/A.png"]],
        },
    }
    plan["fingerprint_sha256"] = lane.fingerprint(plan)
    plan_path = repository / "plans/accepted.json"
    plan_path.parent.mkdir()
    plan_path.write_bytes(lane.canonical_bytes(plan) + b"\n")

    bundle = lane.HeldExecutionBundle.from_plan(
        plan_path, accepted_fingerprint=str(plan["fingerprint_sha256"]),
        validate_document=False,
    )
    try:
        assert set(bundle.descriptors) == {"plans/accepted.json", *payloads}
        bundle.verify_all()
        source_a = repository / "images/A.png"
        renamed_b = repository / "images/B.png"
        source_a.rename(renamed_b)
        source_a.write_bytes(_png_bytes((0, 0, 255)))

        assert bundle.bytes(
            "images/A.png", context="A-to-B held source test",
        ) == payloads["images/A.png"]
        decoded = lane.preprocess_held_image(
            bundle.descriptor("images/A.png"), bundle.pin("images/A.png"), 2,
        )
        assert float(decoded[0].mean()) == pytest.approx(1.0)
        assert float(decoded[2].mean()) == pytest.approx(0.0)
        with pytest.raises(lane.TensorRTR14EError, match="name/inode"):
            bundle.verify_all()
    finally:
        bundle.close()


def test_predecessor_document_holds_its_transitive_pin_and_replays_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(lane, "ROOT", repository)
    artifact_path = repository / "artifacts/predecessor.engine"
    artifact_path.parent.mkdir()
    artifact_path.write_bytes(b"predecessor-engine")
    artifact_pin = {
        "path": "artifacts/predecessor.engine",
        "bytes": len(b"predecessor-engine"),
        "sha256": hashlib.sha256(b"predecessor-engine").hexdigest(),
    }
    predecessor: dict[str, object] = {"artifact": artifact_pin}
    predecessor["fingerprint_sha256"] = lane.fingerprint(predecessor)
    predecessor_path = repository / "records/predecessor.json"
    predecessor_path.parent.mkdir()
    predecessor_payload = lane.canonical_bytes(predecessor) + b"\n"
    predecessor_path.write_bytes(predecessor_payload)
    predecessor_pin = {
        "path": "records/predecessor.json",
        "bytes": len(predecessor_payload),
        "sha256": hashlib.sha256(predecessor_payload).hexdigest(),
        "fingerprint_sha256": predecessor["fingerprint_sha256"],
    }
    plan: dict[str, object] = {"predecessor": predecessor_pin}
    plan["fingerprint_sha256"] = lane.fingerprint(plan)
    plan_path = repository / "plans/accepted.json"
    plan_path.parent.mkdir()
    plan_path.write_bytes(lane.canonical_bytes(plan) + b"\n")

    bundle = lane.HeldExecutionBundle.from_plan(
        plan_path, accepted_fingerprint=str(plan["fingerprint_sha256"]),
        validate_document=False,
    )
    try:
        assert set(bundle.descriptors) == {
            "plans/accepted.json", "records/predecessor.json",
            "artifacts/predecessor.engine",
        }
        assert bundle.documents["records/predecessor.json"] == predecessor
        bundle.verify_all()
    finally:
        bundle.close()

    bad_plan = json.loads(json.dumps(plan))
    bad_plan["predecessor"]["fingerprint_sha256"] = "f" * 64
    bad_plan["fingerprint_sha256"] = lane.fingerprint(bad_plan)
    bad_path = repository / "plans/bad.json"
    bad_path.write_bytes(lane.canonical_bytes(bad_plan) + b"\n")
    with pytest.raises(lane.TensorRTR14EError, match="semantic fingerprint"):
        lane.HeldExecutionBundle.from_plan(
            bad_path, accepted_fingerprint=str(bad_plan["fingerprint_sha256"]),
            validate_document=False,
        )


def test_output_path_redirect_to_second_directory_is_rejected_by_held_dirfd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(lane, "ROOT", repository)
    run_directories = _make_held_run(repository)
    output = run_directories.output
    engine = _write_immutable(output / "engine.staging", b"held-engine")
    log = _write_immutable(output / "trtexec-build.log", b"held-log")
    _write_immutable(output / "worker-result.json", b"{}\n")
    result = {"operation": "build", "outputs": [engine, log]}
    descriptors, pins = lane.hold_worker_output_bundle(result, run_directories)
    try:
        second = repository / "second-output"
        second.mkdir(mode=0o700)
        second.chmod(0o700)
        _write_immutable(second / "engine.staging", b"attacker-engine")
        _write_immutable(second / "trtexec-build.log", b"attacker-log")
        _write_immutable(second / "worker-result.json", b"{}\n")
        output.rename(run_directories.run_root / "held-output")
        second.rename(output)

        assert set(os.listdir(run_directories.output_descriptor)) == {
            "engine.staging", "trtexec-build.log", "worker-result.json",
        }
        assert (output / "engine.staging").read_bytes() == b"attacker-engine"
        with pytest.raises(lane.TensorRTR14EError, match="held directory identity"):
            lane.replay_worker_output_bundle(
                operation="build", run_directories=run_directories,
                descriptors=descriptors, pins=pins,
            )
        with pytest.raises(lane.TensorRTR14EError, match="held directory identity"):
            lane.hold_worker_output_bundle(result, run_directories)
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
        run_directories.close()


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
    with pytest.raises(lane.TensorRTR14EError, match="same-held-FD"):
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
    with pytest.raises(lane.TensorRTR14EError, match="claimed/observed"):
        lane.validate_claimed_predecessor_pin(
            stage=stage, predecessor=predecessor,
            claimed_pin=claimed, observed_pin=observed,
        )


def test_commit_execution_projection_includes_recursive_predecessor_claims() -> None:
    stage = "tensorrt_fp16_960"
    run_id = "r14e-projection-test"
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
        "receipt_id": "rtdetrv4-s-r11-tensorrt-fp16-960-r14e",
        "stage": stage, "profile": 960, "status": "passed",
        "plan": commit["plan"], "r11_stage_receipt": commit["r11_stage_receipt"],
        "worker_result": commit["worker_result"], "fingerprint_sha256": "4" * 64,
        "prior_r14e_commits": [pin],
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
    execution["prior_r14e_commits"] = []
    with pytest.raises(lane.TensorRTR14EError, match="semantic projection"):
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
        "DEEPSAFE_MANAGED_DOCKER_CIDFILE": "/tmp/r14e/control/container.cid",
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
    with pytest.raises(worker.ContainerR14EError, match="effective command digest"):
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
    with pytest.raises(lane.TensorRTR14EError, match="expectation argument"):
        lane.lease_command_digests(tampered_expectation)
    changed = list(sealed)
    changed.insert(2, "--network=none")
    assert lane.gpu_lease_command_sha256(changed) != digest


def test_pathname_cid_read_and_cleanup_are_unconditionally_forbidden() -> None:
    cidfile = lane.ROOT / "validation/results/fake/control/container.cid"
    with pytest.raises(lane.TensorRTR14EError, match="pathname-based CID reads"):
        lane._read_container_id(cidfile)
    with pytest.raises(lane.TensorRTR14EError, match="pathname-based Docker cleanup"):
        lane._force_remove_container(cidfile)


def test_all_r14e_schemas_are_draft_2020_12_valid() -> None:
    schemas = sorted((lane.ROOT / "validation/schemas").glob("*r14e.schema.json"))
    assert len(schemas) == 7
    for path in schemas:
        value = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(value)
        assert value["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert "r14e" in value["$id"]
    plan_schema = json.loads(lane.PLAN_SCHEMA.read_text(encoding="utf-8"))
    assert "frozen_r14d" in plan_schema["required"]
    assert plan_schema["properties"]["frozen_r14d"]["properties"][
        "artifacts"
    ]["const"] == lane.R14D_IMMUTABLE
    execution = plan_schema["properties"]["execution_contract"]
    assert {"held_execution_bundle", "held_run_directories"}.issubset(
        execution["required"],
    )
    assert execution["properties"]["held_execution_bundle"]["const"][
        "onnxruntime_model"
    ] == "same_held_onnx_fd_bytes_without_path_reopen"
    assert execution["properties"]["held_run_directories"]["const"][
        "worker_output_reads"
    ] == "output_dirfd_openat_o_nofollow_exact_ordered_inventory_only"


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


def test_frozen_r14d_inventory_is_byte_exact_read_only_and_unpublished() -> None:
    pins = _r14d_pins()
    assert len(pins) == 11
    assert len({str(pin["path"]) for pin in pins}) == len(pins)
    for pin in pins:
        path = lane.ROOT / str(pin["path"])
        payload = path.read_bytes()
        assert len(payload) == pin["bytes"]
        assert hashlib.sha256(payload).hexdigest() == pin["sha256"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o440
    assert not lane.R14D_PLAN.exists()
    assert not lane.R14D_RUNS_ROOT.exists()
