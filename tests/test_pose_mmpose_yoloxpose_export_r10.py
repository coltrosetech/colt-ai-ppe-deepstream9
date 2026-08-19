from __future__ import annotations

import copy
import base64
import hashlib
import inspect
import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest

from validation import pose_mmpose_yoloxpose_export_r10 as lane
from validation import pose_mmpose_yoloxpose_export_r10_container as container
from validation import pose_mmpose_yoloxpose_onnx_recheck_r10 as host_checker


ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = json.loads(lane.PLAN_SCHEMA.read_text(encoding="utf-8"))
RUN_SCHEMA = json.loads(lane.RECEIPT_SCHEMA.read_text(encoding="utf-8"))
PROFILE_SCHEMA = json.loads(lane.PROFILE_RECEIPT_SCHEMA.read_text(encoding="utf-8"))
FIXED_TIME = "2026-07-18T04:30:00+00:00"


@pytest.fixture(scope="module")
def plan() -> dict:
    return lane.build_plan(created_at=FIXED_TIME)


def comparison(shape: list[int]) -> dict:
    return {
        "shape": shape,
        "max_absolute_error": 0.0,
        "max_relative_error": 0.0,
        "atol": 0.0001,
        "rtol": 0.0001,
        "passed": True,
    }


def profile_receipt(tmp_path: Path, profile: int = 640) -> tuple[Path, dict]:
    output = tmp_path / str(profile)
    output.mkdir()
    model = output / container.PROFILES[profile]
    model.write_bytes(b"synthetic-onnx-for-contract-tests")
    pair = {
        "dets": comparison([1, 100, 5]),
        "keypoints": comparison([1, 100, 17, 3]),
    }
    value = {
        "schema_version": container.SCHEMA_VERSION,
        "status": "passed",
        "run_id": "cpu-export-test",
        "profile": profile,
        "created_at": FIXED_TIME,
        "plan_fingerprint_sha256": "1" * 64,
        "execution_boundary": {
            "effective_uid": 1000,
            "effective_gid": 1000,
            "root_read_only": True,
            "network_interfaces": ["lo"],
            "gpu_device_nodes": [],
            "output_directory_device": 42,
            "output_directory_inode": 84,
            "output_directory_binding_verified": True,
            "source_sha256": {
                str(path): sha256 for path, sha256 in container.SOURCE_PINS.items()
            },
            "runtime": "cpu_only_exact_image",
            "network": "none",
            "root_filesystem": "read_only",
            "gpu_exposed": False,
            "gpu_api_queried": False,
            "gpu_compute_executed": False,
            "model_loaded": True,
            "export_executed": True,
            "onnxruntime_executed": True,
            "tensorrt_executed": False,
            "deepstream_executed": False,
        },
        "profile_configuration": {
            "spatial_size": profile,
            "deploy_onnx_input_shape": [profile, profile],
            "model_input_size": [profile, profile],
            "codec_input_size": [profile, profile],
            "test_bottomup_resize_input_size": [profile, profile],
            "val_bottomup_resize_input_size": [profile, profile],
            "training_pipeline_modified": False,
            "passed": True,
        },
        "publication": {
            "export_staging": "container_private_tmpfs",
            "host_output_used_for_export_or_validation": False,
            "anonymous_inode": True,
            "linkat_empty_path": True,
            "no_overwrite": True,
            "source_sha256": container.sha256_file(model),
            "published_sha256": container.sha256_file(model),
            "source_and_published_match": True,
        },
        "onnx": container.pin(model, output),
        "graph_validation": {
            "checker_passed": True,
            "default_opset": 11,
            "inputs": [
                {
                    "name": "input",
                    "dtype": "FLOAT",
                    "shape": ["batch", 3, profile, profile],
                }
            ],
            "outputs": [
                {"name": "dets", "dtype": "FLOAT", "shape": ["batch", 100, 5]},
                {
                    "name": "keypoints",
                    "dtype": "FLOAT",
                    "shape": ["batch", 100, 17, 3],
                },
            ],
            "external_data": False,
        },
        "numerical_validation": {
            "provider_requested": "CPUExecutionProvider",
            "providers_active": ["CPUExecutionProvider"],
            "batches": {
                "1": {
                    "input_shape": [1, 3, profile, profile],
                    "outputs_finite": True,
                    "pytorch_mmdeploy_vs_onnxruntime": copy.deepcopy(pair),
                },
                "12": {
                    "input_shape": [12, 3, profile, profile],
                    "outputs_finite": True,
                    "pytorch_mmdeploy_vs_onnxruntime": {
                        "dets": comparison([12, 100, 5]),
                        "keypoints": comparison([12, 100, 17, 3]),
                    },
                },
            },
            "batch12_vs_twelve_batch1": {
                "dets": comparison([12, 100, 5]),
                "keypoints": comparison([12, 100, 17, 3]),
            },
            "score_ranges": {
                "dets_inclusive_unit_interval": True,
                "keypoints_inclusive_unit_interval": True,
            },
            "all_outputs_finite": True,
            "passed": True,
        },
        "quality_claims": {
            "trained_resolution": profile == 640,
            "profile_quality_claimed": False,
            "production_ready": False,
        },
    }
    value["receipt_fingerprint_sha256"] = container.fingerprint(
        value, "receipt_fingerprint_sha256"
    )
    receipt = output / "profile-receipt.json"
    receipt.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    model.chmod(0o440)
    receipt.chmod(0o440)
    return receipt, value


def test_r9_passed_receipt_and_exact_image_are_hard_prerequisites() -> None:
    result = lane.validate_r9_prerequisite()
    assert result["status"] == "passed"
    assert result["receipt_file_sha256"] == lane.R9_RECEIPT_FILE_SHA256
    assert result["receipt_payload_sha256"] == lane.R9_RECEIPT_PAYLOAD_SHA256
    assert result["image_id"] == lane.IMAGE_ID
    assert result["runtime_probe_passed"] is True
    assert result["gpu_exposed"] is False
    assert result["probe_receipt_file_sha256"] == lane.R9_PROBE_RECEIPT_SHA256
    assert result["container_onnx_version"] == "1.14.1"


def test_r9_historical_evidence_remains_immutable() -> None:
    assert stat.S_IMODE(lane.R9_RECEIPT.stat().st_mode) == 0o440
    assert stat.S_IMODE(lane.R9_PLAN.stat().st_mode) == 0o440
    assert lane.sha256_file(lane.R9_RECEIPT) == lane.R9_RECEIPT_FILE_SHA256
    assert lane.sha256_file(lane.R9_PLAN) == lane.R9_PLAN_FILE_SHA256
    assert lane.sha256_file(lane.R9_PROBE_RECEIPT) == lane.R9_PROBE_RECEIPT_SHA256


def test_official_source_checkpoint_and_license_pins_are_exact() -> None:
    contract = lane.strict_json(lane.CONTRACT)
    lane.validate_contract(contract)
    sources = contract["official_sources"]
    assert sources["mmpose"]["license_spdx"] == "Apache-2.0"
    assert sources["mmdeploy"]["license_spdx"] == "Apache-2.0"
    assert lane.sha256_file(lane.CHECKPOINT) == sources["checkpoint"]["sha256"]
    assert lane.sha256_file(lane.MODEL_CONFIG) == sources["mmpose"]["config_sha256"]
    assert lane.sha256_file(lane.MMPPOSE_LICENSE) == sources["mmpose"]["license_sha256"]
    assert lane.sha256_file(lane.MMDEPLOY_LICENSE) == sources["mmdeploy"]["license_sha256"]


def test_plan_schema_and_self_fingerprint_are_valid(plan: dict) -> None:
    jsonschema.Draft202012Validator(PLAN_SCHEMA).validate(plan)
    assert plan["plan_fingerprint_sha256"] == lane.fingerprint(
        plan, "plan_fingerprint_sha256"
    )


def test_plan_replay_is_byte_exact(plan: dict) -> None:
    result = lane.verify_plan(plan, plan["plan_fingerprint_sha256"])
    assert result["valid"] is True
    assert result["r9_prerequisite_passed"] is True
    assert result["container_run"] is False
    assert result["model_loaded"] is False
    assert result["export_executed"] is False


def test_host_checker_runtime_and_source_are_accepted_plan_pinned(plan: dict) -> None:
    runtime = plan["host_onnx_recheck_runtime"]
    assert runtime == lane._host_checker_runtime()
    assert runtime["python"]["flags"] == ["-B", "-I", "-S"]
    assert runtime["python"]["sha256"] == lane.HOST_CHECKER_PYTHON_SHA256
    assert runtime["container_onnx_version"] == "1.14.1"
    assert runtime["host_onnx_version"] == "1.22.0"
    assert runtime["container_parity_replaced"] is False
    assert runtime["private_verified_snapshot_required"] is True
    assert runtime["unhashed_pyc_execution_allowed"] is False
    tooling = lane._host_checker_tooling(plan)
    assert tooling["inputs"]["host_onnx_checker"] == plan["inputs"][
        "host_onnx_checker"
    ]
    assert len(tooling["inputs"]) == 7
    assert "sys.path.insert" not in lane.HOST_CHECKER_BOOTSTRAP_SOURCE
    execute_source = inspect.getsource(lane.execute)
    assert "_host_checker_tooling(plan)" in execute_source
    assert "_host_onnx_recheck(" in execute_source


def test_plan_build_does_not_call_subprocess_or_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("planner attempted a subprocess")

    monkeypatch.setattr(lane.subprocess, "run", forbidden)
    value = lane.build_plan(created_at=FIXED_TIME)
    assert value["execution_boundary"]["docker_queried"] is False


def test_plan_external_fingerprint_tamper_fails(plan: dict) -> None:
    with pytest.raises(lane.ExportR10Error, match="external plan fingerprint"):
        lane.verify_plan(plan, "0" * 64)


def test_plan_semantic_tamper_fails_even_when_resealed(plan: dict) -> None:
    forged = copy.deepcopy(plan)
    forged["profiles"]["960"]["quality_claimed"] = True
    forged["plan_fingerprint_sha256"] = lane.fingerprint(
        forged, "plan_fingerprint_sha256"
    )
    with pytest.raises((lane.ExportR10Error, jsonschema.ValidationError)):
        lane.verify_plan(forged, forged["plan_fingerprint_sha256"])


def test_profiles_are_separate_dynamic_batch_fixed_spatial(plan: dict) -> None:
    assert plan["profiles"]["640"]["profile_kind"] == "dynamic_batch_fixed_spatial"
    assert plan["profiles"]["640"]["input_shapes"] == [
        [1, 3, 640, 640],
        [12, 3, 640, 640],
    ]
    assert plan["profiles"]["960"]["input_shapes"] == [
        [1, 3, 960, 960],
        [12, 3, 960, 960],
    ]
    assert plan["profiles"]["960"]["trained_resolution"] is False
    assert plan["profiles"]["960"]["quality_claimed"] is False
    assert plan["profiles"]["960"]["runtime_model_config_overrides"] == {
        "input_size_variable": [960, 960],
        "codec_input_size": [960, 960],
        "test_bottomup_resize_input_size": [960, 960],
        "val_bottomup_resize_input_size": [960, 960],
        "training_pipeline_modified": False,
    }


def test_runtime_profile_configuration_updates_mmpose_preprocessing_not_training() -> None:
    deploy = {"onnx_config": {"input_shape": [640, 640], "save_file": "old.onnx"}}
    train_pipeline = [{"type": "BottomupResize", "input_size": (640, 640)}]
    model = {
        "input_size": (640, 640),
        "codec": {"input_size": (640, 640)},
        "test_dataloader": {
            "dataset": {
                "pipeline": [
                    {"type": "LoadImage"},
                    {"type": "BottomupResize", "input_size": (640, 640)},
                ]
            }
        },
        "val_dataloader": {
            "dataset": {
                "pipeline": [
                    {"type": "LoadImage"},
                    {"type": "BottomupResize", "input_size": (640, 640)},
                ]
            }
        },
        "train_dataloader": {"dataset": {"pipeline": train_pipeline}},
    }
    before_training = copy.deepcopy(train_pipeline)
    result = container.apply_profile_configuration(deploy, model, 960)
    assert result["passed"] is True
    assert deploy["onnx_config"]["input_shape"] == [960, 960]
    assert model["codec"]["input_size"] == (960, 960)
    assert model["test_dataloader"]["dataset"]["pipeline"][1]["input_size"] == (
        960,
        960,
    )
    assert model["val_dataloader"]["dataset"]["pipeline"][1]["input_size"] == (
        960,
        960,
    )
    assert train_pipeline == before_training


def test_runtime_profile_configuration_fails_if_resize_contract_drifts() -> None:
    deploy = {"onnx_config": {}}
    model = {
        "codec": {},
        "test_dataloader": {"dataset": {"pipeline": [{"type": "LoadImage"}]}},
        "val_dataloader": {"dataset": {"pipeline": [{"type": "BottomupResize"}]}},
    }
    with pytest.raises(container.ProfileExportError, match="one BottomupResize"):
        container.apply_profile_configuration(deploy, model, 960)


def test_onnx_names_shapes_opset_and_numeric_gates_are_exact() -> None:
    contract = lane.strict_json(lane.CONTRACT)
    assert contract["onnx"]["opset"] == 11
    assert contract["onnx"]["input"]["name"] == "input"
    assert contract["onnx"]["outputs"][0]["name"] == "dets"
    assert contract["onnx"]["outputs"][0]["shape"] == ["B", 100, 5]
    assert contract["onnx"]["outputs"][1]["name"] == "keypoints"
    assert contract["onnx"]["outputs"][1]["shape"] == ["B", 100, 17, 3]
    acceptance = contract["numerical_acceptance"]
    assert acceptance["provider"] == "CPUExecutionProvider"
    assert acceptance["pytorch_vs_onnxruntime_required_for_batches"] == [1, 12]
    assert acceptance["batch12_vs_twelve_batch1_required"] is True
    assert acceptance["absolute_tolerance"] == 0.0001


def test_docker_commands_are_argv_only_exact_image_and_isolated(plan: dict) -> None:
    for command in plan["commands_not_executed"].values():
        assert isinstance(command, list)
        assert command[:2] == ["docker", "run"]
        assert "--pull=never" in command
        assert "--network=none" in command
        assert "--read-only" in command
        assert "--cap-drop=ALL" in command
        assert "--security-opt=no-new-privileges" in command
        assert "CUDA_VISIBLE_DEVICES=" in command
        assert "NVIDIA_VISIBLE_DEVICES=void" in command
        assert "--gpus" not in command
        assert lane.IMAGE_ID in command
        assert "-c" in command
        assert lane.BOOTSTRAP_SOURCE in command
        assert lane.sha256_file(lane.CONTAINER_RUNNER) in command
        assert "<OUTPUT_DEVICE>" in command
        assert "<OUTPUT_INODE>" in command
        assert "docker pull" not in command
        assert "docker build" not in command


def test_only_profile_output_mount_is_writable(plan: dict) -> None:
    for command in plan["commands_not_executed"].values():
        mounts = [command[index + 1] for index, token in enumerate(command[:-1]) if token == "--mount"]
        assert len(mounts) == 4
        assert all(value.endswith("readonly=true") for value in mounts[:3])
        assert mounts[3].endswith("readonly=false")


def test_plan_explicitly_excludes_model_load_export_gpu_trt_and_ds(plan: dict) -> None:
    assert plan["execution_boundary"] == {
        "planner_only": True,
        "docker_queried": False,
        "container_run": False,
        "model_loaded": False,
        "export_executed": False,
        "onnxruntime_executed": False,
        "gpu_exposed": False,
        "tensorrt_executed": False,
        "deepstream_executed": False,
    }


def test_execution_parser_requires_explicit_plan_acceptance() -> None:
    with pytest.raises(SystemExit):
        lane.parser().parse_args(["execute", "--run-id", "cpu-export-001"])
    parsed = lane.parser().parse_args(
        [
            "execute",
            "--accept-plan-fingerprint",
            "1" * 64,
            "--run-id",
            "cpu-export-001",
        ]
    )
    assert parsed.accept_plan_fingerprint == "1" * 64


@pytest.mark.parametrize("invalid", ["", "UPPER", "ab", "../escape", "with space", "a" * 65])
def test_run_id_contract_rejects_unsafe_values(invalid: str) -> None:
    assert lane.RUN_ID_RE.fullmatch(invalid) is None


def test_atomic_plan_publication_refuses_overwrite(tmp_path: Path, plan: dict) -> None:
    target = tmp_path / "plan.json"
    lane.atomic_json_no_overwrite(target, plan)
    before = target.read_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == 0o440
    with pytest.raises(lane.ExportR10Error, match="refusing to overwrite"):
        lane.atomic_json_no_overwrite(target, plan)
    assert target.read_bytes() == before


def test_atomic_publication_rejects_symlinked_parent(tmp_path: Path, plan: dict) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(lane.ExportR10Error, match="symlink"):
        lane.atomic_json_no_overwrite(linked / "plan.json", plan)


def test_runs_root_creation_and_symlink_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runs"
    monkeypatch.setattr(lane, "RUNS_ROOT", root)
    lane.ensure_runs_root()
    assert root.is_dir()
    root.rmdir()
    real = tmp_path / "real"
    real.mkdir()
    root.symlink_to(real, target_is_directory=True)
    with pytest.raises(lane.ExportR10Error, match="runs root is unsafe"):
        lane.ensure_runs_root()


def test_strict_json_rejects_duplicate_keys_nonfinite_and_symlink(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(lane.ExportR10Error, match="duplicate JSON key"):
        lane.strict_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(lane.ExportR10Error, match="non-finite"):
        lane.strict_json(nonfinite)
    link = tmp_path / "link.json"
    link.symlink_to(duplicate)
    with pytest.raises(lane.ExportR10Error, match="unsafe JSON"):
        lane.strict_json(link)


def test_container_comparison_accepts_tolerance_and_rejects_drift() -> None:
    reference = np.array([1.0, 2.0], dtype=np.float32)
    accepted = reference + np.array([5e-5, -5e-5], dtype=np.float32)
    result = container._comparison(reference, accepted)
    assert result["passed"] is True
    with pytest.raises(container.ProfileExportError, match="numeric parity failed"):
        container._comparison(reference, reference + 0.1)


def test_container_atomic_receipt_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    container.atomic_json_no_overwrite(target, {"status": "test"})
    with pytest.raises(container.ProfileExportError, match="refusing to overwrite"):
        container.atomic_json_no_overwrite(target, {"status": "forged"})
    assert stat.S_IMODE(target.stat().st_mode) == 0o440


def test_anonymous_inode_publication_has_no_temporary_name_window() -> None:
    host_source = inspect.getsource(lane.atomic_bytes_at)
    container_source = inspect.getsource(container.atomic_bytes_at)
    assert "O_TMPFILE" in host_source
    assert "O_TMPFILE" in container_source
    assert "os.link(" not in host_source
    assert "os.link(" not in container_source
    assert "_link_open_inode" in host_source
    assert "_link_open_inode" in container_source


def test_permissions_are_fsynced_after_fchmod_before_publication() -> None:
    for function in (lane.atomic_bytes_at, container.atomic_bytes_at):
        source = inspect.getsource(function)
        assert source.index("os.fchmod(descriptor, mode)") < source.index(
            "os.fsync(descriptor)"
        )
    publication = inspect.getsource(container.publish_file_from_private_staging)
    assert publication.index("os.fchmod(anonymous_fd, 0o440)") < publication.index(
        "os.fsync(anonymous_fd)"
    )
    freeze = inspect.getsource(lane._scan_tree_fd)
    freeze_chmod = freeze.index("os.fchmod(descriptor, 0o440)")
    assert freeze_chmod < freeze.index("os.fsync(descriptor)", freeze_chmod)
    assert lane.BOOTSTRAP_SOURCE.index("os.fchmod(target, 0o400)") < lane.BOOTSTRAP_SOURCE.index(
        "os.fsync(target)"
    )


def test_host_checker_record_snapshot_excludes_unhashed_existing_pyc(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "environment"
    site = environment / "lib/python3.12/site-packages"
    dist_info = site / "demo-1.0.dist-info"
    pycache = site / "__pycache__"
    dist_info.mkdir(parents=True)
    pycache.mkdir()
    source = site / "demo.py"
    source.write_bytes(b"VALUE = 'accepted-source'\n")
    malicious_pyc = pycache / "demo.cpython-312.pyc"
    malicious_pyc.write_bytes(b"unhashed-malicious-bytecode")
    encoded = base64.urlsafe_b64encode(
        hashlib.sha256(source.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    record = dist_info / "RECORD"
    record.write_text(
        f"demo.py,sha256={encoded},{source.stat().st_size}\n"
        "__pycache__/demo.cpython-312.pyc,,\n"
        "demo-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    snapshot_environment = tmp_path / "snapshot"
    snapshot_site = snapshot_environment / "lib/python3.12/site-packages"
    snapshot_site.mkdir(parents=True)
    result = host_checker.verify_record_closure(
        {
            "distribution": "demo",
            "version": "1.0",
            "record_path": str(record),
            "record_relative": "demo-1.0.dist-info/RECORD",
            "record_bytes": record.stat().st_size,
            "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
        },
        site_packages=site,
        environment_root=environment,
        snapshot_site_packages=snapshot_site,
        snapshot_environment_root=snapshot_environment,
    )
    assert (snapshot_site / "demo.py").read_bytes() == source.read_bytes()
    assert not (snapshot_site / "__pycache__/demo.cpython-312.pyc").exists()
    assert result["hashed_entries_verified"] == 1
    assert result["unhashed_record_or_pyc_entries_excluded_from_snapshot"] == 2
    assert result["private_snapshot_entries"] == 2


def test_external_tensor_scan_includes_training_and_function_attribute_graphs() -> None:
    class AttributeProto:
        TENSOR = 1
        TENSORS = 2
        SPARSE_TENSOR = 3
        SPARSE_TENSORS = 4
        GRAPH = 5
        GRAPHS = 6

    fake_onnx = SimpleNamespace(AttributeProto=AttributeProto)
    graph_tensor = SimpleNamespace(name="graph")
    training_tensor = SimpleNamespace(name="training")
    function_tensor = SimpleNamespace(name="function")
    empty_graph = lambda tensors=(): SimpleNamespace(  # noqa: E731
        initializer=list(tensors), sparse_initializer=[], node=[]
    )
    function_attribute = SimpleNamespace(
        type=AttributeProto.TENSOR, t=function_tensor
    )
    model = SimpleNamespace(
        graph=empty_graph([graph_tensor]),
        training_info=[
            SimpleNamespace(
                initialization=empty_graph([training_tensor]),
                algorithm=empty_graph(),
            )
        ],
        functions=[SimpleNamespace(node=[], attribute_proto=[function_attribute])],
    )
    assert container._all_model_tensors(model, fake_onnx) == [
        graph_tensor,
        training_tensor,
        function_tensor,
    ]


def test_private_staging_publication_is_exact_and_no_overwrite(tmp_path: Path) -> None:
    private = tmp_path / "private"
    output = tmp_path / "output"
    private.mkdir()
    output.mkdir()
    source = private / "model.onnx"
    source.write_bytes(b"validated-private-onnx-bytes")
    result = container.publish_file_from_private_staging(
        source, output, "published.onnx"
    )
    assert result == {
        "path": "published.onnx",
        "bytes": len(b"validated-private-onnx-bytes"),
        "sha256": hashlib.sha256(b"validated-private-onnx-bytes").hexdigest(),
    }
    assert (output / "published.onnx").read_bytes() == source.read_bytes()
    assert stat.S_IMODE((output / "published.onnx").stat().st_mode) == 0o440
    with pytest.raises(container.ProfileExportError, match="refusing to overwrite"):
        container.publish_file_from_private_staging(
            source, output, "published.onnx"
        )


def test_inline_bootstrap_executes_only_verified_runner_bytes_and_private_inputs(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "runner.py"
    checkpoint = tmp_path / "checkpoint.pth"
    seed = tmp_path / "seed.jpg"
    runner.write_text(
        "import hashlib,json,pathlib,sys\n"
        "values={}\n"
        "for option in ('--checkpoint','--seed-image'):\n"
        " p=pathlib.Path(sys.argv[sys.argv.index(option)+1]); values[option]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}\n"
        "print(json.dumps(values,sort_keys=True))\n",
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"checkpoint-private-copy")
    seed.write_bytes(b"seed-private-copy")
    command = [
        str(ROOT / ".venv/bin/python"),
        "-B",
        "-c",
        lane.BOOTSTRAP_SOURCE,
        str(runner),
        lane.sha256_file(runner),
        str(runner.stat().st_size),
        str(checkpoint),
        lane.sha256_file(checkpoint),
        str(checkpoint.stat().st_size),
        str(seed),
        lane.sha256_file(seed),
        str(seed.stat().st_size),
        "--",
        "--checkpoint",
        str(checkpoint),
        "--seed-image",
        str(seed),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    result = json.loads(completed.stdout)
    assert result["--checkpoint"]["path"].startswith("/tmp/pose-r10-inputs-")
    assert result["--seed-image"]["path"].startswith("/tmp/pose-r10-inputs-")
    assert result["--checkpoint"]["sha256"] == lane.sha256_file(checkpoint)
    assert result["--seed-image"]["sha256"] == lane.sha256_file(seed)
    forged = list(command)
    forged[5] = "0" * 64
    rejected = subprocess.run(forged, check=False, text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "bootstrap input hash differs" in rejected.stderr


def test_fd_bound_tree_freeze_rejects_symlinks_and_records_exact_closure(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    child = run / "640"
    child.mkdir(parents=True)
    (run / "docker.log").write_bytes(b"log")
    (child / "model.onnx").write_bytes(b"onnx")
    descriptor = os.open(run, lane._directory_flags())
    try:
        artifacts = lane._scan_tree_fd(descriptor, freeze=True, freeze_root=True)
    finally:
        os.close(descriptor)
    assert [item["path"] for item in artifacts] == [
        "640/model.onnx",
        "docker.log",
    ]
    assert stat.S_IMODE(run.stat().st_mode) == 0o550
    assert stat.S_IMODE(child.stat().st_mode) == 0o550
    assert stat.S_IMODE((child / "model.onnx").stat().st_mode) == 0o440

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / "link").symlink_to(run / "docker.log")
    unsafe_fd = os.open(unsafe, lane._directory_flags())
    try:
        with pytest.raises(lane.ExportR10Error, match="special or symlink"):
            lane._scan_tree_fd(unsafe_fd, freeze=True, freeze_root=True)
    finally:
        os.close(unsafe_fd)


def test_profile_receipt_schema_and_host_verifier_accept_exact_pass(tmp_path: Path) -> None:
    receipt_path, value = profile_receipt(tmp_path)
    jsonschema.Draft202012Validator(PROFILE_SCHEMA).validate(value)
    verified = lane.verify_profile_receipt(
        receipt_path,
        profile=640,
        plan_fingerprint="1" * 64,
        run_id="cpu-export-test",
    )
    assert verified["numerical_validation"]["passed"] is True


def test_profile_receipt_rejects_gpu_overclaim(tmp_path: Path) -> None:
    receipt_path, value = profile_receipt(tmp_path)
    value["execution_boundary"]["gpu_exposed"] = True
    value["receipt_fingerprint_sha256"] = container.fingerprint(
        value, "receipt_fingerprint_sha256"
    )
    receipt_path.chmod(0o660)
    receipt_path.write_text(json.dumps(value), encoding="utf-8")
    receipt_path.chmod(0o440)
    with pytest.raises((lane.ExportR10Error, jsonschema.ValidationError)):
        lane.verify_profile_receipt(
            receipt_path,
            profile=640,
            plan_fingerprint="1" * 64,
            run_id="cpu-export-test",
        )


def test_profile_schema_rejects_tensorrt_or_deepstream_claim(tmp_path: Path) -> None:
    _, value = profile_receipt(tmp_path)
    value["execution_boundary"]["tensorrt_executed"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(PROFILE_SCHEMA).validate(value)


def test_host_export_receipt_schema_encodes_fail_closed_pair(plan: dict) -> None:
    value = {
        "schema_version": lane.RECEIPT_SCHEMA_VERSION,
        "status": "failed",
        "run_id": "cpu-export-test",
        "created_at": FIXED_TIME,
        "plan_fingerprint_sha256": "1" * 64,
        "image": {"required_id": lane.IMAGE_ID, "stable": False},
        "run_directory": {
            "name": "cpu-export-test",
            "device": 42,
            "inode": 84,
            "mode": "0550",
            "frozen_before_receipt": True,
            "artifact_count": 1,
        },
        "execution_boundary": {
            "container_runs_attempted": 0,
            "network_policy": "none",
            "root_filesystem_policy": "read_only",
            "non_root_policy": True,
            "gpu_exposed": False,
            "gpu_api_queried": False,
            "docker_pull": False,
            "docker_build": False,
            "tensorrt_executed": False,
            "deepstream_executed": False,
        },
        "host_onnx_recheck_tooling": lane._host_checker_tooling(plan),
        "profiles": {
            str(size): {
                "status": "not_attempted",
                "attempted": False,
                "spatial_size": size,
                "dynamic_batch": [1, 12],
                "receipt": None,
                "host_onnx_recheck": None,
                "docker_exit_code": None,
                "error": None,
            }
            for size in (640, 960)
        },
        "artifacts": [
            {
                "path": "host-failure.json",
                "bytes": 1,
                "sha256": "2" * 64,
                "mode": "0440",
            }
        ],
        "error": "not executed",
        "conclusions": {
            "both_profiles_passed": False,
            "publishable_onnx_pair": False,
            "profile_960_quality_claimed": False,
            "production_model_selected": False,
            "tensorrt_verified": False,
            "deepstream9_verified": False,
            "production_ready": False,
        },
    }
    value["receipt_fingerprint_sha256"] = lane.fingerprint(
        value, "receipt_fingerprint_sha256"
    )
    jsonschema.Draft202012Validator(RUN_SCHEMA).validate(value)


def test_contract_observed_fields_remain_all_false() -> None:
    contract = lane.strict_json(lane.CONTRACT)
    assert all(value is False for value in contract["observed"].values())
    assert contract["runtime_policy"]["docker_pull"] is False
    assert contract["runtime_policy"]["docker_build"] is False
    assert contract["runtime_policy"]["tensorrt"] is False
    assert contract["runtime_policy"]["deepstream"] is False


def test_container_runner_has_no_host_docker_or_gpu_execution_path() -> None:
    source = lane.CONTAINER_RUNNER.read_text(encoding="utf-8")
    assert "subprocess.run" not in source
    assert "docker run" not in source
    assert "torch2onnx" in source
    assert "CPUExecutionProvider" in source
    assert "NVIDIA_VISIBLE_DEVICES" in source
    assert "tensorrt" in source.lower()  # only explicit false boundaries


def test_all_new_json_files_use_unique_keys_and_are_finite() -> None:
    for path in (
        lane.CONTRACT,
        lane.PLAN_SCHEMA,
        lane.RECEIPT_SCHEMA,
        lane.PROFILE_RECEIPT_SCHEMA,
    ):
        value = lane.strict_json(path)
        assert isinstance(value, dict)
        lane.canonical_bytes(value)
