from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_child_image_r4 as lane


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shared_contract():
    return _load(lane.LOCAL_SOURCE_CONTRACT, "local_source_contract")


@pytest.fixture(scope="module")
def producer(shared_contract):
    return _load(lane.MAKE_LOCAL_LOCK, "_r4_make_local_lock_test")


def _manifest(contract, lock: bytes = b"locked source wheels\n") -> dict:
    wheels = [
        {
            "distribution": "mmdeploy",
            "version": "1.3.1",
            "filename": "mmdeploy-1.3.1-py3-none-any.whl",
            "bytes": 445743,
            "sha256": "a" * 64,
            "tags": ["py3-none-any"],
            "parsed_tags": [
                {"interpreter": "py3", "abi": "none", "platform": "any"}
            ],
            "pure_python": True,
        },
        {
            "distribution": "mmpose",
            "version": "1.3.2",
            "filename": "mmpose-1.3.2-py2.py3-none-any.whl",
            "bytes": 1720116,
            "sha256": "b" * 64,
            "tags": ["py2-none-any", "py3-none-any"],
            "parsed_tags": [
                {"interpreter": "py2", "abi": "none", "platform": "any"},
                {"interpreter": "py3", "abi": "none", "platform": "any"},
            ],
            "pure_python": True,
        },
    ]
    value = {
        "schema_version": contract.SCHEMA_VERSION,
        "source_date_epoch": "1581638400",
        "source_modified": False,
        "build_network": "none",
        "build_isolation": False,
        "torch_visible_during_wheel_build": False,
        "tag_policy": dict(contract.TAG_POLICY),
        "wheels": wheels,
        "lock_sha256": hashlib.sha256(lock).hexdigest(),
    }
    value["payload_sha256"] = contract.payload_sha256(value)
    return value


def _resign(contract, value: dict) -> dict:
    value["payload_sha256"] = contract.payload_sha256(value)
    return value


def _wheel(tmp_path: Path, distribution: str, version: str, tags: list[str]) -> Path:
    normalized = distribution.replace("-", "_")
    path = tmp_path / f"{normalized}-{version}-test.whl"
    dist_info = f"{normalized}-{version}.dist-info"
    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {distribution}\n"
        f"Version: {version}\n\n"
    ).encode()
    wheel = (
        "Wheel-Version: 1.0\n"
        "Generator: deepsafe-test\n"
        "Root-Is-Purelib: true\n"
        + "".join(f"Tag: {tag}\n" for tag in tags)
        + "\n"
    ).encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", wheel)
    return path


@pytest.fixture(scope="module")
def plan() -> dict:
    return lane.build_plan(
        created_at="2026-07-18T02:20:00+00:00",
        base_local_observed=True,
        docker_version="29.4.0",
        buildx_version=(
            "github.com/docker/buildx v0.33.0 "
            "f7897eba028583e0071642db3c011e860444f8cf"
        ),
    )


def test_shared_contract_accepts_exact_valid_manifest(shared_contract) -> None:
    value = _manifest(shared_contract)
    assert shared_contract.validate_manifest(value) == value


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("schema", "schema differs"),
        ("extra_top", "top-level fields differ"),
        ("missing_top", "top-level fields differ"),
        ("tag_policy", "tag policy differs"),
        ("extra_wheel", "wheel fields differ"),
        ("missing_wheel", "wheel fields differ"),
        ("bool_bytes", "bytes differ"),
        ("path_escape", "filename escapes"),
        ("duplicate_distribution", "wheel order/set differs"),
        ("pure_false", "pure-wheel claim differs"),
        ("abi", "non-none ABI"),
        ("platform", "platform tag"),
        ("no_py3", "no py3 tag"),
        ("parsed_mismatch", "parsed tags do not replay"),
        ("parsed_extra", "parsed tag fields differ"),
        ("bad_sha", "local source SHA differs"),
        ("bad_lock_sha", "local source lock hash differs"),
        ("wheel_count", "wheel count differs"),
    ],
)
def test_closed_contract_rejects_resigned_structural_mutations(
    shared_contract, case: str, message: str
) -> None:
    value = copy.deepcopy(_manifest(shared_contract))
    if case == "schema":
        value["schema_version"] = "deepsafe.pose-local-source-wheels/v1"
    elif case == "extra_top":
        value["unexpected"] = False
    elif case == "missing_top":
        value.pop("source_modified")
    elif case == "tag_policy":
        value["tag_policy"]["allow_abi3"] = True
    elif case == "extra_wheel":
        value["wheels"][0]["unexpected"] = "x"
    elif case == "missing_wheel":
        value["wheels"][0].pop("sha256")
    elif case == "bool_bytes":
        value["wheels"][0]["bytes"] = True
    elif case == "path_escape":
        value["wheels"][0]["filename"] = "../mmdeploy.whl"
    elif case == "duplicate_distribution":
        value["wheels"][1]["distribution"] = "mmdeploy"
        value["wheels"][1]["version"] = "1.3.1"
    elif case == "pure_false":
        value["wheels"][0]["pure_python"] = False
    elif case == "abi":
        value["wheels"][0]["tags"] = ["py3-abi3-any"]
    elif case == "platform":
        value["wheels"][0]["tags"] = ["py3-none-linux_x86_64"]
    elif case == "no_py3":
        value["wheels"][1]["tags"] = ["py2-none-any"]
    elif case == "parsed_mismatch":
        value["wheels"][0]["parsed_tags"][0]["interpreter"] = "py2"
    elif case == "parsed_extra":
        value["wheels"][0]["parsed_tags"][0]["extra"] = "x"
    elif case == "bad_sha":
        value["wheels"][0]["sha256"] = "not-a-sha"
    elif case == "bad_lock_sha":
        value["lock_sha256"] = "bad"
    else:
        value["wheels"].append(copy.deepcopy(value["wheels"][0]))
    _resign(shared_contract, value)
    with pytest.raises(shared_contract.ContractError, match=message):
        shared_contract.validate_manifest(value)


def test_closed_contract_rejects_payload_tamper_without_resigning(shared_contract) -> None:
    value = _manifest(shared_contract)
    value["wheels"][0]["bytes"] += 1
    with pytest.raises(shared_contract.ContractError, match="self hash differs"):
        shared_contract.validate_manifest(value)


def test_producer_emits_manifest_accepted_by_shared_contract(
    shared_contract, producer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    _wheel(wheel_dir, "mmdeploy", "1.3.1", ["py3-none-any"])
    _wheel(wheel_dir, "mmpose", "1.3.2", ["py2-none-any", "py3-none-any"])
    output_lock = tmp_path / "local.lock"
    output_manifest = tmp_path / "local.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_local_lock.py",
            "--wheel-dir",
            str(wheel_dir),
            "--output-lock",
            str(output_lock),
            "--output-manifest",
            str(output_manifest),
        ],
    )
    assert producer.main() == 0
    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert shared_contract.validate_manifest(manifest) == manifest
    assert hashlib.sha256(output_lock.read_bytes()).hexdigest() == manifest["lock_sha256"]


def test_probe_adapter_replays_same_contract_and_rejects_resigned_extra_field(
    shared_contract, tmp_path: Path
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    shutil.copy2(lane.ENVIRONMENT_PROBE_ADAPTER, runtime / "environment_probe.py")
    shutil.copy2(lane.LOCAL_SOURCE_CONTRACT, runtime / "local_source_contract.py")
    shutil.copy2(lane.ENVIRONMENT_PROBE, runtime / "environment_probe_base.py")
    adapter = _load(runtime / "environment_probe.py", "_r4_probe_adapter_test")
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    lock = b"locked source wheels\n"
    (provenance / "requirements-local-sources.lock.txt").write_bytes(lock)
    manifest_path = provenance / "local-source-wheels.json"
    value = _manifest(shared_contract, lock)
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    adapter._base.PROVENANCE = provenance
    assert adapter.local_source_manifest() == value
    value["unexpected"] = False
    _resign(shared_contract, value)
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(adapter._base.ProbeError, match="top-level fields differ"):
        adapter.local_source_manifest()


def test_r4_plan_is_hash_bound_and_replayable(plan: dict) -> None:
    assert plan["plan_sha256"] == lane.payload_sha256(plan, "plan_sha256")
    result = lane.verify_plan(plan, plan["plan_sha256"])
    assert result["valid"] is True
    assert result["status"] == "planned_build_ready"
    assert result["gpu_exposed"] is False


def test_r4_plan_pins_shared_contract_adapter_and_three_failures(plan: dict) -> None:
    assert plan["inputs"]["local_source_contract"]["path"].endswith(
        "child-image-v4/local_source_contract.py"
    )
    assert plan["inputs"]["environment_probe_adapter"]["path"].endswith(
        "child-image-v4/environment_probe.py"
    )
    assert plan["inputs"]["validator_base_v3"]["sha256"] == lane.V3_VALIDATOR_SHA256
    assert plan["inputs"]["failed_r3_attempt_receipt"]["sha256"] == lane.R3_FAILED_RECEIPT_FILE_SHA256
    assert plan["inputs"]["failed_r3_build_log"]["sha256"] == lane.R3_FAILED_BUILD_LOG_SHA256
    assert [item["attempt_id"] for item in plan["repair_chain"]] == [
        "child-v1-r1-001",
        "child-v2-r2-001",
        "child-v3-r3-001",
    ]
    assert plan["repair_chain"][2]["pip_check_passed"] is True


def test_r4_build_and_runtime_commands_remain_offline_gpu_free(plan: dict) -> None:
    build = plan["commands"]["build"]
    assert "--network=none" in build and "--pull=false" in build
    assert f"wheelbundle={lane.WHEELHOUSE}" in build
    assert f"childhelpers={lane.CHILD_HELPERS_R1}" in build
    assert "--gpus" not in build
    probe = plan["commands"]["runtime_probe_template"]
    assert "--network=none" in probe and "--read-only" in probe
    assert "--cap-drop=ALL" in probe and "--security-opt=no-new-privileges" in probe
    assert "NVIDIA_VISIBLE_DEVICES=void" in probe and "CUDA_VISIBLE_DEVICES=" in probe
    assert "--gpus" not in probe


def test_r4_contract_records_shared_closed_schema_without_overclaim() -> None:
    contract = json.loads(lane.CONTRACT.read_text(encoding="utf-8"))
    shared = contract["build_policy"]["local_source_manifest_contract"]
    assert shared["schema"] == "deepsafe.pose-local-source-wheels/v2"
    assert shared["closed_top_level_fields"] is True
    assert shared["closed_wheel_fields"] is True
    assert shared["producer_and_probe_share_validator"] is True
    assert all(value is False for value in contract["observed"].values())


def test_r4_scripts_parse_as_python38_and_adapter_never_queries_gpu() -> None:
    for path in (
        Path(lane.__file__),
        lane.MAKE_LOCAL_LOCK,
        lane.LOCAL_SOURCE_CONTRACT,
        lane.ENVIRONMENT_PROBE_ADAPTER,
        lane.ENVIRONMENT_PROBE,
    ):
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=(3, 8),
        )
    adapter = lane.ENVIRONMENT_PROBE_ADAPTER.read_text(encoding="utf-8")
    assert "torch.cuda." not in adapter
    assert "nvidia-smi" not in adapter


def test_r4_plan_tamper_fails_against_external_pin(plan: dict) -> None:
    tampered = copy.deepcopy(plan)
    tampered["commands"]["build"].remove("--network=none")
    tampered["plan_sha256"] = lane.payload_sha256(tampered, "plan_sha256")
    with pytest.raises(lane.ChildImageError, match="external plan pin"):
        lane.verify_plan(tampered, plan["plan_sha256"])


def test_r4_source_validation_replays_all_prior_failure_evidence() -> None:
    result = lane._validate_sources()
    assert result["wheelhouse"]["valid"] is True
    assert result["failed_r1_attempt"]["build_exit_code"] == 1
    assert result["failed_r2_attempt"]["source_wheels_built"] is True
    assert result["failed_r3_attempt"]["full_offline_install_passed"] is True
    assert result["failed_r3_attempt"]["pip_check_passed"] is True
