from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_child_image_r3 as lane


def _load_lock_helper():
    spec = importlib.util.spec_from_file_location("_r3_lock_helper_test", lane.MAKE_LOCAL_LOCK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helper():
    return _load_lock_helper()


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
        created_at="2026-07-18T02:10:00+00:00",
        base_local_observed=True,
        docker_version="29.4.0",
        buildx_version=(
            "github.com/docker/buildx v0.33.0 "
            "f7897eba028583e0071642db3c011e860444f8cf"
        ),
    )


def test_r3_accepts_observed_mmpose_universal_tag_pair(helper, tmp_path: Path) -> None:
    result = helper.inspect(
        _wheel(tmp_path, "mmpose", "1.3.2", ["py2-none-any", "py3-none-any"])
    )
    assert result["pure_python"] is True
    assert result["tags"] == ["py2-none-any", "py3-none-any"]
    assert result["parsed_tags"] == [
        {"interpreter": "py2", "abi": "none", "platform": "any"},
        {"interpreter": "py3", "abi": "none", "platform": "any"},
    ]


def test_r3_accepts_compressed_py2_py3_universal_tag(helper, tmp_path: Path) -> None:
    result = helper.inspect(
        _wheel(tmp_path, "mmpose", "1.3.2", ["py2.py3-none-any"])
    )
    assert [tag["interpreter"] for tag in result["parsed_tags"]] == ["py2", "py3"]


@pytest.mark.parametrize(
    ("tags", "message"),
    [
        (["py3-abi3-any"], "non-none ABI"),
        (["cp38-cp38-any", "py3-none-any"], "non-none ABI"),
        (["py3-none-linux_x86_64"], "platform tag"),
        (["py3-none-any", "py2-none-win_amd64"], "platform tag"),
        (["py2-none-any"], "no py3 tag"),
        (["cp38-none-any"], "no py3 tag"),
        (["py3-none-any-extra"], "malformed wheel tag"),
        ([], "wheel has no tags"),
    ],
)
def test_r3_rejects_adversarial_tags(helper, tmp_path: Path, tags: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        helper.inspect(_wheel(tmp_path, "mmpose", "1.3.2", tags))


def test_r3_main_writes_hashed_lock_and_v2_tag_manifest(
    helper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    assert helper.main() == 0
    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "deepsafe.pose-local-source-wheels/v2"
    assert manifest["tag_policy"] == {
        "all_abis": "none",
        "all_platforms": "any",
        "required_interpreter": "py3",
    }
    assert all(item["pure_python"] is True for item in manifest["wheels"])
    lock = output_lock.read_text(encoding="utf-8")
    assert lock.count("--hash=sha256:") == 2


def test_r3_plan_is_hash_bound_and_replayable(plan: dict) -> None:
    assert plan["plan_sha256"] == lane.payload_sha256(plan, "plan_sha256")
    result = lane.verify_plan(plan, plan["plan_sha256"])
    assert result["valid"] is True
    assert result["status"] == "planned_build_ready"
    assert result["gpu_exposed"] is False


def test_r3_plan_pins_both_failed_attempts_and_runner_chain(plan: dict) -> None:
    assert plan["inputs"]["validator_base_v1"]["sha256"] == lane.V1_VALIDATOR_SHA256
    assert plan["inputs"]["validator_base_v2"]["sha256"] == lane.V2_VALIDATOR_SHA256
    assert (
        plan["inputs"]["failed_r2_attempt_receipt"]["sha256"]
        == lane.R2_FAILED_RECEIPT_FILE_SHA256
    )
    assert plan["inputs"]["failed_r2_build_log"]["sha256"] == lane.R2_FAILED_BUILD_LOG_SHA256
    assert [item["attempt_id"] for item in plan["repair_chain"]] == [
        "child-v1-r1-001",
        "child-v2-r2-001",
    ]
    assert plan["repair_chain"][1]["source_wheels_built"] is True


def test_r3_build_command_remains_offline_and_gpu_free(plan: dict) -> None:
    command = plan["commands"]["build"]
    assert "--network=none" in command
    assert "--pull=false" in command
    assert f"wheelbundle={lane.WHEELHOUSE}" in command
    assert f"childhelpers={lane.CHILD_HELPERS_R1}" in command
    assert "--gpus" not in command
    assert command[-1] == str(lane.CHILD_ROOT)


def test_r3_runtime_probe_remains_read_only_cpu_only(plan: dict) -> None:
    command = plan["commands"]["runtime_probe_template"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert "--gpus" not in command


def test_r3_contract_records_narrow_tag_policy_without_overclaim() -> None:
    contract = json.loads(lane.CONTRACT.read_text(encoding="utf-8"))
    assert contract["image_ref"].endswith(":child-v3")
    assert [item["attempt_id"] for item in contract["repair_chain"]] == [
        "child-v1-r1-001",
        "child-v2-r2-001",
    ]
    assert contract["build_policy"]["pure_wheel_tag_policy"] == {
        "all_abis": "none",
        "all_platforms": "any",
        "required_interpreter": "py3",
    }
    assert all(value is False for value in contract["observed"].values())


def test_r3_wrapper_and_helpers_parse_as_python38() -> None:
    for path in (
        Path(lane.__file__),
        lane.BUNDLE_VERIFY,
        lane.MAKE_LOCAL_LOCK,
        lane.ENVIRONMENT_PROBE,
    ):
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=(3, 8),
        )


def test_r3_plan_tamper_fails_against_external_pin(plan: dict) -> None:
    tampered = copy.deepcopy(plan)
    tampered["commands"]["build"].remove("--network=none")
    tampered["plan_sha256"] = lane.payload_sha256(tampered, "plan_sha256")
    with pytest.raises(lane.ChildImageError, match="external plan pin"):
        lane.verify_plan(tampered, plan["plan_sha256"])


def test_r3_source_validation_replays_both_failure_evidence_sets() -> None:
    result = lane._validate_sources()
    assert result["wheelhouse"]["valid"] is True
    assert result["failed_r1_attempt"]["build_exit_code"] == 1
    assert result["failed_r2_attempt"]["source_wheels_built"] is True
    assert result["failed_r2_attempt"]["observed_mmpose_tags"] == [
        "py2-none-any",
        "py3-none-any",
    ]
