from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from validation import pose_mmpose_yoloxpose_tensorrt_ds91_r13i as lane


STAMP = "2026-07-20T16:45:00Z"


@pytest.fixture(scope="session")
def contract() -> dict:
    return lane.build_controller_contract(prepared_at_utc=STAMP)


@pytest.fixture(scope="session")
def template() -> dict:
    with lane.AnchoredWorkspace(lane.ROOT) as workspace:
        return lane.verify_workload_template(workspace)


def _load(relative: str) -> dict:
    return json.loads((lane.ROOT / relative).read_text(encoding="utf-8"))


def _raw_pin(path: Path, relative: str, mode: str) -> dict:
    raw = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mode": mode,
    }


def _schema(relative: str) -> dict:
    value = _load(relative)
    Draft202012Validator.check_schema(value)
    return value


def test_controller_schema_is_valid_exact_and_closed(contract: dict) -> None:
    schema = _schema(lane.CONTROLLER_SCHEMA_REL)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(contract)
    mutated = copy.deepcopy(contract)
    mutated["claim_boundary"]["gpu_executed"] = True
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(mutated)
    extra = copy.deepcopy(contract)
    extra["unexpected"] = True
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(extra)


def test_controller_schema_allows_valid_timestamp_with_replayed_fingerprint(
    contract: dict,
) -> None:
    schema = _schema(lane.CONTROLLER_SCHEMA_REL)
    changed = copy.deepcopy(contract)
    changed["prepared_at_utc"] = "2026-07-20T16:46:00Z"
    changed["fingerprint_sha256"] = lane.fingerprint(changed, "fingerprint_sha256")
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(changed)


def test_template_schema_is_valid_exact_and_non_authorizing(template: dict) -> None:
    schema = _schema(lane.TEMPLATE_SCHEMA_REL)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(template)
    mutated = copy.deepcopy(template)
    mutated["execution_gate"]["authorized"] = True
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(mutated)


def test_controller_and_template_fingerprints_replay(
    contract: dict, template: dict
) -> None:
    assert contract["fingerprint_sha256"] == lane.fingerprint(
        contract, "fingerprint_sha256"
    )
    assert contract["authority"]["self_fingerprint_authoritative"] is False
    assert template["template_fingerprint_sha256"] == lane.TEMPLATE_FINGERPRINT
    assert template["template_fingerprint_sha256"] == lane.fingerprint(
        template, "template_fingerprint_sha256"
    )


@pytest.mark.parametrize("relative", sorted(lane.FROZEN_R13H))
def test_frozen_r13h_inventory_is_exact(relative: str) -> None:
    pin = lane.FROZEN_R13H[relative]
    path = lane.ROOT / relative
    assert _raw_pin(path, relative, "0440") == pin
    assert path.stat().st_nlink == 1


def test_r13h_is_reused_without_modification(contract: dict) -> None:
    assert contract["accepted_foundations"]["r13h"] == {
        "status": "frozen_accepted",
        "files": lane.FROZEN_R13H,
    }
    assert contract["authority"]["accepted_r13h_modified"] is False


def test_r2_independent_acceptance_is_narrow_and_exact(contract: dict) -> None:
    row = contract["accepted_foundations"]["deepstream91_cpu_preflight_r2"]
    assert row["accepted_receipt"] == lane.CONTROL_PINS["preflight_r2_acceptance"]
    assert row["decision"] == "ACCEPT"
    assert row["runtime_qualified"] is False
    assert row["accepted_scope"]["r2_cpu_only_preflight_definition"] is True
    assert row["accepted_scope"]["engine_build_qualified"] is False
    assert row["accepted_scope"]["production_ready"] is False


def test_r2_receipt_fingerprint_and_resource_boundary() -> None:
    receipt = _load(lane.CONTROL_PINS["preflight_r2_acceptance"]["path"])
    assert receipt["severity_counts"] == {"P0": 0, "P1": 0, "P2": 0}
    assert receipt["self_fingerprint"]["canonical_sha256"] == lane.R2_REVIEW_FINGERPRINT
    assert lane.fingerprint(receipt, "self_fingerprint") == lane.R2_REVIEW_FINGERPRINT
    assert all(value is False for value in receipt["permissions"].values())
    assert all(
        value is False
        for value in receipt["verification"]["resource_boundary"].values()
    )


def test_r1c3_scope_is_prepared_closed_only(contract: dict) -> None:
    row = contract["accepted_foundations"]["deepstream91_engine_builder_r1c3"]
    assert row["accepted_scope"] == lane.EXPECTED_BUILDER_SCOPE
    assert row["qualification"] == lane.EXPECTED_BUILDER_QUALIFICATION
    assert row["image"] == {
        "id": lane.IMAGE_ID,
        "manifest": lane.IMAGE_ID,
        "config": lane.IMAGE_CONFIG,
    }
    assert row["accepted_scope"]["prepared_closed_engine_builder_image_identity"] is True
    assert row["accepted_scope"]["gpu_runtime"] is False
    assert row["accepted_scope"]["tensorrt_engine_build"] is False


def test_r1c3_release_plan_embeds_exact_pose_onnx(contract: dict) -> None:
    row = contract["accepted_foundations"]["deepstream91_engine_builder_r1c3"]
    assert row["release_plan"] == lane.CONTROL_PINS["r1c3_release_plan"]
    assert row["release_plan_self_fingerprint"] == lane.R1C3_PLAN_FINGERPRINT
    assert row["embedded_pose_assets"] == {
        "640": lane.MODEL_INPUT_PINS["onnx_640"],
        "960": lane.MODEL_INPUT_PINS["onnx_960"],
    }


def test_gpu_lease_v5_full_locked_policy_is_bound(contract: dict) -> None:
    lease = contract["accepted_foundations"]["gpu_lease_v5"]
    assert lease["contract_fingerprint_sha256"] == lane.LEASE_V5_FINGERPRINT
    assert lease["status"] == "locked-no-live-plan"
    assert lease["activation_policy"] == lane.EXPECTED_LEASE_ACTIVATION_POLICY
    assert lease["receipt_policy"] == lane.EXPECTED_LEASE_RECEIPT_POLICY
    assert lease["receipt_sequence"] == lane.EXPECTED_RECEIPT_SEQUENCE
    assert lease["activation_policy"]["default_plan"] is False
    assert lease["activation_policy"]["caller_argument_overrides"] is False
    assert lease["live_plan_present"] is False
    assert lease["activation_receipt_present"] is False


def test_template_is_not_a_v5_plan(template: dict) -> None:
    assert template["status"] == "execution_closed_non_authorizing_template"
    assert "execution_authorized" not in template
    assert template["authority"]["template_is_gpu_lease_v5_plan"] is False
    assert template["authority"]["template_may_activate_gpu_lease_v5"] is False
    assert template["lease_v5"]["live_plan_in_template"] is None
    assert template["lease_v5"]["activation_receipt_in_template"] is None


def test_template_binds_pose_parser_config_threshold_and_keypoint_inputs(
    template: dict,
) -> None:
    row = template["pose_contract_inputs"]
    for name in (
        "r13b_plan", "adapter_r2_receipt", "parser_handoff",
        "adapter_contract", "bridge_contract", "bridge_header", "bridge_source",
        "packer_header", "packer_source", "decoder_header", "decoder_source",
        "postprocess_contract",
    ):
        assert row[name] == lane.MODEL_INPUT_PINS[name]
    assert row["source_configs"] == {
        "640": lane.MODEL_INPUT_PINS["config_640"],
        "960": lane.MODEL_INPUT_PINS["config_960"],
    }
    assert row["ds91_abi_compatibility_accepted"] is False
    assert row["decoder_thresholds"] == {
        "detection": 0.25,
        "keypoint_visibility": 0.25,
        "detection_inclusive": True,
        "keypoint_visibility_inclusive": True,
    }
    assert row["keypoint_contract"]["schema"] == "COCO17"
    assert row["keypoint_contract"]["count"] == 17


@pytest.mark.parametrize("profile", [640, 960])
def test_profile_shapes_are_batch_1_12_12(template: dict, profile: int) -> None:
    assert template["profiles"][str(profile)]["optimization_profile"] == {
        "input": {
            "min": [1, 3, profile, profile],
            "opt": [12, 3, profile, profile],
            "max": [12, 3, profile, profile],
        }
    }


@pytest.mark.parametrize("profile", [640, 960])
def test_profile_output_and_keypoint_contract(template: dict, profile: int) -> None:
    assert template["profiles"][str(profile)]["output_contract"] == {
        "dets": {"shape": ["B", 100, 5], "dtypes": ["FLOAT", "HALF"]},
        "keypoints": {
            "shape": ["B", 100, 17, 3],
            "dtypes": ["FLOAT", "HALF"],
            "keypoint_count": 17,
            "layout": ["x", "y", "confidence"],
        },
        "same_index_association": True,
    }


@pytest.mark.parametrize("profile", [640, 960])
def test_profile_precision_is_fp16_only(template: dict, profile: int) -> None:
    assert template["profiles"][str(profile)]["precision"] == {
        "fp16": True,
        "int8": False,
        "tf32": False,
    }


@pytest.mark.parametrize("profile", [640, 960])
def test_exact_docker_argv_and_digest(template: dict, profile: int) -> None:
    docker = template["profiles"][str(profile)]["docker"]
    assert docker["argv"] == lane.expected_docker_argv(profile)
    assert docker["argv_sha256"] == lane.command_argv_sha256(docker["argv"])
    assert docker["image_argv_index"] == 13
    assert docker["argv"][13] == lane.IMAGE_REFERENCE


@pytest.mark.parametrize("profile", [640, 960])
def test_docker_argv_is_offline_single_mount_and_tag_free(
    template: dict, profile: int
) -> None:
    row = template["profiles"][str(profile)]
    argv = row["docker"]["argv"]
    assert argv[:2] == ["/usr/bin/docker", "run"]
    assert "--network=none" in argv and "--pull=never" in argv
    assert f"--gpus=device={lane.GPU_UUID}" in argv
    assert "--rm" not in argv and "--detach" not in argv and "-d" not in argv
    assert not any(":latest" in item for item in argv)
    assert not any("com.deepsafe.gpu-lease.plan.v5" in item for item in argv)
    assert row["docker"]["environment"] == {}
    assert len(row["docker"]["mounts"]) == 1
    assert row["docker"]["mounts"][0]["destination"] == "/output"
    assert row["docker"]["caller_overrides"] is False


@pytest.mark.parametrize("profile", [640, 960])
def test_profile_onnx_is_exact_r12c_asset(template: dict, profile: int) -> None:
    row = template["profiles"][str(profile)]
    assert row["onnx"]["repository"] == lane.MODEL_INPUT_PINS[f"onnx_{profile}"]
    assert row["onnx"]["image_path"] == (
        f"/opt/deepsafe/engine-builder/assets/pose-{profile}.onnx"
    )


@pytest.mark.parametrize("profile", [640, 960])
def test_all_future_outputs_are_named_absent_and_unpinned(
    template: dict, profile: int
) -> None:
    outputs = template["profiles"][str(profile)]["outputs"]
    assert set(outputs) == {
        "staging_engine",
        "final_engine",
        "engine_receipt",
        "config",
        "config_receipt",
        "numerical_parity_receipt",
        "deepstream_tensor_meta_parity_receipt",
        "keypoint_contract_receipt",
        "v5_plan",
        "v5_activation_binding_receipt",
    }
    for output in outputs.values():
        assert output["required_absent_now"] is True
        assert output["published_pin"] is None
        assert not (lane.ROOT / output["path"]).exists()


def test_profile_output_paths_are_disjoint(template: dict) -> None:
    left = {item["path"] for item in template["profiles"]["640"]["outputs"].values()}
    right = {item["path"] for item in template["profiles"]["960"]["outputs"].values()}
    assert left.isdisjoint(right)


def test_all_execution_gates_are_false_and_unpinned(contract: dict) -> None:
    gate = contract["execution_gate"]
    assert lane.EXECUTION_AUTHORIZED is False
    assert gate["authorized"] is False
    assert gate["all_satisfied"] is False
    assert all(
        row == {"satisfied": False, "accepted_pin": None}
        for row in gate["required_acceptances"].values()
    )
    assert set(gate["required_acceptances"]) == {
        "driver595_r7_host_qualification",
        "deepstream91_gpu_runtime_smoke",
        "r13i_controller_independent_acceptance",
        "profile_640_fresh_v5_plan",
        "profile_640_fresh_v5_activation_receipt",
        "profile_960_fresh_v5_plan",
        "profile_960_fresh_v5_activation_receipt",
    }


def test_driver_r7_candidate_is_not_treated_as_acceptance(contract: dict) -> None:
    gate = contract["execution_gate"]["required_acceptances"]
    assert gate["driver595_r7_host_qualification"] == {
        "satisfied": False,
        "accepted_pin": None,
    }
    source = (lane.ROOT / lane.THIS_REL).read_text(encoding="utf-8")
    assert "r7-live-candidate/candidate.json" not in source


def test_claim_boundary_is_truthful(contract: dict) -> None:
    claims = contract["claim_boundary"]
    for name in (
        "cpu_control_audit_completed",
        "r13h_lineage_reused_not_modified",
        "preflight_r2_cpu_definition_accepted",
        "r1c3_prepared_closed_image_identity_accepted",
        "lease_v5_locked_definition_accepted",
    ):
        assert claims[name] is True
    for name in (
        "driver595_r7_host_qualified",
        "deepstream91_gpu_runtime_qualified",
        "gpu_executed",
        "docker_executed",
        "model_loaded",
        "onnx_deserialized",
        "tensorrt_executed",
        "deepstream_executed",
        "engine_built",
        "config_published",
        "numerical_parity_completed",
        "deepstream_tensor_meta_parity_completed",
        "keypoint_contract_completed",
        "twelve_camera_benchmark_completed",
        "production_ready",
    ):
        assert claims[name] is False


@pytest.mark.parametrize("name", sorted(lane.MODEL_INPUT_PINS))
def test_every_pose_input_pin_replays_exactly(name: str) -> None:
    pin = lane.MODEL_INPUT_PINS[name]
    path = lane.ROOT / pin["path"]
    assert _raw_pin(path, pin["path"], pin["mode"]) == pin
    assert path.stat().st_nlink == 1


def test_r13b_model_threshold_and_keypoint_semantics(contract: dict) -> None:
    assert contract["model_inputs"] == lane.MODEL_INPUT_PINS
    plan = _load(lane.MODEL_INPUT_PINS["r13b_plan"]["path"])
    assert plan["fingerprint_sha256"] == (
        "b74f377309bf255fc311225cf713e9594ab7b05cd1b94b809643fdb758cb5c7f"
    )
    assert plan["model_contract"]["batch"] == {
        "minimum": 1,
        "optimum": 12,
        "maximum": 12,
    }
    assert plan["model_contract"]["outputs"][1] == {
        "dtype": ["FLOAT", "HALF"],
        "name": "keypoints",
        "shape": ["B", 100, 17, 3],
    }
    thresholds = plan["numerical_parity_contract"]
    assert thresholds["detection_score_max_abs_lte"] == 0.005
    assert thresholds["keypoint_confidence_max_abs_lte"] == 0.005
    assert thresholds["keypoint_xy_max_abs_px_lte"] == {"640": 2.0, "960": 3.0}


def test_r12c_accepted_onnx_receipts_are_bound() -> None:
    run = _load(lane.MODEL_INPUT_PINS["r12c_run_receipt"]["path"])
    assert run["status"] == "passed"
    assert run["receipt_fingerprint_sha256"] == (
        "a70ff1e7c4fa2b3bb176c5a553d52e7f8667adcbb90da5b157587dbc23e506ef"
    )
    assert run["execution_boundary"]["gpu_exposed"] is False
    assert run["execution_boundary"]["tensorrt_executed"] is False


def test_ds90_parser_adapter_is_provenance_only_and_ds91_false(contract: dict) -> None:
    config = contract["config_contract"]
    assert config["parser_source_role"] == (
        "ds9_0_source_and_cpu_abi_provenance_only_not_runtime_mounted"
    )
    assert config["future_v5_runtime_inputs_must_be_held"] is True
    assert config["parser_built_against"] == {
        "deepstream": "9.0.0",
        "artifact": "source_and_cpu_abi_receipt_no_published_shared_library",
    }
    assert config["target_runtime"] == {
        "deepstream": "9.1.0",
        "cuda": "13.2.0.046",
        "tensorrt": "10.16.0.72",
    }
    assert config["ds91_abi_compatibility_accepted"] is False


@pytest.mark.parametrize("profile", [640, 960])
def test_config_contract_binds_exact_source_and_tensor_meta(
    contract: dict, profile: int
) -> None:
    row = contract["config_contract"]["profiles"][str(profile)]
    assert row["batch_size"] == 12
    assert row["network_mode"] == 2
    assert row["network_type"] == 100
    assert row["gie_unique_id"] == 13
    assert row["infer_dims"] == [3, profile, profile]
    assert row["output_blob_names"] == ["dets", "keypoints"]
    assert row["output_tensor_meta"] == 1
    assert row["source_config"] == lane.MODEL_INPUT_PINS[f"config_{profile}"]
    assert row["bridge_contract"] == lane.MODEL_INPUT_PINS["bridge_contract"]
    assert row["adapter_contract"] == lane.MODEL_INPUT_PINS["adapter_contract"]
    assert row["published"] is False


def test_config_threshold_and_keypoint_projection(contract: dict) -> None:
    config = contract["config_contract"]
    assert config["numerical_parity_thresholds"] == {
        "detection_score_max_abs_lte": 0.005,
        "keypoint_confidence_max_abs_lte": 0.005,
        "bbox_max_abs_px_lte": {"640": 2.0, "960": 3.0},
        "keypoint_xy_max_abs_px_lte": {"640": 2.0, "960": 3.0},
    }
    assert config["keypoint_contract"] == {
        "count": 17,
        "source_shape": ["B", 100, 17, 3],
        "canonical_shape": ["B", 300, 57],
        "same_index_association_required": True,
        "second_nms": False,
    }
    assert config["decoder_thresholds"] == {
        "detection": 0.25,
        "keypoint_visibility": 0.25,
        "detection_inclusive": True,
        "keypoint_visibility_inclusive": True,
        "source": lane.MODEL_INPUT_PINS["postprocess_contract"],
    }


def test_command_digest_rejects_argv_mutation(template: dict) -> None:
    argv = list(template["profiles"]["640"]["docker"]["argv"])
    expected = lane.command_argv_sha256(argv)
    argv[-1] = "--version"
    assert lane.command_argv_sha256(argv) != expected


def test_controller_schema_rejects_accepted_foundation_mutation(contract: dict) -> None:
    schema = _schema(lane.CONTROLLER_SCHEMA_REL)
    forged = copy.deepcopy(contract)
    forged["accepted_foundations"]["deepstream91_engine_builder_r1c3"][
        "accepted_scope"
    ]["gpu_runtime"] = True
    forged["fingerprint_sha256"] = lane.fingerprint(forged, "fingerprint_sha256")
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(forged)


def test_strict_json_rejects_duplicate_key_nonfinite_and_bom() -> None:
    with pytest.raises(lane.PoseR13IError, match="duplicate JSON key"):
        lane.strict_json(b'{"x":1,"x":2}', label="duplicate")
    with pytest.raises(lane.PoseR13IError, match="non-finite"):
        lane.strict_json(b'{"x":NaN}', label="nan")
    with pytest.raises(lane.PoseR13IError, match="BOM rejected"):
        lane.strict_json(b'\xef\xbb\xbf{}', label="bom")


def test_anchored_reader_rejects_final_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target.json"
    target.write_bytes(b"{}")
    target.chmod(0o440)
    (root / "link.json").symlink_to("target.json")
    pin = {
        "path": "link.json",
        "bytes": 2,
        "sha256": hashlib.sha256(b"{}").hexdigest(),
        "mode": "0440",
    }
    with lane.AnchoredWorkspace(root) as workspace:
        with pytest.raises(OSError):
            workspace.read_exact(pin, label="symlink")


def test_anchored_reader_rejects_ancestor_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    value = outside / "value.json"
    value.write_bytes(b"{}")
    value.chmod(0o440)
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    pin = {
        "path": "redirect/value.json",
        "bytes": 2,
        "sha256": hashlib.sha256(b"{}").hexdigest(),
        "mode": "0440",
    }
    with lane.AnchoredWorkspace(root) as workspace:
        with pytest.raises(OSError):
            workspace.read_exact(pin, label="ancestor")


def test_anchored_reader_rejects_hardlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    value = root / "value.json"
    value.write_bytes(b"{}")
    value.chmod(0o440)
    os.link(value, root / "alias.json")
    pin = {
        "path": "value.json",
        "bytes": 2,
        "sha256": hashlib.sha256(b"{}").hexdigest(),
        "mode": "0440",
    }
    with lane.AnchoredWorkspace(root) as workspace:
        with pytest.raises(lane.PoseR13IError, match="identity differs"):
            workspace.read_exact(pin, label="hardlink")


def test_exact_pin_rejects_same_length_substitution(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    value = root / "value.json"
    value.write_bytes(b"ab")
    value.chmod(0o440)
    pin = {
        "path": "value.json",
        "bytes": 2,
        "sha256": hashlib.sha256(b"aa").hexdigest(),
        "mode": "0440",
    }
    with lane.AnchoredWorkspace(root) as workspace:
        with pytest.raises(lane.PoseR13IError, match="exact pin differs"):
            workspace.read_exact(pin, label="substitution")


def test_wrong_mode_and_wrong_owner_policy_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    value = root / "value.json"
    value.write_bytes(b"{}")
    value.chmod(0o644)
    pin = {
        "path": "value.json",
        "bytes": 2,
        "sha256": hashlib.sha256(b"{}").hexdigest(),
        "mode": "0440",
    }
    with lane.AnchoredWorkspace(root) as workspace:
        with pytest.raises(lane.PoseR13IError, match="mode differs"):
            workspace.read_exact(pin, label="mode")


def test_template_rejects_any_preexisting_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with lane.AnchoredWorkspace(lane.ROOT) as workspace:
        monkeypatch.setattr(workspace, "exists_no_follow", lambda _path: True)
        with pytest.raises(lane.PoseR13IError, match="future output already exists"):
            lane.verify_workload_template(workspace)


def test_any_compile_time_gate_mutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "LIVE_V5_PLAN_640_ACCEPTED", True)
    with pytest.raises(lane.PoseR13IError, match="compile-time gate"):
        lane._execution_gate()


def test_execution_entrypoint_fails_before_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(lane, "EXECUTION_AUTHORIZED", False)
    monkeypatch.setattr(os, "system", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(lane.PoseR13IError, match="execution gate is closed"):
        lane.require_execution_gate()
    assert calls == []


def test_source_has_no_execution_or_model_runtime_imports() -> None:
    source = (lane.ROOT / lane.THIS_REL).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert not imported_roots & {
        "subprocess",
        "docker",
        "torch",
        "onnx",
        "onnxruntime",
        "tensorrt",
        "pyds",
        "cv2",
        "numpy",
    }
    assert "nvidia-smi" not in source
    assert "Popen(" not in source
    assert "subprocess." not in source


def test_module_and_direct_cli_work_from_clean_cwd(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": str(lane.ROOT),
        "CUDA_VISIBLE_DEVICES": "-1",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    commands = [
        [str(lane.ROOT / ".venv/bin/python"), "-m", lane.__name__, "show-template"],
        [str(lane.ROOT / ".venv/bin/python"), str(lane.ROOT / lane.THIS_REL), "show-template"],
    ]
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["status"] == (
            "execution_closed_non_authorizing_template"
        )


def test_execute_cli_is_closed_and_publishes_nothing(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": str(lane.ROOT),
        "CUDA_VISIBLE_DEVICES": "-1",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [str(lane.ROOT / ".venv/bin/python"), "-m", lane.__name__, "execute"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 2
    assert "execution gate is closed" in completed.stderr
    with lane.AnchoredWorkspace(lane.ROOT) as workspace:
        current = lane.verify_workload_template(workspace)
    assert all(
        not (lane.ROOT / item["path"]).exists()
        for profile in current["profiles"].values()
        for item in profile["outputs"].values()
    )


def test_static_controller_matches_live_projection(contract: dict) -> None:
    path = lane.ROOT / lane.DEFAULT_CONTROLLER_REL
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o440
    assert path.stat().st_nlink == 1
    assert json.loads(path.read_text(encoding="utf-8")) == contract


def test_author_handoff_is_immutable_nonterminal_and_replays_subject() -> None:
    path = lane.ROOT / lane.HANDOFF_REL
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o440
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o550
    handoff = json.loads(path.read_text(encoding="utf-8"))
    assert handoff["status"] == "author_handoff_pending_independent_review"
    assert handoff["decision"] is None
    assert handoff["terminal_accepted"] is False
    assert handoff["execution_authorized"] is False
    body = {key: value for key, value in handoff.items() if key != "self_fingerprint"}
    assert hashlib.sha256(lane.canonical_bytes(body)).hexdigest() == (
        handoff["self_fingerprint"]["canonical_sha256"]
    )
    for pin in handoff["subject"]["artifacts"].values():
        assert _raw_pin(lane.ROOT / pin["path"], pin["path"], pin["mode"]) == pin
    assert handoff["resource_boundary"] == {
        "docker": False,
        "gpu": False,
        "model_load": False,
        "onnx_deserialization": False,
        "tensorrt": False,
        "deepstream": False,
        "network": False,
        "sudo": False,
        "install": False,
        "reboot": False,
    }
