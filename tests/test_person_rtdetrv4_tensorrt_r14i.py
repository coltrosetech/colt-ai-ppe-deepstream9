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

from validation import person_rtdetrv4_tensorrt_r14i as lane


STAMP = "2026-07-20T13:30:00Z"


@pytest.fixture(scope="session")
def contract() -> dict:
    return lane.build_controller_contract(prepared_at_utc=STAMP)


@pytest.fixture(scope="session")
def template() -> dict:
    with lane.AnchoredWorkspace(lane.ROOT) as workspace:
        return lane.verify_workload_template(workspace)


def _load(path: str) -> dict:
    return json.loads((lane.ROOT / path).read_text(encoding="utf-8"))


def _raw_pin(path: Path, relative: str, mode: str) -> dict:
    raw = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mode": mode,
    }


def _validate(value: dict, schema_path: str) -> None:
    schema = _load(schema_path)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)


def test_controller_schema_is_valid_and_closed(contract: dict) -> None:
    schema = _load(lane.CONTROLLER_SCHEMA_REL)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(contract)
    mutated = copy.deepcopy(contract)
    mutated["unexpected"] = True
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(mutated)


def test_template_schema_is_valid_and_closed(template: dict) -> None:
    schema = _load(lane.TEMPLATE_SCHEMA_REL)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(template)
    mutated = copy.deepcopy(template)
    mutated["profiles"]["640"]["docker"]["unexpected"] = True
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(mutated)


def test_controller_fingerprint_replays(contract: dict) -> None:
    assert contract["fingerprint_sha256"] == lane.fingerprint(
        contract, "fingerprint_sha256"
    )
    assert contract["authority"]["self_fingerprint_authoritative"] is False


def test_template_fingerprint_replays(template: dict) -> None:
    assert template["template_fingerprint_sha256"] == lane.TEMPLATE_FINGERPRINT
    assert template["template_fingerprint_sha256"] == lane.fingerprint(
        template, "template_fingerprint_sha256"
    )


@pytest.mark.parametrize("relative", sorted(lane.FROZEN_R14H))
def test_frozen_r14h_inventory_is_exact(relative: str) -> None:
    pin = lane.FROZEN_R14H[relative]
    assert _raw_pin(lane.ROOT / relative, relative, "0440") == pin
    assert (lane.ROOT / relative).stat().st_nlink == 1


def test_r14h_is_reused_without_new_outputs(contract: dict) -> None:
    assert contract["accepted_foundations"]["r14h"] == {
        "status": "frozen_accepted",
        "files": lane.FROZEN_R14H,
    }
    assert contract["authority"]["accepted_r14h_modified"] is False


def test_current_r2_independent_acceptance_is_exact(contract: dict) -> None:
    row = contract["accepted_foundations"]["deepstream91_cpu_preflight_r2"]
    assert row["accepted_receipt"] == lane.CONTROL_PINS["preflight_r2_acceptance"]
    assert row["decision"] == "ACCEPT"
    assert row["runtime_qualified"] is False
    assert row["accepted_scope"]["r2_cpu_only_preflight_definition"] is True
    assert row["accepted_scope"]["engine_build_qualified"] is False
    assert row["accepted_scope"]["production_ready"] is False


def test_r2_receipt_semantic_fingerprint_and_zero_findings() -> None:
    receipt = _load(lane.CONTROL_PINS["preflight_r2_acceptance"]["path"])
    assert receipt["severity_counts"] == {"P0": 0, "P1": 0, "P2": 0}
    assert receipt["self_fingerprint"]["canonical_sha256"] == lane.R2_REVIEW_FINGERPRINT
    assert lane.fingerprint(receipt, "self_fingerprint") == lane.R2_REVIEW_FINGERPRINT
    assert all(value is False for value in receipt["permissions"].values())


def test_r1c3_scope_is_narrow_and_image_is_exact(contract: dict) -> None:
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


def test_r1c3_release_plan_embeds_exact_r11_person_assets(contract: dict) -> None:
    row = contract["accepted_foundations"]["deepstream91_engine_builder_r1c3"]
    assert row["release_plan"] == lane.CONTROL_PINS["r1c3_release_plan"]
    assert row["release_plan_self_fingerprint"] == lane.R1C3_PLAN_FINGERPRINT
    assert row["embedded_person_assets"] == {
        "640": lane.MODEL_INPUT_PINS["onnx_640"],
        "960": lane.MODEL_INPUT_PINS["onnx_960"],
    }


def test_r1c3_candidate_does_not_overclaim_runtime() -> None:
    candidate = _load(lane.CONTROL_PINS["r1c3_candidate"]["path"])
    assert candidate["image"]["id"] == lane.IMAGE_ID
    assert candidate["image"]["manifest"] == lane.IMAGE_ID
    assert candidate["execution_boundary"] == {
        "deepstream_executed": False,
        "gpu_used": False,
        "gpu_visible": False,
        "inference_executed": False,
        "runtime": "runc",
        "tensorrt_executed": False,
    }


def test_gpu_lease_v5_full_policy_is_bound(contract: dict) -> None:
    lease = contract["accepted_foundations"]["gpu_lease_v5"]
    assert lease["contract_fingerprint_sha256"] == lane.LEASE_V5_FINGERPRINT
    assert lease["activation_policy"] == lane.EXPECTED_LEASE_ACTIVATION_POLICY
    assert lease["receipt_policy"] == lane.EXPECTED_LEASE_RECEIPT_POLICY
    assert lease["receipt_sequence"] == lane.EXPECTED_RECEIPT_SEQUENCE


def test_gpu_lease_v5_has_no_default_plan_override_or_receipt(contract: dict) -> None:
    lease = contract["accepted_foundations"]["gpu_lease_v5"]
    assert lease["status"] == "locked-no-live-plan"
    assert lease["activation_policy"]["plan_required"] is True
    assert lease["activation_policy"]["default_plan"] is False
    assert lease["activation_policy"]["caller_argument_overrides"] is False
    assert lease["live_plan_present"] is False
    assert lease["activation_receipt_present"] is False


def test_template_is_not_a_v5_plan_and_cannot_activate(template: dict) -> None:
    assert template["status"] == "execution_closed_non_authorizing_template"
    assert "execution_authorized" not in template
    assert template["authority"]["template_is_gpu_lease_v5_plan"] is False
    assert template["authority"]["template_may_activate_gpu_lease_v5"] is False
    assert template["lease_v5"]["live_plan_in_template"] is None
    assert template["lease_v5"]["activation_receipt_in_template"] is None


@pytest.mark.parametrize("profile", [640, 960])
def test_profile_shapes_are_exact_batch_1_12_12(template: dict, profile: int) -> None:
    shapes = template["profiles"][str(profile)]["optimization_profile"]
    assert shapes["images"] == {
        "min": [1, 3, profile, profile],
        "opt": [12, 3, profile, profile],
        "max": [12, 3, profile, profile],
    }
    assert shapes["orig_target_sizes"] == {
        "min": [1, 2],
        "opt": [12, 2],
        "max": [12, 2],
    }


@pytest.mark.parametrize("profile", [640, 960])
def test_profile_precision_is_fp16_only(template: dict, profile: int) -> None:
    assert template["profiles"][str(profile)]["precision"] == {
        "fp16": True,
        "int8": False,
        "tf32": False,
    }


@pytest.mark.parametrize("profile", [640, 960])
def test_exact_docker_argv_and_digest_replay(template: dict, profile: int) -> None:
    docker = template["profiles"][str(profile)]["docker"]
    assert docker["argv"] == lane.expected_docker_argv(profile)
    assert docker["argv_sha256"] == lane.command_argv_sha256(docker["argv"])
    assert docker["image_argv_index"] == 13
    assert docker["argv"][13] == lane.IMAGE_REFERENCE


@pytest.mark.parametrize("profile", [640, 960])
def test_docker_argv_has_offline_single_mount_no_mutable_tag(
    template: dict, profile: int
) -> None:
    row = template["profiles"][str(profile)]
    argv = row["docker"]["argv"]
    assert "--network=none" in argv
    assert "--pull=never" in argv
    assert f"--gpus=device={lane.GPU_UUID}" in argv
    assert "--rm" not in argv and "--detach" not in argv and "-d" not in argv
    assert not any(":latest" in item for item in argv)
    assert argv.count(lane.IMAGE_REFERENCE) == 1
    assert row["docker"]["environment"] == {}
    assert len(row["docker"]["mounts"]) == 1
    assert row["docker"]["mounts"][0]["destination"] == "/output"
    assert row["docker"]["caller_overrides"] is False


@pytest.mark.parametrize("profile", [640, 960])
def test_profile_onnx_is_exact_r11_asset(template: dict, profile: int) -> None:
    row = template["profiles"][str(profile)]
    assert row["onnx"]["repository"] == lane.MODEL_INPUT_PINS[f"onnx_{profile}"]
    assert row["onnx"]["image_path"] == (
        f"/opt/deepsafe/engine-builder/assets/person-{profile}.onnx"
    )


@pytest.mark.parametrize("profile", [640, 960])
def test_all_future_profile_outputs_are_named_and_absent(template: dict, profile: int) -> None:
    outputs = template["profiles"][str(profile)]["outputs"]
    assert set(outputs) == {
        "staging_engine", "final_engine", "engine_receipt", "config",
        "config_receipt", "numerical_parity_receipt",
        "deepstream_parser_parity_receipt", "v5_plan",
        "v5_activation_binding_receipt",
    }
    for output in outputs.values():
        assert output["required_absent_now"] is True
        assert output["published_pin"] is None
        assert not (lane.ROOT / output["path"]).exists()


def test_profiles_have_distinct_engine_config_parity_and_lease_paths(template: dict) -> None:
    left = template["profiles"]["640"]["outputs"]
    right = template["profiles"]["960"]["outputs"]
    assert set(item["path"] for item in left.values()).isdisjoint(
        item["path"] for item in right.values()
    )


def test_all_execution_gates_are_false_and_unpinned(contract: dict) -> None:
    gate = contract["execution_gate"]
    assert lane.EXECUTION_AUTHORIZED is False
    assert gate["authorized"] is False
    assert gate["all_satisfied"] is False
    assert all(
        row == {"satisfied": False, "accepted_pin": None}
        for row in gate["required_acceptances"].values()
    )
    assert "driver595_r7_host_qualification" in gate["required_acceptances"]
    assert "deepstream91_gpu_runtime_smoke" in gate["required_acceptances"]
    assert "r14i_controller_independent_acceptance" in gate["required_acceptances"]


def test_current_driver_r7_candidate_is_not_misrepresented_as_acceptance(contract: dict) -> None:
    gate = contract["execution_gate"]["required_acceptances"]
    assert gate["driver595_r7_host_qualification"] == {
        "satisfied": False,
        "accepted_pin": None,
    }
    source = (lane.ROOT / lane.THIS_REL).read_text(encoding="utf-8")
    assert "r7-live-candidate/candidate.json" not in source


def test_claim_boundary_is_truthful(contract: dict) -> None:
    claims = contract["claim_boundary"]
    assert claims["cpu_control_audit_completed"] is True
    assert claims["preflight_r2_cpu_definition_accepted"] is True
    assert claims["r1c3_prepared_closed_image_identity_accepted"] is True
    assert claims["lease_v5_locked_definition_accepted"] is True
    for name in (
        "driver595_r7_host_qualified", "deepstream91_gpu_runtime_qualified",
        "gpu_executed", "docker_executed", "model_loaded", "onnx_deserialized",
        "tensorrt_executed", "deepstream_executed", "engine_built",
        "config_published", "numerical_parity_completed", "parser_parity_completed",
        "twelve_camera_benchmark_completed", "production_ready",
    ):
        assert claims[name] is False


def test_r11_model_and_threshold_semantics_are_exact(contract: dict) -> None:
    inputs = contract["model_inputs"]
    assert inputs == lane.MODEL_INPUT_PINS
    threshold = _load(lane.MODEL_INPUT_PINS["threshold_receipt"]["path"])
    assert threshold["payload"]["profiles"]["640"]["threshold"] == 0.4116777181625366
    assert threshold["payload"]["profiles"]["960"]["threshold"] == 0.430930495262146
    assert threshold["payload"]["int8_calibration"] is False


def test_parser_provenance_is_exact_but_ds91_abi_remains_unaccepted(contract: dict) -> None:
    config = contract["config_contract"]
    assert config["parser_source_role"] == "provenance_only_not_runtime_mounted"
    assert config["future_v5_runtime_inputs_must_be_held"] is True
    assert config["parser_built_against"] == {
        "deepstream": "9.0.0",
        "cuda_headers": "13.1",
        "tensorrt_headers": "10.14.1.48",
    }
    assert config["target_runtime"] == {
        "deepstream": "9.1.0",
        "cuda": "13.2.0.046",
        "tensorrt": "10.16.0.72",
    }
    assert config["ds91_abi_compatibility_accepted"] is False
    assert all(row["published"] is False for row in config["profiles"].values())


@pytest.mark.parametrize(
    ("profile", "threshold"),
    [(640, 0.4116777181625366), (960, 0.430930495262146)],
)
def test_config_contract_binds_parser_outputs_and_calibrated_threshold(
    contract: dict, profile: int, threshold: float
) -> None:
    row = contract["config_contract"]["profiles"][str(profile)]
    assert row["batch_size"] == 12
    assert row["network_mode"] == 2
    assert row["cluster_mode"] == 4
    assert row["output_blob_names"] == ["labels", "boxes", "scores"]
    assert row["parse_bbox_func_name"] == "NvDsInferParseCustomRTDETRv4Person"
    assert row["initialize_input_layers_func_name"] == "NvDsInferInitializeInputLayers"
    assert row["custom_library"] == lane.MODEL_INPUT_PINS["parser_library"]
    assert row["labels"] == lane.MODEL_INPUT_PINS["labels_person"]
    assert row["pre_cluster_threshold"] == threshold


def test_command_digest_rejects_any_argv_mutation(template: dict) -> None:
    argv = list(template["profiles"]["640"]["docker"]["argv"])
    original = lane.command_argv_sha256(argv)
    argv[-1] = "--version"
    assert lane.command_argv_sha256(argv) != original


def test_strict_json_rejects_duplicate_keys_and_nonfinite() -> None:
    with pytest.raises(lane.PersonR14IError, match="duplicate JSON key"):
        lane.strict_json(b'{"x":1,"x":2}', label="duplicate")
    with pytest.raises(lane.PersonR14IError, match="non-finite"):
        lane.strict_json(b'{"x":NaN}', label="nan")


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
    (outside / "value.json").write_bytes(b"{}")
    (outside / "value.json").chmod(0o440)
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    pin = {
        "path": "redirect/value.json",
        "bytes": 2,
        "sha256": hashlib.sha256(b"{}").hexdigest(),
        "mode": "0440",
    }
    with lane.AnchoredWorkspace(root) as workspace:
        with pytest.raises(OSError):
            workspace.read_exact(pin, label="ancestor symlink")


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
        with pytest.raises(lane.PersonR14IError, match="identity differs"):
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
        with pytest.raises(lane.PersonR14IError, match="exact pin differs"):
            workspace.read_exact(pin, label="substitution")


def test_template_rejects_any_preexisting_output(monkeypatch: pytest.MonkeyPatch) -> None:
    with lane.AnchoredWorkspace(lane.ROOT) as workspace:
        monkeypatch.setattr(workspace, "exists_no_follow", lambda _path: True)
        with pytest.raises(lane.PersonR14IError, match="future output already exists"):
            lane.verify_workload_template(workspace)


def test_execution_entrypoint_fails_before_any_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(lane, "EXECUTION_AUTHORIZED", False)
    monkeypatch.setattr(os, "system", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(lane.PersonR14IError, match="execution gate is closed"):
        lane.require_execution_gate()
    assert calls == []


def test_source_has_no_execution_or_model_runtime_imports() -> None:
    source = (lane.ROOT / lane.THIS_REL).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not imported & {
        "subprocess", "docker", "torch", "onnx", "onnxruntime", "tensorrt",
        "pyds", "cv2", "numpy",
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
    assert json.loads(path.read_text(encoding="utf-8")) == contract


def test_author_handoff_remains_nonterminal() -> None:
    path = lane.ROOT / lane.HANDOFF_REL
    assert path.is_file()
    handoff = json.loads(path.read_text(encoding="utf-8"))
    assert handoff["status"] == "author_handoff_pending_independent_review"
    assert handoff["decision"] is None
    assert handoff["terminal_accepted"] is False
    assert handoff["execution_authorized"] is False
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
