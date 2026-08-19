from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from validation import deepstream91_full_stack_preflight_r2 as subject


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace() -> subject.AnchoredWorkspace:
    with subject.AnchoredWorkspace(ROOT) as held:
        yield held


def _plan(workspace: subject.AnchoredWorkspace) -> dict:
    return subject.load_default_plan(workspace)


def _resign(plan: dict) -> dict:
    result = copy.deepcopy(plan)
    result.pop("self_fingerprint", None)
    result["self_fingerprint"] = hashlib.sha256(
        subject.canonical_bytes(result)
    ).hexdigest()
    return result


def _load_control(
    workspace: subject.AnchoredWorkspace, name: str, label: str
) -> dict:
    return subject.strict_object(
        workspace.replay_exact(subject.EXTERNAL_CONTROL_PINS[name], label), label
    )


def test_checked_in_candidate_passes_but_never_claims_acceptance(
    workspace: subject.AnchoredWorkspace,
) -> None:
    report = subject.validate_plan(
        _plan(workspace), workspace=workspace, replay_media=False
    )
    assert report["status"] == "author_candidate_pass_blocked_as_designed"
    assert report["subject_accepted"] is False
    assert report["independent_acceptance_required"] is True
    assert report["execution_ready"] is False
    assert report["gpu_or_docker_called"] is False


def test_all_twelve_media_exact_pins_replay(
    workspace: subject.AnchoredWorkspace,
) -> None:
    report = subject.validate_plan(
        _plan(workspace), workspace=workspace, replay_media=True
    )
    assert report["source_count"] == 12
    assert report["distinct_video_types"] == 12
    assert report["profiles"] == [640, 960]
    assert report["simulated_streams"] == 12
    assert report["measurement_seconds_per_profile"] == 300


def test_source_manifest_is_structurally_source_only(
    workspace: subject.AnchoredWorkspace,
) -> None:
    manifest = _load_control(workspace, "source_manifest", "source manifest")
    assert set(manifest) == {"schema_version", "manifest_id", "sources"}
    assert len(manifest["sources"]) == 12
    for row in manifest["sources"]:
        assert set(row) == {"source_id", "video_type", "view_types", "media_pin"}
        assert set(row["media_pin"]) == {"path", "bytes", "sha256"}


def test_source_projection_has_required_camera_views(
    workspace: subject.AnchoredWorkspace,
) -> None:
    manifest = _load_control(workspace, "source_manifest", "source manifest")
    views = {view for row in manifest["sources"] for view in row["view_types"]}
    assert {"medium_close", "overhead_security_camera"}.issubset(views)
    assert len({row["video_type"] for row in manifest["sources"]}) == 12


def test_r2_source_and_import_graph_has_no_r1_legacy_runtime_dependency() -> None:
    source = (ROOT / "validation/deepstream91_full_stack_preflight_r2.py").read_text()
    assert "full_stack_benchmark_contract" not in source
    assert "deepstream-full-stack-benchmark-v1.json" not in source
    assert "deepstream-full-stack-benchmark-plan-v1.schema.json" not in source
    assert "from validation import" not in source
    assert "legacy_ds90_plan_validator_schema_consumed" in source


@pytest.mark.parametrize(
    "schema_name",
    ["plan_schema", "source_schema", "builder_schema", "lease_schema"],
)
def test_every_r2_object_schema_is_closed(
    workspace: subject.AnchoredWorkspace, schema_name: str
) -> None:
    schema = _load_control(workspace, schema_name, schema_name)
    subject._assert_schema_closed(schema, schema_name)


def test_plan_self_fingerprint_is_explicitly_non_authoritative(
    workspace: subject.AnchoredWorkspace,
) -> None:
    plan = _plan(workspace)
    assert plan["authority"]["self_fingerprint_authoritative"] is False
    assert plan["authority"]["plan_embedded_pins_authoritative"] is False
    assert plan["authority"]["external_constants_required"] is True
    assert subject._plan_fingerprint_valid(plan)


def test_resigned_source_manifest_substitution_is_rejected(
    workspace: subject.AnchoredWorkspace,
) -> None:
    plan = _plan(workspace)
    plan["source_matrix"]["manifest"] = {
        "path": "validation/work/fabricated-source.json",
        "bytes": 1,
        "sha256": "0" * 64,
        "mode": "0440",
    }
    with pytest.raises(subject.DS91PreflightR2Error, match="external authority pin"):
        subject.validate_plan(_resign(plan), workspace=workspace, replay_media=False)


def test_resigned_builder_root_substitution_is_rejected(
    workspace: subject.AnchoredWorkspace,
) -> None:
    plan = _plan(workspace)
    plan["runtime_target"]["prepared_closed_engine_builder_root"]["sha256"] = "0" * 64
    with pytest.raises(subject.DS91PreflightR2Error, match="external authority pin"):
        subject.validate_plan(_resign(plan), workspace=workspace, replay_media=False)


def test_resigned_lease_contract_substitution_is_rejected(
    workspace: subject.AnchoredWorkspace,
) -> None:
    plan = _plan(workspace)
    plan["gpu_lease"]["contract"]["sha256"] = "0" * 64
    with pytest.raises(subject.DS91PreflightR2Error, match="external authority pin"):
        subject.validate_plan(_resign(plan), workspace=workspace, replay_media=False)


def test_plan_additional_property_is_rejected(
    workspace: subject.AnchoredWorkspace,
) -> None:
    plan = _plan(workspace)
    plan["nested_runtime_override"] = {"execution_ready": True}
    with pytest.raises(subject.DS91PreflightR2Error, match="schema validation failed"):
        subject.validate_plan(_resign(plan), workspace=workspace, replay_media=False)


def test_plan_authorization_smuggling_is_rejected(
    workspace: subject.AnchoredWorkspace,
) -> None:
    plan = _plan(workspace)
    plan["authorization"]["gpu_workload"] = True
    with pytest.raises(subject.DS91PreflightR2Error, match="schema validation failed"):
        subject.validate_plan(_resign(plan), workspace=workspace, replay_media=False)


def test_builder_schema_rejects_scope_smuggling(
    workspace: subject.AnchoredWorkspace,
) -> None:
    builder = _load_control(workspace, "builder_root", "builder")
    schema = _load_control(workspace, "builder_schema", "builder schema")
    builder["accepted_scope"]["deepstream_runtime"] = True
    builder["accepted_scope"]["hidden"] = {"inference": True}
    with pytest.raises(subject.DS91PreflightR2Error, match="schema validation failed"):
        subject.validate_schema(builder, schema, "fabricated builder")


def test_builder_full_false_scope_and_qualification_are_exact(
    workspace: subject.AnchoredWorkspace,
) -> None:
    plan = _plan(workspace)
    subject._validate_builder(plan, workspace)
    builder = _load_control(workspace, "builder_root", "builder")
    assert builder["accepted_scope"] == subject.EXPECTED_BUILDER_SCOPE
    assert builder["qualification"] == subject.EXPECTED_BUILDER_QUALIFICATION
    assert builder["accepted_scope"]["deepstream_runtime"] is False
    assert builder["accepted_scope"]["inference"] is False


def test_builder_image_is_bound_through_exact_candidate(
    workspace: subject.AnchoredWorkspace,
) -> None:
    plan = _plan(workspace)
    candidate = _load_control(workspace, "builder_candidate", "builder candidate")
    assert candidate["image"] == subject.EXPECTED_IMAGE
    assert plan["runtime_target"]["prepared_closed_image"] == candidate["image"]
    assert candidate["execution_boundary"]["gpu_used"] is False
    assert candidate["execution_boundary"]["inference_executed"] is False


def test_lease_schema_rejects_plan_and_receipt_smuggling(
    workspace: subject.AnchoredWorkspace,
) -> None:
    lease = _load_control(workspace, "lease_contract", "lease")
    schema = _load_control(workspace, "lease_schema", "lease schema")
    lease["activation_policy"]["default_plan"] = True
    lease["activation_policy"]["caller_argument_overrides"] = True
    lease["receipt_policy"]["published_receipt_in_this_delivery"] = True
    lease["hidden_live_plan"] = True
    with pytest.raises(subject.DS91PreflightR2Error, match="schema validation failed"):
        subject.validate_schema(lease, schema, "fabricated lease")


def test_lease_full_locked_boundary_is_exact(
    workspace: subject.AnchoredWorkspace,
) -> None:
    plan = _plan(workspace)
    subject._validate_lease(plan, workspace)
    lease = _load_control(workspace, "lease_contract", "lease")
    assert lease["status"] == "locked-no-live-plan"
    assert lease["activation_policy"] == subject.EXPECTED_ACTIVATION_POLICY
    assert lease["activation_policy"]["plan_required"] is True
    assert lease["activation_policy"]["default_plan"] is False
    assert lease["activation_policy"]["published_live_plan"] is False
    assert lease["activation_policy"]["caller_argument_overrides"] is False
    assert lease["receipt_policy"] == subject.EXPECTED_RECEIPT_POLICY
    assert lease["receipt_policy"]["published_receipt_in_this_delivery"] is False
    assert plan["gpu_lease"]["execution_authorized"] is False


def test_duplicate_json_keys_are_rejected() -> None:
    with pytest.raises(subject.DS91PreflightR2Error, match="duplicate JSON key"):
        subject.strict_object(b'{"a":1,"a":2}', "duplicate fixture")


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "a/../../escape", "a\\b"])
def test_unsafe_workspace_paths_are_rejected(
    workspace: subject.AnchoredWorkspace, relative: str
) -> None:
    with pytest.raises(subject.DS91PreflightR2Error, match="unsafe workspace path"):
        workspace.read_and_pin(relative)


def test_parent_directory_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "control.json").write_text("{}")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with subject.AnchoredWorkspace(root) as workspace:
        with pytest.raises(subject.DS91PreflightR2Error, match="cannot traverse"):
            workspace.read_and_pin("linked/control.json")


def test_final_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "real.json").write_text("{}")
    (root / "alias.json").symlink_to("real.json")
    with subject.AnchoredWorkspace(root) as workspace:
        with pytest.raises(subject.DS91PreflightR2Error, match="cannot open anchored"):
            workspace.read_and_pin("alias.json")


def test_hardlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.json").write_text("{}")
    os.link(root / "a.json", root / "b.json")
    with subject.AnchoredWorkspace(root) as workspace:
        with pytest.raises(subject.DS91PreflightR2Error, match="hard-linked"):
            workspace.read_and_pin("a.json")


def test_root_symlink_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(subject.DS91PreflightR2Error, match="cannot hold workspace root"):
        with subject.AnchoredWorkspace(alias):
            pass


def test_final_name_replacement_during_held_read_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "control.json"
    replacement = root / "replacement.json"
    target.write_bytes(b"original")
    replacement.write_bytes(b"replaced")
    with subject.AnchoredWorkspace(root) as workspace:
        with pytest.raises(subject.DS91PreflightR2Error, match="identity changed"):
            with workspace.held_regular("control.json") as (descriptor, _opened):
                os.replace(replacement, target)
                assert os.read(descriptor, 8) == b"original"


def test_content_mutation_during_held_read_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "control.json"
    target.write_bytes(b"original")
    with subject.AnchoredWorkspace(root) as workspace:
        with pytest.raises(subject.DS91PreflightR2Error, match="identity changed"):
            with workspace.held_regular("control.json") as (descriptor, _opened):
                target.write_bytes(b"modified")
                os.lseek(descriptor, 0, os.SEEK_SET)
                assert os.read(descriptor, 8) == b"modified"


def test_exact_pin_mode_is_enforced(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "control.json"
    target.write_text("{}")
    target.chmod(0o640)
    expected = {
        "path": "control.json",
        "bytes": 2,
        "sha256": hashlib.sha256(b"{}").hexdigest(),
        "mode": "0440",
    }
    with subject.AnchoredWorkspace(root) as workspace:
        with pytest.raises(subject.DS91PreflightR2Error, match="exact pin mismatch"):
            workspace.replay_exact(expected, "mode fixture")


def test_alternate_plan_path_is_never_authoritative(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.deepstream91_full_stack_preflight_r2",
            "--plan",
            "replacement.json",
            "--skip-media-replay",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode != 0
    assert "alternate/self-resigned plan paths are not authoritative" in completed.stderr


def test_package_entrypoint_runs_from_clean_cwd(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.deepstream91_full_stack_preflight_r2",
            "--skip-media-replay",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "author_candidate_pass_blocked_as_designed"
    assert report["subject_accepted"] is False


def test_direct_entrypoint_also_runs_from_clean_cwd(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "validation/deepstream91_full_stack_preflight_r2.py"),
            "--skip-media-replay",
        ],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["execution_ready"] is False


@pytest.mark.parametrize(
    ("relative", "size", "digest"),
    [
        ("validation/deepstream91_full_stack_preflight_v1.py", 16316, "76cb6b20b5d56320531daeba15c8b0ed4c2128f458225e9fb7f30aa308597479"),
        ("validation/plans/deepstream91-full-stack-preflight-v1.json", 4131, "54f5538ac24a514fceb1d038511afb0324eae597c5629cefceb07516432878c8"),
        ("tests/test_deepstream91_full_stack_preflight_v1.py", 3940, "d85d4c15d344a93759503d1d76b51dcc205d9a7b9e5261f340a369697cb6a57a"),
        ("docs/deepstream9-full-stack-architecture.md", 8126, "b36c34fbfe80ac25db29d4697d8e80c68b1c1b72125819de35ac1a330004fa02"),
    ],
)
def test_r1_immutable_subject_pins_remain_exact(
    relative: str, size: int, digest: str
) -> None:
    raw = (ROOT / relative).read_bytes()
    assert len(raw) == size
    assert hashlib.sha256(raw).hexdigest() == digest
