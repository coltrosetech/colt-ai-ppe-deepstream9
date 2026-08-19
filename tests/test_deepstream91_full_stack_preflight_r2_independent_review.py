from __future__ import annotations

import ast
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
RECEIPT = (
    ROOT
    / "validation/results/deepstream91-full-stack-preflight-r2"
    / "independent-review-r1/review.json"
)

EXPECTED_FROZEN_PINS = {
    "verifier": {
        "path": "validation/deepstream91_full_stack_preflight_r2.py",
        "bytes": 30155,
        "sha256": "21cc7ed9abdb4037476fc0aada316c49e5ff26b42eb49330323521cab179f340",
        "mode": "0440",
    },
    "author_tests": {
        "path": "tests/test_deepstream91_full_stack_preflight_r2.py",
        "bytes": 15731,
        "sha256": "fd47235327a837045f9b4a8384570b6fc1b167aed753bacc4086e6ddf7f0b8d3",
        "mode": "0440",
    },
    "plan": {
        "path": "validation/plans/deepstream91-full-stack-preflight-r2.json",
        "bytes": 5057,
        "sha256": "a7f70ee6c366f33acbf448b71594b47ff13c4d0f94e37722a8c58ea6cb17e368",
        "mode": "0440",
    },
    "plan_schema": {
        "path": "validation/schemas/deepstream91-full-stack-preflight-r2.schema.json",
        "bytes": 9048,
        "sha256": "849fb9152059ad5dcf504fdb57deefb58043c70b89995ff907e4553eda61b0c1",
        "mode": "0440",
    },
    "source_manifest": {
        "path": "validation/manifests/deepstream91-source-matrix-r2.json",
        "bytes": 4436,
        "sha256": "ce622a9925c548a360fe68e37f0ce8793dcb5d5ccfe2b2fe1c0f51e3e97677c5",
        "mode": "0440",
    },
    "source_schema": {
        "path": "validation/schemas/deepstream91-source-matrix-r2.schema.json",
        "bytes": 1693,
        "sha256": "a13b58cf7b2dbc74ef877c9d0308a384405e7dcc4e673b58ef30f92db9b21a83",
        "mode": "0440",
    },
    "builder_root": {
        "path": "validation/accepted-roots/ds91-engine-builder-r1c3-terminal/terminal-root-r1c3.json",
        "bytes": 2926,
        "sha256": "57805489b3c0238e9cc5b13802ef644a9f43346629b6d444a7f89de5a9d54a49",
        "mode": "0440",
    },
    "builder_schema": {
        "path": "validation/schemas/deepstream91-r1c3-accepted-root-boundary-r2.schema.json",
        "bytes": 4474,
        "sha256": "86659137d6fb7359859dad54fee03f166dee0c24ddca5489bd48bf9822bffef0",
        "mode": "0440",
    },
    "builder_candidate": {
        "path": "validation/results/ds91-engine-builder/r1c3/candidate-receipt-r1c3.json",
        "bytes": 11849,
        "sha256": "6acdc00e90441d7f7e3f923b597365a74cc6df9bb13619291e08a6db36b6c75a",
        "mode": "0440",
    },
    "lease_contract": {
        "path": "validation/contracts/gpu-lease-v5.json",
        "bytes": 3581,
        "sha256": "039322d84b3ad495e6bf4aa16966c11c9f1dfda981908908bd7f77cc0b58b38d",
        "mode": "0440",
    },
    "lease_schema": {
        "path": "validation/schemas/gpu-lease-v5-locked-boundary-r2.schema.json",
        "bytes": 6140,
        "sha256": "dab835b334f2b720b63bd76ea7fbca98ead241e79192dd00645d0fd42ff2f86e",
        "mode": "0440",
    },
    "r1_review": {
        "path": "validation/results/deepstream91-full-stack-preflight-v1/independent-review-r1/review.json",
        "bytes": 10684,
        "sha256": "02e03cc67acb1aacc5ba89d26ebb7f705b430340091fb1b02689e650d2392e2e",
        "mode": "0440",
    },
}


def _pin(path: Path, relative: str) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mode": f"{path.stat().st_mode & 0o7777:04o}",
    }


def _load_pinned(workspace: subject.AnchoredWorkspace, name: str) -> dict:
    return subject.strict_object(
        workspace.replay_exact(subject.EXTERNAL_CONTROL_PINS[name], name), name
    )


def _resign(plan: dict) -> dict:
    changed = copy.deepcopy(plan)
    changed.pop("self_fingerprint", None)
    changed["self_fingerprint"] = hashlib.sha256(
        subject.canonical_bytes(changed)
    ).hexdigest()
    return changed


@pytest.mark.parametrize("name", sorted(EXPECTED_FROZEN_PINS))
def test_independent_external_pin_and_freeze_replay(name: str) -> None:
    expected = EXPECTED_FROZEN_PINS[name]
    assert _pin(ROOT / str(expected["path"]), str(expected["path"])) == expected
    assert (ROOT / str(expected["path"])).stat().st_nlink == 1


def test_r1_p1_001_source_contract_is_source_only_and_ds90_authority_unreachable() -> None:
    source_text = (ROOT / EXPECTED_FROZEN_PINS["verifier"]["path"]).read_text()
    tree = ast.parse(source_text)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name.startswith("validation.") for name in imported)
    assert "deepstream-full-stack-benchmark-v1.json" not in source_text
    assert "deepstream-full-stack-benchmark-plan-v1.schema.json" not in source_text
    assert set(subject.EXTERNAL_CONTROL_PINS) == {
        "plan",
        "plan_schema",
        "source_manifest",
        "source_schema",
        "builder_root",
        "builder_schema",
        "builder_candidate",
        "lease_contract",
        "lease_schema",
    }
    assert not any(
        "deepstream-full-stack-benchmark-v1" in pin["path"]
        for pin in subject.EXTERNAL_CONTROL_PINS.values()
    )
    with subject.AnchoredWorkspace(ROOT) as workspace:
        plan = subject.load_default_plan(workspace)
        manifest = _load_pinned(workspace, "source_manifest")
        assert plan["source_matrix"]["legacy_ds90_object_reachable"] is False
        assert set(manifest) == {"schema_version", "manifest_id", "sources"}
        assert len(manifest["sources"]) == 12
        assert all(
            set(row) == {"source_id", "video_type", "view_types", "media_pin"}
            and set(row["media_pin"]) == {"path", "bytes", "sha256"}
            for row in manifest["sources"]
        )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("source_matrix", "manifest"),
        ("runtime_target", "prepared_closed_engine_builder_root"),
        ("gpu_lease", "contract"),
    ],
)
def test_r1_p1_002_self_resign_cannot_replace_external_control(
    section: str, field: str
) -> None:
    with subject.AnchoredWorkspace(ROOT) as workspace:
        plan = subject.load_default_plan(workspace)
        plan[section][field]["sha256"] = "0" * 64
        with pytest.raises(subject.DS91PreflightR2Error, match="external authority pin"):
            subject.validate_plan(_resign(plan), workspace=workspace, replay_media=False)


def test_r1_p1_002_self_resign_cannot_claim_independent_acceptance() -> None:
    with subject.AnchoredWorkspace(ROOT) as workspace:
        plan = subject.load_default_plan(workspace)
        plan["authority"]["independent_acceptance_root"] = {
            "path": "fabricated-review.json",
            "bytes": 1,
            "sha256": "0" * 64,
            "mode": "0440",
        }
        with pytest.raises(subject.DS91PreflightR2Error, match="schema validation failed"):
            subject.validate_plan(_resign(plan), workspace=workspace, replay_media=False)


@pytest.mark.parametrize(
    ("group", "field"),
    [
        ("accepted_scope", "gpu_runtime"),
        ("accepted_scope", "tensorrt_engine_build"),
        ("accepted_scope", "deepstream_runtime"),
        ("accepted_scope", "inference"),
        ("accepted_scope", "production"),
        ("qualification", "deepstream_runtime_qualified"),
        ("qualification", "gpu_runtime_qualified"),
        ("qualification", "inference_qualified"),
        ("qualification", "production_ready"),
        ("qualification", "tensorrt_engine_qualified"),
    ],
)
def test_r1_p1_003_builder_runtime_or_qualification_overclaim_is_rejected(
    group: str, field: str
) -> None:
    with subject.AnchoredWorkspace(ROOT) as workspace:
        builder = _load_pinned(workspace, "builder_root")
        schema = _load_pinned(workspace, "builder_schema")
        builder[group][field] = True
        with pytest.raises(subject.DS91PreflightR2Error, match="schema validation failed"):
            subject.validate_schema(builder, schema, "mutated builder")


@pytest.mark.parametrize(
    ("group", "field", "value"),
    [
        ("activation_policy", "default_plan", True),
        ("activation_policy", "published_live_plan", True),
        ("activation_policy", "caller_argument_overrides", True),
        ("receipt_policy", "published_receipt_in_this_delivery", True),
    ],
)
def test_r1_p1_003_lease_plan_or_receipt_overclaim_is_rejected(
    group: str, field: str, value: bool
) -> None:
    with subject.AnchoredWorkspace(ROOT) as workspace:
        lease = _load_pinned(workspace, "lease_contract")
        schema = _load_pinned(workspace, "lease_schema")
        lease[group][field] = value
        with pytest.raises(subject.DS91PreflightR2Error, match="schema validation failed"):
            subject.validate_schema(lease, schema, "mutated lease")


@pytest.mark.parametrize(
    "field",
    [
        "driver_download_install_remove",
        "reboot",
        "gpu_workload",
        "docker_gpu",
        "engine_build",
        "benchmark_execution",
        "endurance_execution",
    ],
)
def test_r2_plan_cannot_smuggle_execution_authorization(field: str) -> None:
    with subject.AnchoredWorkspace(ROOT) as workspace:
        plan = subject.load_default_plan(workspace)
        plan["authorization"][field] = True
        with pytest.raises(subject.DS91PreflightR2Error, match="schema validation failed"):
            subject.validate_plan(_resign(plan), workspace=workspace, replay_media=False)


def test_r1_p1_004_every_workspace_component_is_descriptor_bound_and_nofollow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    nested = root / "one" / "two"
    nested.mkdir(parents=True)
    (nested / "control.json").write_bytes(b"{}")
    real_open = os.open
    calls: list[tuple[object, int, int | None]] = []

    def traced_open(
        path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        calls.append((path, flags, dir_fd))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(subject.os, "open", traced_open)
    with subject.AnchoredWorkspace(root) as workspace:
        raw, observed = workspace.read_and_pin("one/two/control.json")
    assert raw == b"{}"
    assert observed["sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert len(calls) == 4
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    assert nofollow and directory
    assert calls[0][2] is None
    assert calls[0][1] & nofollow and calls[0][1] & directory
    for _path, flags, dir_fd in calls[1:3]:
        assert dir_fd is not None
        assert flags & nofollow and flags & directory
    assert calls[3][2] is not None
    assert calls[3][1] & nofollow
    assert not calls[3][1] & directory


def test_r1_p1_004_ancestor_swap_cannot_redirect_a_held_read(tmp_path: Path) -> None:
    root = tmp_path / "root"
    original_parent = root / "controls"
    outside = tmp_path / "outside"
    original_parent.mkdir(parents=True)
    outside.mkdir()
    (original_parent / "control.json").write_bytes(b"accepted-original")
    (outside / "control.json").write_bytes(b"attacker-substitute")
    with subject.AnchoredWorkspace(root) as workspace:
        with workspace.held_regular("controls/control.json") as (descriptor, opened):
            original_parent.rename(root / "held-original")
            (root / "controls").symlink_to(outside, target_is_directory=True)
            assert os.read(descriptor, opened.st_size) == b"accepted-original"


@pytest.mark.parametrize("kind", ["module", "direct"])
def test_r1_p2_001_clean_cwd_entrypoints_are_reproducible(
    kind: str, tmp_path: Path
) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["NVIDIA_VISIBLE_DEVICES"] = "void"
    if kind == "module":
        command = [
            sys.executable,
            "-m",
            "validation.deepstream91_full_stack_preflight_r2",
            "--skip-media-replay",
        ]
    else:
        command = [
            sys.executable,
            str(ROOT / EXPECTED_FROZEN_PINS["verifier"]["path"]),
            "--skip-media-replay",
        ]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "author_candidate_pass_blocked_as_designed"
    assert report["execution_ready"] is False
    assert report["gpu_or_docker_called"] is False


def test_normal_command_replays_all_twelve_media_without_authorizing_execution() -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["NVIDIA_VISIBLE_DEVICES"] = "void"
    completed = subprocess.run(
        [sys.executable, "-m", "validation.deepstream91_full_stack_preflight_r2"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report == {
        "blockers": subject.EXPECTED_BLOCKERS,
        "distinct_video_types": 12,
        "execution_ready": False,
        "external_control_constants_enforced": True,
        "gpu_or_docker_called": False,
        "independent_acceptance_required": True,
        "legacy_ds90_plan_validator_schema_consumed": False,
        "measurement_seconds_per_profile": 300,
        "profiles": [640, 960],
        "schema_version": "deepsafe.deepstream91-full-stack-preflight-report/v2",
        "self_fingerprint_authoritative": False,
        "simulated_streams": 12,
        "source_count": 12,
        "status": "author_candidate_pass_blocked_as_designed",
        "subject_accepted": False,
    }


def test_independent_review_receipt_is_exact_self_bound_and_non_authorizing() -> None:
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["decision"] == "ACCEPT"
    assert receipt["severity_counts"] == {"P0": 0, "P1": 0, "P2": 0}
    assert [item["finding_id"] for item in receipt["r1_finding_disposition"]] == [
        "DSPF-P1-001",
        "DSPF-P1-002",
        "DSPF-P1-003",
        "DSPF-P1-004",
        "DSPF-P2-001",
    ]
    assert all(
        item["status"] == "fixed" and item["acceptance_blocker"] is False
        for item in receipt["r1_finding_disposition"]
    )
    assert receipt["subject"]["frozen_pins"] == EXPECTED_FROZEN_PINS
    independent_relative = (
        "tests/test_deepstream91_full_stack_preflight_r2_independent_review.py"
    )
    assert receipt["subject"]["independent_tests"] == _pin(
        ROOT / independent_relative, independent_relative
    )
    assert receipt["authority_boundary"] == {
        "model": "cooperative_same_uid",
        "malicious_same_uid_resistance_claimed": False,
        "self_fingerprint_is_authority": False,
        "independent_receipt_is_external_entrypoint": True,
        "receipt_file_mode": "0440",
        "receipt_parent_mode": "0550",
    }
    assert receipt["permissions"] == {
        "benchmark_execution": False,
        "checkpoint_deserialization": False,
        "docker": False,
        "docker_gpu": False,
        "driver_download_install_remove": False,
        "endurance_execution": False,
        "engine_build": False,
        "gpu_workload": False,
        "installation": False,
        "model_load": False,
        "network": False,
        "reboot": False,
        "runtime_execution": False,
        "sudo": False,
    }
    unsigned = dict(receipt)
    fingerprint = unsigned.pop("self_fingerprint")
    assert fingerprint == {
        "algorithm": "sha256-canonical-json-without-self_fingerprint",
        "canonical_sha256": hashlib.sha256(
            subject.canonical_bytes(unsigned)
        ).hexdigest(),
    }
    assert RECEIPT.stat().st_mode & 0o7777 == 0o440
    assert RECEIPT.parent.stat().st_mode & 0o7777 == 0o550
    assert RECEIPT.stat().st_nlink == 1
