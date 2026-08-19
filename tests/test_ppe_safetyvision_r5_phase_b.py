from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest

from validation import ppe_safetyvision_r5_phase_b as gate


ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "models/ppe/export-lanes/safetyvision-yolov8s-v2-cpu-export-r5"


def pin_raw(raw: bytes) -> tuple[int, str]:
    return len(raw), hashlib.sha256(raw).hexdigest()


def canonical_file(path: Path, value: dict) -> bytes:
    raw = gate.canonical_bytes(value) + b"\n"
    path.write_bytes(raw)
    return raw


def thaw(root: Path) -> None:
    for directory, children, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        current.chmod(0o700)
        for name in files:
            path = current / name
            if not path.is_symlink():
                path.chmod(0o600)
        for name in children:
            path = current / name
            if not path.is_symlink():
                path.chmod(0o700)


def freeze(root: Path) -> None:
    for directory, children, files in os.walk(root, topdown=False, followlinks=False):
        current = Path(directory)
        for name in files:
            path = current / name
            if path.is_symlink():
                continue
            path.chmod(0o440 if path.suffix == ".json" else 0o444)
        for name in children:
            path = current / name
            if not path.is_symlink():
                path.chmod(0o555)
        current.chmod(0o550 if current == root else 0o555)


def make_receipt_fixture(tmp_path: Path) -> tuple[Path, dict, dict, dict]:
    context = (tmp_path / "context").resolve()
    (context / "runtime/rootfs").mkdir(parents=True)
    docker_raw = (LANE / "Dockerfile.r5").read_bytes()
    manifest_raw = (LANE / "export-workload-manifest-r5.json").read_bytes()
    (context / "Dockerfile.r5").write_bytes(docker_raw)
    (context / "workload-manifest.json").write_bytes(manifest_raw)
    rootfs = {
        "root": "rootfs",
        "tree_sha256": "a" * 64,
        "regular_files": 0,
        "directories": 1,
        "symlinks": 0,
        "regular_file_bytes": 0,
    }
    snapshot = {
        "schema_version": "deepsafe.ppe-safetyvision-cpu-runtime-snapshot/v1",
        "status": "runtime_rootfs_snapshot_completed_image_not_built",
        "source_closure": {"bytes": 1, "sha256": "b" * 64},
        "rootfs": rootfs,
        "distribution_count": 87,
        "staging_policy": {"directories": "0700", "regular_files": "0600", "symlinks_followed": False},
        "publication": {
            "method": "same_directory_renameat2_noreplace",
            "file_and_directory_fsync": True,
            "parent_directory_fsync": True,
            "final_directory_mode": "0550",
        },
        "checkpoint_included": False,
        "model_export_or_inference_executed": False,
        "gpu_used": False,
    }
    snapshot_raw = canonical_file(context / "runtime/snapshot-receipt.json", snapshot)
    contract_pin = gate.pin_regular(LANE / "source-contract-r5.json")
    receipt = {
        "schema_version": "deepsafe.ppe-safetyvision-cpu-image-context-r5/v1",
        "status": "frozen_context_ready_image_not_built",
        "phase": "export",
        "source_contract": {"bytes": contract_pin[0], "sha256": contract_pin[1]},
        "workload_manifest": {"bytes": len(manifest_raw), "sha256": hashlib.sha256(manifest_raw).hexdigest()},
        "dockerfile": {"bytes": len(docker_raw), "sha256": hashlib.sha256(docker_raw).hexdigest()},
        "runtime_snapshot": {
            "rootfs": rootfs,
            "snapshot_receipt": {"bytes": len(snapshot_raw), "sha256": hashlib.sha256(snapshot_raw).hexdigest()},
        },
        "phase_a1_predecessor_replay": {},
        "context_payload_tree": {},
        "staging_policy": {},
        "publication": {},
        "host_runtime_bind_mounts": False,
        "checkpoint_deserialized": False,
        "model_export_or_inference_executed": False,
        "image_built": False,
        "gpu_used": False,
    }
    receipt_raw = canonical_file(context / "context-receipt.json", receipt)
    plan_context = {
        "receipt": {
            "authority": "exact_raw_bytes_sha256",
            "bytes": len(receipt_raw),
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "canonical_payload_sha256": hashlib.sha256(gate.canonical_bytes(receipt)).hexdigest(),
            "exact_keys": sorted(gate.CONTEXT_RECEIPT_KEYS),
            "self_fingerprint": "required_absent",
        },
        "snapshot_receipt": {
            "bytes": len(snapshot_raw),
            "sha256": hashlib.sha256(snapshot_raw).hexdigest(),
            "exact_keys": sorted(gate.SNAPSHOT_RECEIPT_KEYS),
        },
    }
    freeze(context)
    return context, plan_context, receipt, snapshot


def repin_context_receipt(context: Path, plan_context: dict, receipt: dict) -> None:
    thaw(context)
    raw = canonical_file(context / "context-receipt.json", receipt)
    plan_context["receipt"]["bytes"] = len(raw)
    plan_context["receipt"]["sha256"] = hashlib.sha256(raw).hexdigest()
    plan_context["receipt"]["canonical_payload_sha256"] = hashlib.sha256(gate.canonical_bytes(receipt)).hexdigest()
    freeze(context)


def test_exact_raw_context_and_snapshot_receipts_directly_replay_through_frozen_builder(tmp_path: Path) -> None:
    context, plan_context, _, _ = make_receipt_fixture(tmp_path)
    report = gate._strict_context_receipts({"context": plan_context}, context)
    assert report["context_receipt_self_fingerprint_absent"] is True
    assert report["frozen_image_builder_direct_raw_replay"] is True
    assert report["context_receipt"]["sha256"] == plan_context["receipt"]["sha256"]
    assert report["snapshot_receipt"]["sha256"] == plan_context["snapshot_receipt"]["sha256"]


@pytest.mark.parametrize("mutation", ["extra", "missing", "self_fingerprint"])
def test_repinned_context_receipt_shape_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
    context, plan_context, receipt, _ = make_receipt_fixture(tmp_path)
    altered = copy.deepcopy(receipt)
    if mutation == "extra":
        altered["unexpected"] = True
    elif mutation == "missing":
        altered.pop("phase")
    else:
        altered["self_fingerprint"] = "f" * 64
    repin_context_receipt(context, plan_context, altered)
    with pytest.raises(gate.PhaseBGateR5Error, match="16-key set|self_fingerprint"):
        gate._strict_context_receipts({"context": plan_context}, context)


def test_repinned_snapshot_receipt_extra_or_missing_key_fails_closed(tmp_path: Path) -> None:
    for index, mutation in enumerate(("extra", "missing")):
        context, plan_context, receipt, snapshot = make_receipt_fixture(tmp_path / str(index))
        thaw(context)
        altered = copy.deepcopy(snapshot)
        if mutation == "extra":
            altered["unexpected"] = True
        else:
            altered.pop("distribution_count")
        snapshot_raw = canonical_file(context / "runtime/snapshot-receipt.json", altered)
        plan_context["snapshot_receipt"]["bytes"] = len(snapshot_raw)
        plan_context["snapshot_receipt"]["sha256"] = hashlib.sha256(snapshot_raw).hexdigest()
        receipt["runtime_snapshot"]["snapshot_receipt"] = {
            "bytes": len(snapshot_raw),
            "sha256": hashlib.sha256(snapshot_raw).hexdigest(),
        }
        receipt_raw = canonical_file(context / "context-receipt.json", receipt)
        plan_context["receipt"]["bytes"] = len(receipt_raw)
        plan_context["receipt"]["sha256"] = hashlib.sha256(receipt_raw).hexdigest()
        plan_context["receipt"]["canonical_payload_sha256"] = hashlib.sha256(gate.canonical_bytes(receipt)).hexdigest()
        freeze(context)
        with pytest.raises(gate.PhaseBGateR5Error, match="snapshot receipt exact key set"):
            gate._strict_context_receipts({"context": plan_context}, context)


def make_tree_fixture(tmp_path: Path) -> Path:
    context = (tmp_path / "tree-context").resolve()
    binary = context / "runtime/rootfs/usr/bin/tool"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"tool")
    binary.chmod(0o555)
    link = context / "runtime/rootfs/opt/bin/tool"
    link.parent.mkdir(parents=True)
    link.symlink_to("/usr/bin/tool")
    inputs = context / "inputs"
    inputs.mkdir()
    (inputs / "data.bin").write_bytes(b"data")
    (context / "runtime/snapshot-receipt.json").write_bytes(b"{}\n")
    (context / "context-receipt.json").write_bytes(b"{}\n")
    freeze(context)
    return context


def plan_for_replay(report: dict) -> dict:
    return {
        "context": {
            "attestations": {
                key: copy.deepcopy(report[key])
                for key in ("payload_excluding_context_receipt", "published_full_tree", "runtime_rootfs")
            },
            "mode_inventory": copy.deepcopy(report["mode_inventory"]),
            "allocated_bytes_st_blocks": report["allocated_bytes_st_blocks"],
        }
    }


def test_repinned_altered_tree_and_receipt_full_tree_mismatch_fail_closed(tmp_path: Path) -> None:
    context = make_tree_fixture(tmp_path)
    before = gate.replay_context_trees(context)
    plan = plan_for_replay(before)
    thaw(context)
    (context / "inputs/data.bin").write_bytes(b"repinned altered data")
    freeze(context)
    altered = gate.replay_context_trees(context)
    with pytest.raises(gate.PhaseBGateR5Error, match="tree replay differs"):
        gate._validate_tree_replay(plan, altered)

    mismatch = copy.deepcopy(before)
    mismatch["published_full_tree"]["tree_sha256"] = "f" * 64
    with pytest.raises(gate.PhaseBGateR5Error, match="published_full_tree"):
        gate._validate_tree_replay(plan, mismatch)


def test_mode_hardlink_and_symlink_escape_mutations_fail_closed(tmp_path: Path) -> None:
    context = make_tree_fixture(tmp_path / "mode")
    before = gate.replay_context_trees(context)
    plan = plan_for_replay(before)
    thaw(context)
    (context / "inputs/data.bin").chmod(0o640)
    context.chmod(0o550)
    mode_replay = gate.replay_context_trees(context)
    plan["context"]["attestations"] = {
        key: copy.deepcopy(mode_replay[key])
        for key in ("payload_excluding_context_receipt", "published_full_tree", "runtime_rootfs")
    }
    with pytest.raises(gate.PhaseBGateR5Error, match="mode/link/special inventory"):
        gate._validate_tree_replay(plan, mode_replay)

    hardlink_context = make_tree_fixture(tmp_path / "hardlink")
    thaw(hardlink_context)
    os.link(hardlink_context / "inputs/data.bin", hardlink_context / "inputs/data-hardlink.bin")
    freeze(hardlink_context)
    with pytest.raises(gate.PhaseBGateR5Error, match="hardlink"):
        gate.replay_context_trees(hardlink_context)

    symlink_context = make_tree_fixture(tmp_path / "symlink")
    thaw(symlink_context)
    (symlink_context / "inputs/escape").symlink_to("/etc/passwd")
    freeze(symlink_context)
    with pytest.raises(gate.frozen_builder.ImageBuilderR5Error, match="absolute symlink target outside rootfs"):
        gate.replay_context_trees(symlink_context)


def test_builder_stream_and_effective_argv_must_equal_external_plan(tmp_path: Path) -> None:
    context = make_tree_fixture(tmp_path)
    replay = gate.replay_context_trees(context)
    plan = plan_for_replay(replay)
    plan["build"] = {
        "buildx_argv": ["docker", "buildx", "build", "-"],
        "clean_build_environment": {"CUDA_VISIBLE_DEVICES": "-1"},
        "base_image": {"id": "sha256:" + "a" * 64, "reference": "base@example"},
        "toolchain": {
            "builder": "default",
            "buildkit": "v0.29.0",
            "buildx_commit": "b" * 40,
            "buildx_version": "v0.33.0",
            "driver": "docker",
            "gpu_devices_automatically_allowed": False,
            "platform": "linux/amd64",
        },
    }
    good = {
        "context_stream": {"tree": {**replay["published_full_tree"], "root": str(context)}},
        "build_argv": plan["build"]["buildx_argv"],
        "build_environment": plan["build"]["clean_build_environment"],
        "base_image": plan["build"]["base_image"],
        "build_toolchain": {
            "buildx": {"version": "v0.33.0", "commit": "b" * 40},
            "builder": {
                "builder": "default",
                "driver": "docker",
                "buildkit": "v0.29.0",
                "gpu_devices_automatically_allowed": False,
            },
        },
    }
    gate._validate_builder_result(plan, context, good)
    bad_tree = copy.deepcopy(good)
    bad_tree["context_stream"]["tree"]["tree_sha256"] = "f" * 64
    with pytest.raises(gate.PhaseBGateR5Error, match="stream tree"):
        gate._validate_builder_result(plan, context, bad_tree)
    bad_argv = copy.deepcopy(good)
    bad_argv["build_argv"] = ["docker", "buildx", "build", "--network=host", "-"]
    with pytest.raises(gate.PhaseBGateR5Error, match="effective Buildx argv"):
        gate._validate_builder_result(plan, context, bad_argv)


def minimal_plan() -> dict:
    plan = {
        "schema_version": "deepsafe.ppe-safetyvision-cpu-phase-b-authority-r5/v1",
        "status": "phase_b_exact_build_gate_ready_not_executed",
        "phase": "build",
        "authorization": {
            "docker_build_requires_external_exact_gate": True,
            "external_exact_gate_granted_at_plan_publication": False,
            "export_or_inference": False,
            "gpu": False,
            "model_acceptance": False,
            "network": False,
            "nvidia_runtime": False,
            "tensorrt_or_deepstream": False,
        },
        "authority_files": [],
        "context": {},
        "build": {},
        "cooperative_same_uid_toctou_boundary": {
            "prevented_against_non_cooperative_same_uid_mutator": False,
            "explicitly_accepted_for_gate": True,
            "prebuild_full_replay": True,
            "builder_stream_before_after_replay": True,
            "postbuild_full_replay": True,
        },
        "execution": {},
    }
    plan["self_fingerprint"] = hashlib.sha256(gate.canonical_bytes(plan)).hexdigest()
    return plan


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_phase_b_plan_top_level_shape_mutations_fail_even_when_repinned(tmp_path: Path, mutation: str) -> None:
    plan = minimal_plan()
    plan.pop("self_fingerprint")
    if mutation == "extra":
        plan["unexpected"] = True
    else:
        plan.pop("execution")
    plan["self_fingerprint"] = hashlib.sha256(gate.canonical_bytes(plan)).hexdigest()
    raw = canonical_file(tmp_path / "plan.json", plan)
    (tmp_path / "plan.json").chmod(0o440)
    with pytest.raises(gate.PhaseBGateR5Error, match="top-level shape"):
        gate.load_plan(tmp_path / "plan.json", pin_raw(raw))


def test_audit_path_has_no_build_or_subprocess_execution_surface() -> None:
    source = inspect.getsource(gate.audit_plan)
    assert "build_image(" not in source
    assert "subprocess" not in source
    assert "Popen" not in source
    assert "docker" not in source.lower()
