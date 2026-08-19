from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from validation import ppe_safetyvision_r5_phase_a3 as phase_a3
from validation import ppe_safetyvision_r5_phase_a3_gate as phase_a3_gate


ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "models/ppe/export-lanes/safetyvision-yolov8s-v2-cpu-export-r5a3"
BASE_LANE = ROOT / "models/ppe/export-lanes/safetyvision-yolov8s-v2-cpu-export-r5"


def load_module(filename: str, name: str) -> ModuleType:
    for lane in (str(BASE_LANE), str(LANE)):
        if lane not in sys.path:
            sys.path.insert(0, lane)
    spec = importlib.util.spec_from_file_location(name, LANE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pin(path: Path) -> tuple[int, str]:
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def test_virtual_root_resolves_absolute_and_required_libpython_alias_chain(tmp_path: Path) -> None:
    runtime = load_module("runtime_closure_a3.py", "ppe_r5a3_runtime_success_test")
    rootfs = tmp_path / "rootfs"
    binary = rootfs / "usr/bin/python3.12"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"python")
    absolute = rootfs / "opt/venv/bin/python3"
    absolute.parent.mkdir(parents=True)
    absolute.symlink_to("/usr/bin/python3.12")

    real = rootfs / "usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"libpython")
    (real.parent / "libpython3.12.so.1").symlink_to(real.name)
    config = rootfs / "usr/lib/python3.12/config-3.12-x86_64-linux-gnu/libpython3.12.so"
    config.parent.mkdir(parents=True)
    config.symlink_to("../../x86_64-linux-gnu/libpython3.12.so.1")

    report = runtime.validate_symlink_closure(rootfs)
    assert report["verified"] is True
    assert report["symlinks"] == 3
    assert report["dangling"] == report["cycles"] == report["root_escapes"] == 0
    config_row = next(row for row in report["rows"] if row["path"].endswith("config-3.12-x86_64-linux-gnu/libpython3.12.so"))
    assert config_row["terminal"] == "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0"
    assert config_row["hops"] == 2


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("dangling", "dangling symlink component"),
        ("absolute_dangling", "dangling symlink component"),
        ("cycle", "symlink cycle"),
        ("escape", "escapes virtual root"),
    ],
)
def test_virtual_root_rejects_dangling_cycle_absolute_missing_and_escape(
    tmp_path: Path, kind: str, match: str
) -> None:
    runtime = load_module("runtime_closure_a3.py", f"ppe_r5a3_runtime_{kind}_test")
    rootfs = tmp_path / kind
    link = rootfs / "usr/lib/link"
    link.parent.mkdir(parents=True)
    if kind == "dangling":
        link.symlink_to("missing.so")
    elif kind == "absolute_dangling":
        link.symlink_to("/usr/lib/missing.so")
    elif kind == "cycle":
        link.symlink_to("other")
        (link.parent / "other").symlink_to("link")
    else:
        escape = rootfs / "escape"
        escape.symlink_to("../../outside")
        link = escape
    with pytest.raises(runtime.RuntimeClosureA3Error, match=match):
        runtime.resolve_virtual_symlink(rootfs, link.relative_to(rootfs).as_posix())


def test_wrong_libpython_hash_alias_and_package_metadata_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = load_module("runtime_closure_a3.py", "ppe_r5a3_runtime_host_pin_test")
    monkeypatch.setattr(runtime, "LIBPYTHON_PIN", (1, "0" * 64))
    with pytest.raises(runtime.RuntimeClosureA3Error, match="regular target drifted"):
        runtime.validate_successor_host_sources()

    alias = tmp_path / "alias"
    alias.symlink_to("wrong.so")
    with pytest.raises(runtime.RuntimeClosureA3Error, match="alias drifted"):
        runtime._validate_alias(alias, "expected.so", "unit")

    correct_package = "libpython3.12t64:amd64\t3.12.3-1ubuntu0.15\tamd64\n"
    correct_owner = "libpython3.12t64:amd64: /usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0\n"
    with pytest.raises(runtime.RuntimeClosureA3Error, match="version/architecture"):
        runtime._validate_package_outputs(correct_package.replace("0.15", "0.14"), correct_owner)
    with pytest.raises(runtime.RuntimeClosureA3Error, match="ownership drifted"):
        runtime._validate_package_outputs(correct_package, correct_owner.replace("libpython3.12t64", "wrong"))


def test_frozen_a3_source_closure_pins_package_file_alias_and_no_silent_drop() -> None:
    closure = json.loads((LANE / "runtime-source-closure-a3.json").read_text(encoding="utf-8"))
    assert closure["status"] == "phase_a3_source_closure_hashed_runtime_image_not_built"
    assert closure["successor_external_regular_files"] == [
        {
            "path": "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0",
            "bytes": 9061000,
            "sha256": "a4c35494d197a92f08a9d0a94975d9558e7a50880c42947484f01b720b68d423",
            "package": {
                "binary_package": "libpython3.12t64:amd64",
                "version": "3.12.3-1ubuntu0.15",
                "architecture": "amd64",
            },
        }
    ]
    assert closure["successor_external_aliases"] == [
        {
            "path": "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1",
            "target": "libpython3.12.so.1.0",
        }
    ]
    assert closure["source_gap"]["stdlib_symlink_target"] == "../../x86_64-linux-gnu/libpython3.12.so.1"
    assert closure["symlink_policy"]["drop_source_symlinks"] is False
    assert closure["expected_rootfs_inventory"] == {
        "directories": 9101,
        "regular_file_bytes": 2167081281,
        "regular_files": 57077,
        "symlinks": 47,
    }
    unsigned = dict(closure)
    fingerprint = unsigned.pop("self_fingerprint")
    runtime = load_module("runtime_closure_a3.py", "ppe_r5a3_runtime_fingerprint_test")
    assert hashlib.sha256(runtime.canonical_bytes(unsigned)).hexdigest() == fingerprint


def test_phase_a2_historical_lane_is_unchanged_and_a3_has_distinct_target_contract() -> None:
    assert pin(BASE_LANE / "runtime_closure.py") == (
        30355,
        "17d9b63d273439c47063cbdea14e70313078fe6ed63140135cd7203c01498a80",
    )
    assert pin(BASE_LANE / "image_builder.py") == (
        48776,
        "15c2fd1b8aba7d62947618db6272345db934800ca402e48e2557ce0d12a4c280",
    )
    source = (LANE / "image_builder_a3.py").read_text(encoding="utf-8")
    assert "phase_a2_context_mutated\": False" in source
    assert "runtime_a3.prepare_snapshot" in source
    assert "build_image(" not in source
    assert "docker buildx" not in source.lower()


def test_a3_context_receipt_adds_self_fingerprint_before_atomic_publication() -> None:
    source = (LANE / "image_builder_a3.py").read_text(encoding="utf-8")
    fingerprint_at = source.index('receipt["self_fingerprint"]')
    raw_at = source.index("receipt_raw = canonical_bytes(receipt)")
    rename_at = source.index("base_runtime.rename_noreplace")
    assert fingerprint_at < raw_at < rename_at
    assert '"method": "same_directory_renameat2_noreplace"' in source
    assert '"phase_a2_context_mutated": False' in source


def test_a3_static_gate_replays_every_pin_without_heavy_copy_or_execution() -> None:
    report = phase_a3.static_audit()
    assert report["status"] == "phase_a3_static_sources_closed_context_not_prepared"
    assert report["phase_a_revision"] == 3
    assert report["verified_sources"] == 24
    assert report["verified_embedded_sources"] == 15
    assert report["runtime_source_closure"]["live_replay_equal"] is True
    assert report["successor_context"] == {
        "path": "validation/work/ppe/safetyvision-r5-cpu-image-a3-001/context",
        "absent": True,
        "distinct_from_phase_a2": True,
    }
    assert report["historical_phase_a2_context_mutated"] is False
    assert report["checkpoint_deserialized"] is False
    assert report["model_export_or_inference_executed"] is False
    assert report["image_built"] is False
    assert report["gpu_used"] is False


def test_a3_contract_manifest_and_host_dependency_are_exact() -> None:
    contract_path = LANE / "source-contract-a3.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest_path = LANE / "export-workload-manifest-a3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert contract["schema_version"] == "deepsafe.ppe-safetyvision-cpu-source-contract-a3/v1"
    assert contract["status"] == "phase_a3_static_sources_closed_context_not_prepared"
    assert contract["phase_a_revision"] == 3
    assert contract["authorization"] == {
        "checkpoint_deserialization_or_export": False,
        "context_prepare_under_external_exact_gate": True,
        "cpu_onnxruntime_parity": False,
        "dedicated_cpu_image_build": False,
        "gpu": False,
        "model_acceptance": False,
        "runtime_snapshot_copy_under_external_exact_gate": True,
        "tensorrt_or_deepstream": False,
    }
    assert contract["host_source_dependencies"] == [
        {
            "aliases": [
                {
                    "path": "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1",
                    "target": "libpython3.12.so.1.0",
                },
                {
                    "path": "/usr/lib/python3.12/config-3.12-x86_64-linux-gnu/libpython3.12.so",
                    "target": "../../x86_64-linux-gnu/libpython3.12.so.1",
                },
            ],
            "architecture": "amd64",
            "bytes": 9061000,
            "package": "libpython3.12t64:amd64",
            "path": "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0",
            "sha256": "a4c35494d197a92f08a9d0a94975d9558e7a50880c42947484f01b720b68d423",
            "version": "3.12.3-1ubuntu0.15",
        }
    ]
    assert manifest["schema_version"] == "deepsafe.ppe-safetyvision-cpu-workload-manifest-a3/v1"
    assert manifest["status"] == "phase_a3_frozen_export_workload_context_not_prepared"
    assert manifest["phase_a_revision"] == 3
    assert manifest["runtime_source_closure"]["expected_rootfs"] == {
        "directories": 9101,
        "regular_file_bytes": 2167081281,
        "regular_files": 57077,
        "symlinks": 47,
    }
    for value, canonical in ((contract, phase_a3.canonical_bytes), (manifest, phase_a3.canonical_bytes)):
        unsigned = dict(value)
        fingerprint = unsigned.pop("self_fingerprint")
        assert hashlib.sha256(canonical(unsigned)).hexdigest() == fingerprint

    builder = load_module("image_builder_a3.py", "ppe_r5a3_builder_load_inputs_test")
    loaded_contract, loaded_manifest = builder.load_inputs(
        contract_path,
        pin(contract_path),
        manifest_path,
        pin(manifest_path),
    )
    assert loaded_contract == contract
    assert loaded_manifest == manifest


def test_a3_official_command_is_cpu_only_and_cannot_prepare_or_build() -> None:
    command = ROOT / "scripts/test-ppe-safetyvision-r5-phase-a3"
    source = command.read_text(encoding="utf-8")
    assert command.stat().st_mode & 0o111
    assert command.stat().st_mode & 0o222 == 0
    for required in (
        "set -euo pipefail",
        "PYTHONHASHSEED=0",
        "PYTHONDONTWRITEBYTECODE=1",
        "CUDA_VISIBLE_DEVICES=-1",
        "NVIDIA_VISIBLE_DEVICES=void",
        "tests/test_ppe_safetyvision_r5_phase_a3.py",
    ):
        assert required in source
    for forbidden in ("prepare-context", "docker build", "nvidia-smi", "best.pt"):
        assert forbidden not in source.lower()


def test_a3_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    with pytest.raises(phase_a3.PhaseA3GateError, match="duplicate JSON key"):
        phase_a3.strict_object(b'{"x":1,"x":2}', "duplicate")
    with pytest.raises(phase_a3.PhaseA3GateError, match="non-finite JSON token"):
        phase_a3.strict_object(b'{"x":NaN}', "nonfinite")


def test_a3_expected_report_is_exact_readonly_and_replayed_pre_post(tmp_path: Path) -> None:
    runtime = load_module("runtime_closure_a3.py", "ppe_r5a3_expected_report_test")
    report = tmp_path / "closure.json"
    report.write_bytes(b"{}\n")
    expected = pin(report)
    with pytest.raises(runtime.RuntimeClosureA3Error, match="writable"):
        runtime._read_exact(report, expected)
    report.chmod(0o444)
    assert runtime._read_exact(report, expected) == b"{}\n"
    report.chmod(0o644)
    report.write_bytes(b'{"drift":true}\n')
    report.chmod(0o444)
    with pytest.raises(runtime.RuntimeClosureA3Error, match="pin differs"):
        runtime._read_exact(report, expected)
    source = (LANE / "runtime_closure_a3.py").read_text(encoding="utf-8")
    assert source.count("_read_exact(expected_report, expected_report_pin)") == 3


def test_a3_output_is_exact_lexical_symlink_free_absent_and_has_no_stale_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = load_module("image_builder_a3.py", "ppe_r5a3_output_preconditions_test")
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(builder, "ROOT", root)
    contract = {"successor_context": {"path": "validation/work/ppe/run/context"}}
    output = root / "validation/work/ppe/run/context"
    report = builder._prepare_exact_output_parent(output, contract)
    assert report["ancestors_symlink_free"] is True
    assert report["target_absent"] is True
    assert report["stale_stages_absent"] is True

    stale = output.parent / ".context.stage-crash"
    stale.mkdir()
    with pytest.raises(builder.ImageBuilderA3Error, match="stale"):
        builder._prepare_exact_output_parent(output, contract)
    stale.rmdir()
    with pytest.raises(builder.ImageBuilderA3Error, match="exact canonical"):
        builder._prepare_exact_output_parent(root / "validation/work/ppe/other/context", contract)

    symlink_root = tmp_path / "symlink-workspace"
    symlink_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (symlink_root / "validation").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(builder, "ROOT", symlink_root)
    symlink_output = symlink_root / "validation/work/ppe/run/context"
    with pytest.raises(builder.ImageBuilderA3Error, match="real directory"):
        builder._prepare_exact_output_parent(symlink_output, contract)


def test_a3_failed_stage_cleanup_removes_readonly_tree_and_proves_no_stale_stage(
    tmp_path: Path,
) -> None:
    builder = load_module("image_builder_a3.py", "ppe_r5a3_stage_cleanup_test")
    output = tmp_path / "context"
    stage = tmp_path / ".context.stage-unit"
    child = stage / "rootfs/subdir"
    child.mkdir(parents=True)
    file = child / "payload"
    file.write_bytes(b"payload")
    file.chmod(0o444)
    child.chmod(0o555)
    (stage / "rootfs").chmod(0o555)
    stage.chmod(0o550)
    builder._cleanup_failed_stage(stage, output)
    assert not stage.exists()
    assert not list(tmp_path.glob(".context.stage-*"))


def test_a3_historical_tree_exact_pin_and_strict_receipt_replay_fail_closed(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "historical"
    historical.mkdir()
    payload = historical / "payload"
    payload.write_bytes(b"frozen")
    payload.chmod(0o444)
    historical.chmod(0o555)
    expected = phase_a3_gate.base_runtime.tree_attestation(historical)
    expected_without_root = {key: value for key, value in expected.items() if key != "root"}
    assert phase_a3_gate.attest_exact_tree(historical, expected_without_root, "fixture") == expected_without_root
    payload.chmod(0o644)
    payload.write_bytes(b"tampered")
    payload.chmod(0o444)
    with pytest.raises(phase_a3_gate.PhaseA3ExecutionGateError, match="attestation differs"):
        phase_a3_gate.attest_exact_tree(historical, expected_without_root, "fixture")

    receipt = {key: None for key in phase_a3_gate.builder_a3.CONTEXT_RECEIPT_KEYS}
    receipt["schema_version"] = "deepsafe.ppe-safetyvision-cpu-image-context-a3/v1"
    receipt["status"] = "phase_a3_frozen_context_ready_image_not_built"
    unsigned = dict(receipt)
    del unsigned["self_fingerprint"]
    receipt["self_fingerprint"] = hashlib.sha256(phase_a3_gate.canonical_bytes(unsigned)).hexdigest()
    raw = phase_a3_gate.canonical_bytes(receipt)
    replay = phase_a3_gate._validate_receipt(
        raw,
        phase_a3_gate.builder_a3.CONTEXT_RECEIPT_KEYS,
        "deepsafe.ppe-safetyvision-cpu-image-context-a3/v1",
        "phase_a3_frozen_context_ready_image_not_built",
        "fixture receipt",
    )
    assert replay == receipt
    missing = dict(receipt)
    del missing["gpu_used"]
    with pytest.raises(phase_a3_gate.PhaseA3ExecutionGateError, match="key set"):
        phase_a3_gate._validate_receipt(
            phase_a3_gate.canonical_bytes(missing),
            phase_a3_gate.builder_a3.CONTEXT_RECEIPT_KEYS,
            "deepsafe.ppe-safetyvision-cpu-image-context-a3/v1",
            "phase_a3_frozen_context_ready_image_not_built",
            "fixture receipt",
        )
    wrong_schema = dict(receipt)
    wrong_schema["schema_version"] = "wrong"
    unsigned = dict(wrong_schema)
    del unsigned["self_fingerprint"]
    wrong_schema["self_fingerprint"] = hashlib.sha256(
        phase_a3_gate.canonical_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(phase_a3_gate.PhaseA3ExecutionGateError, match="schema/status"):
        phase_a3_gate._validate_receipt(
            phase_a3_gate.canonical_bytes(wrong_schema),
            phase_a3_gate.builder_a3.CONTEXT_RECEIPT_KEYS,
            "deepsafe.ppe-safetyvision-cpu-image-context-a3/v1",
            "phase_a3_frozen_context_ready_image_not_built",
            "fixture receipt",
        )


def test_a3_gate_replays_historical_post_tree_even_when_builder_invocation_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = {
        "successor_context": {"historical_phase_a2_path": "historical"},
        "sources": [
            {
                "logical_name": "runtime_source_closure_a3",
                "path": "closure.json",
                "bytes": 1,
                "sha256": "0" * 64,
            }
        ],
    }
    monkeypatch.setattr(phase_a3_gate, "_read_exact_readonly", lambda *_args, **_kwargs: b"x")
    monkeypatch.setattr(phase_a3_gate.static_a3, "strict_object", lambda *_args, **_kwargs: contract)
    monkeypatch.setattr(phase_a3_gate.static_a3, "_validate_fingerprint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(phase_a3_gate, "_assert_exact_paths", lambda *_args, **_kwargs: (tmp_path, tmp_path))
    monkeypatch.setattr(
        phase_a3_gate.static_a3,
        "static_audit",
        lambda *_args, **_kwargs: {"successor_context": {"absent": True}},
    )
    monkeypatch.setattr(phase_a3_gate, "_stale_stage_names", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(phase_a3_gate, "_workspace_path", lambda value: tmp_path / value)
    monkeypatch.setattr(
        phase_a3_gate,
        "_validate_real_workspace_ancestors",
        lambda *_args, **_kwargs: None,
    )
    attestations: list[str] = []

    def attest(_root: Path, _expected: dict, label: str) -> dict:
        attestations.append(label)
        return dict(phase_a3_gate.HISTORICAL_TREE)

    monkeypatch.setattr(phase_a3_gate, "attest_exact_tree", attest)
    monkeypatch.setattr(
        phase_a3_gate.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            phase_a3_gate.subprocess.TimeoutExpired(cmd="builder", timeout=900)
        ),
    )
    with pytest.raises(phase_a3_gate.PhaseA3ExecutionGateError, match="TimeoutExpired"):
        phase_a3_gate.execute(
            tmp_path / "contract.json",
            (1, "0" * 64),
            tmp_path / "manifest.json",
            (1, "1" * 64),
            tmp_path / "context",
            tmp_path / "receipt.json",
        )
    assert attestations == [
        "historical Phase-A2 context pre",
        "historical Phase-A2 context post",
    ]
