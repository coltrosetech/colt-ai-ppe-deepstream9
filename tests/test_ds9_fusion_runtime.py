from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deepstream import fusion_runtime


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "deepstream/fusion-runtime-build-plan.json"


def test_frozen_plan_and_local_inputs_are_exact() -> None:
    plan, digest = fusion_runtime.load_frozen_plan(PLAN)
    assert digest == fusion_runtime.FROZEN_PLAN_SHA256
    assert len(plan["local_inputs"]) == 39
    paths = [item["path"] for item in plan["local_inputs"]]
    assert paths == sorted(paths)
    for item in plan["local_inputs"]:
        content = (ROOT / item["path"]).read_bytes()
        assert len(content) == item["bytes"]
        assert hashlib.sha256(content).hexdigest() == item["sha256"]


def test_plan_is_networkless_runc_and_gpu_free() -> None:
    plan, _ = fusion_runtime.load_frozen_plan(PLAN)
    argv = plan["build"]["docker_argv_template"]
    assert "--network=none" in argv
    assert "--runtime=runc" in argv
    assert "--env=NVIDIA_VISIBLE_DEVICES=void" in argv
    assert "--env=CUDA_VISIBLE_DEVICES=-1" in argv
    assert not any(token.startswith(("--gpus", "--device", "--runtime=nvidia")) for token in argv)
    assert plan["build"]["uses_gpu"] is False
    assert plan["build"]["runs_inference"] is False


def test_capability_is_static_ready_but_production_blocked() -> None:
    artifacts = {
        "deepstream-parallel-infer": {"sha256": "1" * 64, "size_bytes": 1},
        "libdeepsafe_fusion.so.1": {"sha256": "2" * 64, "size_bytes": 1},
        "fusion-runtime.conf": {"sha256": "3" * 64, "size_bytes": 1},
    }
    probe = {
        "plugin": {
            "exported_c_abi": sorted(fusion_runtime.EXPECTED_EXPORTS),
            "abi": {"unresolved_sonames": []},
        },
        "app": {"abi": {"unresolved_sonames": []}},
    }
    value = fusion_runtime._capability(artifacts, "4" * 64, probe)
    assert value["status"] == "blocked"
    assert value["fusion_plugin_ready"] is True
    assert value["gpu_integration_validated"] is False
    assert value["runtime_ready"] is False
    assert value["features"]["pose_tensor_track_association"] is True
    assert value["features"]["ppe_person_association"] is True
    assert value["fusion_contract"]["missing_ppe_means"] == "unknown"
    assert value["fusion_contract"]["unknown_generates_violation"] is False


def test_tampered_plan_is_rejected_before_parsing(tmp_path: Path) -> None:
    target = tmp_path / "plan.json"
    target.write_bytes(PLAN.read_bytes() + b" ")
    with pytest.raises(fusion_runtime.FusionRuntimeError, match="SHA-256 mismatch"):
        fusion_runtime.load_frozen_plan(target)


def test_export_parser_requires_exact_versioned_c_abi() -> None:
    output = b"\n".join(
        f"0000000000000000 T {name}@@DEEPSAFE_FUSION_1.0".encode()
        for name in sorted(fusion_runtime.EXPECTED_EXPORTS)
    )
    assert fusion_runtime._parse_exports(output) == sorted(fusion_runtime.EXPECTED_EXPORTS)
    with pytest.raises(fusion_runtime.FusionRuntimeError, match="export set drifted"):
        fusion_runtime._parse_exports(output + b"\n0 T deepsafe_fusion_unversioned_v1")


def test_plan_json_is_strict_and_canonical_values_are_boolean() -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    assert value["publication"]["no_overwrite"] is True
    assert value["build"]["uses_gpu"] is False
    assert value["build"]["runs_inference"] is False
