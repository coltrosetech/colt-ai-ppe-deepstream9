from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from deepstream import fusion_runtime_r2


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "deepstream/fusion-runtime-r2-build-plan.json"
REPO = ROOT / "third_party/deepstream_reference_apps"
APP_ROOT = (
    "deepstream_parallel_inference_app/tritonclient/sample/apps/"
    "deepstream-parallel-infer"
)


def test_overlay_and_effective_source_set_are_exact() -> None:
    overlay, digest = fusion_runtime_r2.load_frozen_overlay(PLAN)
    assert digest == fusion_runtime_r2.FROZEN_PLAN_SHA256
    effective, base_digest = fusion_runtime_r2._materialize(overlay, ROOT)
    assert base_digest == fusion_runtime_r2.EXPECTED_BASE_PLAN_SHA256
    paths = [item["path"] for item in effective["local_inputs"]]
    assert len(paths) == 39
    assert paths == sorted(paths)
    assert "deepstream/fusion-r2/build-runtime.sh" in paths
    assert "deepstream/patches/deepsafe-fusion-ds9-app-r2.patch" in paths
    assert "deepstream/fusion/build-runtime.sh" not in paths
    assert "deepstream/patches/deepsafe-fusion-ds9-app.patch" not in paths
    for item in effective["local_inputs"]:
        content = (ROOT / item["path"]).read_bytes()
        assert len(content) == item["bytes"]
        assert hashlib.sha256(content).hexdigest() == item["sha256"]


def test_exact_patch_removes_both_unsafe_openpose_probe_registrations(
    tmp_path: Path,
) -> None:
    for name in ("Makefile", "deepstream_parallel_infer_app.cpp"):
        content = subprocess.run(
            [
                "git",
                "-C",
                str(REPO),
                "show",
                f"9946965e8adb1aa93b1b66983ec4196351c9190c:{APP_ROOT}/{name}",
            ],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        (tmp_path / name).write_bytes(content)
    subprocess.run(
        [
            "git",
            "apply",
            "--no-index",
            "--unsafe-paths",
            f"--directory={tmp_path}",
            str(ROOT / "deepstream/patches/deepsafe-fusion-ds9-app-r2.patch"),
        ],
        check=True,
        cwd=ROOT,
    )
    patched = (tmp_path / "deepstream_parallel_infer_app.cpp").read_bytes()
    assert (
        patched.count(
            b"body_pose_gie_src_pad_buffer_probe, GST_PAD_PROBE_TYPE_BUFFER"
        )
        == 0
    )
    assert patched.count(b"deepsafe_fusion_app_hook_install(") == 1


def test_r2_keeps_network_gpu_and_inference_disabled() -> None:
    overlay, _ = fusion_runtime_r2.load_frozen_overlay(PLAN)
    effective, _ = fusion_runtime_r2._materialize(overlay, ROOT)
    argv = effective["build"]["docker_argv_template"]
    assert "--network=none" in argv
    assert "--runtime=runc" in argv
    assert "--env=NVIDIA_VISIBLE_DEVICES=void" in argv
    assert "--env=CUDA_VISIBLE_DEVICES=-1" in argv
    assert effective["build"]["uses_gpu"] is False
    assert effective["build"]["runs_inference"] is False
    assert argv[-1] == fusion_runtime_r2.EXPECTED_BUILD_SCRIPT


def test_r2_prerequisite_replay_pins_two_removals() -> None:
    result = fusion_runtime_r2.verify_prerequisites()
    assert result["legacy_openpose_probe_removals_pinned"] == 2
    assert result["local_inputs_verified"] == 39
    assert result["uses_gpu"] is False
    assert result["runs_inference"] is False
