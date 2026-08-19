from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from deepstream import fusion_runtime_r3


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "deepstream/fusion-runtime-r3-publication-plan.json"


def _fixture_files() -> dict[str, tuple[bytes, int]]:
    return {
        "build-provenance.json": (b'{"fixture":"provenance"}\n', 0o440),
        "capability-manifest.json": (b'{"fixture":"capability"}\n', 0o440),
        "deepstream-parallel-infer": (b"fixture-app\n", 0o550),
        "fusion-runtime.conf": (b"[fusion]\nfixture=true\n", 0o440),
        "libdeepsafe_fusion.so.1": (b"fixture-plugin\n", 0o550),
    }


def _make_test_plan(destination: str) -> tuple[dict, str]:
    plan, digest = fusion_runtime_r3.load_frozen_plan(PLAN)
    value = copy.deepcopy(plan)
    value["publication"]["destination"] = destination
    return value, digest


def _writable_rmtree(path: Path) -> None:
    if not path.exists():
        return
    for directory, names, _files in os.walk(path):
        Path(directory).chmod(0o750)
        for name in names:
            (Path(directory) / name).chmod(0o750)
    shutil.rmtree(path)


def test_frozen_r3_plan_and_r2_source_pins_are_exact() -> None:
    plan, digest = fusion_runtime_r3.load_frozen_plan(PLAN)
    assert digest == fusion_runtime_r3.FROZEN_PLAN_SHA256
    assert plan["publication"]["primitive"] == fusion_runtime_r3.PRIMITIVE
    assert plan["source_publication"]["build_plan_sha256"] == (
        "d84fae28b35514c86c3536c693174e9083f5ea35d27aaa1ef28109bb6ca2ed39"
    )
    assert [item["name"] for item in plan["source_publication"]["files"]] == sorted(
        fusion_runtime_r3.SOURCE_NAMES
    )
    for item in plan["source_publication"]["files"]:
        path = ROOT / plan["source_publication"]["path"] / item["name"]
        content = path.read_bytes()
        assert len(content) == item["bytes"]
        assert hashlib.sha256(content).hexdigest() == item["sha256"]


def test_r3_prerequisite_replay_is_gpu_network_and_inference_free() -> None:
    result = fusion_runtime_r3.verify_prerequisites()
    assert result["source_files_verified"] == 5
    for key in (
        "artifact_rebuild",
        "container_started",
        "network_used",
        "gpu_device_injected",
        "gpu_queried",
        "inference_executed",
        "model_or_engine_loaded",
        "endurance_executed",
    ):
        assert result[key] is False


def test_descriptor_bound_publisher_replays_exact_inode_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    plan, digest = _make_test_plan("published/r3-valid")
    files = _fixture_files()
    builder_sha = "a" * 64
    published, receipt_content = fusion_runtime_r3._publish(
        plan,
        project_root=tmp_path,
        plan_sha256=digest,
        builder_sha256=builder_sha,
        files=files,
    )
    try:
        receipt = json.loads(receipt_content)
        info = published.stat()
        assert receipt["directory_identity"] == {
            "device": info.st_dev,
            "inode": info.st_ino,
        }
        assert sorted(path.name for path in published.iterdir()) == sorted(
            fusion_runtime_r3.PUBLISHED_NAMES
        )
        assert (published / fusion_runtime_r3.RECEIPT_NAME).read_bytes() == receipt_content
        with pytest.raises(
            fusion_runtime_r3.FusionRuntimeR3Error, match="already exists"
        ):
            fusion_runtime_r3._publish(
                plan,
                project_root=tmp_path,
                plan_sha256=digest,
                builder_sha256=builder_sha,
                files=files,
            )
        assert (published / "deepstream-parallel-infer").read_bytes() == files[
            "deepstream-parallel-infer"
        ][0]
    finally:
        _writable_rmtree(tmp_path / "published")


def test_stage_name_substitution_between_fsync_and_rename_is_rejected_by_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, digest = _make_test_plan("published/r3-raced")
    real_rename = fusion_runtime_r3._rename_noreplace

    def substitute_then_rename(parent_fd: int, source_name: str, target_name: str) -> None:
        os.rename(
            source_name,
            source_name + ".held-original",
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.mkdir(source_name, mode=0o700, dir_fd=parent_fd)
        real_rename(parent_fd, source_name, target_name)

    monkeypatch.setattr(fusion_runtime_r3, "_rename_noreplace", substitute_then_rename)
    with pytest.raises(
        fusion_runtime_r3.FusionRuntimeR3Error,
        match="destination inode differs from held stage FD",
    ):
        fusion_runtime_r3._publish(
            plan,
            project_root=tmp_path,
            plan_sha256=digest,
            builder_sha256="b" * 64,
            files=_fixture_files(),
        )

    raced = tmp_path / "published/r3-raced"
    held = [
        path
        for path in (tmp_path / "published").iterdir()
        if path.name.endswith(".held-original")
    ]
    try:
        assert raced.is_dir()
        assert list(raced.iterdir()) == []
        assert len(held) == 1
        assert (held[0] / fusion_runtime_r3.RECEIPT_NAME).is_file()
    finally:
        _writable_rmtree(tmp_path / "published")


def test_repository_r3_receipt_replays_exact_publisher_and_directory_pins() -> None:
    result = fusion_runtime_r3.inspect_publication()
    assert result["publication_plan_sha256"] == (
        "5b14391a10f1bcd2a7500abb165de5d45002845df4262db720158ad6bb9a9d8e"
    )
    assert result["publisher_sha256"] == (
        "cc3e62d4b604f75a053dc6e4dc27177c48460ab5ebdfd5ceb820971d430887a0"
    )
    assert result["provenance_sha256"] == (
        "ebe9b5158c1ba49ffdd50ba348ebe456cc7c28da2b6bd87a6aef9256f6ddb4a8"
    )
    assert result["capability_manifest_sha256"] == (
        "4ce3163678f5f3a8a730b5b8e3a9ace6c4f6b4171858a476b2ad77d4c59ae841"
    )
    assert result["publication_receipt_sha256"] == (
        "af2a9731886674cd33141d7577c40d47c511f4d595ee1c1595435a7c80c22f7f"
    )
    assert result["directory_identity"] == {"device": 66310, "inode": 24264047}
    assert result["post_rename_inode_verified"] is True
    assert result["descriptor_relative_artifact_replay"] is True
    assert result["exact_file_set_verified"] is True
    assert result["canonical_path_reopened"] is True
    assert result["gpu_integration_validated"] is False
    assert result["runtime_ready"] is False
