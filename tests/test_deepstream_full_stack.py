import copy
import configparser
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from deepstream import full_stack


ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, relative: str, content: bytes, *, executable: bool = False) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o755 if executable else 0o644)
    return {
        "host_path": relative,
        "container_path": "/fixture/" + relative,
        "sha256": hashlib.sha256(content).hexdigest(),
        "json_requirements": None,
    }


def _write_json_artifact(
    root: Path,
    relative: str,
    value: dict,
    *,
    schema_version: str,
    status: str = "runtime_ready",
) -> dict:
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    slot = _write(root, relative, content)
    slot["json_requirements"] = {
        "schema_version": schema_version,
        "status": status,
    }
    return slot


def _infer_config(role: str, engine_path: str, *, tensor_meta: bool = False) -> bytes:
    lines = [
        "[property]",
        f"model-engine-file={engine_path}",
        "batch-size=12",
        f"gie-unique-id={full_stack.ROLE_GIE_IDS[role]}",
        "process-mode=1",
        "network-type=0",
    ]
    if tensor_meta:
        lines.append("output-tensor-meta=1")
    lines.extend(["network-mode=2", "", "[class-attrs-all]", "pre-cluster-threshold=0.25", ""])
    return "\n".join(lines).encode()


def _ready_contract(tmp_path: Path) -> Path:
    app = _write(
        tmp_path,
        "runtime/deepstream-parallel-infer",
        b"#!/bin/sh\nexit 99\n",
        executable=True,
    )
    fusion_plugin = _write(
        tmp_path,
        "runtime/libdeepsafe_fusion.so.1",
        b"fixture-fusion-plugin\n",
        executable=True,
    )
    fusion_config = _write(
        tmp_path,
        "runtime/fusion-runtime.conf",
        full_stack._fusion_runtime_config(),
    )
    provenance = _write(
        tmp_path,
        "runtime/build-provenance.json",
        b'{"fixture":"descriptor-bound-publication"}\n',
    )
    capabilities = {
        "schema_version": full_stack.CAPABILITY_SCHEMA_VERSION,
        "status": "runtime_ready",
        "deepstream_version": full_stack.ACTIVE_DEEPSTREAM_VERSION,
        "parallel_pattern": full_stack.PARALLEL_PATTERN,
        "parallel_app_binary_sha256": app["sha256"],
        "fusion_plugin_sha256": fusion_plugin["sha256"],
        "fusion_config_sha256": hashlib.sha256(
            full_stack._fusion_runtime_config()
        ).hexdigest(),
        "build_provenance_sha256": provenance["sha256"],
        "fusion_plugin_ready": True,
        "gpu_integration_validated": True,
        "runtime_ready": True,
        "static_evidence": {
            "legacy_openpose_probe_registered": False,
            "canonical_fusion_probe_install_count": 1,
        },
        "publication_contract": {
            "receipt_schema_version": full_stack.PUBLICATION_RECEIPT_SCHEMA_VERSION,
            "receipt_name": full_stack.PUBLICATION_RECEIPT_NAME,
            "primitive": full_stack.PUBLICATION_PRIMITIVE,
            "directory_identity_bound": True,
            "post_rename_inode_verified": True,
            "descriptor_relative_artifact_replay": True,
            "exact_file_set_verified": True,
            "canonical_path_reopened": True,
        },
        "features": {
            "nvdsmetamux": True,
            "full_frame_camera_batch": True,
            "nvdcf_tracker": True,
            "pose_tensor_track_association": True,
            "ppe_person_association": True,
            "headless_performance": True,
            "component_latency": True,
        },
        "fusion_contract": {
            "abi_version": "0x00010000",
            "canonical_person_gie_id": 1,
            "pose_tensor_gie_id": 2,
            "ppe_object_gie_id": 3,
            "max_sources": 12,
            "missing_ppe_means": "unknown",
            "unknown_generates_violation": False,
            "ambiguous_or_occluded": "unknown_unassociated",
            "fp16_pose_adapter": True,
            "fp32_pose_adapter": True,
        },
    }
    capability_slot = _write_json_artifact(
        tmp_path,
        "runtime/capability-manifest.json",
        capabilities,
        schema_version=full_stack.CAPABILITY_SCHEMA_VERSION,
    )
    runtime_directory = tmp_path / "runtime"
    runtime_directory.chmod(0o755)
    runtime_files = {}
    for name in (
        "build-provenance.json",
        "capability-manifest.json",
        "deepstream-parallel-infer",
        "fusion-runtime.conf",
        "libdeepsafe_fusion.so.1",
    ):
        path = runtime_directory / name
        content = path.read_bytes()
        runtime_files[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "mode": f"{path.stat().st_mode & 0o7777:04o}",
        }
    directory_info = runtime_directory.stat()
    publication_receipt = {
        "schema_version": full_stack.PUBLICATION_RECEIPT_SCHEMA_VERSION,
        "status": "published",
        "publication_id": "fixture-descriptor-bound-r3",
        "destination": "runtime",
        "publication_plan_sha256": "5" * 64,
        "publisher_sha256": "6" * 64,
        "primitive": full_stack.PUBLICATION_PRIMITIVE,
        "directory_identity": {
            "device": directory_info.st_dev,
            "inode": directory_info.st_ino,
        },
        "post_rename_inode_verified": True,
        "descriptor_relative_artifact_replay": True,
        "exact_file_set_verified": True,
        "canonical_path_reopened": True,
        "exact_file_names": sorted(full_stack.PUBLICATION_BUNDLE_NAMES),
        "files": runtime_files,
    }
    publication_receipt_slot = _write_json_artifact(
        tmp_path,
        "runtime/publication-receipt.json",
        publication_receipt,
        schema_version=full_stack.PUBLICATION_RECEIPT_SCHEMA_VERSION,
        status="published",
    )

    artifacts = {}
    for role in full_stack.ROLE_ORDER:
        engine = _write(tmp_path, f"models/{role}.engine", f"{role}-engine\n".encode())
        config = _write(
            tmp_path,
            f"models/{role}.txt",
            _infer_config(role, engine["container_path"], tensor_meta=role == "pose"),
        )
        artifacts[role] = {"engine": engine, "infer_config": config}

    pose_contract = {
        "schema_version": "deepsafe.pose.postprocess-contract/v1",
        "status": "runtime_ready",
        "track_association": {
            "identity_source": "canonical_person_tracker",
            "duplicate_track_id_policy": "fail_closed",
        },
    }
    ppe_contract = {
        "schema_version": "deepsafe.ppe.postprocess-contract/v1",
        "status": "runtime_ready",
        "identity_boundary": {
            "track_id_source": "canonical_ephemeral_person_tracker",
            "biometric_identity": False,
        },
        "association": {
            "duplicate_track_id_policy": "fail_closed",
            "confirmed_tracks_required": True,
        },
        "equipment_observation": {
            "equipment": ["helmet", "hi_vis"],
            "missing_detection_means": "unknown",
            "unknown_generates_violation": False,
        },
    }
    artifacts["pose"]["postprocess_library"] = _write(
        tmp_path, "models/libpose.so", b"pose-runtime-library\n"
    )
    pose_contract["postprocess_library_sha256"] = artifacts["pose"][
        "postprocess_library"
    ]["sha256"]
    artifacts["pose"]["association_contract"] = _write_json_artifact(
        tmp_path,
        "models/pose-association.json",
        pose_contract,
        schema_version=pose_contract["schema_version"],
    )
    artifacts["ppe"]["postprocess_library"] = _write(
        tmp_path, "models/libppe.so", b"ppe-runtime-library\n"
    )
    ppe_contract["postprocess_library_sha256"] = artifacts["ppe"][
        "postprocess_library"
    ]["sha256"]
    artifacts["ppe"]["association_contract"] = _write_json_artifact(
        tmp_path,
        "models/ppe-association.json",
        ppe_contract,
        schema_version=ppe_contract["schema_version"],
    )

    models = {
        "person": {
            "role": "person",
            "gie_unique_id": 1,
            "inference_mode": "full_frame_primary",
            "batch_semantics": "camera_batch",
            "tracker": "nvdcf",
            "metadata_output": "nvds_object_meta",
            "association_target_gie": None,
            "artifacts": artifacts["person"],
        },
        "pose": {
            "role": "pose",
            "gie_unique_id": 2,
            "inference_mode": "full_frame_primary",
            "batch_semantics": "camera_batch",
            "tracker": "none",
            "metadata_output": "nvds_infer_tensor_meta",
            "association_target_gie": 1,
            "artifacts": artifacts["pose"],
        },
        "ppe": {
            "role": "ppe",
            "gie_unique_id": 3,
            "inference_mode": "full_frame_primary",
            "batch_semantics": "camera_batch",
            "tracker": "none",
            "metadata_output": "nvds_object_meta",
            "association_target_gie": 1,
            "artifacts": artifacts["ppe"],
        },
    }
    contract = {
        "schema_version": full_stack.CONTRACT_SCHEMA_VERSION,
        "runtime": {
            "active_deepstream_version": full_stack.ACTIVE_DEEPSTREAM_VERSION,
            "migration_deepstream_version": full_stack.MIGRATION_DEEPSTREAM_VERSION,
            "migration_status": full_stack.MIGRATION_STATUS,
            "parallel_app_binary": app,
            "fusion_plugin": fusion_plugin,
            "capability_manifest": capability_slot,
            "publication_receipt": publication_receipt_slot,
        },
        "limits": {"max_sources": 12, "max_batch_size": 12},
        "topology": {
            "pattern": full_stack.PARALLEL_PATTERN,
            "nvdsmetamux_required": True,
            "metamux_active_pad": "sink_0",
            "metamux_pts_tolerance_us": 60000,
            "headless": True,
            "perf_measurement_interval_seconds": 5,
            "component_latency_measurement": True,
            "streammux_width": 1920,
            "streammux_height": 1080,
            "tracker_library": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
            "tracker_config": "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
        },
        "profiles": {
            "fixture": {
                "input_width": 640,
                "input_height": 640,
                "models": models,
            }
        },
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return path


def _sources(count: int = 12) -> list[str]:
    return [f"file:///fixtures/camera-{index:02d}.mp4" for index in range(count)]


def test_repository_contract_is_honestly_blocked_and_writes_no_runtime_config(tmp_path):
    plan = full_stack.build_plan(
        ROOT / "deepstream/full-stack-contract.json",
        profile_id="640",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=ROOT,
    )

    assert plan["execution_ready"] is False
    assert plan["launch_authorized"] is False
    runtime_checks = {
        item["label"]: item
        for item in plan["artifact_checks"]
        if item["label"].startswith("runtime.")
    }
    assert set(runtime_checks) == {
        "runtime.parallel_app_binary",
        "runtime.fusion_plugin",
        "runtime.capability_manifest",
        "runtime.publication_receipt",
    }
    assert all("-r3/" in item["host_path"] for item in runtime_checks.values())
    assert runtime_checks["runtime.publication_receipt"]["status"] == "ready"
    assert runtime_checks["runtime.publication_receipt"]["expected_sha256"] == (
        "af2a9731886674cd33141d7577c40d47c511f4d595ee1c1595435a7c80c22f7f"
    )
    assert not any(
        blocker.startswith("runtime.publication_receipt:")
        for blocker in plan["readiness_blockers"]
    )
    assert any(blocker.startswith("artifact:pose.engine:unconfigured") for blocker in plan["readiness_blockers"])
    assert any(blocker.startswith("artifact:ppe.engine:unconfigured") for blocker in plan["readiness_blockers"])
    assert any("pose.association_contract:json_status_mismatch" in blocker for blocker in plan["readiness_blockers"])
    assert any(
        blocker.startswith("artifact:ppe.association_contract:")
        for blocker in plan["readiness_blockers"]
    )
    assert plan["rendered_outputs"] == []
    assert plan["launch"] is None
    assert {path.name for path in (tmp_path / "blocked").iterdir()} == {"full-stack-plan.json"}


def test_ready_fixture_renders_official_parallel_topology_and_is_deterministic(tmp_path):
    contract = _ready_contract(tmp_path)
    first = full_stack.build_plan(
        contract,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "first",
        project_root=tmp_path,
    )
    second = full_stack.build_plan(
        contract,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "second",
        project_root=tmp_path,
    )

    assert first == second
    assert first["execution_ready"] is True
    assert first["launch_authorized"] is False
    assert first["sources"]["ids"] == list(range(12))
    assert [branch["gie_unique_id"] for branch in first["topology"]["branches"]] == [1, 2, 3]
    assert all(branch["source_ids"] == list(range(12)) for branch in first["topology"]["branches"])

    yaml_text = (tmp_path / "first/parallel-inference.yml").read_text()
    assert yaml_text.count("primary-gie0:") == 1
    assert yaml_text.count("primary-gie1:") == 1
    assert yaml_text.count("primary-gie2:") == 1
    assert "secondary-gie" not in yaml_text
    assert "gie-unique-id: 1" in yaml_text
    assert "gie-unique-id: 2" in yaml_text
    assert "gie-unique-id: 3" in yaml_text
    assert "tracker0:\n  enable: 1" in yaml_text
    assert "cfg-file-path: /opt/deepsafe/generated/full-stack/tracker.yml" in yaml_text
    assert "tracker1:\n  enable: 0" in yaml_text
    assert "tracker2:\n  enable: 0" in yaml_text
    assert "meta-mux:\n  enable: 1" in yaml_text
    assert "sink0:\n  enable: 1\n  type: 1\n  sync: 0\n  qos: 0" in yaml_text
    assert "enable-perf-measurement: 1" in yaml_text
    tracker = (tmp_path / "first/tracker.yml").read_text()
    assert "tracker:\n  tracker-width: 640\n  tracker-height: 384" in tracker
    assert "ll-lib-file: /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so" in tracker
    assert "ll-config-file: /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml" in tracker

    metamux = (tmp_path / "first/metamux.txt").read_text()
    assert "active-pad=sink_0" in metamux
    for gie_id in (1, 2, 3):
        assert f"src-ids-model-{gie_id}=" + ";".join(map(str, range(12))) in metamux
    fusion = json.loads((tmp_path / "first/fusion-contract.json").read_text())
    assert fusion["pose"]["association_target_gie"] == 1
    assert fusion["ppe"]["required_attributes"] == ["helmet", "hi_vis"]
    fusion_runtime = (tmp_path / "first/fusion-runtime.conf").read_bytes()
    assert fusion_runtime == full_stack._fusion_runtime_config()
    assert first["launch"]["environment"]["DEEPSAFE_FUSION_CONFIG"].endswith(
        "/fusion-runtime.conf"
    )
    assert first["launch"]["environment"]["DEEPSAFE_FUSION_CONFIG_SHA256"] == hashlib.sha256(
        fusion_runtime
    ).hexdigest()
    assert full_stack.verify_persisted_plan(
        tmp_path / "first/full-stack-plan.json", project_root=tmp_path
    ) == first


@pytest.mark.parametrize("count", [0, 13])
def test_source_count_must_be_one_to_twelve(tmp_path, count):
    contract = _ready_contract(tmp_path)
    with pytest.raises(full_stack.FullStackContractError, match="source count"):
        full_stack.build_plan(
            contract,
            profile_id="fixture",
            sources=_sources(count),
            output_dir=tmp_path / "out",
            project_root=tmp_path,
        )


def test_every_branch_has_every_contiguous_source_and_unique_ids(tmp_path):
    contract = _ready_contract(tmp_path)
    plan = full_stack.build_plan(
        contract,
        profile_id="fixture",
        sources=["rtsp://camera.local/live", "file:///fixtures/b.mp4"],
        output_dir=tmp_path / "out",
        project_root=tmp_path,
    )
    assert plan["sources"]["ids"] == [0, 1]
    assert plan["sources"]["batch_size"] == 2
    assert all(branch["source_ids"] == [0, 1] for branch in plan["topology"]["branches"])
    csv_text = (tmp_path / "out/sources.csv").read_text()
    assert "1,4,rtsp://camera.local/live,1,0,0" in csv_text
    assert "1,3,file:///fixtures/b.mp4,1,0,0" in csv_text

    tampered = copy.deepcopy(plan)
    tampered["topology"]["branches"][2]["source_ids"] = [0]
    with pytest.raises(full_stack.FullStackContractError, match="ppe branch topology drifted"):
        full_stack.validate_plan(tampered)


def test_generated_yaml_csv_and_metamux_are_structurally_parseable(tmp_path):
    contract = _ready_contract(tmp_path)
    sources = [
        "rtsp://camera.local/live,name",
        "file:///fixtures/camera%20two.mp4",
    ]
    output = tmp_path / "out"
    full_stack.build_plan(
        contract,
        profile_id="fixture",
        sources=sources,
        output_dir=output,
        project_root=tmp_path,
    )

    parsed_yaml = yaml.safe_load((output / "parallel-inference.yml").read_text())
    assert list(parsed_yaml) == [
        "application",
        "tiled-display",
        "source",
        "sink0",
        "osd",
        "streammux",
        "primary-gie0",
        "branch0",
        "tracker0",
        "primary-gie1",
        "branch1",
        "tracker1",
        "primary-gie2",
        "branch2",
        "tracker2",
        "meta-mux",
        "tests",
    ]
    assert parsed_yaml["streammux"]["batch-size"] == 2
    assert parsed_yaml["branch0"]["src-ids"] == "0;1"
    assert parsed_yaml["branch1"]["src-ids"] == "0;1"
    assert parsed_yaml["branch2"]["src-ids"] == "0;1"

    with (output / "sources.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["uri"] for row in rows] == sources
    assert [row["type"] for row in rows] == ["4", "3"]
    assert all(row["num-sources"] == "1" for row in rows)

    metamux = configparser.ConfigParser(interpolation=None, strict=True)
    metamux.read(output / "metamux.txt")
    assert metamux["property"]["active-pad"] == "sink_0"
    for gie_id in (1, 2, 3):
        assert metamux["group-0"][f"src-ids-model-{gie_id}"] == "0;1"


@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("rtsp://operator:secret@camera.local/live", "userinfo/credentials"),
        ("rtsps://camera.local/live?token=secret", "query/fragment"),
        ("file:///fixtures/camera.mp4#secret", "query/fragment"),
        ("file://remote-host/fixtures/camera.mp4", "authority"),
        ("rtsp://camera.local:invalid/live", "malformed"),
        ("rtsp://[::1/live", "malformed"),
        ("rtsp://camera.local/live stream", "whitespace/control"),
        ("RTSP://camera.local/live", "lowercase"),
        ("rtsp://camera.local/live\\escaped", "RFC-safe"),
        ("rtsp://camera.local/live%ZZ", "RFC-safe"),
    ],
)
def test_persisted_source_uris_cannot_embed_credentials_or_tokens(tmp_path, uri, message):
    contract = _ready_contract(tmp_path)
    with pytest.raises(full_stack.FullStackContractError, match=message):
        full_stack.build_plan(
            contract,
            profile_id="fixture",
            sources=[uri],
            output_dir=tmp_path / "out",
            project_root=tmp_path,
        )


def test_source_uri_must_be_valid_utf8(tmp_path):
    contract = _ready_contract(tmp_path)
    with pytest.raises(full_stack.FullStackContractError, match="valid UTF-8"):
        full_stack.build_plan(
            contract,
            profile_id="fixture",
            sources=["rtsp://camera.local/\ud800"],
            output_dir=tmp_path / "out",
            project_root=tmp_path,
        )


def test_pose_tensor_meta_is_a_hard_config_gate(tmp_path):
    contract_path = _ready_contract(tmp_path)
    contract = json.loads(contract_path.read_text())
    slot = contract["profiles"]["fixture"]["models"]["pose"]["artifacts"]["infer_config"]
    path = tmp_path / slot["host_path"]
    content = path.read_bytes().replace(b"output-tensor-meta=1\n", b"")
    path.write_bytes(content)
    slot["sha256"] = hashlib.sha256(content).hexdigest()
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=tmp_path,
    )
    assert plan["execution_ready"] is False
    assert "pose.infer_config:output_tensor_meta_required" in plan["readiness_blockers"]
    assert {path.name for path in (tmp_path / "blocked").iterdir()} == {"full-stack-plan.json"}


def test_ppe_object_metadata_requires_detector_network_type(tmp_path):
    contract_path = _ready_contract(tmp_path)
    contract = json.loads(contract_path.read_text())
    slot = contract["profiles"]["fixture"]["models"]["ppe"]["artifacts"][
        "infer_config"
    ]
    path = tmp_path / slot["host_path"]
    content = path.read_bytes().replace(b"network-type=0\n", b"network-type=1\n")
    path.write_bytes(content)
    slot["sha256"] = hashlib.sha256(content).hexdigest()
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=tmp_path,
    )
    assert "ppe.infer_config:network-type_mismatch" in plan["readiness_blockers"]
    assert plan["execution_ready"] is False


def test_artifact_hash_drift_blocks_before_any_config_is_rendered(tmp_path):
    contract_path = _ready_contract(tmp_path)
    (tmp_path / "models/ppe.engine").write_bytes(b"drift")
    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=tmp_path,
    )
    assert plan["execution_ready"] is False
    assert "artifact:ppe.engine:sha256_mismatch" in plan["readiness_blockers"]
    assert plan["rendered_outputs"] == []


def test_empty_hash_pinned_artifact_is_not_launch_ready(tmp_path):
    contract_path = _ready_contract(tmp_path)
    contract = json.loads(contract_path.read_text())
    slot = contract["profiles"]["fixture"]["models"]["ppe"]["artifacts"]["engine"]
    path = tmp_path / slot["host_path"]
    path.write_bytes(b"")
    slot["sha256"] = hashlib.sha256(b"").hexdigest()
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=tmp_path,
    )
    assert "artifact:ppe.engine:empty_file" in plan["readiness_blockers"]
    assert plan["execution_ready"] is False


def test_artifact_mutation_during_descriptor_probe_fails_closed(tmp_path, monkeypatch):
    contract_path = _ready_contract(tmp_path)
    raced_path = tmp_path / "models/ppe.engine"
    real_read = full_stack.os.read
    raced = False

    def racing_read(descriptor, size):
        nonlocal raced
        content = real_read(descriptor, size)
        try:
            target = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
        except OSError:
            target = None
        if target == raced_path and not raced:
            raced = True
            raced_path.write_bytes(b"mutated-during-probe\n")
        return content

    monkeypatch.setattr(full_stack.os, "read", racing_read)
    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=tmp_path,
    )
    assert raced is True
    assert "artifact:ppe.engine:changed_during_probe" in plan["readiness_blockers"]
    assert plan["execution_ready"] is False


def test_artifact_symlink_is_never_accepted_as_hash_equivalent(tmp_path):
    contract_path = _ready_contract(tmp_path)
    engine = tmp_path / "models/ppe.engine"
    content = engine.read_bytes()
    target = tmp_path / "outside.engine"
    target.write_bytes(content)
    engine.unlink()
    engine.symlink_to(target)
    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=tmp_path,
    )
    assert plan["execution_ready"] is False
    assert "artifact:ppe.engine:path_escape_or_symlink" in plan["readiness_blockers"]


def test_artifact_hardlink_alias_is_not_accepted_as_an_isolated_snapshot(tmp_path):
    contract_path = _ready_contract(tmp_path)
    engine = tmp_path / "models/ppe.engine"
    alias = tmp_path / "outside-hardlink.engine"
    alias.hardlink_to(engine)
    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=tmp_path,
    )
    assert "artifact:ppe.engine:hardlink_forbidden" in plan["readiness_blockers"]
    assert alias.read_bytes() == engine.read_bytes()


def test_launch_authorization_replays_artifacts_and_never_starts_a_process(tmp_path):
    contract_path = _ready_contract(tmp_path)
    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "authorized",
        project_root=tmp_path,
        authorize_launch=True,
    )
    assert plan["launch_authorized"] is True
    authorization = full_stack.require_launch_ready(
        tmp_path / "authorized/full-stack-plan.json", project_root=tmp_path
    )
    assert authorization["argv"][0] == "/fixture/runtime/deepstream-parallel-infer"
    assert authorization["environment"]["NVDS_ENABLE_LATENCY_MEASUREMENT"] == "1"

    (tmp_path / "models/pose.engine").write_bytes(b"post-plan-drift")
    with pytest.raises(full_stack.LaunchRejected, match="artifact replay differs"):
        full_stack.require_launch_ready(
            tmp_path / "authorized/full-stack-plan.json", project_root=tmp_path
        )


def test_replay_rejects_hash_equivalent_generated_output_symlink(tmp_path):
    contract_path = _ready_contract(tmp_path)
    output = tmp_path / "authorized"
    full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=output,
        project_root=tmp_path,
        authorize_launch=True,
    )
    generated = output / "parallel-inference.yml"
    external = tmp_path / "same-bytes.yml"
    external.write_bytes(generated.read_bytes())
    generated.unlink()
    generated.symlink_to(external)

    with pytest.raises(full_stack.LaunchRejected, match="regular and not a symlink"):
        full_stack.require_launch_ready(
            output / "full-stack-plan.json", project_root=tmp_path
        )


def test_replay_rejects_symlinked_plan_even_when_bytes_are_identical(tmp_path):
    contract_path = _ready_contract(tmp_path)
    output = tmp_path / "authorized"
    full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=output,
        project_root=tmp_path,
        authorize_launch=True,
    )
    plan_path = output / "full-stack-plan.json"
    external = tmp_path / "same-plan.json"
    external.write_bytes(plan_path.read_bytes())
    plan_path.unlink()
    plan_path.symlink_to(external)

    with pytest.raises(full_stack.LaunchRejected, match="regular and not a symlink"):
        full_stack.require_launch_ready(plan_path, project_root=tmp_path)


def test_launch_binary_argv_is_bound_to_the_runtime_artifact(tmp_path):
    contract_path = _ready_contract(tmp_path)
    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "authorized",
        project_root=tmp_path,
        authorize_launch=True,
    )
    plan["launch"]["argv"][0] = "/tmp/unpinned-launcher"
    with pytest.raises(full_stack.FullStackContractError, match="launch config argv drifted"):
        full_stack.validate_plan(plan)


def test_replay_rejects_alternate_generator_even_with_identical_bytes_and_hash(tmp_path):
    contract_path = _ready_contract(tmp_path)
    output = tmp_path / "ready"
    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=output,
        project_root=tmp_path,
    )
    alternate = tmp_path / "alternate-generator.py"
    alternate.write_bytes(Path(full_stack.__file__).read_bytes())
    plan["generator"] = {
        "path": alternate.name,
        "sha256": hashlib.sha256(alternate.read_bytes()).hexdigest(),
    }
    (output / "full-stack-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(full_stack.FullStackContractError, match="not this full-stack"):
        full_stack.verify_persisted_plan(
            output / "full-stack-plan.json", project_root=tmp_path
        )


@pytest.mark.parametrize(
    "malicious_path",
    [
        "deepstream/full_stack.py\nlaunch: injected",
        "deepstream\\full_stack.py",
        "//network-like/full_stack.py",
        "deepstream/\ud800/full_stack.py",
    ],
)
def test_plan_contract_and_generator_paths_reject_control_or_ambiguous_text(
    tmp_path, malicious_path
):
    contract_path = _ready_contract(tmp_path)
    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "ready",
        project_root=tmp_path,
    )
    for field in ("contract", "generator"):
        tampered = copy.deepcopy(plan)
        tampered[field]["path"] = malicious_path
        with pytest.raises(
            full_stack.FullStackContractError,
            match="normalized path|valid UTF-8",
        ):
            full_stack.validate_plan(tampered)


def test_replay_recomputes_semantic_readiness_instead_of_trusting_plan_boolean(tmp_path):
    contract_path = _ready_contract(tmp_path)
    output = tmp_path / "ready"
    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=output,
        project_root=tmp_path,
    )

    contract = json.loads(contract_path.read_text())
    slot = contract["profiles"]["fixture"]["models"]["pose"]["artifacts"]["infer_config"]
    config_path = tmp_path / slot["host_path"]
    content = config_path.read_bytes().replace(b"output-tensor-meta=1\n", b"")
    config_path.write_bytes(content)
    new_sha = hashlib.sha256(content).hexdigest()
    slot["sha256"] = new_sha
    contract_bytes = (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode()
    contract_path.write_bytes(contract_bytes)

    plan["contract"]["sha256"] = hashlib.sha256(contract_bytes).hexdigest()
    check = next(item for item in plan["artifact_checks"] if item["label"] == "pose.infer_config")
    check["expected_sha256"] = new_sha
    check["observed_sha256"] = new_sha
    check["size_bytes"] = len(content)
    (output / "full-stack-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(full_stack.FullStackContractError, match="live readiness derivation"):
        full_stack.verify_persisted_plan(
            output / "full-stack-plan.json", project_root=tmp_path
        )


def test_validate_plan_rejects_false_ready_artifact_check_with_null_pins(tmp_path):
    contract_path = _ready_contract(tmp_path)
    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "ready",
        project_root=tmp_path,
    )
    check = plan["artifact_checks"][0]
    check.update(
        {
            "host_path": None,
            "container_path": None,
            "expected_sha256": None,
            "observed_sha256": None,
            "size_bytes": None,
            "status": "ready",
        }
    )
    with pytest.raises(full_stack.FullStackContractError, match="all-null form"):
        full_stack.validate_plan(plan)


def test_dry_run_plan_is_never_a_launch_authorization(tmp_path):
    contract = _ready_contract(tmp_path)
    full_stack.build_plan(
        contract,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "dry",
        project_root=tmp_path,
    )
    with pytest.raises(full_stack.LaunchRejected, match="dry-run plan"):
        full_stack.require_launch_ready(
            tmp_path / "dry/full-stack-plan.json", project_root=tmp_path
        )


def test_missing_repository_pose_ppe_rejects_authorization_without_subprocess(tmp_path):
    result = full_stack.main(
        [
            "--contract",
            str(ROOT / "deepstream/full-stack-contract.json"),
            "--project-root",
            str(ROOT),
            "--profile",
            "960",
            "--output-dir",
            str(tmp_path / "blocked"),
            "--source",
            "file:///fixtures/camera.mp4",
            "--authorize-launch",
        ]
    )
    assert result == 3
    plan = json.loads((tmp_path / "blocked/full-stack-plan.json").read_text())
    assert plan["execution_ready"] is False
    assert plan["safety"] == {
        "docker_called": False,
        "gpu_process_started": False,
        "inference_started": False,
    }


def test_deepstream_91_cannot_replace_active_90_contract(tmp_path):
    contract_path = _ready_contract(tmp_path)
    contract = json.loads(contract_path.read_text())
    contract["runtime"]["active_deepstream_version"] = "9.1"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    with pytest.raises(full_stack.FullStackContractError, match="must remain DeepStream 9.0.0"):
        full_stack.load_contract(contract_path)


def test_contract_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"deepsafe.deepstream-full-stack-contract/v1",'
        '"schema_version":"duplicate"}\n'
    )
    with pytest.raises(full_stack.FullStackContractError, match="strict contract JSON"):
        full_stack.load_contract(path)


@pytest.mark.parametrize(
    ("slot_path", "requirements", "message"),
    [
        (
            ("runtime", "capability_manifest"),
            None,
            "must exactly require status runtime_ready",
        ),
        (
            ("runtime", "publication_receipt"),
            {
                "schema_version": full_stack.PUBLICATION_RECEIPT_SCHEMA_VERSION,
                "status": "runtime_ready",
            },
            "must exactly require status published",
        ),
        (
            (
                "profiles",
                "fixture",
                "models",
                "pose",
                "artifacts",
                "association_contract",
            ),
            {
                "schema_version": full_stack.POSE_ASSOCIATION_SCHEMA_VERSION,
                "status": "core_tested_model_not_integrated",
            },
            "must exactly require status runtime_ready",
        ),
        (
            ("profiles", "fixture", "models", "person", "artifacts", "engine"),
            {"schema_version": "fake-engine-json/v1", "status": "runtime_ready"},
            "must be null for a non-JSON artifact",
        ),
    ],
)
def test_contract_cannot_weaken_exact_json_readiness_policy(
    tmp_path, slot_path, requirements, message
):
    contract_path = _ready_contract(tmp_path)
    contract = json.loads(contract_path.read_text())
    slot = contract
    for key in slot_path:
        slot = slot[key]
    slot["json_requirements"] = requirements
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    with pytest.raises(full_stack.FullStackContractError, match=message):
        full_stack.load_contract(contract_path)


@pytest.mark.parametrize(
    ("slot_path", "binding_key", "blocker"),
    [
        (
            ("runtime", "capability_manifest"),
            "parallel_app_binary_sha256",
            "runtime.capability_manifest:parallel_app_binary_sha256_mismatch",
        ),
        (
            ("runtime", "capability_manifest"),
            "fusion_plugin_sha256",
            "runtime.capability_manifest:fusion_plugin_sha256_mismatch",
        ),
        (
            (
                "profiles",
                "fixture",
                "models",
                "pose",
                "artifacts",
                "association_contract",
            ),
            "postprocess_library_sha256",
            "pose.association_contract:postprocess_library_sha256_mismatch",
        ),
        (
            (
                "profiles",
                "fixture",
                "models",
                "ppe",
                "artifacts",
                "association_contract",
            ),
            "postprocess_library_sha256",
            "ppe.association_contract:postprocess_library_sha256_mismatch",
        ),
    ],
)
def test_runtime_json_attestations_are_bound_to_their_exact_binary_or_library(
    tmp_path, slot_path, binding_key, blocker
):
    contract_path = _ready_contract(tmp_path)
    contract = json.loads(contract_path.read_text())
    slot = contract
    for key in slot_path:
        slot = slot[key]
    artifact_path = tmp_path / slot["host_path"]
    value = json.loads(artifact_path.read_text())
    value[binding_key] = "0" * 64
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    artifact_path.write_bytes(content)
    slot["sha256"] = hashlib.sha256(content).hexdigest()
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=tmp_path,
    )
    assert blocker in plan["readiness_blockers"]
    assert plan["execution_ready"] is False


@pytest.mark.parametrize(
    ("field", "value", "status"),
    [
        ("fusion_plugin_ready", False, "runtime_fusion_plugin_not_ready"),
        ("gpu_integration_validated", False, "runtime_gpu_integration_not_validated"),
        ("runtime_ready", False, "runtime_ready_claim_mismatch"),
    ],
)
def test_static_fusion_success_cannot_authorize_gpu_runtime(
    tmp_path, field, value, status
):
    contract_path = _ready_contract(tmp_path)
    contract = json.loads(contract_path.read_text())
    slot = contract["runtime"]["capability_manifest"]
    artifact_path = tmp_path / slot["host_path"]
    capability = json.loads(artifact_path.read_text())
    capability[field] = value
    content = (json.dumps(capability, indent=2, sort_keys=True) + "\n").encode()
    artifact_path.write_bytes(content)
    slot["sha256"] = hashlib.sha256(content).hexdigest()
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=tmp_path,
    )
    assert f"artifact:runtime.capability_manifest:{status}" in plan["readiness_blockers"]
    assert plan["execution_ready"] is False
    assert plan["launch"] is None


@pytest.mark.parametrize(
    ("static_evidence", "status"),
    [
        (None, "runtime_static_evidence_missing"),
        (
            {
                "legacy_openpose_probe_registered": True,
                "canonical_fusion_probe_install_count": 1,
            },
            "runtime_legacy_openpose_probe_registered",
        ),
        (
            {
                "legacy_openpose_probe_registered": False,
                "canonical_fusion_probe_install_count": 0,
            },
            "runtime_canonical_fusion_probe_count_mismatch",
        ),
        (
            {
                "legacy_openpose_probe_registered": False,
                "canonical_fusion_probe_install_count": True,
            },
            "runtime_canonical_fusion_probe_count_mismatch",
        ),
    ],
)
def test_runtime_requires_exact_single_fusion_hook_and_no_legacy_pose_probe(
    tmp_path, static_evidence, status
):
    contract_path = _ready_contract(tmp_path)
    contract = json.loads(contract_path.read_text())
    slot = contract["runtime"]["capability_manifest"]
    artifact_path = tmp_path / slot["host_path"]
    capability = json.loads(artifact_path.read_text())
    capability["static_evidence"] = static_evidence
    content = (json.dumps(capability, indent=2, sort_keys=True) + "\n").encode()
    artifact_path.write_bytes(content)
    slot["sha256"] = hashlib.sha256(content).hexdigest()
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=tmp_path,
    )
    assert f"artifact:runtime.capability_manifest:{status}" in plan["readiness_blockers"]
    assert plan["execution_ready"] is False
    assert plan["launch"] is None


def test_byte_identical_runtime_directory_substitution_is_rejected_by_receipt_inode(
    tmp_path,
):
    contract_path = _ready_contract(tmp_path)
    original = tmp_path / "runtime"
    held = tmp_path / "runtime-held"
    original.rename(held)
    shutil.copytree(held, original, copy_function=shutil.copy2)

    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=tmp_path,
    )
    checks = {item["label"]: item for item in plan["artifact_checks"]}
    assert checks["runtime.publication_receipt"]["status"] == "ready"
    assert (
        "runtime.publication_receipt:directory_identity_mismatch"
        in plan["readiness_blockers"]
    )
    assert plan["execution_ready"] is False
    assert plan["launch"] is None


@pytest.mark.parametrize("drift", ["hash", "mode", "extra_file"])
def test_later_consumer_descriptor_replay_rejects_bundle_drift(tmp_path, drift):
    contract_path = _ready_contract(tmp_path)
    runtime = tmp_path / "runtime"
    provenance = runtime / "build-provenance.json"
    if drift == "hash":
        provenance.write_bytes(b'{"fixture":"post-publication-drift"}\n')
        expected = "descriptor_replay_mismatch:build-provenance.json"
    elif drift == "mode":
        provenance.chmod(0o600)
        expected = "descriptor_replay_mismatch:build-provenance.json"
    else:
        (runtime / "unreceipted.txt").write_text("not authorized\n")
        expected = "exact_file_set_mismatch"

    plan = full_stack.build_plan(
        contract_path,
        profile_id="fixture",
        sources=_sources(),
        output_dir=tmp_path / "blocked",
        project_root=tmp_path,
    )
    assert f"runtime.publication_receipt:{expected}" in plan["readiness_blockers"]
    assert plan["execution_ready"] is False
    assert plan["launch"] is None


def test_blocked_output_invalidates_plan_and_removes_stale_launchable_configs(tmp_path):
    output = tmp_path / "blocked"
    output.mkdir()
    (output / "parallel-inference.yml").write_text("stale\n")
    (output / "sources.csv").write_text("stale\n")
    (output / "full-stack-plan.json").write_text('{"execution_ready":true}\n')

    plan = full_stack.build_plan(
        ROOT / "deepstream/full-stack-contract.json",
        profile_id="640",
        sources=_sources(),
        output_dir=output,
        project_root=ROOT,
    )
    assert plan["execution_ready"] is False
    assert {path.name for path in output.iterdir()} == {"full-stack-plan.json"}
    assert json.loads((output / "full-stack-plan.json").read_text())["execution_ready"] is False


def test_blocked_cleanup_unlinks_stale_symlink_without_touching_target(tmp_path):
    output = tmp_path / "blocked"
    output.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("do-not-touch\n")
    (output / "parallel-inference.yml").symlink_to(target)

    full_stack.build_plan(
        ROOT / "deepstream/full-stack-contract.json",
        profile_id="640",
        sources=_sources(),
        output_dir=output,
        project_root=ROOT,
    )
    assert target.read_text() == "do-not-touch\n"
    assert not (output / "parallel-inference.yml").exists()


@pytest.mark.parametrize(
    "container_path",
    [
        "/opt/deepsafe/generated\nsink9:",
        "/opt/deepsafe/generated\t# injected",
        "/opt/deepsafe/generated:tag",
        "//network-like/generated",
    ],
)
def test_container_paths_cannot_inject_yaml_structure(tmp_path, container_path):
    contract = _ready_contract(tmp_path)
    with pytest.raises(full_stack.FullStackContractError, match="injection-safe"):
        full_stack.build_plan(
            contract,
            profile_id="fixture",
            sources=_sources(),
            output_dir=tmp_path / "out",
            container_output_dir=container_path,
            project_root=tmp_path,
        )


def test_contract_container_artifact_path_cannot_inject_yaml(tmp_path):
    contract_path = _ready_contract(tmp_path)
    contract = json.loads(contract_path.read_text())
    contract["profiles"]["fixture"]["models"]["pose"]["artifacts"][
        "infer_config"
    ]["container_path"] = "/models/pose.txt\nmeta-mux:"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    with pytest.raises(full_stack.FullStackContractError, match="injection-safe"):
        full_stack.load_contract(contract_path)


def test_endurance_current_is_never_a_valid_output_directory(tmp_path):
    contract = _ready_contract(tmp_path)
    forbidden_root = tmp_path / "validation/results/endurance/current"
    with pytest.raises(full_stack.FullStackContractError, match="endurance/current"):
        full_stack.build_plan(
            contract,
            profile_id="fixture",
            sources=_sources(),
            output_dir=forbidden_root / "nested",
            project_root=tmp_path,
        )


def test_container_output_directory_cannot_be_filesystem_root(tmp_path):
    contract = _ready_contract(tmp_path)
    with pytest.raises(full_stack.FullStackContractError, match="absolute container path"):
        full_stack.build_plan(
            contract,
            profile_id="fixture",
            sources=_sources(),
            output_dir=tmp_path / "out",
            container_output_dir="/",
            project_root=tmp_path,
        )


def test_schemas_are_strict_and_pin_runtime_topology_versions():
    contract_schema = json.loads(
        (ROOT / "validation/schemas/deepstream-full-stack-contract-v1.schema.json").read_text()
    )
    plan_schema = json.loads(
        (ROOT / "validation/schemas/deepstream-full-stack-plan-v1.schema.json").read_text()
    )
    assert contract_schema["additionalProperties"] is False
    assert plan_schema["additionalProperties"] is False
    assert contract_schema["properties"]["runtime"]["properties"]["active_deepstream_version"]["const"] == "9.0.0"
    assert contract_schema["properties"]["runtime"]["properties"]["migration_deepstream_version"]["const"] == "9.1"
    assert contract_schema["properties"]["topology"]["properties"]["nvdsmetamux_required"]["const"] is True
    assert plan_schema["properties"]["artifact_checks"]["minItems"] == 14


def test_direct_file_cli_help_bootstraps_project_imports():
    result = subprocess.run(
        [sys.executable, str(ROOT / "deepstream/full_stack.py"), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "fail-closed DeepStream 9.0" in result.stdout
