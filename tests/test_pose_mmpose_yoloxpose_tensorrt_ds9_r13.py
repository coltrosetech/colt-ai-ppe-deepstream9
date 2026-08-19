from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13 as lane
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13_container as worker


STAMP = "2026-07-18T12:00:00+00:00"


def test_fresh_import_and_plan_build_never_load_gpu_modules_or_spawn_processes() -> None:
    code = r'''
import builtins, subprocess
real_import = builtins.__import__
blocked = {"tensorrt", "cuda", "pynvml", "pycuda"}
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("GPU module import: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
def forbidden(*args, **kwargs):
    raise AssertionError("process launch during import/prepare")
subprocess.Popen = forbidden
subprocess.run = forbidden
subprocess.check_output = forbidden
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13 as lane
plan = lane.build_plan(prepared_at_utc="2026-07-18T12:00:00+00:00")
assert plan["claim_boundary"]["gpu_executed"] is False
print(plan["fingerprint_sha256"])
'''
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=lane.ROOT,
        check=False, capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(completed.stdout.strip()) == 64


def test_plan_binds_exact_r12c_runtime_profiles_and_contracts() -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
    assert plan["upstream_r12c"]["run_receipt"]["sha256"] == lane.R12C_RECEIPT_SHA256
    assert plan["upstream_r12c"]["run_receipt"]["fingerprint_sha256"] == lane.R12C_FINGERPRINT
    assert plan["runtime"]["tensorrt"] == "10.14.1.48"
    assert plan["runtime"]["deepstream"] == "9.0.0"
    assert plan["runtime"]["image_id"] == lane.IMAGE_ID
    assert plan["profiles"]["640"]["input"]["min"] == [1, 3, 640, 640]
    assert plan["profiles"]["960"]["input"]["max"] == [12, 3, 960, 960]
    assert plan["model_contract"]["outputs"][0]["shape"] == ["B", 100, 5]
    assert plan["model_contract"]["outputs"][1]["shape"] == ["B", 100, 17, 3]
    assert plan["adapter_contract"]["second_nms"] is False
    assert plan["fingerprint_sha256"] == lane.fingerprint(plan)


def test_plan_stage_order_and_dependencies_are_exact() -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
    assert tuple(plan["stage_order"]) == lane.STAGES
    assert plan["execution_contract"]["stage_dependencies"] == {
        key: list(value) for key, value in lane.PREDECESSORS.items()
    }
    assert lane.STAGES.index("tensorrt_fp16_640") < lane.STAGES.index("tensorrt_fp16_960")
    assert lane.STAGES.index("numerical_parity_640") < lane.STAGES.index("numerical_parity_960")
    assert lane.STAGES.index("deepstream_tensor_meta_640") < lane.STAGES.index("deepstream_tensor_meta_960")


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["model_contract"]["outputs"].reverse(),
        lambda value: value["model_contract"]["outputs"][0].__setitem__("shape", ["B", 99, 5]),
        lambda value: value["model_contract"].__setitem__("graph_non_max_suppression_nodes", 2),
        lambda value: value["model_contract"].__setitem__("integer_output_dtype", True),
        lambda value: value["deepstream_contract"].__setitem__("second_nms", True),
    ],
)
def test_strict_plan_schema_rejects_output_swap_shape_duplicate_nms_and_int_dtype(mutator) -> None:
    changed = copy.deepcopy(lane.build_plan(prepared_at_utc=STAMP))
    mutator(changed)
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.PoseR13Error, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)


def test_plan_schema_rejects_unknown_property() -> None:
    changed = lane.build_plan(prepared_at_utc=STAMP)
    changed["silent_fallback"] = True
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.PoseR13Error, match="Additional properties"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)


def test_prepare_verify_are_no_overwrite_and_cpu_only(monkeypatch, tmp_path: Path) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU preparation must not invoke subprocess/Docker/GPU")

    monkeypatch.setattr(lane.subprocess if hasattr(lane, "subprocess") else subprocess, "run", forbidden)
    output = tmp_path / "plan.json"
    plan = lane.prepare_plan(output, prepared_at_utc=STAMP)
    assert lane.verify_plan(output, expected_fingerprint=plan["fingerprint_sha256"]) == plan
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.raises(Exception, match="already exists"):
        lane.prepare_plan(output, prepared_at_utc=STAMP)


def test_verify_rejects_stale_plan_hash_and_pinned_source_drift(tmp_path: Path) -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
    stale = copy.deepcopy(plan)
    stale["implementation"]["bridge_source"]["sha256"] = "0" * 64
    stale["fingerprint_sha256"] = lane.fingerprint(stale)
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(lane.PoseR13Error, match="semantic/file-pin replay differs"):
        lane.verify_plan(path, expected_fingerprint=stale["fingerprint_sha256"])
    with pytest.raises(lane.PoseR13Error, match="external plan fingerprint differs"):
        lane.verify_plan(path, expected_fingerprint="f" * 64)


def test_stale_upstream_receipt_is_rejected_before_plan(monkeypatch) -> None:
    original = lane.file_pin

    def stale_pin(path: Path):
        value = original(path)
        if path == lane.R12C_RECEIPT:
            value["sha256"] = "0" * 64
        return value

    monkeypatch.setattr(lane, "file_pin", stale_pin)
    with pytest.raises(lane.PoseR13Error, match="receipt file pin differs"):
        lane.verify_upstream()


def test_receipt_intent_and_commit_builders_match_strict_schemas() -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
    file_value = {"path": "validation/results/unit.bin", "bytes": 7, "sha256": "a" * 64}
    plan_pin = {"path": "models/pose/unit-plan.json", "bytes": 11, "sha256": "b" * 64}
    worker_pin = {"path": "validation/results/worker.json", "bytes": 13, "sha256": "c" * 64, "fingerprint_sha256": "d" * 64}
    receipt = lane.build_receipt(
        plan=plan, plan_pin=plan_pin, stage="tensorrt_fp16_640",
        run_id="pose-unit-r13", worker_pin=worker_pin,
        worker_outputs=[file_value], primary_published=file_value,
        predecessors=[], numerical=None, created_at_utc=STAMP,
    )
    receipt_source = {"path": "validation/results/source-receipt.json", "bytes": 17, "sha256": "e" * 64}
    receipt_published = {**receipt_source, "path": "validation/results/published-receipt.json", "fingerprint_sha256": receipt["fingerprint_sha256"]}
    intent = lane.build_intent(
        plan=plan, plan_pin=plan_pin, stage="tensorrt_fp16_640",
        run_id="pose-unit-r13", primary_source=file_value,
        primary_published=file_value, receipt_source=receipt_source,
        receipt_published=receipt_published, prepared_at_utc=STAMP,
    )
    intent_pin = {"path": "validation/results/intent.json", "bytes": 19, "sha256": "f" * 64, "fingerprint_sha256": intent["fingerprint_sha256"]}
    commit = lane.build_commit(
        plan=plan, plan_pin=plan_pin, stage="tensorrt_fp16_640",
        run_id="pose-unit-r13", intent_pin=intent_pin,
        primary_pin=file_value, receipt_pin=receipt_published,
        worker_pin=worker_pin, predecessors=[], committed_at_utc=STAMP,
    )
    assert receipt["fingerprint_sha256"] == lane.fingerprint(receipt)
    assert intent["fingerprint_sha256"] == lane.fingerprint(intent)
    assert commit["fingerprint_sha256"] == lane.fingerprint(commit)


def test_docker_command_is_exact_foreground_offline_readonly_and_fp16_plan(tmp_path: Path) -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
    command = lane.render_docker_command(
        plan=plan, stage="tensorrt_fp16_640", output_dir=tmp_path,
        cidfile=tmp_path / "cid",
    )
    assert command[:5] == ["docker", "run", "--rm", "--pull=never", "--network=none"]
    assert "--read-only" in command
    assert "--gpus=device=0" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges:true" in command
    assert lane.IMAGE_ID in command and lane.IMAGE_REFERENCE not in command
    assert "-d" not in command and "--detach" not in command
    assert command[-1] == lane.UPSTREAM[640]["onnx"][2]
    wrapped = lane.render_lease_command(command)
    assert wrapped[-len(command):] == command


def test_show_command_construction_does_not_execute(monkeypatch, tmp_path: Path) -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
    invoked = []
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: invoked.append(True))
    fake_engine = tmp_path / "model.engine"
    fake_engine.write_bytes(b"future-engine")
    monkeypatch.setattr(lane, "engine_path", lambda _profile: fake_engine)
    command = lane.render_docker_command(
        plan=plan, stage="deepstream_tensor_meta_960", output_dir=tmp_path,
        cidfile=tmp_path / "cid", config=lane.CONFIGS[960],
    )
    assert invoked == []
    assert "bridge" in command
    assert "--bridge-source-sha256" in command


def _parity_arrays(profile: int) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for prefix, batch in (("b1", 1), ("b12", 12)):
        dets = np.zeros((batch, 100, 5), dtype=np.float32)
        keypoints = np.zeros((batch, 100, 17, 3), dtype=np.float32)
        dets[:, 0, :] = np.asarray([1, 2, profile / 2, profile / 2, 0.75], dtype=np.float32)
        keypoints[:, 0, :, 0] = np.arange(17, dtype=np.float32)
        keypoints[:, 0, :, 1] = 10.0
        keypoints[:, 0, :, 2] = 0.5
        values[f"{prefix}_dets"] = dets
        values[f"{prefix}_keypoints"] = keypoints
    return values


def test_numerical_parity_accepts_exact_b1_b12_and_reports_all_fields(tmp_path: Path) -> None:
    values = _parity_arrays(640)
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(reference, **values)
    np.savez(candidate, **values)
    result = lane.compare_numerical(640, reference, candidate)
    assert result["passed"] is True and result["samples"] == 13
    assert result["same_index"] is True and result["invalid_rows_exact_zero"] is True


@pytest.mark.parametrize("mutation", ["output_swap", "shape", "score", "keypoint", "invalid_nonzero"])
def test_numerical_parity_fail_closed_negative_cases(tmp_path: Path, mutation: str) -> None:
    reference_values = _parity_arrays(640)
    candidate_values = {key: value.copy() for key, value in reference_values.items()}
    if mutation == "output_swap":
        candidate_values["b1_dets"] = candidate_values.pop("b1_keypoints")
    elif mutation == "shape":
        candidate_values["b12_keypoints"] = candidate_values["b12_keypoints"][:, :99]
    elif mutation == "score":
        candidate_values["b1_dets"][0, 0, 4] += 0.1
    elif mutation == "keypoint":
        candidate_values["b12_keypoints"][0, 0, 0, 0] += 10
    else:
        candidate_values["b1_dets"][0, 99, 0] = 1.0
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(reference, **reference_values)
    np.savez(candidate, **candidate_values)
    with pytest.raises(lane.PoseR13Error, match="parity|shape"):
        lane.compare_numerical(640, reference, candidate)


def test_bridge_contract_and_sources_bind_nvds_batch_source_frame_and_no_second_nms() -> None:
    contract = json.loads(lane.BRIDGE_CONTRACT.read_text(encoding="utf-8"))
    assert contract["deepstream9_integration"]["association"] == ["batch_id", "pad_index", "source_id", "frame_num"]
    assert contract["canonical_mapping"]["second_nms"] is False
    header = lane.BRIDGE_HEADER.read_text(encoding="utf-8")
    source = lane.BRIDGE_SOURCE.read_text(encoding="utf-8")
    tests = lane.BRIDGE_TESTS.read_text(encoding="utf-8")
    assert "pack_nvds_batch_meta" in header and "NvDsBatchMeta" in header
    assert "NVDSINFER_TENSOR_OUTPUT_META" in source
    assert "AssociationMismatch" in source
    assert "rejects_source_or_frame_swap_without_partial_output" in tests
    assert "any_late_failure_discards_all_prior_canonical_rows" in tests


def test_packer_preserves_class0_bbox_confidence_keypoints_and_has_no_nms() -> None:
    source = lane.PACKER_SOURCE.read_text(encoding="utf-8")
    assert "config.implicit_class_id" in source
    assert "kImplicitPersonClassId" in source
    assert "keypoint_offset" in source
    assert "pack_mmdeploy_outputs" in source
    assert "nms" not in source.lower()


def test_container_worker_import_has_no_gpu_or_docker_side_effects() -> None:
    source = lane.WORKER.read_text(encoding="utf-8")
    prefix = source[: source.index("def main(")]
    assert "if __name__ == \"__main__\":" in source
    assert "subprocess.run(" not in prefix
    assert "nvidia-smi" not in prefix
    assert '"--fp16"' in source and '"--noTF32"' in source
    assert "for batch in (1, 12):" in source


def test_worker_binding_inventory_rejects_integer_pose_outputs() -> None:
    class Mode:
        INPUT = "input"
        OUTPUT = "output"

    class Trt:
        TensorIOMode = Mode
        float32 = "f32"
        float16 = "f16"
        int32 = "i32"
        int64 = "i64"

    class Engine:
        num_io_tensors = 3
        names = ["input", "dets", "keypoints"]

        def get_tensor_name(self, index): return self.names[index]
        def get_tensor_mode(self, name): return Mode.INPUT if name == "input" else Mode.OUTPUT
        def get_tensor_dtype(self, name): return "f32" if name == "input" else "i32"
        def get_tensor_profile_shape(self, _name, _index): return ((1, 3, 640, 640), (12, 3, 640, 640), (12, 3, 640, 640))

    with pytest.raises(worker.PoseWorkerR13Error, match="integer dtype forbidden"):
        worker.binding_inventory(Engine(), Trt, 640)


def test_atomic_publication_survives_source_name_swap(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane.atomic, "ROOT", tmp_path)
    source = tmp_path / "source.bin"
    source.write_bytes(b"accepted")
    descriptor, source_pin = lane.atomic.open_held_source(source)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"attacker")
    source.rename(tmp_path / "old-source.bin")
    replacement.rename(source)
    destination = tmp_path / "published.bin"
    expected = {"path": "published.bin", "bytes": source_pin["bytes"], "sha256": source_pin["sha256"]}
    try:
        assert lane.atomic.publish_held_fd(descriptor, destination, expected, allow_existing_exact=False) == expected
    finally:
        os.close(descriptor)
    assert destination.read_bytes() == b"accepted"


def test_commit_is_last_and_partial_members_require_recovery(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    monkeypatch.setattr(lane.atomic, "ROOT", tmp_path)
    sources = {}
    descriptors = {}
    for role, content in (("primary_artifact", b"artifact"), ("execution_receipt", b"receipt")):
        path = tmp_path / f"source-{role}"
        path.write_bytes(content)
        descriptor, pin = lane.atomic.open_held_source(path)
        descriptors[role] = descriptor
        sources[role] = pin
    commit_source = tmp_path / "source-commit"
    commit_source.write_bytes(b"commit")
    commit_descriptor, commit_source_pin = lane.atomic.open_held_source(commit_source)
    intent = {
        "members": [
            {"role": "primary_artifact", "published": {**sources["primary_artifact"], "path": "primary.bin"}},
            {"role": "execution_receipt", "published": {**sources["execution_receipt"], "path": "receipt.json"}},
        ],
        "commit_destination": "commit.json",
    }
    commit_pin = {**commit_source_pin, "path": "commit.json"}
    original = lane.atomic.publish_held_fd
    calls = []

    def fail_second(descriptor, destination, expected, *, allow_existing_exact):
        calls.append(destination.name)
        if destination.name == "receipt.json":
            raise RuntimeError("injected partial")
        return original(descriptor, destination, expected, allow_existing_exact=allow_existing_exact)

    monkeypatch.setattr(lane.atomic, "publish_held_fd", fail_second)
    try:
        with pytest.raises(RuntimeError, match="partial"):
            lane.publish_transaction(intent=intent, member_descriptors=descriptors, commit_descriptor=commit_descriptor, commit_pin=commit_pin, recovery=False)
        assert (tmp_path / "primary.bin").exists()
        assert not (tmp_path / "commit.json").exists()
        monkeypatch.setattr(lane.atomic, "publish_held_fd", original)
        result = lane.publish_transaction(intent=intent, member_descriptors=descriptors, commit_descriptor=commit_descriptor, commit_pin=commit_pin, recovery=True)
        assert result["status"] == "committed"
        assert (tmp_path / "receipt.json").exists() and (tmp_path / "commit.json").exists()
    finally:
        for descriptor in descriptors.values(): os.close(descriptor)
        os.close(commit_descriptor)


def test_bounded_process_timeout_kills_term_ignoring_group(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane.atomic, "PROCESS_TERM_GRACE_SECONDS", 1)
    command = [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(30)"]
    with pytest.raises(Exception, match="timed out"):
        lane.atomic._run_bounded_process_group(command, log_path=tmp_path / "timeout.log", timeout_seconds=1)
    assert (tmp_path / "timeout.log").read_text(encoding="utf-8").startswith("ready")


def test_cpu_stub_bridge_build_and_all_ten_cases(tmp_path: Path) -> None:
    include = tmp_path / "include"
    (include / "gst").mkdir(parents=True)
    (include / "gst/gst.h").write_text(
        """#pragma once
#include <cstddef>
using gboolean=int; inline constexpr int TRUE=1; inline constexpr int FALSE=0;
struct GList{void* data; GList* next;};
inline GList* g_list_append(GList* head, void* data){auto* n=new GList{data,nullptr}; if(!head)return n; auto* p=head; while(p->next)p=p->next; p->next=n; return head;}
inline void g_list_free(GList* p){while(p){auto* n=p->next; delete p; p=n;}}
""", encoding="utf-8")
    (include / "gstnvdsinfer.h").write_text(
        """#pragma once
#include <cstdint>
#include <gst/gst.h>
enum NvDsInferDataType { FLOAT=0, HALF=1, INT8=2, INT32=3 };
struct NvDsInferDims { unsigned int numDims=0; unsigned int d[8]{}; unsigned int numElements=0; };
struct NvDsInferLayerInfo { NvDsInferDataType dataType=FLOAT; NvDsInferDims inferDims{}; int bindingIndex=0; const char* layerName=nullptr; void* buffer=nullptr; int isInput=0; };
struct NvDsInferNetworkInfo { unsigned int width=0,height=0,channels=0; };
struct NvDsInferTensorMeta { unsigned int unique_id=0,num_output_layers=0; NvDsInferLayerInfo* output_layers_info=nullptr; void** out_buf_ptrs_host=nullptr; void** out_buf_ptrs_dev=nullptr; unsigned int gpu_id=0; void* priv_data=nullptr; NvDsInferNetworkInfo network_info{}; gboolean maintain_aspect_ratio=FALSE; gboolean symmetric_padding=FALSE; };
""", encoding="utf-8")
    (include / "nvdsmeta.h").write_text(
        """#pragma once
#include <cstdint>
#include <gst/gst.h>
using NvDsMetaList=GList; inline constexpr std::uint32_t NVDSINFER_TENSOR_OUTPUT_META=4097;
struct NvDsBaseMeta{std::uint32_t meta_type=0;};
struct NvDsUserMeta{NvDsBaseMeta base_meta{}; void* user_meta_data=nullptr;};
struct NvDsFrameMeta{unsigned int batch_id=0,pad_index=0,source_id=0; std::uint64_t frame_num=0; NvDsMetaList* frame_user_meta_list=nullptr;};
struct NvDsBatchMeta{unsigned int num_frames_in_batch=0; NvDsMetaList* frame_meta_list=nullptr;};
""", encoding="utf-8")
    binary = tmp_path / "bridge-tests"
    command = [
        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
        f"-I{include}", f"-I{lane.ADAPTER / 'include'}",
        f"-I{lane.ROOT / 'models/pose/postprocess/include'}",
        f"-I{lane.ROOT / 'models/pose/postprocess/tests'}",
        str(lane.PACKER_SOURCE), str(lane.BRIDGE_SOURCE), str(lane.DECODER_SOURCE),
        str(lane.BRIDGE_TESTS), "-o", str(binary),
    ]
    compiled = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run([str(binary)], check=False, capture_output=True, text=True, timeout=60)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "10/10 tests passed" in executed.stdout
