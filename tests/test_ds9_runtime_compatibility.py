import copy
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from validation import ds9_runtime_compatibility as gate
from validation import ds9_gpu_smoke as smoke


IMAGE_ID = "sha256:" + "1" * 64
BASE_DIGEST = "sha256:" + "2" * 64
PARSER_SHA = "3" * 64


@pytest.fixture(autouse=True)
def _semantic_gpu_smoke_replay_stub(monkeypatch):
    """This module tests the outer DS9 gate; raw replay has its own suite."""

    monkeypatch.setattr(
        smoke,
        "validate_production_evidence",
        lambda payload, **kwargs: {
            "status": "pass",
            "checks": payload["checks"],
        },
    )


def _project(tmp_path: Path) -> Path:
    for name, relative in gate.RUNTIME_CONTROL_PATHS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"control={name}\n", encoding="utf-8")
    pins = gate.runtime_control_pins(tmp_path)
    manifest = {
        "schema_version": gate.CONTROL_MANIFEST_SCHEMA_VERSION,
        "artifacts": pins,
    }
    path = tmp_path / gate.RUNTIME_CONTROL_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return tmp_path


def _controls(root: Path) -> dict:
    return gate.validate_runtime_control_manifest(root)


def _image(root: Path) -> dict:
    controls = _controls(root)
    labels = {
        gate.LABELS["schema"]: gate.BUILD_LINEAGE_SCHEMA_VERSION,
        gate.LABELS["base_ref"]: f"{gate.DEEPSTREAM_BASE_TAG}@{BASE_DIGEST}",
        gate.LABELS["base_digest"]: BASE_DIGEST,
        gate.LABELS["deepstream"]: gate.DEEPSTREAM_VERSION,
        gate.LABELS["cuda"]: gate.CUDA_VERSION,
        gate.LABELS["tensorrt"]: gate.TENSORRT_VERSION,
        gate.LABELS["gstreamer"]: gate.GSTREAMER_VERSION,
        gate.LABELS["repository"]: gate.DEEPSTREAM_YOLO_REPOSITORY,
        gate.LABELS["commit"]: gate.DEEPSTREAM_YOLO_COMMIT,
        gate.LABELS["tree"]: gate.DEEPSTREAM_YOLO_TREE,
        gate.LABELS["patch_sha256"]: gate.DEEPSTREAM_YOLO_PATCH_SHA256,
        gate.LABELS["upstream_source_sha256"]: gate.DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256,
        gate.LABELS["patched_source_sha256"]: gate.DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256,
        gate.LABELS["patched_tree"]: gate.DEEPSTREAM_YOLO_PATCHED_TREE,
        gate.LABELS[
            "parser_build_makefile_sha256"
        ]: gate.DEEPSTREAM_YOLO_PARSER_BUILD_MAKEFILE_SHA256,
        gate.LABELS[
            "parser_cuda_cubin_architecture"
        ]: gate.DEEPSTREAM_YOLO_PARSER_CUDA_CUBIN_ARCHITECTURE,
        gate.LABELS[
            "parser_cuda_ptx_architecture"
        ]: gate.DEEPSTREAM_YOLO_PARSER_CUDA_PTX_ARCHITECTURE,
        gate.LABELS["parser_cuda_gencode_flags"]: ";".join(
            gate.DEEPSTREAM_YOLO_PARSER_CUDA_GENCODE_FLAGS
        ),
        gate.LABELS[
            "parser_build_command_sha256"
        ]: gate.DEEPSTREAM_YOLO_PARSER_BUILD_COMMAND_SHA256,
        gate.LABELS[
            "parser_post_link_tool_path"
        ]: gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_PATH,
        gate.LABELS[
            "parser_post_link_tool_sha256"
        ]: gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_SHA256,
        gate.LABELS[
            "parser_post_link_tool_version"
        ]: gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_VERSION,
        gate.LABELS[
            "parser_post_link_command"
        ]: gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_COMMAND,
        gate.LABELS[
            "parser_post_link_command_sha256"
        ]: gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_COMMAND_SHA256,
        gate.LABELS["parser_post_link_removed_sections"]: ";".join(
            gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_REMOVED_SECTIONS
        ),
        gate.LABELS["parser_post_link_retained_sections"]: ";".join(
            gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_RETAINED_SECTIONS
        ),
        gate.LABELS["kernel_proof_schema"]: gate.CUDA_KERNEL_PROOF_SCHEMA_VERSION,
        gate.LABELS["parser_sha256"]: PARSER_SHA,
        gate.LABELS["controller_sha256"]: controls["artifacts"][
            "ds9_runtime_compatibility"
        ]["sha256"],
        gate.LABELS["control_manifest_sha256"]: controls["pin"]["sha256"],
        gate.LABELS["dockerignore_sha256"]: controls["artifacts"][
            "docker_build_context_policy"
        ]["sha256"],
    }
    return {
        "requested_image": "deepsafe-deepstream:9.0",
        "resolved_image_id": IMAGE_ID,
        "architecture": "amd64",
        "os": "linux",
        "repo_digests": [],
        "labels": labels,
    }


def _static_probe(root: Path, image: dict) -> dict:
    controls = _controls(root)
    command_names = {
        "deepstream",
        "tensorrt",
        "cuda",
        "gstreamer",
        "readelf",
        "readelf_sections",
        "ldd",
        "nm",
        "cuobjdump_elf",
        "cuobjdump_ptx",
        "cuobjdump_sass",
        "abi_compile",
    }
    commands = {
        name: {"argv": [name], "returncode": 0, "stdout": "ok", "stderr": ""}
        for name in command_names
    }
    commands["nm"] = {
        "argv": gate._nm_command_argv(),
        "returncode": 0,
        "stdout": (
            "000000000008a92e T NvDsInferParseYoloCuda\n"
            "000000000005d034 T NvDsInferYoloCudaEngineGet\n"
        ),
        "stderr": "",
        "stdout_original_bytes": 221132,
        "stdout_original_sha256": (
            "29fad6aded282f961dd75c535812aa89505f66fa2989c059afc36bbbb91890e8"
        ),
        "stdout_original_utf8_decoded": True,
        "stdout_original_within_limit": True,
        "stdout_projection_policy": gate.NM_PROJECTION_POLICY,
        "stdout_projection_complete": True,
        "stdout_projection_match_count": 2,
    }
    commands.update(
        {
            name: {"argv": argv, "returncode": 0, "stdout": "", "stderr": ""}
            for name, argv in gate._nvinfer_static_command_argv().items()
        }
    )
    commands["deepstream_version_manifest"]["stdout"] = (
        "Version: 9.0.0\nDATE: Mon Mar  2 19:07:41 UTC 2026\n"
    )
    commands["nvinfer_realpath"]["stdout"] = gate.EXPECTED_NVINFER_PATH + "\n"
    commands["nvinfer_stat"]["stdout"] = (
        f"81ed|{gate.EXPECTED_NVINFER_BYTES}|755|0|0\n"
    )
    commands["nvinfer_sha256"]["stdout"] = (
        f"{gate.EXPECTED_NVINFER_SHA256} *{gate.EXPECTED_NVINFER_PATH}\n"
    )
    commands["nvinfer_readelf_header"]["stdout"] = """ELF Header:
  Class:                             ELF64
  Data:                              2's complement, little endian
  Type:                              DYN (Shared object file)
  Machine:                           Advanced Micro Devices X86-64
"""
    commands["nvinfer_readelf_dynamic"]["stdout"] = "".join(
        f" 0x0000000000000001 (NEEDED) Shared library: [{library}]\n"
        for library in gate.EXPECTED_NVINFER_NEEDED
    )
    commands["nvinfer_readelf_symbols"]["stdout"] = "".join(
        f"  1: 0000000000001000 12 FUNC GLOBAL DEFAULT 14 {symbol}\n"
        for symbol in gate.NVINFER_DESCRIPTOR_SYMBOLS
    )
    commands["nvinfer_strings"]["stdout"] = (
        "nvinfer plugin\n9.0.0\nnvdsgst_infer\n"
    )
    commands["cuobjdump_elf"] = {
        "argv": [
            "/usr/local/cuda-13.1/bin/cuobjdump",
            "--list-elf",
            gate.PARSER_LIBRARY.as_posix(),
        ],
        "returncode": 0,
        "stdout": "".join(
            "ELF file    "
            f"{index}: libnvdsinfer_custom_impl_Yolo.{index}.sm_86.cubin\n"
            for index in gate.EXPECTED_PARSER_CUDA_ENTRY_INDEXES
        ),
        "stderr": "",
    }
    commands["cuobjdump_ptx"] = {
        "argv": [
            "/usr/local/cuda-13.1/bin/cuobjdump",
            "--list-ptx",
            gate.PARSER_LIBRARY.as_posix(),
        ],
        "returncode": 0,
        "stdout": "".join(
            "PTX file    "
            f"{index}: libnvdsinfer_custom_impl_Yolo.{index}.sm_86.ptx\n"
            for index in gate.EXPECTED_PARSER_CUDA_ENTRY_INDEXES
        ),
        "stderr": "",
    }
    commands["cuobjdump_sass"] = {
        "argv": [
            "/usr/local/cuda-13.1/bin/cuobjdump",
            "--dump-sass",
            gate.PARSER_LIBRARY.as_posix(),
        ],
        "returncode": 0,
        "stdout": "",
        "stderr": "",
    }
    commands["readelf_sections"] = {
        "argv": gate._parser_section_command_argv(),
        "returncode": 0,
        "stdout": """
Section Headers:
  [ 1] .dynsym           DYNSYM          0000000000000000 000100 000180 18   A  4   1  8
  [ 2] .dynstr           STRTAB          0000000000000000 000280 000100 00   A  0   0  1
  [ 3] .text             PROGBITS        0000000000000000 000380 000200 00  AX  0   0 16
""",
        "stderr": "",
    }
    labels = image["labels"]
    lineage = {
        "schema_version": gate.BUILD_LINEAGE_SCHEMA_VERSION,
        "base_ref": labels[gate.LABELS["base_ref"]],
        "base_digest": labels[gate.LABELS["base_digest"]],
        "deepstream_version": gate.DEEPSTREAM_VERSION,
        "cuda_version": gate.CUDA_VERSION,
        "tensorrt_version": gate.TENSORRT_VERSION,
        "gstreamer_version": gate.GSTREAMER_VERSION,
        "deepstream_yolo_repository": gate.DEEPSTREAM_YOLO_REPOSITORY,
        "deepstream_yolo_commit": gate.DEEPSTREAM_YOLO_COMMIT,
        "deepstream_yolo_tree": gate.DEEPSTREAM_YOLO_TREE,
        "deepstream_yolo_patch_sha256": gate.DEEPSTREAM_YOLO_PATCH_SHA256,
        "deepstream_yolo_upstream_source_sha256": gate.DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256,
        "deepstream_yolo_patched_source_sha256": gate.DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256,
        "deepstream_yolo_patched_tree": gate.DEEPSTREAM_YOLO_PATCHED_TREE,
        "deepstream_yolo_parser_build_makefile_sha256": gate.DEEPSTREAM_YOLO_PARSER_BUILD_MAKEFILE_SHA256,
        "deepstream_yolo_parser_cuda_cubin_architecture": gate.DEEPSTREAM_YOLO_PARSER_CUDA_CUBIN_ARCHITECTURE,
        "deepstream_yolo_parser_cuda_ptx_architecture": gate.DEEPSTREAM_YOLO_PARSER_CUDA_PTX_ARCHITECTURE,
        "deepstream_yolo_parser_cuda_gencode_flags": ";".join(
            gate.DEEPSTREAM_YOLO_PARSER_CUDA_GENCODE_FLAGS
        ),
        "deepstream_yolo_parser_build_command_sha256": gate.DEEPSTREAM_YOLO_PARSER_BUILD_COMMAND_SHA256,
        "deepstream_yolo_parser_post_link_tool_path": gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_PATH,
        "deepstream_yolo_parser_post_link_tool_sha256": gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_SHA256,
        "deepstream_yolo_parser_post_link_tool_version": gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_VERSION,
        "deepstream_yolo_parser_post_link_command": gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_COMMAND,
        "deepstream_yolo_parser_post_link_command_sha256": gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_COMMAND_SHA256,
        "deepstream_yolo_parser_post_link_removed_sections": ";".join(
            gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_REMOVED_SECTIONS
        ),
        "deepstream_yolo_parser_post_link_retained_sections": ";".join(
            gate.DEEPSTREAM_YOLO_PARSER_POST_LINK_RETAINED_SECTIONS
        ),
        "cuda_kernel_proof_schema": gate.CUDA_KERNEL_PROOF_SCHEMA_VERSION,
        "parser_sha256": PARSER_SHA,
        "controller_sha256": controls["artifacts"]["ds9_runtime_compatibility"][
            "sha256"
        ],
        "runtime_control_manifest_sha256": controls["pin"]["sha256"],
        "dockerignore_sha256": controls["artifacts"][
            "docker_build_context_policy"
        ]["sha256"],
    }
    facts = {
        "deepstream_app_version": gate.DEEPSTREAM_VERSION,
        "deepstream_sdk_version": gate.DEEPSTREAM_VERSION,
        "tensorrt_version": gate.TENSORRT_VERSION,
        "cuda_version": gate.CUDA_VERSION,
        "gstreamer_version": gate.GSTREAMER_VERSION,
        "parser_sha256": PARSER_SHA,
        "parser_post_link_section_counts": {
            ".symtab": 0,
            ".strtab": 0,
            ".dynsym": 1,
            ".dynstr": 1,
        },
        "readelf_needed": ["libnvinfer.so.10", "libcudart.so.13"],
        "ldd_missing": [],
        "nm_symbol_counts": {name: 1 for name in gate.REQUIRED_SYMBOLS},
        "nm_symbol_projection_invalid_lines": [],
        "nm_original_stdout": {
            "bytes": commands["nm"]["stdout_original_bytes"],
            "sha256": commands["nm"]["stdout_original_sha256"],
            "utf8_decoded": True,
            "within_limit": True,
        },
        "dlsym": {name: True for name in gate.REQUIRED_SYMBOLS},
        "dlsym_error": None,
        "abi_compile_passed": True,
        "sm86_cubin_present": True,
        "sm86_only_cubin_set": True,
        "cubin_elf_entries": [
            {
                "index": index,
                "name": (
                    f"libnvdsinfer_custom_impl_Yolo.{index}.sm_86.cubin"
                ),
                "architecture": 86,
            }
            for index in gate.EXPECTED_PARSER_CUDA_ENTRY_INDEXES
        ],
        "ptx_entries": [
            {
                "index": index,
                "name": f"libnvdsinfer_custom_impl_Yolo.{index}.sm_86.ptx",
                "architecture": 86,
            }
            for index in gate.EXPECTED_PARSER_CUDA_ENTRY_INDEXES
        ],
        "ptx_targets": ["86"],
        "compute86_only_ptx_set": True,
        "forward_compatible_ptx_present": True,
        "build_lineage": lineage,
        "runtime_control_manifest_sha256": controls["pin"]["sha256"],
        "dockerignore_sha256": controls["artifacts"][
            "docker_build_context_policy"
        ]["sha256"],
        "gpu_smoke_harness_sha256": controls["artifacts"]["ds9_gpu_smoke"][
            "sha256"
        ],
        "file_errors": [],
    }
    facts.update(gate._derive_nvinfer_static_facts(commands))
    return {
        "schema_version": gate.STATIC_PROBE_SCHEMA_VERSION,
        "commands": commands,
        "facts": facts,
    }


def _gpu_evidence(root: Path, controls: dict) -> Path:
    path = root / "validation/results/ds9-runtime-compatibility/gpu-smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": gate.GPU_SMOKE_SCHEMA_VERSION,
                "status": "pass",
                "resolved_image_id": IMAGE_ID,
                "runtime_control_manifest_sha256": controls["pin"]["sha256"],
                "checks": {
                    name: "pass" for name in gate.REQUIRED_GPU_SMOKE_CHECKS
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o440)
    return path


def _production_receipt(root: Path) -> tuple[Path, dict, dict]:
    controls = _controls(root)
    image = _image(root)
    gpu = _gpu_evidence(root, controls)
    receipt = gate.create_static_receipt(
        requested_image="deepsafe-deepstream:9.0",
        image=image,
        probe_command=gate.build_static_probe_command(IMAGE_ID),
        probe=_static_probe(root, image),
        project_root=root,
        gpu_smoke_evidence=gpu,
        created_at_utc="2026-07-16T10:00:00Z",
    )
    path = root / gate.DEFAULT_RECEIPT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o440)
    return path, receipt, image


def test_static_probe_command_is_immutable_offline_and_gpu_free():
    command = gate.build_static_probe_command(IMAGE_ID)
    assert command[:2] == ["docker", "run"]
    assert "--pull=never" in command
    assert "--network=none" in command
    assert "--gpus" not in command
    assert command.count(IMAGE_ID) == 1
    with pytest.raises(gate.Ds9CompatibilityError, match="immutable"):
        gate.build_static_probe_command("deepsafe-deepstream:9.0")


def test_nvinfer_static_contract_uses_only_non_loading_artifact_commands(tmp_path):
    root = _project(tmp_path)
    probe = _static_probe(root, _image(root))
    assert "gst_nvinfer" not in probe["commands"]
    assert len(probe["commands"]) == 20
    assert all(
        "gst-inspect-1.0" not in command["argv"]
        for command in probe["commands"].values()
    )
    facts = probe["facts"]
    assert facts["nvinfer_verification_scope"] == "static_binary_metadata_only"
    assert facts["nvinfer_runtime_plugin_load_attempted"] is False
    assert facts["nvinfer_artifact_sha256"] == gate.EXPECTED_NVINFER_SHA256
    assert facts["nvinfer_sdk_manifest_version"] == gate.DEEPSTREAM_VERSION
    assert facts["nvinfer_descriptor_symbol_counts"] == {
        name: 1 for name in gate.NVINFER_DESCRIPTOR_SYMBOLS
    }


def test_nvinfer_binary_sha_tamper_is_rejected_even_if_fact_is_rewritten(tmp_path):
    root = _project(tmp_path)
    image = _image(root)
    probe = _static_probe(root, image)
    probe["commands"]["nvinfer_sha256"]["stdout"] = (
        f"{'9' * 64} *{gate.EXPECTED_NVINFER_PATH}\n"
    )
    probe["facts"].update(gate._derive_nvinfer_static_facts(probe["commands"]))
    with pytest.raises(gate.Ds9CompatibilityError, match="artifact/version/ABI"):
        gate.create_static_receipt(
            requested_image="deepsafe-deepstream:9.0",
            image=image,
            probe_command=gate.build_static_probe_command(IMAGE_ID),
            probe=probe,
            project_root=root,
        )


def test_nvinfer_missing_libcuda_abi_edge_is_rejected_coherently(tmp_path):
    root = _project(tmp_path)
    image = _image(root)
    probe = _static_probe(root, image)
    dynamic = probe["commands"]["nvinfer_readelf_dynamic"]
    dynamic["stdout"] = dynamic["stdout"].replace(
        " 0x0000000000000001 (NEEDED) Shared library: [libcuda.so.1]\n", ""
    )
    probe["facts"].update(gate._derive_nvinfer_static_facts(probe["commands"]))
    with pytest.raises(gate.Ds9CompatibilityError, match="artifact/version/ABI"):
        gate.create_static_receipt(
            requested_image="deepsafe-deepstream:9.0",
            image=image,
            probe_command=gate.build_static_probe_command(IMAGE_ID),
            probe=probe,
            project_root=root,
        )


def test_nvinfer_command_substitution_and_legacy_load_facts_fail_closed(tmp_path):
    root = _project(tmp_path)
    image = _image(root)
    probe = _static_probe(root, image)
    probe["commands"]["nvinfer_strings"]["argv"] = [
        "gst-inspect-1.0",
        "nvinfer",
    ]
    with pytest.raises(gate.Ds9CompatibilityError, match="runtime plugin loading"):
        gate.create_static_receipt(
            requested_image="deepsafe-deepstream:9.0",
            image=image,
            probe_command=gate.build_static_probe_command(IMAGE_ID),
            probe=probe,
            project_root=root,
        )

    probe = _static_probe(root, image)
    probe["facts"]["gst_nvinfer_version"] = gate.DEEPSTREAM_VERSION
    with pytest.raises(gate.Ds9CompatibilityError, match="legacy runtime-loaded"):
        gate.create_static_receipt(
            requested_image="deepsafe-deepstream:9.0",
            image=image,
            probe_command=gate.build_static_probe_command(IMAGE_ID),
            probe=probe,
            project_root=root,
        )


def test_nm_collector_finds_early_exports_before_bounded_projection(monkeypatch):
    required = (
        b"000000000008a92e T NvDsInferParseYoloCuda\n"
        b"000000000005d034 T NvDsInferYoloCudaEngineGet\n"
    )
    # Reproduces the real 221132-byte shape: both exports precede the final
    # 131072 bytes and were therefore absent from the former tail evidence.
    filler_line = b"0000000000000000 T unrelated_export_symbol\n"
    original = b"nm header\n" + required + filler_line * 5000
    assert len(original) > 131072
    assert b"NvDsInferParseYoloCuda" not in original[-131072:]
    assert b"NvDsInferYoloCudaEngineGet" not in original[-131072:]

    def fake_run(argv, **kwargs):
        assert argv == gate._nm_command_argv()
        assert kwargs["text"] is False
        return subprocess.CompletedProcess(argv, 0, original, b"")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    evidence = gate._run_bounded_nm_symbol_command(gate._nm_command_argv())
    assert evidence["stdout"] == required.decode("utf-8")
    assert len(evidence["stdout"].encode("utf-8")) < 4096
    assert evidence["stdout_original_bytes"] == len(original)
    assert evidence["stdout_original_sha256"] == hashlib.sha256(original).hexdigest()
    assert evidence["stdout_original_within_limit"] is True
    assert evidence["stdout_projection_complete"] is True
    assert evidence["stdout_projection_match_count"] == 2


def test_nm_collector_marks_oversized_original_as_incomplete(monkeypatch):
    original = b"x" * (gate.MAX_NM_ORIGINAL_STDOUT_BYTES + 1)

    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, original, b""),
    )
    evidence = gate._run_bounded_nm_symbol_command(gate._nm_command_argv())
    assert evidence["stdout"] == ""
    assert evidence["stdout_original_bytes"] == len(original)
    assert evidence["stdout_original_within_limit"] is False
    assert evidence["stdout_projection_complete"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_nm_missing_or_duplicate_export_fails_closed(tmp_path, mutation):
    root = _project(tmp_path)
    image = _image(root)
    probe = _static_probe(root, image)
    nm = probe["commands"]["nm"]
    lines = nm["stdout"].splitlines()
    if mutation == "missing":
        lines = lines[1:]
    else:
        lines.append(lines[0])
    nm["stdout"] = "\n".join(lines) + "\n"
    nm["stdout_projection_match_count"] = len(lines)
    projection = gate._parse_nm_symbol_projection(nm["stdout"])
    probe["facts"]["nm_symbol_counts"] = projection["counts"]
    probe["facts"]["nm_symbol_projection_invalid_lines"] = projection[
        "invalid_lines"
    ]
    with pytest.raises(gate.Ds9CompatibilityError, match="nm export projection"):
        gate.create_static_receipt(
            requested_image="deepsafe-deepstream:9.0",
            image=image,
            probe_command=gate.build_static_probe_command(IMAGE_ID),
            probe=probe,
            project_root=root,
        )


def test_nm_argv_substitution_fails_closed(tmp_path):
    root = _project(tmp_path)
    image = _image(root)
    probe = _static_probe(root, image)
    probe["commands"]["nm"]["argv"] = [
        "nm",
        "-D",
        "--defined-only",
        "/tmp/substituted-parser.so",
    ]
    with pytest.raises(gate.Ds9CompatibilityError, match="bounded parser nm"):
        gate.create_static_receipt(
            requested_image="deepsafe-deepstream:9.0",
            image=image,
            probe_command=gate.build_static_probe_command(IMAGE_ID),
            probe=probe,
            project_root=root,
        )


def test_nm_incomplete_or_truncated_projection_fails_closed(tmp_path):
    root = _project(tmp_path)
    image = _image(root)
    probe = _static_probe(root, image)
    nm = probe["commands"]["nm"]
    nm["stdout"] = ""
    nm["stdout_projection_complete"] = False
    nm["stdout_projection_match_count"] = 0
    probe["facts"]["nm_symbol_counts"] = {name: 0 for name in gate.REQUIRED_SYMBOLS}
    with pytest.raises(gate.Ds9CompatibilityError, match="bounded parser nm"):
        gate.create_static_receipt(
            requested_image="deepsafe-deepstream:9.0",
            image=image,
            probe_command=gate.build_static_probe_command(IMAGE_ID),
            probe=probe,
            project_root=root,
        )


def test_cuobjdump_list_collector_keeps_complete_bounded_stdout(monkeypatch):
    stdout = b"".join(
        (
            "ELF file    "
            f"{index}: libnvdsinfer_custom_impl_Yolo.{index}.sm_86.cubin\n"
        ).encode("utf-8")
        for index in gate.EXPECTED_PARSER_CUDA_ENTRY_INDEXES
    )
    argv = ["cuobjdump", "--list-elf", gate.PARSER_LIBRARY.as_posix()]

    def fake_run(observed, **kwargs):
        assert observed == argv
        assert kwargs["text"] is False
        return subprocess.CompletedProcess(observed, 0, stdout, b"")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    evidence = gate._run_bounded_cuobjdump_list_command(argv)
    assert evidence["stdout"].encode("utf-8") == stdout
    assert evidence["stderr"] == ""


def test_cuobjdump_list_collector_rejects_oversize_or_non_utf8(monkeypatch):
    argv = ["cuobjdump", "--list-elf", gate.PARSER_LIBRARY.as_posix()]
    outputs = iter(
        [
            b"x" * (gate.MAX_CUOBJDUMP_LIST_BYTES + 1),
            b"\xff",
        ]
    )

    def fake_run(observed, **kwargs):
        return subprocess.CompletedProcess(observed, 0, next(outputs), b"")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    with pytest.raises(gate.Ds9CompatibilityError, match="full-capture limit"):
        gate._run_bounded_cuobjdump_list_command(argv)
    with pytest.raises(gate.Ds9CompatibilityError, match="not UTF-8"):
        gate._run_bounded_cuobjdump_list_command(argv)


@pytest.mark.parametrize(
    ("parser", "line"),
    [
        (
            gate._parse_cuobjdump_elf_entries,
            "warning prefix\nELF file 1: libnvdsinfer_custom_impl_Yolo.1.sm_86.cubin\n",
        ),
        (
            gate._parse_cuobjdump_ptx_entries,
            "warning prefix\nPTX file 1: libnvdsinfer_custom_impl_Yolo.1.sm_86.ptx\n",
        ),
    ],
)
def test_cuobjdump_list_parsers_reject_every_unknown_nonblank_line(parser, line):
    with pytest.raises(gate.Ds9CompatibilityError, match="invalid line"):
        parser(line)


@pytest.mark.parametrize("kind", ["elf", "ptx"])
def test_cuobjdump_list_parsers_reject_duplicate_indexes(kind):
    noun = "ELF" if kind == "elf" else "PTX"
    suffix = "cubin" if kind == "elf" else "ptx"
    parser = (
        gate._parse_cuobjdump_elf_entries
        if kind == "elf"
        else gate._parse_cuobjdump_ptx_entries
    )
    text = "".join(
        f"{noun} file 1: libnvdsinfer_custom_impl_Yolo.1.sm_86.{suffix}\n"
        for _ in range(2)
    )
    with pytest.raises(gate.Ds9CompatibilityError, match="indexes are invalid"):
        parser(text)


@pytest.mark.parametrize("kind", ["cubin", "ptx"])
def test_static_probe_rejects_coherent_mixed_architecture_entry_set(tmp_path, kind):
    root = _project(tmp_path)
    image = _image(root)
    probe = _static_probe(root, image)
    if kind == "cubin":
        command = probe["commands"]["cuobjdump_elf"]
        command["stdout"] = command["stdout"].replace(
            ".4.sm_86.cubin", ".4.sm_75.cubin"
        )
        entries = gate._parse_cuobjdump_elf_entries(command["stdout"])
        probe["facts"]["cubin_elf_entries"] = entries
        probe["facts"]["sm86_cubin_present"] = True
        probe["facts"]["sm86_only_cubin_set"] = False
    else:
        command = probe["commands"]["cuobjdump_ptx"]
        command["stdout"] = command["stdout"].replace(
            ".4.sm_86.ptx", ".4.sm_75.ptx"
        )
        entries = gate._parse_cuobjdump_ptx_entries(command["stdout"])
        probe["facts"]["ptx_entries"] = entries
        probe["facts"]["ptx_targets"] = ["75", "86"]
        probe["facts"]["compute86_only_ptx_set"] = False
        probe["facts"]["forward_compatible_ptx_present"] = True
    with pytest.raises(gate.Ds9CompatibilityError, match="exact four-entry"):
        gate.create_static_receipt(
            requested_image="deepsafe-deepstream:9.0",
            image=image,
            probe_command=gate.build_static_probe_command(IMAGE_ID),
            probe=probe,
            project_root=root,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_static_probe_requires_exact_four_cubin_indexes(tmp_path, mutation):
    root = _project(tmp_path)
    image = _image(root)
    probe = _static_probe(root, image)
    command = probe["commands"]["cuobjdump_elf"]
    if mutation == "missing":
        command["stdout"] = "\n".join(command["stdout"].splitlines()[:-1]) + "\n"
    else:
        command["stdout"] += (
            "ELF file    5: libnvdsinfer_custom_impl_Yolo.5.sm_86.cubin\n"
        )
    entries = gate._parse_cuobjdump_elf_entries(command["stdout"])
    probe["facts"]["cubin_elf_entries"] = entries
    probe["facts"]["sm86_cubin_present"] = True
    probe["facts"]["sm86_only_cubin_set"] = False
    with pytest.raises(gate.Ds9CompatibilityError, match="exact four-entry"):
        gate.create_static_receipt(
            requested_image="deepsafe-deepstream:9.0",
            image=image,
            probe_command=gate.build_static_probe_command(IMAGE_ID),
            probe=probe,
            project_root=root,
        )


def test_dry_run_writes_only_pending_report_without_subprocess(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry-run must not call Docker/subprocess"),
    )
    report = gate.write_pending_report(
        root / "pending.json",
        requested_image="deepsafe-deepstream:9.0",
        project_root=root,
        launch_scope="test",
    )
    assert report["status"] == "pending_static_probe"
    assert report["production_ready"] is False
    assert report["docker_called"] is False
    assert report["gpu_process_started"] is False


def test_static_pass_remains_pending_without_gpu_smoke(tmp_path):
    root = _project(tmp_path)
    image = _image(root)
    receipt = gate.create_static_receipt(
        requested_image="deepsafe-deepstream:9.0",
        image=image,
        probe_command=gate.build_static_probe_command(IMAGE_ID),
        probe=_static_probe(root, image),
        project_root=root,
        created_at_utc="2026-07-16T10:00:00Z",
    )
    assert receipt["status"] == "pending_gpu_smoke"
    assert receipt["production_ready"] is False


def test_pending_and_production_receipts_match_published_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    root = _project(tmp_path)
    image = _image(root)
    pending = gate.create_static_receipt(
        requested_image="deepsafe-deepstream:9.0",
        image=image,
        probe_command=gate.build_static_probe_command(IMAGE_ID),
        probe=_static_probe(root, image),
        project_root=root,
        created_at_utc="2026-07-16T10:00:00Z",
    )
    _path, production, _image_value = _production_receipt(root)
    schema = json.loads(
        (
            gate.PROJECT_ROOT
            / "validation/schemas/ds9-runtime-compatibility-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(pending)
    validator.validate(production)


def test_production_receipt_passes_exact_image_and_live_controls(tmp_path):
    root = _project(tmp_path)
    path, _receipt, image = _production_receipt(root)
    result = gate.require_runtime_compatibility(
        path,
        project_root=root,
        requested_image="deepsafe-deepstream:9.0",
        resolved_image_id=IMAGE_ID,
        now=datetime(2026, 7, 16, 11, 0, tzinfo=timezone.utc),
        image_inspector=lambda _image: image,
    )
    assert result["status"] == "production_ready"
    assert result["resolved_image_id"] == IMAGE_ID


def test_image_substitution_is_rejected(tmp_path):
    root = _project(tmp_path)
    path, _receipt, image = _production_receipt(root)
    with pytest.raises(gate.Ds9CompatibilityError, match="image ID differs"):
        gate.require_runtime_compatibility(
            path,
            project_root=root,
            requested_image="deepsafe-deepstream:9.0",
            resolved_image_id="sha256:" + "9" * 64,
            now=datetime(2026, 7, 16, 11, 0, tzinfo=timezone.utc),
            image_inspector=lambda _image: image,
        )


def test_prevalidation_rejects_requested_tag_repin_before_launch(tmp_path):
    root = _project(tmp_path)
    path, _receipt, image = _production_receipt(root)
    repinned = copy.deepcopy(image)
    repinned["resolved_image_id"] = "sha256:" + "9" * 64
    inspected = []

    def inspect(value):
        inspected.append(value)
        return repinned if value == "deepsafe-deepstream:9.0" else image

    with pytest.raises(gate.Ds9CompatibilityError, match="no longer resolves"):
        gate.prevalidate_runtime_compatibility(
            path,
            project_root=root,
            requested_image="deepsafe-deepstream:9.0",
            now=datetime(2026, 7, 16, 11, 0, tzinfo=timezone.utc),
            image_inspector=inspect,
        )
    assert inspected == ["deepsafe-deepstream:9.0"]


def test_parser_symbol_or_sha_tamper_is_rejected(tmp_path):
    root = _project(tmp_path)
    image = _image(root)
    probe = _static_probe(root, image)
    probe["facts"]["nm_symbol_counts"][gate.REQUIRED_SYMBOLS[0]] = 0
    with pytest.raises(gate.Ds9CompatibilityError, match="symbol counts"):
        gate.create_static_receipt(
            requested_image="deepsafe-deepstream:9.0",
            image=image,
            probe_command=gate.build_static_probe_command(IMAGE_ID),
            probe=probe,
            project_root=root,
        )


def test_sm86_token_in_sass_or_stderr_is_not_cubin_evidence(tmp_path):
    root = _project(tmp_path)
    image = _image(root)
    probe = _static_probe(root, image)
    probe["commands"]["cuobjdump_elf"]["stdout"] = (
        "ELF file    1: libnvdsinfer_custom_impl_Yolo.1.sm_75.cubin\n"
    )
    probe["commands"]["cuobjdump_elf"]["stderr"] = "sm_86.cubin\n"
    probe["commands"]["cuobjdump_sass"]["stdout"] = "code for sm_86\n"
    with pytest.raises(gate.Ds9CompatibilityError, match="cuobjdump"):
        gate.create_static_receipt(
            requested_image="deepsafe-deepstream:9.0",
            image=image,
            probe_command=gate.build_static_probe_command(IMAGE_ID),
            probe=probe,
            project_root=root,
        )


def test_image_lineage_requires_exact_kernel_patch_and_post_patch_tree(tmp_path):
    root = _project(tmp_path)
    image = _image(root)
    image["labels"][gate.LABELS["patch_sha256"]] = "9" * 64
    with pytest.raises(gate.Ds9CompatibilityError, match="patch-sha256"):
        gate.validate_image_lineage(image, controls=_controls(root))


def test_image_lineage_requires_exact_sm86_build_flags_label(tmp_path):
    root = _project(tmp_path)
    image = _image(root)
    image["labels"][gate.LABELS["parser_cuda_gencode_flags"]] = (
        "-gencode=arch=compute_75,code=sm_75"
    )
    with pytest.raises(gate.Ds9CompatibilityError, match="gencode-flags"):
        gate.validate_image_lineage(image, controls=_controls(root))


def test_image_lineage_requires_exact_post_link_tool_sha_label(tmp_path):
    root = _project(tmp_path)
    image = _image(root)
    image["labels"][gate.LABELS["parser_post_link_tool_sha256"]] = "9" * 64
    with pytest.raises(gate.Ds9CompatibilityError, match="post-link-tool-sha256"):
        gate.validate_image_lineage(image, controls=_controls(root))


def test_static_probe_rejects_unstripped_parser_section_table(tmp_path):
    root = _project(tmp_path)
    image = _image(root)
    probe = _static_probe(root, image)
    probe["commands"]["readelf_sections"]["stdout"] += (
        "  [ 4] .symtab           SYMTAB          0000000000000000 000580 "
        "000100 18      5   1  8\n"
    )
    probe["facts"]["parser_post_link_section_counts"][".symtab"] = 1
    with pytest.raises(gate.Ds9CompatibilityError, match="section canonicalization"):
        gate.create_static_receipt(
            requested_image="deepsafe-deepstream:9.0",
            image=image,
            probe_command=gate.build_static_probe_command(IMAGE_ID),
            probe=probe,
            project_root=root,
        )


def test_coherent_control_repin_cannot_cross_immutable_image_labels(tmp_path):
    root = _project(tmp_path)
    path, receipt, original_image = _production_receipt(root)
    os.chmod(path, 0o600)
    changed = root / gate.RUNTIME_CONTROL_PATHS["gpu_guard"]
    changed.write_text("coherently replaced guard\n", encoding="utf-8")
    new_pins = gate.runtime_control_pins(root)
    manifest_path = root / gate.RUNTIME_CONTROL_MANIFEST
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": gate.CONTROL_MANIFEST_SCHEMA_VERSION,
                "artifacts": new_pins,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    new_controls = gate.validate_runtime_control_manifest(root)
    forged = copy.deepcopy(receipt)
    forged["runtime_controls"] = new_controls
    forged["image"]["labels"][gate.LABELS["control_manifest_sha256"]] = new_controls[
        "pin"
    ]["sha256"]
    path.write_text(json.dumps(forged, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o440)
    with pytest.raises(gate.Ds9CompatibilityError, match="live immutable image metadata"):
        gate.require_runtime_compatibility(
            path,
            project_root=root,
            requested_image="deepsafe-deepstream:9.0",
            resolved_image_id=IMAGE_ID,
            now=datetime(2026, 7, 16, 11, 0, tzinfo=timezone.utc),
            image_inspector=lambda _image: original_image,
        )


def test_receipt_permissions_and_expiry_fail_closed(tmp_path):
    root = _project(tmp_path)
    path, _receipt, image = _production_receipt(root)
    os.chmod(path, 0o644)
    with pytest.raises(gate.Ds9CompatibilityError, match="mode 0440"):
        gate.require_runtime_compatibility(
            path,
            project_root=root,
            requested_image="deepsafe-deepstream:9.0",
            resolved_image_id=IMAGE_ID,
            image_inspector=lambda _image: image,
        )
    os.chmod(path, 0o440)
    with pytest.raises(gate.Ds9CompatibilityError, match="not current"):
        gate.require_runtime_compatibility(
            path,
            project_root=root,
            requested_image="deepsafe-deepstream:9.0",
            resolved_image_id=IMAGE_ID,
            now=datetime(2026, 7, 18, 11, 0, tzinfo=timezone.utc),
            image_inspector=lambda _image: image,
        )


def test_symlinked_receipt_is_rejected_even_when_target_is_inside_project(tmp_path):
    root = _project(tmp_path)
    path, _receipt, image = _production_receipt(root)
    target = path.with_name("real-receipt.json")
    path.rename(target)
    path.symlink_to(target.name)
    with pytest.raises(gate.Ds9CompatibilityError, match="cannot contain a symlink"):
        gate.require_runtime_compatibility(
            path,
            project_root=root,
            requested_image="deepsafe-deepstream:9.0",
            resolved_image_id=IMAGE_ID,
            now=datetime(2026, 7, 16, 11, 0, tzinfo=timezone.utc),
            image_inspector=lambda _image: image,
        )
