from __future__ import annotations

import copy
import json
import struct
import subprocess
from pathlib import Path

import pytest

from validation.site_distance_attestation_v1 import (
    ENGINE_EVENT_SEQUENCE,
    ENGINE_LOG_PREFIX,
    ENGINE_SCHEMA_VERSION,
    EXPECTED_CUDA_VERSION,
    EXPECTED_DEEPSTREAM_VERSION,
    EXPECTED_ENGINE_CREATE_FUNCTION,
    EXPECTED_PARSE_FUNCTION,
    EXPECTED_TENSORRT_VERSION,
    SiteDistanceAttestationError,
    canonical_sha256,
    file_pin,
    finalize_engine_load_attestation,
    inspect_onnx,
    inspect_onnx_pair,
    inspect_parser,
    seal_receipt,
    validate_engine_load_attestation,
    validate_engine_load_attestation_pair,
    validate_onnx_attestation,
    validate_parser_attestation,
)


def _varint(value: int) -> bytes:
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number: int, wire: int, value: bytes | int) -> bytes:
    tag = _varint((number << 3) | wire)
    if wire == 0:
        assert isinstance(value, int)
        return tag + _varint(value)
    assert wire == 2 and isinstance(value, bytes)
    return tag + _varint(len(value)) + value


def _message(number: int, value: bytes) -> bytes:
    return _field(number, 2, value)


def _text(number: int, value: str) -> bytes:
    return _field(number, 2, value.encode())


def _dimension(value: int | str) -> bytes:
    return _field(1, 0, value) if isinstance(value, int) else _text(2, value)


def _value_info(name: str, dtype: int, dimensions: list[int | str]) -> bytes:
    shape = b"".join(_message(1, _dimension(item)) for item in dimensions)
    tensor_type = _field(1, 0, dtype) + _message(2, shape)
    type_proto = _message(1, tensor_type)
    return _text(1, name) + _message(2, type_proto)


def _tiny_onnx(*, height: int | str = "height", width: int | str = "width", dtype: int = 1) -> bytes:
    node = (
        _text(1, "input")
        + _text(2, "output")
        + _text(3, "head")
        + _text(4, "Identity")
    )
    graph = (
        _message(1, node)
        + _text(2, "tiny_graph")
        + _message(11, _value_info("input", dtype, ["batch", 3, height, width]))
        + _message(12, _value_info("output", 1, ["batch", "anchors", 6]))
    )
    opset = _field(2, 0, 18)
    return (
        _field(1, 0, 10)
        + _text(2, "unit-test")
        + _text(3, "1")
        + _message(7, graph)
        + _message(8, opset)
    )


def _tensor_initializer(
    name: str, dtype: int, dimensions: list[int], raw_data: bytes
) -> bytes:
    return (
        b"".join(_field(1, 0, item) for item in dimensions)
        + _field(2, 0, dtype)
        + _text(8, name)
        + _field(9, 2, raw_data)
    )


def _tiny_static_pair_model(
    profile: int,
    *,
    weight: float = 1.25,
    reshape_extent: int | None = None,
    expose_weight_as_input: bool = False,
    conv_group: int = 1,
) -> bytes:
    weight_tensor = _tensor_initializer("weight", 1, [1], struct.pack("<f", weight))
    shape_tensor = _tensor_initializer(
        "reshape_shape",
        7,
        [3],
        struct.pack("<3q", -1, 4, reshape_extent or profile // 32),
    )
    conv = (
        _text(1, "input")
        + _text(1, "weight")
        + _text(2, "conv_output")
        + _text(3, "conv")
        + _text(4, "Conv")
        + _message(
            5,
            _text(1, "group")
            + _field(3, 0, conv_group)
            + _field(20, 0, 2),
        )
    )
    reshape = (
        _text(1, "conv_output")
        + _text(1, "reshape_shape")
        + _text(2, "output")
        + _text(3, "reshape")
        + _text(4, "Reshape")
    )
    graph = (
        _message(1, conv)
        + _message(1, reshape)
        + _text(2, "paired_graph")
        + _message(5, weight_tensor)
        + _message(5, shape_tensor)
        + _message(11, _value_info("input", 1, ["batch", 3, profile, profile]))
        + (
            _message(11, _value_info("weight", 1, [1]))
            if expose_weight_as_input
            else b""
        )
        + _message(12, _value_info("output", 1, ["batch", 4, "anchors"]))
    )
    return (
        _field(1, 0, 10)
        + _text(2, "unit-test")
        + _text(3, "1")
        + _message(7, graph)
        + _message(8, _field(2, 0, 18))
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _checkpoint_triplet(
    root: Path, content: bytes = b"one-shared-checkpoint"
) -> tuple[Path, Path, Path]:
    paths = (
        root / "canonical.pt",
        root / "checkpoint-640.pt",
        root / "checkpoint-960.pt",
    )
    for path in paths:
        path.write_bytes(content)
    return paths


def test_onnx_dynamic_model_proves_both_profiles_and_head_metadata(tmp_path: Path) -> None:
    model = tmp_path / "dynamic.onnx"
    model.write_bytes(_tiny_onnx())

    receipt = inspect_onnx(
        model,
        project_root=tmp_path,
        created_at_utc="2026-07-16T00:00:00Z",
    )

    export = receipt["exports"][0]
    assert export["input"]["name"] == "input"
    assert export["input"]["dtype"] == "FLOAT32"
    assert export["input"]["dimensions"][2]["parameter"] == "height"
    assert [item["profile"] for item in export["input"]["declared_profile_compatibility"]] == [640, 960]
    assert export["outputs"][0]["dimensions"][1]["kind"] == "dynamic"
    assert export["heads"] == [
        {
            "output_name": "output",
            "producer_node_index": 0,
            "producer_name": "head",
            "operator_type": "Identity",
            "domain": "",
            "input_names": ["input"],
        }
    ]
    assert receipt["safety"] == {
        "cpu_inference_executed": False,
        "docker_called": False,
        "gpu_accessed": False,
    }


def test_onnx_static_single_profile_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / "static-640.onnx"
    model.write_bytes(_tiny_onnx(height=640, width=640))
    with pytest.raises(SiteDistanceAttestationError, match="inspect_onnx_pair"):
        inspect_onnx(model, project_root=tmp_path)


@pytest.mark.parametrize(
    "content, message",
    [
        (_tiny_onnx()[:-1], "truncated"),
        (_tiny_onnx(dtype=99), "unsupported ONNX tensor dtype"),
        (b"\x00", "field number zero"),
    ],
)
def test_onnx_malformed_wire_or_dtype_is_rejected(
    tmp_path: Path, content: bytes, message: str
) -> None:
    model = tmp_path / "bad.onnx"
    model.write_bytes(content)
    with pytest.raises(SiteDistanceAttestationError, match=message):
        inspect_onnx(model, project_root=tmp_path)


def test_onnx_receipt_tamper_and_live_model_change_are_rejected(tmp_path: Path) -> None:
    model = tmp_path / "dynamic.onnx"
    model.write_bytes(_tiny_onnx())
    receipt = inspect_onnx(model, project_root=tmp_path)

    raw_tamper = copy.deepcopy(receipt)
    raw_tamper["exports"][0]["input"]["name"] = "tampered"
    with pytest.raises(SiteDistanceAttestationError, match="fingerprint mismatch"):
        validate_onnx_attestation(raw_tamper, project_root=tmp_path, verify_live=False)

    extra = copy.deepcopy(receipt)
    extra["exports"][0]["input"]["unexpected"] = True
    extra = seal_receipt(extra)
    with pytest.raises(SiteDistanceAttestationError, match="unexpected|unknown fields"):
        validate_onnx_attestation(extra, project_root=tmp_path, verify_live=False)

    model.write_bytes(_tiny_onnx() + b"\x08\x01")
    with pytest.raises(SiteDistanceAttestationError, match="live artifact"):
        validate_onnx_attestation(receipt, project_root=tmp_path)


def test_paired_static_onnx_requires_shared_checkpoint_and_equal_parameters(
    tmp_path: Path,
) -> None:
    model_640 = tmp_path / "640.onnx"
    model_960 = tmp_path / "960.onnx"
    checkpoint, checkpoint_640, checkpoint_960 = _checkpoint_triplet(tmp_path)
    model_640.write_bytes(_tiny_static_pair_model(640))
    model_960.write_bytes(_tiny_static_pair_model(960))

    receipt = inspect_onnx_pair(
        model_640,
        model_960,
        checkpoint,
        checkpoint_640,
        checkpoint_960,
        project_root=tmp_path,
        created_at_utc="2026-07-16T00:00:00Z",
    )
    assert receipt["mode"] == "paired_static_shared_checkpoint"
    assert receipt["lineage"]["trainable_initializer_count"] == 1
    assert receipt["lineage"]["all_source_checkpoint_contents_identical"] is True
    checkpoint_pins = receipt["lineage"]["source_checkpoints"]
    assert set(checkpoint_pins) == {"canonical", "profile_640", "profile_960"}
    assert len({pin["path"] for pin in checkpoint_pins.values()}) == 3
    assert len(
        {(pin["bytes"], pin["sha256"]) for pin in checkpoint_pins.values()}
    ) == 1
    with pytest.raises(SiteDistanceAttestationError, match="distinct checkpoint paths"):
        inspect_onnx_pair(
            model_640,
            model_960,
            checkpoint,
            checkpoint,
            checkpoint_960,
            project_root=tmp_path,
        )
    assert receipt["lineage"]["resolution_derived_exceptions"] == [
        {
            "name": "reshape_shape",
            "classification": "reshape_linear_or_area_scaling_640_to_960",
            "uses": [
                {
                    "operator_type": "Reshape",
                    "input_index": 1,
                    "role": "non_parameter",
                }
            ],
            "exports": receipt["lineage"]["resolution_derived_exceptions"][0][
                "exports"
            ],
        }
    ]

    checkpoint_960.write_bytes(b"different-checkpoint")
    with pytest.raises(SiteDistanceAttestationError, match="not identical"):
        inspect_onnx_pair(
            model_640,
            model_960,
            checkpoint,
            checkpoint_640,
            checkpoint_960,
            project_root=tmp_path,
        )
    checkpoint_960.write_bytes(b"one-shared-checkpoint")

    model_960.write_bytes(_tiny_static_pair_model(960, weight=9.0))
    with pytest.raises(SiteDistanceAttestationError, match="trainable initializer"):
        inspect_onnx_pair(
            model_640,
            model_960,
            checkpoint,
            checkpoint_640,
            checkpoint_960,
            project_root=tmp_path,
        )

    model_960.write_bytes(_tiny_static_pair_model(960, conv_group=2))
    with pytest.raises(SiteDistanceAttestationError, match="graph/opset"):
        inspect_onnx_pair(
            model_640,
            model_960,
            checkpoint,
            checkpoint_640,
            checkpoint_960,
            project_root=tmp_path,
        )


def test_paired_static_rejects_non_derived_exception_and_live_lineage_forgery(
    tmp_path: Path,
) -> None:
    model_640 = tmp_path / "640.onnx"
    model_960 = tmp_path / "960.onnx"
    checkpoint, checkpoint_640, checkpoint_960 = _checkpoint_triplet(
        tmp_path, b"checkpoint"
    )
    model_640.write_bytes(_tiny_static_pair_model(640))
    model_960.write_bytes(_tiny_static_pair_model(960))
    receipt = inspect_onnx_pair(
        model_640,
        model_960,
        checkpoint,
        checkpoint_640,
        checkpoint_960,
        project_root=tmp_path,
    )
    forged = copy.deepcopy(receipt)
    forged["lineage"]["resolution_derived_exceptions"][0]["classification"] = (
        "yolo_8_16_32_stride_vector"
    )
    forged = seal_receipt(forged)
    with pytest.raises(SiteDistanceAttestationError, match="lineage projection"):
        validate_onnx_attestation(forged, project_root=tmp_path)

    model_960.write_bytes(_tiny_static_pair_model(960, reshape_extent=31))
    with pytest.raises(SiteDistanceAttestationError, match="not 640/960-derived"):
        inspect_onnx_pair(
            model_640,
            model_960,
            checkpoint,
            checkpoint_640,
            checkpoint_960,
            project_root=tmp_path,
        )


def test_current_real_static_pair_has_closed_weight_and_resolution_lineage() -> None:
    root = Path(__file__).resolve().parents[1]
    receipt = inspect_onnx_pair(
        root / "models/person/640/yolo11s.onnx",
        root / "models/person/960/yolo11s.onnx",
        root / "models/person/yolo11s.pt",
        root / "models/person/640/yolo11s.pt",
        root / "models/person/960/yolo11s.pt",
        project_root=root,
        created_at_utc="2026-07-16T00:00:00Z",
    )
    assert {
        pin["sha256"]
        for pin in receipt["lineage"]["source_checkpoints"].values()
    } == {"85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"}
    assert receipt["lineage"]["trainable_initializer_count"] == 175
    assert len(receipt["lineage"]["resolution_derived_exceptions"]) == 12
    assert {
        item["classification"]
        for item in receipt["lineage"]["resolution_derived_exceptions"]
    } == {
        "reshape_linear_or_area_scaling_640_to_960",
        "yolo_8_16_32_anchor_grid",
        "yolo_8_16_32_stride_vector",
    }


def test_onnx_rejects_duplicate_overlong_and_nested_external_tensor_fields(
    tmp_path: Path,
) -> None:
    malformed = {
        "duplicate_ir": _field(1, 0, 10) + _tiny_onnx(),
        "overlong_tag": b"\x88\x00\x0a" + _tiny_onnx(),
    }
    for name, content in malformed.items():
        path = tmp_path / f"{name}.onnx"
        path.write_bytes(content)
        with pytest.raises(SiteDistanceAttestationError, match="duplicate|overlong"):
            inspect_onnx(path, project_root=tmp_path)

    external_tensor = (
        _field(2, 0, 1)
        + _field(13, 2, b"")
        + _field(14, 0, 1)
    )
    attribute = (
        _text(1, "value")
        + _field(20, 0, 4)
        + _message(5, external_tensor)
    )
    node = (
        _text(1, "input")
        + _text(2, "output")
        + _text(3, "head")
        + _text(4, "Identity")
        + _message(5, attribute)
    )
    graph = (
        _message(1, node)
        + _text(2, "external_graph")
        + _message(11, _value_info("input", 1, ["batch", 3, "height", "width"]))
        + _message(12, _value_info("output", 1, ["batch", "anchors", 6]))
    )
    external_model = (
        _field(1, 0, 10)
        + _message(7, graph)
        + _message(8, _field(2, 0, 18))
    )
    path = tmp_path / "nested-external.onnx"
    path.write_bytes(external_model)
    with pytest.raises(SiteDistanceAttestationError, match="nested tensor/graph"):
        inspect_onnx(path, project_root=tmp_path)

    for name, suffix, message in (
        ("training", _message(20, b""), "training graphs"),
        ("function", _message(25, b""), "local functions"),
    ):
        path = tmp_path / f"{name}.onnx"
        path.write_bytes(_tiny_onnx() + suffix)
        with pytest.raises(SiteDistanceAttestationError, match=message):
            inspect_onnx(path, project_root=tmp_path)

    model_640 = tmp_path / "overridable-640.onnx"
    model_960 = tmp_path / "overridable-960.onnx"
    model_640.write_bytes(_tiny_static_pair_model(640, expose_weight_as_input=True))
    model_960.write_bytes(_tiny_static_pair_model(960, expose_weight_as_input=True))
    checkpoint, checkpoint_640, checkpoint_960 = _checkpoint_triplet(
        tmp_path, b"overridable-checkpoint"
    )
    with pytest.raises(SiteDistanceAttestationError, match="overridable graph inputs"):
        inspect_onnx_pair(
            model_640,
            model_960,
            checkpoint,
            checkpoint_640,
            checkpoint_960,
            project_root=tmp_path,
        )


def test_receipts_reject_nonfinite_values_even_before_schema_projection(
    tmp_path: Path,
) -> None:
    model = tmp_path / "dynamic.onnx"
    model.write_bytes(_tiny_onnx())
    receipt = inspect_onnx(model, project_root=tmp_path)
    receipt["exports"][0]["inspection"]["node_count"] = float("nan")
    with pytest.raises(SiteDistanceAttestationError, match="NaN or infinity"):
        validate_onnx_attestation(receipt, project_root=tmp_path, verify_live=False)

    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(SiteDistanceAttestationError, match="cyclic"):
        canonical_sha256(cyclic)


def _kernel_payload(profile: int) -> dict[str, object]:
    return {
        "schema_version": "deepsafe.ds9-cuda-kernel-proof/v1",
        "run_id": f"{profile}-cuda",
        "kernel": "decodeTensorYoloCuda",
        "binary_version": 86,
        "ptx_version": 86,
        "launch_count_at_marker": 1,
        "marker_write_count": 1,
        "cuda_get_last_error": 0,
        "cuda_device_synchronize": 0,
        "cuda_func_get_attributes": 0,
    }


def _fake_parser_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    parser = tmp_path / "libnvdsinfer_custom_impl_Yolo.so"
    parser.write_bytes(b"fake-elf-with-cuda-fatbin")
    readelf = tmp_path / "readelf"
    readelf.write_bytes(b"fake-readelf")
    readelf.chmod(0o755)
    cuobjdump = tmp_path / "cuobjdump"
    cuobjdump.write_bytes(b"fake-cuobjdump")
    cuobjdump.chmod(0o755)

    marker_records: dict[str, object] = {}
    for profile in (640, 960):
        marker_path = tmp_path / f"{profile}-cuda-marker.json"
        payload = _kernel_payload(profile)
        _write_json(marker_path, payload)
        marker_records[f"{profile}-cuda"] = {
            "payload": payload,
            "pin": file_pin(marker_path, project_root=tmp_path),
        }
    image_id = "sha256:" + "a" * 64
    evidence = {
        "schema_version": "deepsafe.ds9-gpu-smoke-evidence/v1",
        "status": "pass",
        "resolved_image_id": image_id,
        "checks": {"cuda_parser_kernel_launch_sm86": "pass"},
        "metrics": {
            "gpu_identity": {"compute_capability": "8.6"},
            "cuda_parser": {
                "kernel_proof_method": "source_pinned_immediate_post_launch_marker",
                "ptx_jit_disabled_for_kernel_proof": True,
                "static_sm86_cubin": True,
                "valid_kernel_marker_count": 2,
                "kernel_markers": marker_records,
            },
        },
    }
    evidence_path = tmp_path / "gpu-evidence.json"
    _write_json(evidence_path, evidence)
    parser_sha = file_pin(parser, project_root=tmp_path)["sha256"]
    compatibility = {
        "schema_version": "deepsafe.ds9-runtime-compatibility-receipt/v1",
        "status": "production_ready",
        "production_ready": True,
        "created_at_utc": "2026-07-15T23:00:00Z",
        "expires_at_utc": "2026-07-16T23:00:00Z",
        "image": {
            "resolved_image_id": image_id,
            "labels": {
                "com.deepsafe.deepstream-yolo.parser-sha256": parser_sha,
                "com.deepsafe.deepstream.version": EXPECTED_DEEPSTREAM_VERSION,
                "com.deepsafe.tensorrt.version": EXPECTED_TENSORRT_VERSION,
                "com.deepsafe.cuda.version": EXPECTED_CUDA_VERSION,
            },
        },
        "static_probe": {
            "status": "pass",
            "evidence": {
                "status": "pass",
                "facts": {
                    "parser_sha256": parser_sha,
                    "abi_compile_passed": True,
                    "dlsym": {
                        EXPECTED_PARSE_FUNCTION: True,
                        EXPECTED_ENGINE_CREATE_FUNCTION: True,
                    },
                    "nm_symbol_counts": {
                        EXPECTED_PARSE_FUNCTION: 1,
                        EXPECTED_ENGINE_CREATE_FUNCTION: 1,
                    },
                    "sm86_only_cubin_set": True,
                    "compute86_only_ptx_set": True,
                    "cubin_elf_entries": [
                        {
                            "index": 1,
                            "architecture": 86,
                            "name": "libnvdsinfer_custom_impl_Yolo.1.sm_86.cubin",
                        }
                    ],
                    "ptx_entries": [
                        {
                            "index": 1,
                            "architecture": 86,
                            "name": "libnvdsinfer_custom_impl_Yolo.1.sm_86.ptx",
                        }
                    ],
                },
            },
        },
        "gpu_smoke": {
            "status": "pass",
            "checks": {"cuda_parser_kernel_launch_sm86": "pass"},
            "evidence": file_pin(evidence_path, project_root=tmp_path),
        },
    }
    compatibility_path = tmp_path / "compatibility.json"
    _write_json(compatibility_path, compatibility)
    return parser, compatibility_path, evidence_path, readelf, cuobjdump


def _fake_command_runner(
    argv: list[str] | tuple[str, ...], *, symbol_arch: int = 86, include_engine_symbol: bool = True
) -> subprocess.CompletedProcess[str]:
    flag = argv[1]
    if flag == "-hW":
        stdout = """ELF Header:
  Class:                             ELF64
  Data:                              2's complement, little endian
  Type:                              DYN (Shared object file)
  Machine:                           Advanced Micro Devices X86-64
"""
    elif flag == "-Ws":
        lines = [
            "  41: 00000001 64 FUNC GLOBAL DEFAULT 12 NvDsInferParseYoloCuda",
        ]
        if include_engine_symbol:
            lines.append(
                "  42: 00000002 64 FUNC GLOBAL DEFAULT 12 NvDsInferYoloCudaEngineGet"
            )
        stdout = "\n".join(lines) + "\n"
    elif flag == "-dW":
        stdout = """ 0x1 (NEEDED) Shared library: [libnvinfer.so.10]
 0x1 (NEEDED) Shared library: [libnvonnxparser.so.10]
 0x1 (NEEDED) Shared library: [libcudart.so.13]
"""
    elif flag == "-SW":
        stdout = """  [ 1] .dynsym DYNSYM
  [ 2] .dynstr STRTAB
  [ 3] .text PROGBITS
"""
    elif flag == "--list-elf":
        stdout = (
            f"ELF file    1: libnvdsinfer_custom_impl_Yolo.1.sm_{symbol_arch}.cubin\n"
        )
    elif flag == "--list-ptx":
        stdout = f"PTX file    1: libnvdsinfer_custom_impl_Yolo.1.sm_{symbol_arch}.ptx\n"
    else:  # pragma: no cover - a regression would expose the argv
        raise AssertionError(argv)
    return subprocess.CompletedProcess(argv, 0, stdout, "")


def _build_parser_receipt(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    parser, compatibility, evidence, readelf, cuobjdump = _fake_parser_sources(tmp_path)
    receipt = inspect_parser(
        parser,
        compatibility,
        kernel_evidence_path=evidence,
        readelf_path=readelf,
        cuobjdump_path=cuobjdump,
        project_root=tmp_path,
        command_runner=_fake_command_runner,
        created_at_utc="2026-07-16T00:00:00Z",
    )
    return receipt, parser, compatibility


def test_parser_static_elf_abi_and_prior_sm86_kernel_proofs(tmp_path: Path) -> None:
    receipt, _, _ = _build_parser_receipt(tmp_path)
    assert [item["name"] for item in receipt["elf"]["dynamic_symbols"]] == [
        EXPECTED_PARSE_FUNCTION,
        EXPECTED_ENGINE_CREATE_FUNCTION,
    ]
    assert receipt["elf"]["section_counts"] == {
        ".dynsym": 1,
        ".dynstr": 1,
        ".symtab": 0,
        ".strtab": 0,
    }
    assert receipt["cuda_static"]["cubin_entries"][0]["architecture"] == 86
    assert [item["profile"] for item in receipt["kernel_runtime_proof"]["markers"]] == [
        640,
        960,
    ]
    assert receipt["safety"] == {
        "library_loaded_this_run": False,
        "docker_called": False,
        "gpu_accessed": False,
    }


def test_parser_rejects_missing_symbol_and_wrong_cuda_architecture(tmp_path: Path) -> None:
    parser, compatibility, evidence, readelf, cuobjdump = _fake_parser_sources(tmp_path)

    with pytest.raises(SiteDistanceAttestationError, match="exactly one dynamic symbol"):
        inspect_parser(
            parser,
            compatibility,
            kernel_evidence_path=evidence,
            readelf_path=readelf,
            cuobjdump_path=cuobjdump,
            project_root=tmp_path,
            command_runner=lambda argv: _fake_command_runner(argv, include_engine_symbol=False),
            created_at_utc="2026-07-16T00:00:00Z",
        )
    with pytest.raises(SiteDistanceAttestationError, match="SM86"):
        inspect_parser(
            parser,
            compatibility,
            kernel_evidence_path=evidence,
            readelf_path=readelf,
            cuobjdump_path=cuobjdump,
            project_root=tmp_path,
            command_runner=lambda argv: _fake_command_runner(argv, symbol_arch=75),
            created_at_utc="2026-07-16T00:00:00Z",
        )


def test_parser_receipt_tamper_and_live_parser_change_are_rejected(tmp_path: Path) -> None:
    receipt, parser, _ = _build_parser_receipt(tmp_path)
    raw_tamper = copy.deepcopy(receipt)
    raw_tamper["elf"]["dynamic_symbols"][0]["count"] = 0
    with pytest.raises(SiteDistanceAttestationError):
        validate_parser_attestation(raw_tamper, project_root=tmp_path, verify_live=False)

    extra = copy.deepcopy(receipt)
    extra["cuda_static"]["unexpected"] = True
    extra = seal_receipt(extra)
    with pytest.raises(SiteDistanceAttestationError, match="unexpected|unknown fields"):
        validate_parser_attestation(extra, project_root=tmp_path, verify_live=False)

    parser.write_bytes(b"changed parser")
    with pytest.raises(SiteDistanceAttestationError, match="live artifact"):
        validate_parser_attestation(receipt, project_root=tmp_path)


def test_parser_refingerprinted_projection_is_replayed_against_exact_tool_output(
    tmp_path: Path,
) -> None:
    receipt, _, _ = _build_parser_receipt(tmp_path)
    forged = copy.deepcopy(receipt)
    forged["elf"]["needed_libraries"].append("libforged.so")
    forged = seal_receipt(forged)
    with pytest.raises(SiteDistanceAttestationError, match="projection differs|replay differs"):
        validate_parser_attestation(
            forged,
            project_root=tmp_path,
            command_runner=_fake_command_runner,
        )


def test_parser_rejects_unparsed_cuobjdump_lines_changed_argv_and_symlink(
    tmp_path: Path,
) -> None:
    parser, compatibility, evidence, readelf, cuobjdump = _fake_parser_sources(tmp_path)

    def extra_line(argv):
        result = _fake_command_runner(argv)
        if argv[1] == "--list-elf":
            result.stdout += "warning: forged sm_90 entry\n"
        return result

    with pytest.raises(SiteDistanceAttestationError, match="unparsed line"):
        inspect_parser(
            parser,
            compatibility,
            kernel_evidence_path=evidence,
            readelf_path=readelf,
            cuobjdump_path=cuobjdump,
            project_root=tmp_path,
            command_runner=extra_line,
            created_at_utc="2026-07-16T00:00:00Z",
        )

    with pytest.raises(SiteDistanceAttestationError, match="non-canonical readelf"):
        inspect_parser(
            parser,
            compatibility,
            kernel_evidence_path=evidence,
            readelf_path=readelf,
            cuobjdump_path=cuobjdump,
            project_root=tmp_path,
            created_at_utc="2026-07-16T00:00:00Z",
        )

    def changed_argv(argv):
        result = _fake_command_runner(argv)
        result.args = ["/forged/tool", *list(argv)[1:]]
        return result

    with pytest.raises(SiteDistanceAttestationError, match="changed argv"):
        inspect_parser(
            parser,
            compatibility,
            kernel_evidence_path=evidence,
            readelf_path=readelf,
            cuobjdump_path=cuobjdump,
            project_root=tmp_path,
            command_runner=changed_argv,
            created_at_utc="2026-07-16T00:00:00Z",
        )

    symlink = tmp_path / "parser-link.so"
    symlink.symlink_to(parser.name)
    with pytest.raises(SiteDistanceAttestationError, match="symlink"):
        inspect_parser(
            symlink,
            compatibility,
            kernel_evidence_path=evidence,
            readelf_path=readelf,
            cuobjdump_path=cuobjdump,
            project_root=tmp_path,
            command_runner=_fake_command_runner,
            created_at_utc="2026-07-16T00:00:00Z",
        )


def test_parser_rejects_kernel_payload_not_equal_to_pinned_marker(
    tmp_path: Path,
) -> None:
    parser, compatibility_path, evidence_path, readelf, cuobjdump = _fake_parser_sources(
        tmp_path
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["metrics"]["cuda_parser"]["kernel_markers"]["640-cuda"]["payload"][
        "launch_count_at_marker"
    ] = 2
    _write_json(evidence_path, evidence)
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    compatibility["gpu_smoke"]["evidence"] = file_pin(
        evidence_path, project_root=tmp_path
    )
    _write_json(compatibility_path, compatibility)
    with pytest.raises(SiteDistanceAttestationError, match="payload/pin projection"):
        inspect_parser(
            parser,
            compatibility_path,
            kernel_evidence_path=evidence_path,
            readelf_path=readelf,
            cuobjdump_path=cuobjdump,
            project_root=tmp_path,
            command_runner=_fake_command_runner,
            created_at_utc="2026-07-16T00:00:00Z",
        )


def _engine_receipt(
    tmp_path: Path,
    profile: int,
    *,
    pair_id: str = "c" * 64,
) -> dict[str, object]:
    model = tmp_path / "dynamic.onnx"
    model.write_bytes(_tiny_onnx())
    onnx_receipt = inspect_onnx(
        model,
        project_root=tmp_path,
        created_at_utc="2026-07-16T00:00:00Z",
    )
    onnx_path = tmp_path / "onnx-attestation.json"
    _write_json(onnx_path, onnx_receipt)

    parser_receipt, _, _ = _build_parser_receipt(tmp_path)
    parser_path = tmp_path / "parser-attestation.json"
    _write_json(parser_path, parser_receipt)

    engine = tmp_path / f"person-{profile}.engine"
    config = tmp_path / f"person-{profile}.txt"
    runtime_log = tmp_path / f"person-{profile}.log"
    engine.write_bytes(f"serialized-tensorrt-engine-{profile}".encode())
    config.write_text(
        "[property]\n"
        "gpu-id=0\n"
        f"onnx-file=/models/person-{profile}.onnx\n"
        f"model-engine-file=/models/person-{profile}.engine\n"
        "batch-size=1\n"
        f"infer-dims=3;{profile};{profile}\n"
        "network-input-order=0\n"
        "network-mode=2\n"
        f"parse-bbox-func-name={EXPECTED_PARSE_FUNCTION}\n"
        "custom-lib-path=/opt/deepsafe/libnvdsinfer_custom_impl_Yolo.so\n"
        f"engine-create-func-name={EXPECTED_ENGINE_CREATE_FUNCTION}\n",
        encoding="utf-8",
    )
    shape = [1, 3, profile, profile]
    image_id = parser_receipt["abi_runtime_proof"]["resolved_image_id"]
    if profile == 640:
        event_times = [
            "2026-07-16T00:01:00Z",
            "2026-07-16T00:01:15Z",
            "2026-07-16T00:01:30Z",
            "2026-07-16T00:01:45Z",
            "2026-07-16T00:02:00Z",
        ]
        run_id = "6" * 64
    else:
        event_times = [
            "2026-07-16T00:03:00Z",
            "2026-07-16T00:03:15Z",
            "2026-07-16T00:03:30Z",
            "2026-07-16T00:03:45Z",
            "2026-07-16T00:04:00Z",
        ]
        run_id = "9" * 64
    bindings = {
        "input": {
            "name": "input",
            "is_input": True,
            "dtype": "FLOAT32",
            "format": "NCHW",
            "rank": 4,
            "shape": shape,
            "profile_min": shape,
            "profile_opt": [12, 3, profile, profile],
            "profile_max": [12, 3, profile, profile],
        },
        "outputs": [
            {
                "name": "output",
                "is_input": False,
                "dtype": "FLOAT32",
                "rank": 3,
                "dimensions": [1, 8400 if profile == 640 else 18900, 6],
            }
        ],
    }
    body = {
        "schema_version": ENGINE_SCHEMA_VERSION,
        "attestation_type": "deepstream9_tensorrt_engine_load",
        "created_at_utc": event_times[-1],
        "status": "pass",
        "run": {
            "pair_id": pair_id,
            "run_id": run_id,
            "profile": profile,
            "started_at_utc": event_times[0],
            "finished_at_utc": event_times[-1],
            "event_prefix": ENGINE_LOG_PREFIX,
            "event_count": len(ENGINE_EVENT_SEQUENCE),
            "event_stream_sha256": "0" * 64,
        },
        "parents": {
            "onnx_attestation": file_pin(onnx_path, project_root=tmp_path),
            "onnx_receipt_fingerprint_sha256": onnx_receipt[
                "receipt_fingerprint_sha256"
            ],
            "parser_attestation": file_pin(parser_path, project_root=tmp_path),
            "parser_receipt_fingerprint_sha256": parser_receipt[
                "receipt_fingerprint_sha256"
            ],
        },
        "artifacts": {
            "engine": file_pin(engine, project_root=tmp_path),
            "inference_config": file_pin(config, project_root=tmp_path),
        },
        "contract": {
            "profile": profile,
            "engine_precision": "FP16",
            "input_layout": "NCHW",
            "inference_batch_size": 1,
            "requested_shape": shape,
            "model_sha256": onnx_receipt["exports"][0]["model"]["sha256"],
            "parser_sha256": parser_receipt["parser"]["sha256"],
            "parse_function": EXPECTED_PARSE_FUNCTION,
            "engine_create_function": EXPECTED_ENGINE_CREATE_FUNCTION,
        },
        "runtime": {
            "deepstream_version": EXPECTED_DEEPSTREAM_VERSION,
            "tensorrt_version": EXPECTED_TENSORRT_VERSION,
            "cuda_version": EXPECTED_CUDA_VERSION,
            "requested_image": "deepsafe-deepstream:9.0",
            "resolved_image_id": image_id,
            "gpu": {
                "host_index": 0,
                "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                "compute_capability": "8.6",
                "container_device_ordinal": 0,
                "docker_device_request": {
                    "driver": "nvidia",
                    "count": 1,
                    "device_ids": ["0"],
                    "capabilities": [["gpu"]],
                },
            },
        },
        "load": {
            "onnx_path_in_container": f"/models/person-{profile}.onnx",
            "engine_path_in_container": f"/models/person-{profile}.engine",
            "parser_path_in_container": "/opt/deepsafe/libnvdsinfer_custom_impl_Yolo.so",
            "inference_config_path_in_container": f"/contract/person-{profile}.txt",
            "deserialized": True,
            "deserialized_marker_count": 1,
            "use_deserialized_marker_count": 1,
            "load_success_marker_count": 1,
            "onnx_parse_marker_count": 0,
            "engine_build_marker_count": 0,
            "fallback_marker_count": 0,
            "cuda_error_marker_count": 0,
        },
        "bindings": bindings,
        "safety": {
            "engine_load_executed_on_gpu": True,
            "validator_loads_engine": False,
            "cpu_inference_executed": False,
            "receipt_replay_only": True,
        },
    }
    common_event = {
        "schema_version": "deepsafe.engine-load-log-event/v1",
        "pair_id": pair_id,
        "run_id": run_id,
        "profile": profile,
        "engine_sha256": body["artifacts"]["engine"]["sha256"],
        "inference_config_sha256": body["artifacts"]["inference_config"]["sha256"],
        "model_sha256": body["contract"]["model_sha256"],
        "parser_sha256": body["contract"]["parser_sha256"],
        "onnx_receipt_fingerprint_sha256": body["parents"][
            "onnx_receipt_fingerprint_sha256"
        ],
        "parser_receipt_fingerprint_sha256": body["parents"][
            "parser_receipt_fingerprint_sha256"
        ],
        "resolved_image_id": image_id,
        "gpu_uuid": body["runtime"]["gpu"]["uuid"],
        "binding_manifest_sha256": canonical_sha256(bindings),
    }
    events = [
        {
            **common_event,
            "sequence": sequence,
            "event": event,
            "timestamp_utc": event_times[sequence],
        }
        for sequence, event in enumerate(ENGINE_EVENT_SEQUENCE)
    ]
    body["run"]["event_stream_sha256"] = canonical_sha256(events)
    runtime_log.write_text(
        "\n".join(
            [
                f"deserialized trt engine from :/models/person-{profile}.engine",
                f"Use deserialized engine model: /models/person-{profile}.engine",
                f"Load new model:/contract/person-{profile}.txt sucessfully",
                *[
                    ENGINE_LOG_PREFIX + json.dumps(event, sort_keys=True)
                    for event in events
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    body["artifacts"]["runtime_log"] = file_pin(runtime_log, project_root=tmp_path)
    return finalize_engine_load_attestation(body)


@pytest.mark.parametrize("profile", [640, 960])
def test_engine_load_receipt_contract_accepts_both_profiles(
    tmp_path: Path, profile: int
) -> None:
    receipt = _engine_receipt(tmp_path, profile)
    validate_engine_load_attestation(receipt, project_root=tmp_path)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda item: item["contract"].__setitem__("requested_shape", [1, 3, 960, 960]),
            "requested shape",
        ),
        (
            lambda item: item["load"].__setitem__("fallback_marker_count", 1),
            "fallback_marker_count|permits build",
        ),
        (
            lambda item: item["runtime"].__setitem__(
                "resolved_image_id", "sha256:" + "b" * 64
            ),
            "image differs",
        ),
        (
            lambda item: item["runtime"]["gpu"]["docker_device_request"].__setitem__(
                "device_ids", ["1"]
            ),
            "runtime/GPU binding",
        ),
        (
            lambda item: item["bindings"]["outputs"][0].__setitem__(
                "dimensions", [1, 8400]
            ),
            "rank/dimensions",
        ),
        (
            lambda item: item["load"].__setitem__(
                "engine_path_in_container", "/models/../person-640.engine"
            ),
            "not canonical",
        ),
    ],
)
def test_engine_load_receipt_semantic_tamper_is_rejected(
    tmp_path: Path, mutator, message: str
) -> None:
    receipt = _engine_receipt(tmp_path, 640)
    mutator(receipt)
    receipt = seal_receipt(receipt)
    with pytest.raises(SiteDistanceAttestationError, match=message):
        validate_engine_load_attestation(receipt, project_root=tmp_path)


def test_engine_receipt_unknown_field_and_live_engine_change_are_rejected(tmp_path: Path) -> None:
    receipt = _engine_receipt(tmp_path, 640)
    extra = copy.deepcopy(receipt)
    extra["load"]["trust_me"] = True
    extra = seal_receipt(extra)
    with pytest.raises(SiteDistanceAttestationError, match="trust_me|unknown fields"):
        validate_engine_load_attestation(extra, project_root=tmp_path, verify_live=False)

    engine = tmp_path / "person-640.engine"
    engine.write_bytes(b"tampered")
    with pytest.raises(SiteDistanceAttestationError, match="live artifact"):
        validate_engine_load_attestation(receipt, project_root=tmp_path)

    boolean_profile = _engine_receipt(tmp_path, 640)
    boolean_profile["contract"]["profile"] = True
    boolean_profile = seal_receipt(boolean_profile)
    with pytest.raises(SiteDistanceAttestationError, match="forbidden value"):
        validate_engine_load_attestation(
            boolean_profile, project_root=tmp_path, verify_live=False
        )


def test_engine_log_counters_and_machine_events_are_recomputed_from_pinned_log(
    tmp_path: Path,
) -> None:
    receipt = _engine_receipt(tmp_path, 640)
    log_path = tmp_path / "person-640.log"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    log_path.write_text(
        "\n".join(line for line in lines if not line.startswith("Use deserialized"))
        + "\n",
        encoding="utf-8",
    )
    receipt["artifacts"]["runtime_log"] = file_pin(log_path, project_root=tmp_path)
    receipt = seal_receipt(receipt)
    with pytest.raises(SiteDistanceAttestationError, match="marker counters"):
        validate_engine_load_attestation(receipt, project_root=tmp_path)

    receipt = _engine_receipt(tmp_path, 640)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    marker_index = next(
        index for index, line in enumerate(lines) if line.startswith(ENGINE_LOG_PREFIX)
    )
    marker_json = lines[marker_index][len(ENGINE_LOG_PREFIX) :]
    lines[marker_index] = ENGINE_LOG_PREFIX + '{"profile":640,' + marker_json[1:]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt["artifacts"]["runtime_log"] = file_pin(log_path, project_root=tmp_path)
    receipt = seal_receipt(receipt)
    with pytest.raises(SiteDistanceAttestationError, match="duplicate JSON key"):
        validate_engine_load_attestation(receipt, project_root=tmp_path)


def test_engine_config_and_binding_claims_are_replayed_not_trusted(
    tmp_path: Path,
) -> None:
    receipt = _engine_receipt(tmp_path, 640)
    config = tmp_path / "person-640.txt"
    config.write_text(
        config.read_text(encoding="utf-8") + "infer-dims=3;960;960\n",
        encoding="utf-8",
    )
    receipt["artifacts"]["inference_config"] = file_pin(
        config, project_root=tmp_path
    )
    receipt = seal_receipt(receipt)
    with pytest.raises(SiteDistanceAttestationError, match="duplicate DeepStream config key"):
        validate_engine_load_attestation(receipt, project_root=tmp_path)

    receipt = _engine_receipt(tmp_path, 640)
    receipt["bindings"]["outputs"][0]["dimensions"][1] = 9999
    receipt = seal_receipt(receipt)
    with pytest.raises(SiteDistanceAttestationError, match="binding_manifest_sha256"):
        validate_engine_load_attestation(receipt, project_root=tmp_path)


def test_engine_pair_requires_same_parents_gpu_and_separate_nonoverlapping_runs(
    tmp_path: Path,
) -> None:
    receipt_640 = _engine_receipt(tmp_path, 640)
    receipt_960 = _engine_receipt(tmp_path, 960)
    validate_engine_load_attestation_pair(
        [receipt_960, receipt_640], project_root=tmp_path
    )

    overlap = copy.deepcopy(receipt_960)
    overlap["run"]["started_at_utc"] = "2026-07-16T00:01:30Z"
    overlap["run"]["finished_at_utc"] = "2026-07-16T00:02:30Z"
    overlap["created_at_utc"] = "2026-07-16T00:02:30Z"
    overlap = seal_receipt(overlap)
    with pytest.raises(SiteDistanceAttestationError, match="overlap"):
        validate_engine_load_attestation_pair(
            [receipt_640, overlap], project_root=tmp_path, verify_live=False
        )

    different_pair = copy.deepcopy(receipt_960)
    different_pair["run"]["pair_id"] = "d" * 64
    different_pair = seal_receipt(different_pair)
    with pytest.raises(SiteDistanceAttestationError, match="different pair IDs"):
        validate_engine_load_attestation_pair(
            [receipt_640, different_pair],
            project_root=tmp_path,
            verify_live=False,
        )


def test_engine_validator_does_not_offer_an_engine_load_or_inference_api() -> None:
    import validation.site_distance_attestation_v1 as module

    forbidden = {
        "load_engine",
        "infer",
        "benchmark",
        "run_cpu_inference",
        "run_gpu_inference",
    }
    assert forbidden.isdisjoint(vars(module))
