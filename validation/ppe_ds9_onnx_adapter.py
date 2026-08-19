#!/usr/bin/env python3
"""Adapt raw Ultralytics YOLOv8 detection output for DeepStream-Yolo.

SafetyVision exports ``[B, 4 + C, N]`` tensors containing ``xywh`` boxes and
per-class scores.  The DeepStream 9 CUDA parser in the local runtime consumes
``[B, N, 6]`` records containing ``xyxy, max_score, class_id``.  This module
appends only deterministic ONNX tensor operations; it does not alter model
weights, run inference, or apply NMS.  DeepStream nvinfer remains responsible
for thresholding and NMS.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SCHEMA_VERSION = "colt-ai.ppe-ds9-onnx-adapter/v1"


class PpeOnnxAdapterError(ValueError):
    """Raised when a graph is not the expected raw SafetyVision export."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.dim_param:
            result.append(dimension.dim_param)
        else:
            result.append(dimension.dim_value)
    return result


def _metadata(model: onnx.ModelProto) -> dict[str, str]:
    return {item.key: item.value for item in model.metadata_props}


def _class_count(model: onnx.ModelProto) -> int:
    metadata = _metadata(model)
    try:
        names = ast.literal_eval(metadata["names"])
    except (KeyError, SyntaxError, ValueError) as exc:
        raise PpeOnnxAdapterError("ONNX names metadata is missing or invalid") from exc
    if (
        not isinstance(names, dict)
        or not names
        or sorted(names) != list(range(len(names)))
        or any(not isinstance(value, str) or not value for value in names.values())
    ):
        raise PpeOnnxAdapterError("ONNX class-name mapping is not contiguous")
    return len(names)


def expected_ds9_output(raw: np.ndarray) -> np.ndarray:
    """Reference NumPy transform used for ONNX Runtime parity tests."""

    if raw.ndim != 3 or raw.shape[1] < 6:
        raise PpeOnnxAdapterError("raw tensor must be [B, 4+C, N]")
    xy = raw[:, 0:2, :]
    half_wh = raw[:, 2:4, :] * np.float32(0.5)
    xyxy = np.concatenate((xy - half_wh, xy + half_wh), axis=1)
    scores = raw[:, 4:, :]
    maximum = scores.max(axis=1, keepdims=True)
    class_id = scores.argmax(axis=1, keepdims=True).astype(raw.dtype)
    return np.concatenate((xyxy, maximum, class_id), axis=1).transpose(0, 2, 1)


def adapt_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise PpeOnnxAdapterError("expected exactly one input and one raw output")
    raw_output = model.graph.output[0]
    input_shape = _shape(model.graph.input[0])
    output_shape = _shape(raw_output)
    classes = _class_count(model)
    if (
        len(input_shape) != 4
        or input_shape[1] != 3
        or input_shape[2] not in (640, 960)
        or input_shape[3] != input_shape[2]
    ):
        raise PpeOnnxAdapterError(f"unsupported input shape: {input_shape}")
    if (
        len(output_shape) != 3
        or output_shape[1] != classes + 4
        or output_shape[2] not in (8400, 18900)
    ):
        raise PpeOnnxAdapterError(f"unsupported raw output shape: {output_shape}")

    output_name = raw_output.name
    constants = {
        "ppe_ds9_starts_0": np.asarray([0], dtype=np.int64),
        "ppe_ds9_starts_2": np.asarray([2], dtype=np.int64),
        "ppe_ds9_starts_4": np.asarray([4], dtype=np.int64),
        "ppe_ds9_ends_2": np.asarray([2], dtype=np.int64),
        "ppe_ds9_ends_4": np.asarray([4], dtype=np.int64),
        "ppe_ds9_ends_classes": np.asarray([classes + 4], dtype=np.int64),
        "ppe_ds9_axes_1": np.asarray([1], dtype=np.int64),
        "ppe_ds9_axes_2": np.asarray([2], dtype=np.int64),
        "ppe_ds9_half": np.asarray([0.5], dtype=np.float32),
    }
    model.graph.initializer.extend(
        numpy_helper.from_array(value, name)
        for name, value in constants.items()
    )
    model.graph.node.extend(
        [
            helper.make_node(
                "Slice",
                [
                    output_name,
                    "ppe_ds9_starts_0",
                    "ppe_ds9_ends_2",
                    "ppe_ds9_axes_1",
                ],
                ["ppe_ds9_xy"],
                name="ppe_ds9_slice_xy",
            ),
            helper.make_node(
                "Slice",
                [
                    output_name,
                    "ppe_ds9_starts_2",
                    "ppe_ds9_ends_4",
                    "ppe_ds9_axes_1",
                ],
                ["ppe_ds9_wh"],
                name="ppe_ds9_slice_wh",
            ),
            helper.make_node(
                "Mul",
                ["ppe_ds9_wh", "ppe_ds9_half"],
                ["ppe_ds9_half_wh"],
                name="ppe_ds9_half_wh",
            ),
            helper.make_node(
                "Sub",
                ["ppe_ds9_xy", "ppe_ds9_half_wh"],
                ["ppe_ds9_top_left"],
                name="ppe_ds9_top_left",
            ),
            helper.make_node(
                "Add",
                ["ppe_ds9_xy", "ppe_ds9_half_wh"],
                ["ppe_ds9_bottom_right"],
                name="ppe_ds9_bottom_right",
            ),
            helper.make_node(
                "Concat",
                ["ppe_ds9_top_left", "ppe_ds9_bottom_right"],
                ["ppe_ds9_xyxy_chw"],
                name="ppe_ds9_concat_xyxy",
                axis=1,
            ),
            helper.make_node(
                "Transpose",
                ["ppe_ds9_xyxy_chw"],
                ["ppe_ds9_xyxy"],
                name="ppe_ds9_transpose_xyxy",
                perm=[0, 2, 1],
            ),
            helper.make_node(
                "Slice",
                [
                    output_name,
                    "ppe_ds9_starts_4",
                    "ppe_ds9_ends_classes",
                    "ppe_ds9_axes_1",
                ],
                ["ppe_ds9_scores_chw"],
                name="ppe_ds9_slice_scores",
            ),
            helper.make_node(
                "Transpose",
                ["ppe_ds9_scores_chw"],
                ["ppe_ds9_scores"],
                name="ppe_ds9_transpose_scores",
                perm=[0, 2, 1],
            ),
            helper.make_node(
                "ReduceMax",
                ["ppe_ds9_scores", "ppe_ds9_axes_2"],
                ["ppe_ds9_max_score"],
                name="ppe_ds9_max_score",
                keepdims=1,
            ),
            helper.make_node(
                "ArgMax",
                ["ppe_ds9_scores"],
                ["ppe_ds9_class_i64"],
                name="ppe_ds9_argmax_class",
                axis=2,
                keepdims=1,
            ),
            helper.make_node(
                "Cast",
                ["ppe_ds9_class_i64"],
                ["ppe_ds9_class_f32"],
                name="ppe_ds9_cast_class",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Concat",
                [
                    "ppe_ds9_xyxy",
                    "ppe_ds9_max_score",
                    "ppe_ds9_class_f32",
                ],
                ["ppe_ds9_output"],
                name="ppe_ds9_concat_output",
                axis=2,
            ),
        ]
    )
    batch_dimension = input_shape[0]
    anchors = output_shape[2]
    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                "ppe_ds9_output",
                TensorProto.FLOAT,
                [batch_dimension, anchors, 6],
            )
        ]
    )
    metadata = _metadata(model)
    metadata.update(
        {
            "end2end": "True",
            "deepsafe_ds9_adapter": SCHEMA_VERSION,
            "deepsafe_ds9_output": "xyxy,max_score,class_id",
            "deepsafe_ds9_nms": "nvinfer_cluster_mode_2",
        }
    )
    del model.metadata_props[:]
    for key in sorted(metadata):
        item = model.metadata_props.add()
        item.key = key
        item.value = metadata[key]
    onnx.checker.check_model(model, full_check=True)
    return model, {
        "input_shape": input_shape,
        "raw_output_shape": output_shape,
        "adapted_output_shape": [batch_dimension, anchors, 6],
        "class_count": classes,
    }


def adapt_file(source: Path, destination: Path, receipt: Path) -> dict[str, Any]:
    if not source.is_file():
        raise PpeOnnxAdapterError(f"source ONNX is absent: {source}")
    model = onnx.load(source, load_external_data=False)
    source_bytes = source.stat().st_size
    source_sha256 = _sha256(source)
    model, graph = adapt_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(raw_name)
    try:
        onnx.save_model(model, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "status": "adapted_static_verified",
        "production_accepted": False,
        "license_id": "AGPL-3.0",
        "source": {
            "path": source.as_posix(),
            "bytes": source_bytes,
            "sha256": source_sha256,
        },
        "artifact": {
            "path": destination.as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        },
        "graph": graph,
        "transform": {
            "boxes": "xywh_to_xyxy",
            "scores": "reduce_max",
            "class_id": "argmax_cast_float32",
            "nms_embedded": False,
        },
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            terminal,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor, raw_name = tempfile.mkstemp(
        prefix=f".{receipt.name}.", suffix=".tmp", dir=receipt.parent
    )
    temporary = Path(raw_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, receipt)
    finally:
        temporary.unlink(missing_ok=True)
    return terminal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append a DeepStream raw6 output adapter to a PPE ONNX"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    terminal = adapt_file(args.source, args.output, args.receipt)
    print(json.dumps(terminal, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
