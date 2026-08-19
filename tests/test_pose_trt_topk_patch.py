from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from models.pose.patch_trt_topk import (
    PATCH_INPUT_NAME,
    TRT_TOPK_LIMIT,
    patch_model,
)


def _fixture(path: Path) -> None:
    cap = helper.make_node(
        "Constant",
        [],
        ["cap"],
        name="/Constant_cap",
        value=numpy_helper.from_array(np.asarray([5000], dtype=np.int64)),
    )
    k = helper.make_node("Identity", ["cap"], ["k"], name="/KIdentity")
    topk = helper.make_node(
        "TopK",
        ["input", "k"],
        ["dets", "keypoints"],
        name="/TopK",
        axis=1,
        largest=1,
        sorted=1,
    )
    graph = helper.make_graph(
        [cap, k, topk],
        "pose-topk-fixture",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 5000])],
        [
            helper.make_tensor_value_info(
                "dets", TensorProto.FLOAT, [1, "candidates"]
            ),
            helper.make_tensor_value_info(
                "keypoints", TensorProto.INT64, [1, "candidates"]
            ),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_inspection_is_read_only(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    output = tmp_path / "patched.onnx"
    _fixture(source)

    receipt = patch_model(source, output, write=False)

    assert receipt["status"] == "validated_not_written"
    assert receipt["patch"]["old_cap"] == 5000
    assert receipt["patch"]["new_cap"] == TRT_TOPK_LIMIT
    assert not output.exists()


def test_patch_replaces_only_target_topk_k_edge(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    output = tmp_path / "patched.onnx"
    _fixture(source)

    receipt = patch_model(source, output, write=True)
    model = onnx.load(output)
    target = next(node for node in model.graph.node if node.name == "/TopK")
    initializers = {
        item.name: numpy_helper.to_array(item)
        for item in model.graph.initializer
    }

    assert receipt["status"] == "patched"
    assert target.input == ["input", PATCH_INPUT_NAME]
    assert initializers[PATCH_INPUT_NAME].tolist() == [TRT_TOPK_LIMIT]
    assert list(target.output) == ["dets", "keypoints"]
    onnx.checker.check_model(model)
