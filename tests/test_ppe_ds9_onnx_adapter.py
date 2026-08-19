import numpy as np
import pytest

pytest.importorskip("onnx")

from validation.ppe_ds9_onnx_adapter import (
    PpeOnnxAdapterError,
    expected_ds9_output,
)


def test_reference_transform_converts_xywh_scores_to_xyxy_raw6():
    raw = np.zeros((1, 7, 2), dtype=np.float32)
    raw[0, 0:4, 0] = [100, 50, 20, 10]
    raw[0, 4:, 0] = [0.1, 0.8, 0.2]
    raw[0, 0:4, 1] = [20, 30, 8, 12]
    raw[0, 4:, 1] = [0.9, 0.2, 0.1]
    output = expected_ds9_output(raw)
    assert output.shape == (1, 2, 6)
    np.testing.assert_allclose(
        output[0, 0], [90, 45, 110, 55, 0.8, 1]
    )
    np.testing.assert_allclose(
        output[0, 1], [16, 24, 24, 36, 0.9, 0]
    )


def test_reference_transform_rejects_wrong_rank():
    with pytest.raises(PpeOnnxAdapterError, match="raw tensor"):
        expected_ds9_output(np.zeros((17, 8400), dtype=np.float32))
