import numpy as np
import pytest

from vllm_bonsai2.pq2 import dequantize, join_blocks, split_blocks


def test_codec_bit_order_and_scale_broadcast():
    # 0xe4 stores 00, 01, 10, 11 from least to most significant slot.
    codes = np.full((2, 2, 32), 0xE4, dtype=np.uint8)
    scales = np.array([[0.5, 2], [-1, 0]], dtype="<f2")
    result = dequantize(codes, scales).reshape(2, 2, 128)
    for row in range(2):
        for group in range(2):
            expected = np.tile([-1, 0, 1, 2], 32) * float(scales[row, group])
            np.testing.assert_array_equal(result[row, group], expected)


def test_every_fp16_bit_pattern_survives_split_join():
    # Includes signed zeros, infinities, subnormals, and NaN payload bits.
    raw = np.zeros((65536, 34), dtype=np.uint8)
    raw[:, :2] = np.arange(65536, dtype="<u2").view(np.uint8).reshape(-1, 2)
    raw[:, 2:] = np.arange(32, dtype=np.uint8)
    codes, scales = split_blocks(raw, (256, 256 * 128))
    assert join_blocks(codes, scales).tobytes() == raw.tobytes()


@pytest.mark.parametrize("shape", [(1, 127), (0, 128), (1, 0), (2, 128)])
def test_bad_logical_shape_rejected(shape):
    with pytest.raises(ValueError):
        split_blocks(np.zeros(34, dtype=np.uint8), shape)


def test_bad_codes_or_scales_rejected():
    with pytest.raises(ValueError):
        join_blocks(np.zeros((1, 1, 31), dtype=np.uint8), np.ones((1, 1), dtype="<f2"))
    with pytest.raises(ValueError):
        join_blocks(np.zeros((1, 1, 32), dtype=np.uint8), np.ones((1, 1), dtype=np.float32))
