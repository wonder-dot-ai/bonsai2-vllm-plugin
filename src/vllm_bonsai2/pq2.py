"""Lossless PQ2_0 block layout, matching the pinned Prism ggml-quants.c."""

from __future__ import annotations

import numpy as np

GROUP_SIZE = 128
CODE_BYTES = 32
BLOCK_BYTES = 34


def split_blocks(raw: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Split row-major [out, in] blocks without changing code or scale bits."""
    rows, width = shape
    if rows <= 0 or width <= 0 or width % GROUP_SIZE:
        raise ValueError("PQ2_0 requires positive rows and input width divisible by 128")
    if raw.dtype != np.uint8 or raw.size != rows * (width // GROUP_SIZE) * BLOCK_BYTES:
        raise ValueError("invalid PQ2_0 byte buffer or shape")
    blocks = raw.reshape(rows, width // GROUP_SIZE, BLOCK_BYTES)
    codes = np.ascontiguousarray(blocks[..., 2:])
    scales = np.ascontiguousarray(blocks[..., :2]).view("<f2").reshape(blocks.shape[:2])
    return codes, scales


def join_blocks(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Recreate the original interleaved little-endian GGUF bytes."""
    if codes.dtype != np.uint8 or codes.ndim != 3 or codes.shape[-1] != CODE_BYTES:
        raise ValueError("qweight must be uint8 [out, in/128, 32]")
    if scales.dtype != np.dtype("<f2") or scales.shape != codes.shape[:2]:
        raise ValueError("scales must be little-endian float16 [out, in/128]")
    blocks = np.empty((*scales.shape, BLOCK_BYTES), dtype=np.uint8)
    blocks[..., :2] = np.ascontiguousarray(scales).view(np.uint8).reshape(*scales.shape, 2)
    blocks[..., 2:] = codes
    return blocks.reshape(codes.shape[0], -1)


def dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Decode to FP32; low two bits are first, and each code maps to code - 1.

    Code 3 represents +2 in the codec. Preserve it even though ternary model
    weights normally use only 0, 1, and 2. No requantization is performed.
    """
    join_blocks(codes, scales)  # Validate before interpreting the arrays.
    shifts = np.arange(4, dtype=np.uint8) * 2
    values = ((codes[..., None] >> shifts) & 3).astype(np.int8) - 1
    values = values.reshape(*scales.shape, GROUP_SIZE).astype(np.float32)
    return (values * scales.astype(np.float32)[..., None]).reshape(codes.shape[0], -1)
