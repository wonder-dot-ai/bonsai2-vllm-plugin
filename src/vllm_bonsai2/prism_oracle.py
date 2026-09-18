"""ctypes wrapper for the pinned, instrumented Prism GGML projection graph."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from vllm_bonsai2.convert import PRISM_REVISION
from vllm_bonsai2.reference import Rotation


class PrismOracle:
    def __init__(self, library: Path):
        self.library = ctypes.CDLL(str(library.resolve()))
        self.library.bonsai_prism_commit.restype = ctypes.c_char_p
        self.commit = self.library.bonsai_prism_commit().decode()
        if len(self.commit) < 7 or not PRISM_REVISION.startswith(self.commit):
            raise ValueError(f"Prism commit mismatch: {self.commit}")
        self.run = self.library.bonsai_projection
        self.run.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 7 + [ctypes.c_void_p] * 3
        self.run.restype = ctypes.c_int

    def project(self, raw: np.ndarray, x: np.ndarray, rotation: Rotation, backend: str) -> dict:
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend must be cpu or cuda")
        width = len(rotation.signs)
        if raw.dtype != np.uint8 or raw.ndim != 2 or raw.shape[1] != width // 128 * 34:
            raise ValueError("invalid packed buffer")
        if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != width:
            raise ValueError("oracle input must be FP32 [batch, width]")
        raw, x = np.ascontiguousarray(raw), np.ascontiguousarray(x)
        signs = np.array(rotation.signs, dtype=np.float32)
        rotated = np.empty_like(x)
        packed = np.empty((len(x), len(raw)), dtype=np.float32)
        dense = np.empty_like(packed)
        status = self.run(
            raw.ctypes.data,
            signs.ctypes.data,
            x.ctypes.data,
            width,
            len(raw),
            len(x),
            rotation.block_size,
            rotation.perm_nk,
            rotation.perm_rep,
            int(backend == "cuda"),
            rotated.ctypes.data,
            packed.ctypes.data,
            dense.ctypes.data,
        )
        if status:
            raise RuntimeError(f"Prism projection failed with status {status}")
        return {"rotated": rotated, "packed": packed, "dense": dense}
