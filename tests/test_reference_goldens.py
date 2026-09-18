"""Offline regressions against immutable outputs from the Prism graph harness."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_bonsai2.convert import sha256_file
from vllm_bonsai2.reference import Rotation, linear_reference, transform_activation
from vllm_bonsai2.validation import compare

DATA = Path(__file__).parent / "data" / "m2"


@pytest.mark.parametrize("name", ["gate", "down", "ssm_out"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_frozen_prism_projection(name, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    record = json.loads((DATA / "manifest.json").read_text())["fixtures"][name]
    path = DATA / f"{name}.npz"
    assert sha256_file(path) == record["sha256"]
    with np.load(path) as data:
        rotation = Rotation(signs=tuple(int(s) for s in data["signs"]), **record["rotation"])
        x = torch.from_numpy(data["input"]).to(device)
        codes = torch.from_numpy(data["qweight"]).to(device)
        scales = torch.from_numpy(data["scales"]).to(device)
        rotated = transform_activation(x, rotation).cpu().numpy()
        result = linear_reference(x, codes, scales, rotation).cpu().numpy()
        for backend in ("cpu", "cuda"):
            assert compare(rotated, data[backend + "_rotated"], "transform")["passed"]
            assert compare(data[backend + "_dense"], result, backend + "_dense")["passed"]
            assert compare(data[backend + "_packed"], result, "packed")["passed"]
        # A missing signs/permutation implementation must fail, not hide in tolerances.
        wrong = linear_reference(x, codes, scales, Rotation(1024, (1,) * x.shape[-1]))
        assert not compare(wrong.cpu().numpy(), data["cpu_dense"], "packed")["passed"]


def test_comparison_rejects_one_bad_sequence_and_nonfinite_values():
    expected = np.ones((32, 128), dtype=np.float32)
    actual = expected.copy()
    actual[0] += 0.1
    assert not compare(actual, expected, "packed")["passed"]
    actual[0, 0] = np.nan
    assert not compare(actual, expected, "packed")["passed"]
    assert compare(np.zeros((1, 128)), np.zeros((1, 128)), "packed")["passed"]
