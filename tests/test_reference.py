import copy

import numpy as np
import pytest
import torch

from vllm_bonsai2.pq2 import dequantize
from vllm_bonsai2.reference import (
    Rotation,
    hadamard,
    linear_reference,
    rotation_from_manifest,
    transform_activation,
    unpack_pq2,
)


def matrix_hadamard(n):
    # Independent dense definition, not the butterfly algorithm under test.
    return (
        torch.tensor([[(-1.0) ** ((i & j).bit_count()) for j in range(n)] for i in range(n)])
        / n**0.5
    )


@pytest.mark.parametrize("block", [1, 2, 8, 128, 1024])
def test_fwht_matches_dense_and_preserves_norm(block):
    x = torch.randn(3, block * 2, generator=torch.Generator().manual_seed(12))
    original = x.clone()
    expected = (x.reshape(-1, block) @ matrix_hadamard(block)).reshape_as(x)
    got = hadamard(x, block)
    torch.testing.assert_close(got, expected, atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(hadamard(got, block), x, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.vector_norm(got), torch.linalg.vector_norm(x))
    assert torch.equal(original, x)


def test_noncontiguous_half_input_and_leading_dimensions():
    x = torch.randn(2, 32, 3).half().transpose(1, 2)
    assert not x.is_contiguous()
    got = hadamard(x, 16)
    expected = (x.float().reshape(-1, 16) @ matrix_hadamard(16)).reshape_as(x)
    torch.testing.assert_close(got, expected)
    assert got.dtype == torch.float32


def test_signed_grouped_transform_feature_order():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    rotation = Rotation(4, (1, -1) * 6, perm_nk=2, perm_rep=3)
    # Explicit feature indexing for hd=2, nk=2, rep=3.
    order = [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]
    grouped = x[:, order] * torch.tensor(rotation.signs)
    expected = (grouped.reshape(-1, 4) @ matrix_hadamard(4)).reshape_as(x)
    torch.testing.assert_close(transform_activation(x, rotation), expected)


def test_torch_unpack_matches_numpy_codec_and_linear_definition():
    rng = np.random.default_rng(33)
    codes = rng.integers(0, 256, (7, 2, 32), dtype=np.uint8)
    scales = rng.normal(size=(7, 2)).astype(np.float16)
    c, s = torch.from_numpy(codes), torch.from_numpy(scales)
    weights = torch.from_numpy(dequantize(codes, scales))
    assert torch.equal(unpack_pq2(c, s), weights)
    x = torch.randn(2, 3, 256)
    rotation = Rotation(128, (-1, 1) * 128)
    rotated = (x * torch.tensor(rotation.signs)).reshape(-1, 128) @ matrix_hadamard(128)
    expected = rotated.reshape_as(x) @ weights.T
    torch.testing.assert_close(linear_reference(x, c, s, rotation), expected, atol=2e-5, rtol=1e-5)


def manifest():
    return {
        "tensor_name": "blk.0.ssm_out.weight",
        "logical_shape": [8, 12],
        "context": {
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": 4,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.sign_widths": [4, 12],
            "prism.hadamard.sign_values": [1] * 4 + [-1, 1] * 6,
            "prism.hadamard.weight_names": ["blk.0.ssm_out.weight"],
            "prism.hadamard.gdn_v_grouped": True,
            "qwen35.ssm.group_count": 2,
            "qwen35.ssm.time_step_rank": 6,
        },
    }


def test_manifest_resolves_width_specific_signs_and_geometry():
    m = manifest()
    assert rotation_from_manifest(m) == Rotation(4, (-1, 1) * 6, 2, 3)
    m["context"]["prism.hadamard.sign_mode"] = "identity"
    assert rotation_from_manifest(m).signs == (1,) * 12


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", 2),
        ("block_size", 3),
        ("axis", "output"),
        ("transform", "unknown"),
        ("sign_mode", "random"),
        ("sign_widths", [4, 4]),
        ("sign_values", [0] * 16),
        ("weight_names", ["token_embd.weight"]),
    ],
)
def test_bad_rotation_metadata_rejected(key, value):
    m = copy.deepcopy(manifest())
    m["context"]["prism.hadamard." + key] = value
    with pytest.raises(ValueError):
        rotation_from_manifest(m)


@pytest.mark.parametrize("block", [0, 3, 32])
def test_invalid_hadamard_dimensions(block):
    with pytest.raises(ValueError):
        hadamard(torch.ones(16), block)
