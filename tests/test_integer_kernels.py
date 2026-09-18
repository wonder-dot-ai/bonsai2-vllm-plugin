"""The Q8 backend has its own native-Prism oracle; FP32 tolerances stay unchanged."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_bonsai2.convert import sha256_file
from vllm_bonsai2.integer_kernels import _transform_q8, integer_linear, integer_project
from vllm_bonsai2.kernels import packed_linear
from vllm_bonsai2.reference import hadamard

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DATA = Path(__file__).parent / "data" / "m2"


@pytest.mark.parametrize("name", ["gate", "down", "ssm_out"])
@pytest.mark.parametrize("batch", [1, 4])
def test_matches_immutable_native_packed_goldens(name, batch):
    path = DATA / f"{name}.npz"
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert sha256_file(path) == manifest["fixtures"][name]["sha256"]
    with np.load(path) as arrays:
        x = torch.from_numpy(arrays["input"]).cuda()
        if name == "ssm_out":
            x = x.reshape(-1, 3, 16, x.shape[-1] // 48).transpose(1, 2).reshape(x.shape)
        codes = torch.from_numpy(arrays["qweight"]).cuda()
        scales = torch.from_numpy(arrays["scales"]).cuda()
        signs = torch.from_numpy(arrays["signs"]).cuda().float()
        result = torch.cat(
            [integer_linear(row, codes, scales, signs, 1024) for row in x.split(batch)]
        )
        expected = torch.from_numpy(arrays["cuda_packed"]).cuda()
        torch.testing.assert_close(result, expected, atol=3e-6, rtol=3e-6)
        assert ((result - expected).norm() / expected.norm()).item() < 5e-7


def test_quantizer_zero_and_rounding_ties_and_fp16_scales():
    transformed = torch.zeros(2, 1024, device="cuda")
    transformed[1] = torch.tensor(
        [127, -127, 2.5, -2.5, 3.5, -3.5, 0, 1] * 128, device="cuda", dtype=torch.float32
    )
    x = hadamard(transformed, 1024)
    signs = torch.ones(1024, device="cuda")
    q = torch.empty(2, 256, device="cuda", dtype=torch.int32)
    d = torch.empty(2, 32, device="cuda", dtype=torch.float16)
    _transform_q8[(2, 1)](x, signs, q, d, 1024, 1024, enable_fp_fusion=False)
    expected = transformed.sign() * (transformed.abs() + 0.5).floor()
    torch.testing.assert_close(q.view(torch.int8).float(), expected, atol=0, rtol=0)
    torch.testing.assert_close(d[0], torch.zeros_like(d[0]), atol=0, rtol=0)
    torch.testing.assert_close(d[1], torch.ones_like(d[1]), atol=0, rtol=0)


@pytest.mark.parametrize("batch", [1, 4, 7])
def test_graph_replay_tail_rows_and_prefill_fallback(batch):
    torch.manual_seed(914)
    x = torch.randn(batch, 5120, device="cuda", dtype=torch.bfloat16)
    codes = torch.randint(0, 256, (37, 40, 32), dtype=torch.uint8, device="cuda")
    scales = torch.rand(37, 40, device="cuda").half()
    signs = (torch.randint(0, 2, (5120,), device="cuda") * 2 - 1).float()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            integer_linear(x, codes, scales, signs, 1024)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = integer_linear(x, codes, scales, signs, 1024)
    for _ in range(3):
        x.normal_()
        graph.replay()
        expected = integer_linear(x, codes, scales, signs, 1024)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if batch > 4:
            torch.testing.assert_close(
                actual, packed_linear(x, codes, scales, signs, 1024), atol=0, rtol=0
            )


@pytest.mark.parametrize("batch", [4, 6, 8, 16])
@pytest.mark.parametrize("reorder", [False, True])
def test_interleaved_verification_matches_individual_q8_rows(monkeypatch, batch, reorder):
    monkeypatch.setenv("BONSAI_VERIFY_KERNEL", "interleave")
    torch.manual_seed(230)
    x = torch.randn(batch, 5120, device="cuda", dtype=torch.bfloat16)
    codes = torch.randint(0, 256, (37, 40, 32), dtype=torch.uint8, device="cuda")
    scales = torch.rand(37, 40, device="cuda").half()
    signs = (torch.randint(0, 2, (5120,), device="cuda") * 2 - 1).float()
    rows = torch.randperm(37, device="cuda") if reorder else None
    result = integer_project(x, codes, scales, signs, 1024, rows)
    reference = torch.cat([integer_linear(row, codes, scales, signs, 1024) for row in x.split(1)])
    if rows is not None:
        reference = reference.index_select(-1, rows)
    torch.testing.assert_close(result, reference.to(x.dtype), atol=0, rtol=0)
