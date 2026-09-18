"""Check FP32 Gemma weight arithmetic and the unrounded residual contract."""

import pytest
import torch

from vllm_bonsai2.normalization import gemma_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1024, 5120])
@pytest.mark.parametrize("with_residual", [False, True])
def test_norm_fp32_semantics(dtype, width, with_residual):
    torch.manual_seed(305)
    x = torch.randn(7, width, device="cuda", dtype=dtype)
    weight = torch.randn(width, device="cuda", dtype=dtype) * 0.25
    residual = torch.randn_like(x) if with_residual else None
    value = x.float() if residual is None else x.float() + residual.float()
    expected = (
        value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * (weight.float() + 1)
    ).to(dtype)
    actual, next_r = gemma_norm(x, weight, residual, 1e-6)
    torch.testing.assert_close(next_r, value.to(dtype), atol=0, rtol=0)
    # Parallel reduction order can move values on a rounding boundary by one ULP.
    torch.testing.assert_close(actual, expected, atol=0, rtol=2 * torch.finfo(dtype).eps)
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 2e-4


def test_graph_replay_preserves_unrounded_residual():
    torch.manual_seed(672)
    x = torch.randn(1, 5120, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gemma_norm(x, weight, residual, 1e-6)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, next_r = gemma_norm(x, weight, residual, 1e-6)
    for _ in range(3):
        x.normal_()
        residual.normal_()
        graph.replay()
        value = x.float() + residual.float()
        expected = (
            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * (weight.float() + 1)
        ).to(x.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(next_r, value.to(x.dtype), rtol=0, atol=0)
