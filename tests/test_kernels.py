"""Numerics, tail masks and capture/replay contracts for the A100 kernels."""

import pytest
import torch

from vllm_bonsai2.kernels import packed_embedding, packed_linear
from vllm_bonsai2.reference import hadamard, unpack_pq2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def inputs(width=5120, rows=37):
    torch.manual_seed(71)
    codes = torch.randint(0, 256, (rows, width // 128, 32), device="cuda", dtype=torch.uint8)
    scales = (torch.rand(rows, width // 128, device="cuda") * 0.1).half()
    scales[0] = 0
    signs = (torch.randint(0, 2, (width,), device="cuda") * 2 - 1).float()
    return codes, scales, signs


@pytest.mark.parametrize("batch,width", [(1, 5120), (4, 6144), (5, 5120), (17, 17408), (128, 5120)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_projection_matches_fp32_definition(batch, width, dtype):
    torch.backends.cuda.matmul.allow_tf32 = False
    codes, scales, signs = inputs(width)
    # Noncontiguous activations, partial row tiles and every two-bit code.
    x = torch.randn(width, batch, device="cuda", dtype=dtype).T
    expected = hadamard(x.float() * signs, 1024) @ unpack_pq2(codes, scales).T
    actual = packed_linear(x, codes, scales, signs, 1024)
    torch.testing.assert_close(actual, expected, atol=8e-5, rtol=2e-5)
    assert ((actual - expected).norm() / expected.norm()).item() < 2e-6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_embedding_is_exact_with_repeated_ids(dtype):
    codes, scales, signs = inputs()
    ids = torch.tensor([[0, 36, 2], [2, 17, 36]], device="cuda").T
    expected = ((hadamard(unpack_pq2(codes, scales), 1024) * signs)[ids]).to(dtype)
    actual = packed_embedding(ids, codes, scales, signs, 1024, dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("batch", [1, 7])
def test_cuda_graph_replay_uses_updated_inputs(batch):
    codes, scales, signs = inputs()
    x = torch.randn(batch, 5120, device="cuda")
    ids = torch.zeros(batch, dtype=torch.long, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            packed_linear(x, codes, scales, signs, 1024)
            packed_embedding(ids, codes, scales, signs, 1024, torch.bfloat16)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = packed_linear(x, codes, scales, signs, 1024)
        e = packed_embedding(ids, codes, scales, signs, 1024, torch.bfloat16)
    for token in [1, 36, 0]:
        x.normal_()
        ids.fill_(token)
        graph.replay()
        torch.testing.assert_close(y, packed_linear(x, codes, scales, signs, 1024), atol=0, rtol=0)
        torch.testing.assert_close(
            e, packed_embedding(ids, codes, scales, signs, 1024, torch.bfloat16), atol=0, rtol=0
        )


def test_compile_dynamic_batch_and_empty_inputs():
    codes, scales, signs = inputs()
    compiled = torch.compile(packed_linear, fullgraph=True, dynamic=True)
    for batch in [1, 8, 17]:
        x = torch.randn(batch, 5120, device="cuda")
        torch.testing.assert_close(
            compiled(x, codes, scales, signs, 1024),
            packed_linear(x, codes, scales, signs, 1024),
            atol=0,
            rtol=0,
        )
    x = torch.empty(0, 5120, device="cuda")
    assert packed_linear(x, codes, scales, signs, 1024).shape == (0, 37)
    ids = torch.empty(0, dtype=torch.long, device="cuda")
    assert packed_embedding(ids, codes, scales, signs, 1024, torch.float32).shape == (0, 5120)
