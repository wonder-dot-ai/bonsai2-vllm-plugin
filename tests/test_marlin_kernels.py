import pytest
import torch

from vllm_bonsai2.marlin_kernels import _pack_gptq, marlin_project, prepare_marlin
from vllm_bonsai2.reference import hadamard, unpack_pq2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_all_pq2_word_values_repack_losslessly_with_reordered_rows():
    n, k = 512, 1024
    codes = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.uint16)
    codes = codes.view(torch.uint8).reshape(n, k // 128, 32)
    rows = torch.arange(n - 1, -1, -1, device="cuda")
    packed = torch.empty(k // 8, n, device="cuda", dtype=torch.int32)
    _pack_gptq[((n * (k // 8) + 255) // 256,)](codes, packed, rows, n, k, True)
    unpacked = ((packed.T.long()[:, :, None] >> (4 * torch.arange(8, device="cuda"))) & 15) - 7
    expected = (
        codes[rows].reshape(n, -1).long()[:, :, None] >> (2 * torch.arange(4, device="cuda"))
    ) & 3
    torch.testing.assert_close(unpacked.reshape(n, k), expected.reshape(n, k), atol=0, rtol=0)


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_marlin_matches_fp32_definition_and_graph_replay(batch):
    torch.manual_seed(658)
    n, k = 128, 1024
    codes = torch.randint(0, 256, (n, k // 128, 32), device="cuda", dtype=torch.uint8)
    scales = (torch.rand(n, k // 128, device="cuda") * 0.1).half()
    signs = (2 * torch.randint(0, 2, (k,), device="cuda") - 1).float()
    rows = torch.randperm(n, device="cuda")
    weight, scale, workspace = prepare_marlin(codes, scales, rows)
    x = torch.randn(batch, k, device="cuda")
    expected = hadamard(x * signs, 1024) @ unpack_pq2(codes, scales)[rows].T
    actual = marlin_project(x, weight, scale, signs, workspace, 1024, n)
    relative = ((actual - expected).norm() / expected.norm()).item()
    assert relative < 5e-4, relative
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            marlin_project(x, weight, scale, signs, workspace, 1024, n)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = marlin_project(x, weight, scale, signs, workspace, 1024, n)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(
        captured, marlin_project(x, weight, scale, signs, workspace, 1024, n), atol=0, rtol=0
    )
