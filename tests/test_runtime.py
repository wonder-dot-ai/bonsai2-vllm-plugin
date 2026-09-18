import numpy as np
import pytest
import torch
from torch import nn

from vllm_bonsai2.full_checkpoint import untile_rows
from vllm_bonsai2.quantization import Bonsai2Method
from vllm_bonsai2.reference import hadamard, unpack_pq2


def method(inverse=False, chunks=3):
    name = "token_embd.weight" if inverse else "blk.0.ffn_gate.weight"
    config = {
        "context": {
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": 128,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.sign_widths": [128],
            "prism.hadamard.sign_values": [-1, 1] * 64,
            "prism.hadamard.weight_names": [name],
            "prism.hadamard.inverse_weight_names": [name],
        },
        "reference_chunk_rows": chunks,
        "layers": {
            "test": {
                "inverse": inverse,
                "segments": [{"name": name, "shape": [7, 128], "output_order": "none"}],
            }
        },
    }
    m = Bonsai2Method(config, "test")
    layer = nn.Module()
    m.create_weights(layer, 128, [7], 128, 7, torch.float32)
    rng = np.random.default_rng(57)
    layer.qweight_0.copy_(torch.from_numpy(rng.integers(0, 256, (7, 1, 32), dtype=np.uint8)))
    layer.scales_0.copy_(torch.linspace(0.1, 0.7, 7).half().reshape(7, 1))
    return m, layer


def test_chunked_projection_matches_dense_definition():
    m, layer = method()
    x = torch.randn(2, 3, 128)
    expected = (
        hadamard(x * layer.bonsai_signs_0, 128) @ unpack_pq2(layer.qweight_0, layer.scales_0).T
    )
    torch.testing.assert_close(m.apply(layer, x), expected)


def test_embedding_inverse_order_repeated_tokens_and_shape():
    m, layer = method(inverse=True)
    ids = torch.tensor([[0, 6, 0], [2, 3, 1]])
    weights = unpack_pq2(layer.qweight_0, layer.scales_0)
    expected = (hadamard(weights, 128) * layer.bonsai_signs_0)[ids]
    torch.testing.assert_close(m.embedding(layer, ids), expected)
    wrong = hadamard(weights * layer.bonsai_signs_0, 128)[ids]
    assert not torch.allclose(expected, wrong)


def test_inverse_head_order():
    # grouped: K0_v0,K0_v1,K0_v2,K1_v0,K1_v1,K1_v2
    tiled = torch.tensor([0, 3, 1, 4, 2, 5])[:, None]
    assert untile_rows(tiled, 2, 3, 1).flatten().tolist() == [0, 1, 2, 3, 4, 5]


def test_reject_partitioned_weight_shapes():
    m, _ = method()
    with pytest.raises(ValueError, match="TP=1"):
        m.create_weights(nn.Module(), 64, [7], 128, 7, torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", [1, 4, 7])
@pytest.mark.parametrize(
    "segments",
    [
        [(17, "none"), (17, "none")],
        [(20, "qkv_untile"), (12, "untile")],
        [(16, "none"), (4, "none"), (4, "none")],
    ],
)
def test_merged_integer_projections_preserve_rows_and_parameter_values(
    monkeypatch, batch, segments
):
    monkeypatch.setenv("BONSAI_BACKEND", "integer")
    base, _ = method()
    config = base.config
    config["context"].update(
        {
            "qwen35.ssm.group_count": 2,
            "qwen35.ssm.time_step_rank": 6,
            "qwen35.ssm.inner_size": 12,
            "qwen35.ssm.state_size": 2,
        }
    )
    name = config["layers"]["test"]["segments"][0]["name"]
    config["layers"]["test"]["segments"] = [
        {"name": name, "shape": [n, 128], "output_order": order} for n, order in segments
    ]
    m = Bonsai2Method(config, "test")
    layer = nn.Module()
    sizes = [s[0] for s in segments]
    m.create_weights(layer, 128, sizes, 128, sum(sizes), torch.bfloat16)
    layer.cuda()
    torch.manual_seed(394)
    for i in range(len(sizes)):
        getattr(layer, f"qweight_{i}").random_(0, 256)
        getattr(layer, f"scales_{i}").uniform_(0.01, 0.1)
    original = {name: p.clone() for name, p in layer.named_parameters()}
    x = torch.randn(batch, 128, dtype=torch.bfloat16, device="cuda")
    expected = m.apply(layer, x)
    m.process_weights_after_loading(layer)
    actual = m.apply(layer, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, param in layer.named_parameters():
        torch.testing.assert_close(param, original[name], rtol=0, atol=0)
    assert (
        layer.qweight_0.untyped_storage().data_ptr()
        == layer.bonsai_merged_codes.untyped_storage().data_ptr()
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            m.apply(layer, x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = m.apply(layer, x)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(captured, m.apply(layer, x), rtol=0, atol=0)
