"""Fused Gemma RMSNorm preserving FP32 weights and unrounded residual arithmetic."""

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _norm(
    X,
    R,
    W,
    Y,
    NEXT_R,
    K: tl.constexpr,
    EPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * K + col, col < K, other=0).to(tl.float32)
    if HAS_RESIDUAL:
        value += tl.load(R + row * K + col, col < K, other=0).to(tl.float32)
    tl.store(NEXT_R + row * K + col, value, col < K)
    variance = tl.sum(value * value, 0) / K
    normalized = value * tl.rsqrt(variance + EPS)
    weight = tl.load(W + col, col < K, other=0).to(tl.float32) + 1.0
    tl.store(Y + row * K + col, normalized * weight, col < K)


@torch.library.custom_op("bonsai2::gemma_norm", mutates_args=())
def gemma_norm(
    x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor | None, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    y, next_r = torch.empty_like(x), torch.empty_like(x)
    k = x.shape[-1]
    if x.numel():
        _norm[(x.numel() // k,)](
            x,
            x if residual is None else residual.contiguous(),
            weight,
            y,
            next_r,
            k,
            epsilon,
            residual is not None,
            triton.next_power_of_2(k),
            enable_fp_fusion=False,
        )
    return y, next_r


@gemma_norm.register_fake
def _norm_fake(x, weight, residual, epsilon):
    return torch.empty_like(x), torch.empty_like(x)


class FusedGemmaRMSNorm(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.variance_epsilon = original.variance_epsilon

    def forward(self, x, residual=None):
        result, next_residual = gemma_norm(x, self.weight, residual, self.variance_epsilon)
        return result if residual is None else (result, next_residual)
