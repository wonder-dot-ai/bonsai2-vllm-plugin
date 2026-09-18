"""PQ2 x Q8 decode using the pinned Prism quantization convention and DP4A.

This is a separate numerical backend. The FP32 M4 kernels and tests remain intact.
"""

import os

import torch
import triton
import triton.language as tl

from .kernels import _fwht, _validate, packed_linear


@triton.jit
def _transform_q8(X, SIGNS, Q, D, K: tl.constexpr, BLOCK: tl.constexpr):
    m, block = tl.program_id(0), tl.program_id(1)
    idx = tl.arange(0, BLOCK)
    k = block * BLOCK + idx
    v = tl.load(X + m * K + k).to(tl.float32) * tl.load(SIGNS + k)
    v = _fwht(v, idx, BLOCK).reshape((BLOCK // 32, 32))
    scale = tl.max(tl.abs(v), axis=1) / 127.0
    # Match the pinned Prism build's -use_fast_math division near rounding ties.
    normalized = tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[v, tl.where(scale[:, None] > 0, scale[:, None], 1.0)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    # C roundf: nearest with ties away from zero, not round-to-even.
    quant = (tl.floor(tl.abs(normalized) + 0.5) * tl.where(normalized < 0, -1, 1)).to(tl.int32)
    groups = block * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    tl.store(D + m * (K // 32) + groups, scale)
    quant = quant.reshape((BLOCK // 4, 4))
    shifts = 8 * tl.arange(0, 4)
    packed = tl.sum((quant & 255).to(tl.uint32) << shifts[None, :], axis=1)
    b = block * (BLOCK // 4) + tl.arange(0, BLOCK // 4)
    tl.store(Q + m * (K // 4) + b, packed)


@triton.jit
def _dp4a(W, X):
    # prmt selects {-1,0,1,2} bytes from a fixed table, one per two-bit symbol.
    selectors = (W & 3) | ((W & 12) << 2) | ((W & 48) << 4) | ((W & 192) << 6)
    return tl.inline_asm_elementwise(
        "{ .reg .b32 w; prmt.b32 w, 0x020100ff, 0x020100ff, $1; dp4a.s32.s32 $0, w, $2, 0; }",
        constraints="=r,r,r",
        args=[selectors, X],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _integer_gemv(
    Q,
    D,
    W,
    S,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BG: tl.constexpr,
    ROWS=None,
    REORDER: tl.constexpr = False,
    INTERLEAVE: tl.constexpr = False,
):
    m = tl.program_id(0) if INTERLEAVE else tl.program_id(1)
    n_pid = tl.program_id(1) if INTERLEAVE else tl.program_id(0)
    n = n_pid * BN + tl.arange(0, BN)
    weight_row = tl.load(ROWS + n, n < N, other=0) if REORDER else n
    groups = tl.arange(0, BG)
    bytes_ = tl.arange(0, 8)
    acc = tl.full((BN, BG), 0, tl.float32)
    for start in range(tl.cdiv(K // 32, BG)):
        g = start * BG + groups
        b = g[:, None] * 8 + bytes_[None, :]
        codes = tl.load(
            W + weight_row[:, None, None] * (K // 4) + b[None, :, :],
            (n[:, None, None] < N) & (g[None, :, None] < K // 32),
            other=0,
        ).to(tl.int32)
        x = tl.load(Q + m * (K // 4) + b, g[:, None] < K // 32, other=0)
        dot = tl.sum(_dp4a(codes, x[None, :, :]), axis=2).to(tl.float32)
        ws = tl.load(
            S + weight_row[:, None] * (K // 128) + g[None, :] // 4,
            (n[:, None] < N) & (g[None, :] < K // 32),
            other=0,
        ).to(tl.float32)
        xs = tl.load(D + m * (K // 32) + g, g < K // 32, other=0).to(tl.float32)
        acc = acc + (ws * xs[None, :]) * dot
    tl.store(Y + m * N + n, tl.sum(acc, axis=1), n < N)


@torch.library.custom_op("bonsai2::integer_linear", mutates_args=())
def integer_linear(
    x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, signs: torch.Tensor, block: int
) -> torch.Tensor:
    k = _validate(codes, scales, signs, block)
    if block < 32 or x.shape[-1] != k or x.device != codes.device:
        raise ValueError("Q8 path requires matching CUDA activations and block >= 32")
    x = x.reshape(-1, k).contiguous()
    m, n = x.shape[0], codes.shape[0]
    if m > 4:
        return packed_linear(x, codes, scales, signs, block)
    q = torch.empty((m, k // 4), device=x.device, dtype=torch.int32)
    d = torch.empty((m, k // 32), device=x.device, dtype=torch.float16)
    y = torch.empty((m, n), device=x.device, dtype=torch.float32)
    if m and n:
        _transform_q8[(m, k // block)](x, signs, q, d, k, block, enable_fp_fusion=False)
        _integer_gemv[(triton.cdiv(n, 16), m)](
            q, d, codes, scales, y, n, k, BN=16, BG=32, num_warps=4, enable_fp_fusion=False
        )
    return y


@integer_linear.register_fake
def _integer_fake(x, codes, scales, signs, block):
    return x.new_empty((x.numel() // x.shape[-1], codes.shape[0]), dtype=torch.float32)


@torch.library.custom_op("bonsai2::integer_project", mutates_args=())
def integer_project(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    signs: torch.Tensor,
    block: int,
    rows: torch.Tensor | None,
) -> torch.Tensor:
    """Shared transform for merged projections; write activation dtype in HF row order."""
    k = _validate(codes, scales, signs, block)
    if block < 32 or x.shape[-1] != k or x.device != codes.device:
        raise ValueError("Q8 path requires matching CUDA activations and block >= 32")
    x = x.reshape(-1, k).contiguous()
    m, n = x.shape[0], codes.shape[0]
    verification = os.environ.get("BONSAI_VERIFY_KERNEL", "legacy")
    if verification not in ("legacy", "interleave", "marlin"):
        raise ValueError("BONSAI_VERIFY_KERNEL must be legacy, interleave or marlin")
    interleave = verification == "interleave"
    if m > (16 if interleave else 4):
        result = packed_linear(x, codes, scales, signs, block)
        if rows is not None:
            result = result.index_select(-1, rows)
        return result.to(x.dtype)
    q = torch.empty((m, k // 4), device=x.device, dtype=torch.int32)
    d = torch.empty((m, k // 32), device=x.device, dtype=torch.float16)
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m and n:
        _transform_q8[(m, k // block)](x, signs, q, d, k, block, enable_fp_fusion=False)
        grid = (m, triton.cdiv(n, 16)) if interleave else (triton.cdiv(n, 16), m)
        _integer_gemv[grid](
            q,
            d,
            codes,
            scales,
            y,
            n,
            k,
            BN=16,
            BG=32,
            num_warps=4,
            ROWS=rows,
            REORDER=rows is not None,
            INTERLEAVE=interleave,
            enable_fp_fusion=False,
        )
    return y


@integer_project.register_fake
def _project_fake(x, codes, scales, signs, block, rows):
    return x.new_empty((x.numel() // x.shape[-1], codes.shape[0]))
