"""A100 PQ2 kernels: FP32 FWHT, fused packed GEMV/GEMM, inverse embedding.

No dense model-weight cache or host synchronization. The torch custom-op boundary
keeps dynamic batch dispatch outside torch.compile and permits CUDA graph replay.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwht(v, idx, BLOCK: tl.constexpr):
    v = v * (BLOCK**-0.5)
    for stage in tl.static_range(0, tl.constexpr(BLOCK.bit_length() - 1)):
        stride = 1 << stage
        other = tl.gather(v, idx ^ stride, 0)
        v = tl.where((idx & stride) == 0, v + other, other - v)
    return v


@triton.jit
def _transform(X, S, Y, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    idx = tl.arange(0, BLOCK)
    k = block * BLOCK + idx
    x = tl.load(X + row * K + k).to(tl.float32)
    signs = tl.load(S + k)
    y = _fwht(x * signs, idx, BLOCK)
    tl.store(Y + row * K + k, y)


@triton.jit
def _gemv(X, W, S, Y, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    m = tl.program_id(1)
    kk = tl.arange(0, BK)
    acc = tl.full((BN, BK), 0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        k = base * BK + kk
        code = tl.load(
            W + row[:, None] * (K // 4) + k[None, :] // 4,
            (row[:, None] < N) & (k[None, :] < K),
            other=0,
        ).to(tl.int32)
        scale = tl.load(
            S + row[:, None] * (K // 128) + k[None, :] // 128,
            (row[:, None] < N) & (k[None, :] < K),
            other=0,
        ).to(tl.float32)
        weight = (((code >> (2 * (k[None, :] % 4))) & 3) - 1).to(tl.float32) * scale
        x = tl.load(X + m * K + k, k < K, other=0)
        acc = acc + weight * x[None, :]
    y = tl.sum(acc, 1)
    tl.store(Y + m * N + row, y, row < N)


@triton.jit(do_not_specialize=["M"])
def _gemm(
    X,
    W,
    S,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        k = base * BK + kk
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), other=0)
        code = tl.load(
            W + n[None, :] * (K // 4) + k[:, None] // 4,
            (n[None, :] < N) & (k[:, None] < K),
            other=0,
        ).to(tl.int32)
        scale = tl.load(
            S + n[None, :] * (K // 128) + k[:, None] // 128,
            (n[None, :] < N) & (k[:, None] < K),
            other=0,
        ).to(tl.float32)
        w = (((code >> (2 * (k[:, None] % 4))) & 3) - 1).to(tl.float32) * scale
        acc = tl.dot(x, w, acc, input_precision="tf32x3")
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _embedding(IDS, W, S, SIGNS, Y, K: tl.constexpr, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    block = tl.program_id(1)
    row = tl.load(IDS + m)
    idx = tl.arange(0, BLOCK)
    k = block * BLOCK + idx
    code = tl.load(W + row * (K // 4) + k // 4).to(tl.int32)
    scale = tl.load(S + row * (K // 128) + k // 128).to(tl.float32)
    weight = (((code >> (2 * (k % 4))) & 3) - 1).to(tl.float32) * scale
    y = _fwht(weight, idx, BLOCK) * tl.load(SIGNS + k)
    tl.store(Y + m * K + k, y)


def _validate(codes, scales, signs, block):
    if codes.dtype != torch.uint8 or codes.ndim != 3 or codes.shape[-1] != 32:
        raise ValueError("qweight must be uint8 [out, in/128, 32]")
    if scales.dtype != torch.float16 or scales.shape != codes.shape[:2]:
        raise ValueError("scales must be float16 [out, in/128]")
    width = codes.shape[1] * 128
    if block < 2 or block > 1024 or block & (block - 1) or width % block:
        raise ValueError("kernel supports power-of-two Hadamard blocks in [2,1024]")
    if signs.shape != (width,) or signs.dtype != torch.float32:
        raise ValueError("signs must be FP32 [input width]")
    if not codes.is_cuda or any(t.device != codes.device for t in (scales, signs)):
        raise ValueError("kernel inputs must share a CUDA device")
    if not all(t.is_contiguous() for t in (codes, scales, signs)):
        raise ValueError("packed weights, scales and signs must be contiguous")
    return width


@torch.library.custom_op("bonsai2::packed_linear", mutates_args=())
def packed_linear(
    x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, signs: torch.Tensor, block: int
) -> torch.Tensor:
    k = _validate(codes, scales, signs, block)
    if x.device != codes.device or x.shape[-1] != k or not x.is_floating_point():
        raise ValueError("invalid activation shape, dtype or device")
    x = x.reshape(-1, k).contiguous()
    m, n = x.shape[0], codes.shape[0]
    transformed = torch.empty((m, k), device=x.device, dtype=torch.float32)
    output = torch.empty((m, n), device=x.device, dtype=torch.float32)
    if m and n:
        _transform[(m, k // block)](x, signs, transformed, k, block)
        if m <= 4:
            _gemv[(triton.cdiv(n, 8), m)](
                transformed,
                codes,
                scales,
                output,
                n,
                k,
                BN=8,
                BK=512,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            bm, bn = (16, 64) if m <= 16 else (64, 128)
            _gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                transformed, codes, scales, output, m, n, k, BM=bm, BN=bn, BK=32, num_warps=4
            )
    return output


@packed_linear.register_fake
def _linear_fake(x, codes, scales, signs, block):
    return x.new_empty((x.numel() // x.shape[-1], codes.shape[0]), dtype=torch.float32)


@torch.library.custom_op("bonsai2::packed_embedding", mutates_args=())
def packed_embedding(
    ids: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    signs: torch.Tensor,
    block: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    k = _validate(codes, scales, signs, block)
    if ids.device != codes.device or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("embedding IDs must be int32/int64 on the weight device")
    output = torch.empty((*ids.shape, k), device=ids.device, dtype=dtype)
    if ids.numel():
        _embedding[(ids.numel(), k // block)](
            ids.contiguous(), codes, scales, signs, output, k, block
        )
    return output


@packed_embedding.register_fake
def _embedding_fake(ids, codes, scales, signs, block, dtype):
    return ids.new_empty((*ids.shape, codes.shape[1] * 128), dtype=dtype)
