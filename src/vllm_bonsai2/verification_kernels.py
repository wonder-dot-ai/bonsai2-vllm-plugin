"""Experimental Q8 tensor-core verification: preserve per-32 activation scales."""

import triton
import triton.language as tl


@triton.jit
def _q8_shared_gemv(
    Q,
    D,
    W,
    S,
    Y,
    ROWS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BG: tl.constexpr,
    REORDER: tl.constexpr,
):
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    weight_row = tl.load(ROWS + n, n < N, other=0) if REORDER else n
    groups = tl.arange(0, BG)
    bytes_ = tl.arange(0, 8)
    acc = tl.full((BM, BN, BG), 0, tl.float32)
    for start in range(tl.cdiv(K // 32, BG)):
        g = start * BG + groups
        b = g[:, None] * 8 + bytes_[None, :]
        code = tl.load(
            W + weight_row[:, None, None] * (K // 4) + b[None, :, :],
            (n[:, None, None] < N) & (g[None, :, None] < K // 32),
            other=0,
        ).to(tl.int32)
        selector = (code | (code << 4)) & 0x0F0F
        selector = (selector | (selector << 2)) & 0x3333
        decoded = tl.inline_asm_elementwise(
            "prmt.b32 $0,0x020100ff,0x020100ff,$1;",
            constraints="=r,r",
            args=[selector],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        x = tl.load(
            Q + m[:, None, None] * (K // 4) + b[None, :, :],
            (m[:, None, None] < M) & (g[None, :, None] < K // 32),
            other=0,
        )
        dots = tl.inline_asm_elementwise(
            "dp4a.s32.s32 $0,$1,$2,0;",
            constraints="=r,r,r",
            args=[decoded[None, :, :, :], x[:, None, :, :]],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        dot = tl.sum(dots, 3).to(tl.float32)
        ws = tl.load(
            S + weight_row[:, None] * (K // 128) + g[None, :] // 4,
            (n[:, None] < N) & (g[None, :] < K // 32),
            other=0,
        ).to(tl.float32)
        xs = tl.load(
            D + m[:, None] * (K // 32) + g[None, :],
            (m[:, None] < M) & (g[None, :] < K // 32),
            other=0,
        ).to(tl.float32)
        acc += (ws[None, :, :] * xs[:, None, :]) * dot
    tl.store(Y + m[:, None] * N + n[None, :], tl.sum(acc, 2), (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _pair_dot(W, X0, X1):
    return tl.inline_asm_elementwise(
        "{ .reg .b32 a,b,c,d,e,f; "
        "and.b32 a,$1,0x7777; shr.u32 b,$1,2; and.b32 b,b,0x7777; "
        "prmt.b32 c,0x020100ff,0x020100ff,a; "
        "prmt.b32 d,0x020100ff,0x020100ff,b; "
        "prmt.b32 e,c,d,0x5140; prmt.b32 f,c,d,0x7362; "
        "dp4a.s32.s32 a,e,$2,0; dp4a.s32.s32 $0,f,$3,a; }",
        constraints="=r,r,r,r",
        args=[W, X0, X1],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _q8_pair_gemv(
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
):
    m = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    weight_row = tl.load(ROWS + n, n < N, other=0) if REORDER else n
    groups = tl.arange(0, BG)
    pairs = tl.arange(0, 4)
    acc = tl.full((BN, BG), 0, tl.float32)
    W16 = W.to(tl.pointer_type(tl.uint16))
    for start in range(tl.cdiv(K // 32, BG)):
        g = start * BG + groups
        b = g[:, None] * 4 + pairs[None, :]
        code = tl.load(
            W16 + weight_row[:, None, None] * (K // 8) + b[None, :, :],
            (n[:, None, None] < N) & (g[None, :, None] < K // 32),
            other=0,
        ).to(tl.int32)
        x0 = tl.load(Q + m * (K // 4) + b * 2, g[:, None] < K // 32, other=0)
        x1 = tl.load(Q + m * (K // 4) + b * 2 + 1, g[:, None] < K // 32, other=0)
        dot = tl.sum(_pair_dot(code, x0[None, :, :], x1[None, :, :]), axis=2).to(tl.float32)
        ws = tl.load(
            S + weight_row[:, None] * (K // 128) + g[None, :] // 4,
            (n[:, None] < N) & (g[None, :] < K // 32),
            other=0,
        ).to(tl.float32)
        xs = tl.load(D + m * (K // 32) + g, g < K // 32, other=0).to(tl.float32)
        acc += (ws * xs[None, :]) * dot
    tl.store(Y + m * N + n, tl.sum(acc, 1), n < N)


@triton.jit
def _q8_float_mma(
    Q,
    D,
    W,
    S,
    Y,
    ROWS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    REORDER: tl.constexpr,
    HALF2: tl.constexpr = False,
):
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    weight_row = tl.load(ROWS + n, n < N, other=0) if REORDER else n
    kk = tl.arange(0, BK)
    byte = tl.arange(0, BK // 4)
    shift = tl.arange(0, 4) * 2
    acc = tl.full((BM, BN), 0, tl.float32)
    Q8 = Q.to(tl.pointer_type(tl.int8))
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        x = tl.load(
            Q8 + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), other=0
        ).to(tl.float32)
        xs = tl.load(
            D + m[:, None] * (K // 32) + k[None, :] // 32,
            (m[:, None] < M) & (k[None, :] < K),
            other=0,
        ).to(tl.float32)
        code = tl.load(
            W + weight_row[:, None] * (K // 4) + start * (BK // 4) + byte[None, :],
            (n[:, None] < N) & (start * (BK // 4) + byte[None, :] < K // 4),
            other=0,
        )
        w = (
            (((code[:, :, None].to(tl.int32) >> shift[None, None, :]) & 3) - 1)
            .reshape((BN, BK))
            .to(tl.float32)
        )
        ws = tl.load(
            S + weight_row[:, None] * (K // 128) + k[None, :] // 128,
            (n[:, None] < N) & (k[None, :] < K),
            other=0,
        ).to(tl.float32)
        if HALF2:
            value = x * xs
            high = value.to(tl.float16)
            low = (value - high.to(tl.float32)).to(tl.float16)
            weight = tl.trans((w * ws).to(tl.float16))
            acc = tl.dot(high, weight, acc)
            acc = tl.dot(low, weight, acc)
        else:
            acc = tl.dot(x * xs, tl.trans(w * ws), acc, input_precision="tf32x3")
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _q8_mma(
    Q,
    D,
    W,
    S,
    Y,
    ROWS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BG: tl.constexpr,
    REORDER: tl.constexpr,
):
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    weight_row = tl.load(ROWS + n, n < N, other=0) if REORDER else n
    groups = tl.arange(0, BG)
    kk = tl.arange(0, 32)
    byte = tl.arange(0, 8)
    shift = tl.arange(0, 4) * 2
    acc = tl.full((BM, BN), 0, tl.float32)
    Q8 = Q.to(tl.pointer_type(tl.int8))
    for start in range(tl.cdiv(K // 32, BG)):
        g = start * BG + groups
        x = tl.load(
            Q8 + m[None, :, None] * K + g[:, None, None] * 32 + kk[None, None, :],
            (m[None, :, None] < M) & (g[:, None, None] < K // 32),
            other=0,
        )
        code = tl.load(
            W + weight_row[None, :, None] * (K // 4) + g[:, None, None] * 8 + byte[None, None, :],
            (n[None, :, None] < N) & (g[:, None, None] < K // 32),
            other=0,
        )
        weight = ((code[:, :, :, None].to(tl.int32) >> shift[None, None, None, :]) & 3) - 1
        weight = weight.reshape((BG, BN, 32)).trans(0, 2, 1).to(tl.int8)
        dot = tl.dot(x, weight).to(tl.float32)
        ws = tl.load(
            S + weight_row[None, :] * (K // 128) + g[:, None] // 4,
            (n[None, :] < N) & (g[:, None] < K // 32),
            other=0,
        ).to(tl.float32)
        xs = tl.load(
            D + m[None, :] * (K // 32) + g[:, None],
            (m[None, :] < M) & (g[:, None] < K // 32),
            other=0,
        ).to(tl.float32)
        acc += tl.sum(dot * (xs[:, :, None] * ws[:, None, :]), axis=0)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))
