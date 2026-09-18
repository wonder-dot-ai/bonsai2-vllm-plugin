"""Experimental FP16 tensor-core projection that reads packed two-bit weights."""

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from m46_verification_bench import measure

from vllm_bonsai2.kernels import _transform
from vllm_bonsai2.marlin_kernels import marlin_project, prepare_marlin


@triton.jit
def pq2_mma(
    X,
    W,
    S,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    m = tl.program_id(0) * 16 + tl.arange(0, 16)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    kk = tl.arange(0, BK)
    acc = tl.full((16, BN), 0, tl.float32)
    for base in range(split, tl.cdiv(K, BK), SPLIT):
        k = base * BK + kk
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        q = tl.load(
            W + (k[:, None] // 4) * N + n[None, :], (k[:, None] < K) & (n[None, :] < N), 0
        ).to(tl.int32)
        s = tl.load(
            S + (k[:, None] // 128) * N + n[None, :], (k[:, None] < K) & (n[None, :] < N), 0
        )
        w = (((q >> (2 * (k[:, None] % 4))) & 3) - 1).to(tl.float16) * s
        acc = tl.dot(x, w, acc)
    tl.store(
        Y + split * M * N + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.manual_seed(608)
    flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    records = []
    for n, k in [(34816, 5120), (5120, 17408), (248320, 5120)]:
        c = torch.randint(0, 256, (n, k // 128, 32), device="cuda", dtype=torch.uint8)
        s = torch.rand(n, k // 128, device="cuda").half()
        signs = torch.ones(k, device="cuda")
        w = c.reshape(n, k // 4).T.contiguous()
        scale = s.T.contiguous()
        mw, ms, workspace = prepare_marlin(c, s)
        m = 8
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        rotated = torch.empty_like(x, dtype=torch.float16)
        _transform[(m, k // 1024)](x, signs, rotated, k, 1024, enable_fp_fusion=False)
        ref = marlin_project(x, mw, ms, signs, workspace, 1024, n)
        baseline = measure(lambda: marlin_project(x, mw, ms, signs, workspace, 1024, n), flush)
        rows = []
        for bn in [32, 64, 128]:
            for bk in [64, 128, 256]:
                for split in [1, 2, 4]:
                    y = torch.empty((split, m, n), device="cuda")

                    def run():
                        pq2_mma[(1, triton.cdiv(n, bn), split)](
                            rotated, w, scale, y, m, n, k, bn, bk, split, num_warps=4, num_stages=3
                        )
                        return y.sum(0).half().bfloat16()

                    try:
                        out = run()
                    except triton.OutOfResources:
                        continue
                    error = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                    ms_time = measure(run, flush)
                    row = {
                        "bn": bn,
                        "bk": bk,
                        "split": split,
                        "ms": ms_time,
                        "relative_l2_vs_marlin_bf16": error,
                    }
                    rows.append(row)
        record = {
            "shape": [m, n, k],
            "marlin_ms": baseline,
            "best": sorted(rows, key=lambda r: r["ms"])[:5],
            "all": rows,
        }
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "all"}), flush=True)


if __name__ == "__main__":
    main()
