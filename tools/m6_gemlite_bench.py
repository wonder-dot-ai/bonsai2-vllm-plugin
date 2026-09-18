"""Compare exact PQ2 repacking into pinned upstream GemLite against Marlin."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("artifacts/m6-gemlite-src").resolve()))
import gemlite
import torch
from m46_verification_bench import measure

from vllm_bonsai2.kernels import _transform
from vllm_bonsai2.marlin_kernels import marlin_project, prepare_marlin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    torch.manual_seed(606)
    gemlite.set_acc_dtype(gemlite.DType.FP32)
    gemlite.set_autotune("fast")
    flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    records = []
    for n, k in [(34816, 5120), (5120, 17408), (248320, 5120)]:
        c = torch.randint(0, 256, (n, k // 128, 32), device="cuda", dtype=torch.uint8)
        s = torch.rand(n, k // 128, device="cuda").half()
        signs = torch.ones(k, device="cuda")
        mw, ms, workspace = prepare_marlin(c, s)
        layer = gemlite.GemLiteLinearTriton(
            W_nbits=2,
            group_size=128,
            in_features=k,
            out_features=n,
            input_dtype=gemlite.DType.FP16,
            output_dtype=gemlite.DType.FP16,
            acc_dtype=gemlite.DType.FP32,
        ).pack(c.reshape(n, k // 4), s, 1, packing_bitwidth=8, packed=True, fma_mode=False)
        for m in [4, 8, 16]:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            rotated = torch.empty_like(x, dtype=torch.float16)

            def run():
                _transform[(m, k // 1024)](x, signs, rotated, k, 1024, enable_fp_fusion=False)
                return layer(rotated).bfloat16()

            actual = run()
            ref = marlin_project(x, mw, ms, signs, workspace, 1024, n)
            row = {
                "shape": [m, n, k],
                "gemlite_ms": measure(run, flush),
                "marlin_ms": measure(
                    lambda: marlin_project(x, mw, ms, signs, workspace, 1024, n), flush
                ),
                "relative_l2_vs_marlin_bf16": (
                    (actual.float() - ref.float()).norm() / ref.float().norm()
                ).item(),
            }
            records.append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(records, indent=2) + "\n")
            gemlite.cache_config("artifacts/m6-gemlite-config.json")


if __name__ == "__main__":
    main()
