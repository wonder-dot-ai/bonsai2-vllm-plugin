"""Measure Marlin reduction modes without changing the selected runtime."""

import json
from pathlib import Path

import torch
from m46_verification_bench import measure
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types

from vllm_bonsai2.marlin_kernels import prepare_marlin


def main():
    path = Path("artifacts/m6-marlin-modes.json")
    assert not path.exists()
    torch.manual_seed(661)
    flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    records = []
    for n, k in [(34816, 5120), (5120, 17408), (248320, 5120)]:
        codes = torch.randint(0, 256, (n, k // 128, 32), device="cuda", dtype=torch.uint8)
        scales = torch.rand(n, k // 128, device="cuda").half()
        weight, scale, workspace = prepare_marlin(codes, scales)
        x = torch.randn(8, k, device="cuda", dtype=torch.float16)

        def run(atomic=False, fp32=True, out=None):
            return ops.marlin_gemm(
                x,
                out,
                weight,
                None,
                scale,
                None,
                None,
                None,
                None,
                None,
                workspace,
                scalar_types.uint4b8,
                8,
                n,
                k,
                use_atomic_add=atomic,
                use_fp32_reduce=fp32,
            )

        ref = run().clone()
        for atomic, fp32 in [(False, True), (False, False), (True, True), (True, False)]:
            out = run(atomic, fp32)
            row = {
                "shape": [8, n, k],
                "atomic": atomic,
                "fp32": fp32,
                "ms": measure(lambda: run(atomic, fp32), flush),
                "relative_l2_vs_default": (
                    (out.float() - ref.float()).norm() / ref.float().norm()
                ).item(),
                "repeat_exact": torch.equal(run(atomic, fp32), run(atomic, fp32)),
            }
            records.append(row)
            print(json.dumps(row), flush=True)
            path.write_text(json.dumps(records, indent=2) + "\n")
        out_bf16 = torch.empty((8, n), device="cuda", dtype=torch.bfloat16)
        try:
            actual = run(out=out_bf16)
            print(
                "bf16 output",
                actual.dtype,
                ((actual.float() - ref.float()).norm() / ref.float().norm()).item(),
                flush=True,
            )
        except RuntimeError as exc:
            print("bf16 output rejected:", str(exc).splitlines()[0], flush=True)


if __name__ == "__main__":
    main()
