import argparse
import json
from pathlib import Path

import torch
from m46_verification_bench import measure

from vllm_bonsai2.integer_kernels import integer_project
from vllm_bonsai2.marlin_kernels import marlin_project, prepare_marlin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output file")
    flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    records = []
    for n, k in [(34816, 5120), (5120, 17408), (248320, 5120)]:
        codes = torch.randint(0, 256, (n, k // 128, 32), device="cuda", dtype=torch.uint8)
        scales = torch.rand(n, k // 128, device="cuda").half()
        signs = torch.ones(k, device="cuda")
        weight, scale, workspace = prepare_marlin(codes, scales)
        for m in [1, 4, 8]:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            row = {
                "shape": [m, n, k],
                "marlin_ms": measure(
                    lambda: marlin_project(x, weight, scale, signs, workspace, 1024, n), flush
                ),
                "legacy_ms": measure(
                    lambda: integer_project(x, codes, scales, signs, 1024, None), flush
                ),
            }
            records.append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
