"""Cold-cache DP4A tile search; measurements are not full-model throughput."""

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch
import triton

from vllm_bonsai2.integer_kernels import _integer_gemv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output file")
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    cases = []
    configurations = list(itertools.product([1, 2, 4, 8, 16], [16, 32, 64, 128], [1, 2, 4]))
    for n, k in [(34816, 5120), (5120, 17408), (248320, 5120)]:
        w = torch.randint(0, 256, (n, k // 4), dtype=torch.uint8, device="cuda")
        s = torch.rand(n, k // 128, dtype=torch.float16, device="cuda")
        q = torch.randint(0, 2**30, (1, k // 4), dtype=torch.int32, device="cuda")
        d = torch.rand(1, k // 32, dtype=torch.float16, device="cuda")
        y = torch.empty(1, n, device="cuda")
        baseline = torch.empty_like(y)
        _integer_gemv[(triton.cdiv(n, 16), 1)](
            q, d, w, s, baseline, n, k, 16, 32, num_warps=4, enable_fp_fusion=False
        )
        rows = []
        for bn, bg, warps in configurations:

            def launch():
                return _integer_gemv[(triton.cdiv(n, bn), 1)](
                    q, d, w, s, y, n, k, bn, bg, num_warps=warps, enable_fp_fusion=False
                )

            try:
                compiled = launch()
                relative = ((y - baseline).norm() / baseline.norm()).item()
                assert relative < 1e-6, relative
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    launch()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                samples = []
                for _ in range(9):
                    flush.zero_()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                record = {
                    "tile": [bn, bg, warps],
                    "ms": statistics.median(samples),
                    "registers": compiled.n_regs,
                    "spills": compiled.n_spills,
                    "relative_l2": relative,
                }
            except triton.OutOfResources as exc:
                record = {"tile": [bn, bg, warps], "error": str(exc)}
            rows.append(record)
        rows.sort(key=lambda r: r.get("ms", float("inf")))
        cases.append({"shape": [n, k], "configs": rows})
        args.output.write_text(json.dumps(cases, indent=2) + "\n")
        print(json.dumps({"shape": [n, k], "best": rows[:5]}), flush=True)


if __name__ == "__main__":
    main()
