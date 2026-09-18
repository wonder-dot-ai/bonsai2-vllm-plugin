"""Compare multi-row DP4A with per-Q8-group tensor-core verification."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from vllm_bonsai2.integer_kernels import _integer_gemv
from vllm_bonsai2.verification_kernels import _q8_float_mma, _q8_mma, _q8_pair_gemv, _q8_shared_gemv


def measure(fn, flush):
    fn()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(9):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--float-mma", action="store_true")
    parser.add_argument("--half2", action="store_true")
    parser.add_argument("--interleave-only", action="store_true")
    parser.add_argument("--pair", action="store_true")
    parser.add_argument("--shared", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output file")
    torch.manual_seed(740)
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    cases = []
    for n, k in [(37, 1024), (34816, 5120), (5120, 17408)]:
        w = torch.randint(0, 256, (n, k // 4), device="cuda", dtype=torch.uint8)
        s = torch.rand(n, k // 128, device="cuda", dtype=torch.float16)
        for m in [1, 4, 8]:
            q = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8).view(torch.int32)
            d = torch.rand(m, k // 32, device="cuda", dtype=torch.float16)
            base, y = torch.empty(m, n, device="cuda"), torch.empty(m, n, device="cuda")

            def dp4a():
                return _integer_gemv[(triton.cdiv(n, 16), m)](
                    q, d, w, s, base, n, k, 16, 32, num_warps=4, enable_fp_fusion=False
                )

            dp_ms = measure(dp4a, flush)
            results = []
            configurations = [(32, 1, 4), (32, 4, 4), (64, 4, 4), (32, 8, 4), (64, 4, 8)]
            if args.float_mma:
                configurations = [
                    (64, 128, 4),
                    (64, 256, 4),
                    (128, 128, 4),
                    (32, 128, 4),
                    (64, 64, 4),
                ]
            if args.interleave_only:
                configurations = []
            if args.pair:
                configurations = [(8, 32, 2), (16, 32, 4), (16, 64, 4), (32, 32, 4), (8, 64, 2)]
            if args.shared:
                configurations = [(2, 32, 4), (4, 32, 4), (8, 16, 4), (8, 32, 4), (4, 32, 8)]
            for bn, bg, warps in configurations:

                def mma():
                    if args.shared:
                        return _q8_shared_gemv[(triton.cdiv(n, bn), triton.cdiv(m, 4))](
                            q,
                            d,
                            w,
                            s,
                            y,
                            None,
                            m,
                            n,
                            k,
                            4,
                            bn,
                            bg,
                            False,
                            num_warps=warps,
                            enable_fp_fusion=False,
                        )
                    if args.pair:
                        return _q8_pair_gemv[(triton.cdiv(n, bn), m)](
                            q,
                            d,
                            w,
                            s,
                            y,
                            n,
                            k,
                            bn,
                            bg,
                            num_warps=warps,
                            enable_fp_fusion=False,
                        )
                    kernel = _q8_float_mma if args.float_mma else _q8_mma
                    extra = {"HALF2": args.half2} if args.float_mma else {}
                    return kernel[(triton.cdiv(n, bn), 1)](
                        q,
                        d,
                        w,
                        s,
                        y,
                        None,
                        m,
                        n,
                        k,
                        16,
                        bn,
                        bg,
                        False,
                        num_warps=warps,
                        enable_fp_fusion=False,
                        **extra,
                    )

                compiled = mma()
                relative = ((y - base).norm() / base.norm()).item()
                if not args.float_mma:
                    assert relative < 5e-7, relative
                results.append(
                    {
                        "tile": [bn, bg, warps],
                        "ms": measure(mma, flush),
                        "relative_l2": relative,
                        "passes_dp4a_tight_gate": relative < 5e-7,
                        "registers": compiled.n_regs,
                        "spills": compiled.n_spills,
                    }
                )
            case = {"shape": [m, n, k], "dp4a_ms": dp_ms, "mma": results}

            def interleaved():
                return _integer_gemv[(m, triton.cdiv(n, 16))](
                    q,
                    d,
                    w,
                    s,
                    y,
                    n,
                    k,
                    16,
                    32,
                    num_warps=4,
                    INTERLEAVE=True,
                    enable_fp_fusion=False,
                )

            case["interleaved_ms"] = measure(interleaved, flush)
            torch.testing.assert_close(y, base, atol=0, rtol=0)
            cases.append(case)
            args.output.write_text(json.dumps(cases, indent=2) + "\n")
            print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
