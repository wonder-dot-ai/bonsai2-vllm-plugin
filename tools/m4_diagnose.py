"""Read-only diagnosis: cold-cache packed GEMV timing and longer model generation."""

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import triton

from vllm_bonsai2.kernels import _gemv, _transform


def kernels():
    config = json.loads(Path("converted/bonsai2-pq2/config.json").read_text())[
        "quantization_config"
    ]
    shapes = Counter(
        tuple(s["shape"])
        for spec in config["layers"].values()
        if not spec["inverse"]
        for s in spec["segments"]
    )
    flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    results = []
    for (n, k), count in shapes.items():
        x = torch.randn(1, k, device="cuda")
        signs = torch.ones(k, device="cuda")
        rotated = torch.empty_like(x)
        w = torch.randint(0, 256, (n, k // 128, 32), dtype=torch.uint8, device="cuda")
        s = torch.ones(n, k // 128, dtype=torch.float16, device="cuda")
        y = torch.empty(1, n, device="cuda")

        def gemv():
            return _gemv[(triton.cdiv(n, 8), 1)](
                rotated, w, s, y, n, k, BN=8, BK=512, num_warps=4, enable_fp_fusion=False
            )

        def transform():
            _transform[(1, k // 1024)](x, signs, rotated, k, 1024)

        transform()
        compiled = gemv()
        record = {
            "shape": [n, k],
            "count": count,
            "registers": compiled.n_regs,
            "spills": compiled.n_spills,
        }
        for name, fn in [("gemv", gemv), ("transform", transform)]:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fn()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            samples = []
            for _ in range(21):
                flush.zero_()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            record[name + "_ms"] = statistics.median(samples)
        results.append(record)
    report = {
        "cases": results,
        "weighted_gemv_ms": sum(r["count"] * r["gemv_ms"] for r in results),
        "weighted_transform_ms": sum(r["count"] * r["transform_ms"] for r in results),
    }
    Path("artifacts/m4-kernel-diagnosis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def model():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="converted/bonsai2-pq2",
        dtype="bfloat16",
        enforce_eager=False,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 3, 4],
        },
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        attention_backend="FLASH_ATTN",
    )
    ids = json.loads(Path("tests/data/m3/corpus.json").read_text())["cases"][0]["input_ids"]
    rows = []
    for length in [16, 128]:
        params = SamplingParams(temperature=0, max_tokens=length, ignore_eos=True)
        llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            out = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)
            samples.append(time.perf_counter() - start)
            assert len(out[0].outputs[0].token_ids) == length
        rows.append(
            {
                "output_tokens": length,
                "seconds": samples,
                "median_seconds": statistics.median(samples),
                "tokens_per_second": length / statistics.median(samples),
            }
        )
    report = {
        "cases": rows,
        "incremental_token_ms": (rows[1]["median_seconds"] - rows[0]["median_seconds"])
        / 112
        * 1000,
    }
    Path("artifacts/m4-longer-generation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["kernels", "model"])
    args = parser.parse_args()
    kernels() if args.mode == "kernels" else model()
