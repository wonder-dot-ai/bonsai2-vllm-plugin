"""M4 frozen full-vocabulary parity and fixed-work end-to-end timing.

Uses M3's saved Prism vectors so the oracle occupies no GPU memory during timing.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["reference", "triton", "integer"], default="triton")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--speculative-config", type=json.loads)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-seqs", type=int, default=4)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output file")
    os.environ["BONSAI_BACKEND"] = args.backend
    corpus = json.loads(Path("tests/data/m3/corpus.json").read_text())
    compilation = {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [
            i * (1 + (args.speculative_config or {}).get("num_speculative_tokens", 0))
            for i in range(1, args.max_seqs + 1)
        ],
    }
    llm = LLM(
        model="converted/bonsai2-pq2",
        enforce_eager=not args.graph,
        compilation_config=compilation if args.graph else None,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=args.max_seqs,
        max_num_batched_tokens=128,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        attention_backend="FLASH_ATTN",
        max_logprobs=-1,
        speculative_config=args.speculative_config,
    )
    report = {
        "backend": args.backend,
        "graph": args.graph,
        "max_seqs": args.max_seqs,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "speculative_config": args.speculative_config,
        "verification_kernel": os.environ.get("BONSAI_VERIFY_KERNEL", "legacy"),
    }
    report["source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path("src/vllm_bonsai2/kernels.py"),
            Path("src/vllm_bonsai2/integer_kernels.py"),
            Path("src/vllm_bonsai2/quantization.py"),
            Path("src/vllm_bonsai2/model.py"),
            Path("src/vllm_bonsai2/normalization.py"),
            Path("src/vllm_bonsai2/verification_kernels.py"),
            Path("src/vllm_bonsai2/marlin_kernels.py"),
            Path("tests/data/m3/corpus.json"),
        ]
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    if not args.skip_parity:
        generated = llm.generate(
            [{"prompt_token_ids": c["input_ids"]} for c in corpus["cases"]],
            SamplingParams(temperature=0, max_tokens=corpus["max_new_tokens"]),
        )
        cases = []
        prefixes = []
        for case, result in zip(corpus["cases"], generated, strict=True):
            tokens = list(result.outputs[0].token_ids)
            cases.append(
                {"name": case["name"], "tokens": tokens, "exact": tokens == case["expected_tokens"]}
            )
            for step in range(min(corpus["teacher_forcing_steps"], len(case["expected_tokens"]))):
                prefixes.append(
                    {
                        "name": case["name"],
                        "step": step,
                        "ids": case["input_ids"] + case["expected_tokens"][:step],
                    }
                )
        predicted = llm.generate(
            [{"prompt_token_ids": p["ids"]} for p in prefixes],
            SamplingParams(temperature=0, max_tokens=1, logprobs=-1),
        )
        decisions = []
        for i, (prefix, result) in enumerate(zip(prefixes, predicted, strict=True)):
            with np.load(f"artifacts/m3-parity-bf16/logprobs-{i:03d}.npz") as data:
                assert data["input_ids"].tolist() == prefix["ids"]
                p = data["prism"].copy()
                m3 = data["vllm"].copy()
            q = np.full_like(p, -np.inf)
            for token, value in result.outputs[0].logprobs[0].items():
                q[token] = value.logprob
            pp, qq = np.exp(p), np.exp(q)
            pp, qq = pp / pp.sum(), qq / qq.sum()
            m3p = np.exp(m3)
            m3p /= m3p.sum()
            decisions.append(
                {
                    "name": prefix["name"],
                    "step": prefix["step"],
                    "top1_equal": bool(p.argmax() == q.argmax()),
                    "finite_common": int((np.isfinite(p) & np.isfinite(q)).sum()),
                    "total_variation": float(np.abs(pp - qq).sum() / 2),
                    "tv_vs_m3": float(np.abs(m3p - qq).sum() / 2),
                }
            )
        report["parity"] = {
            "cases": cases,
            "decisions": decisions,
            "passed": all(c["exact"] for c in cases)
            and all(
                d["top1_equal"]
                and d["finite_common"] == 248320
                and d["total_variation"] <= corpus["max_total_variation"]
                for d in decisions
            ),
        }
        save()
        if not report["parity"]["passed"]:
            raise SystemExit("Frozen M3 parity gate failed")
        print("PARITY PASSED", flush=True)

    if not args.skip_bench:
        measurements = []
        modes = [("single", 1, False)]
        if args.max_seqs > 1:
            modes += [("serial", args.max_seqs, True), ("concurrent", args.max_seqs, False)]
        for mode, concurrent, serial in modes:
            prompts = [{"prompt_token_ids": c["input_ids"]} for c in corpus["cases"][:concurrent]]
            params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)

            def generate_workload():
                if serial:
                    return [llm.generate([p], params, use_tqdm=False)[0] for p in prompts]
                return llm.generate(prompts, params, use_tqdm=False)

            generate_workload()
            durations = []
            for _ in range(3):
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = generate_workload()
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
                assert sum(len(r.outputs[0].token_ids) for r in outputs) == 16 * concurrent
            measurements.append(
                {
                    "mode": mode,
                    "requests": concurrent,
                    "concurrent": 1 if serial else concurrent,
                    "input_tokens": sum(len(p["prompt_token_ids"]) for p in prompts),
                    "output_tokens": 16 * concurrent,
                    "seconds": durations,
                    "median_seconds": statistics.median(durations),
                    "output_tokens_per_second": 16 * concurrent / statistics.median(durations),
                }
            )
            report["benchmark"] = measurements
            save()
            print(json.dumps(measurements[-1]), flush=True)
    save()


if __name__ == "__main__":
    main()
