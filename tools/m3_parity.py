"""Frozen-token full-model comparison against the local Prism oracle server."""

import argparse
import json
from pathlib import Path

import numpy as np
import requests
from vllm import LLM, SamplingParams


def parity_passes(report, corpus):
    expected = {case["name"]: case["expected_tokens"] for case in corpus["cases"]}
    wanted = {
        (case["name"], step)
        for case in corpus["cases"]
        for step in range(min(corpus["teacher_forcing_steps"], len(case["expected_tokens"])))
    }
    return (
        len(report["cases"]) == len(expected)
        and {c["name"] for c in report["cases"]} == set(expected)
        and all(
            c["prism_tokens"] == c["vllm_tokens"] == expected[c["name"]] for c in report["cases"]
        )
        and len(report["decisions"]) == len(wanted)
        and {(d["name"], d["step"]) for d in report["decisions"]} == wanted
        and all(
            d["top1_equal"]
            and d["finite_common"] == 248320
            and np.isfinite(d["total_variation"])
            and d["total_variation"] <= corpus["max_total_variation"]
            for d in report["decisions"]
        )
    )


def oracle(ids, count, probs=20):
    response = requests.post(
        "http://127.0.0.1:8091/completion",
        json={
            "prompt": ids,
            "n_predict": count,
            "temperature": 0,
            "n_probs": probs,
            "return_tokens": True,
            "cache_prompt": False,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/m3-parity"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--corpus", type=Path, default=Path("tests/data/m3/corpus.json"))
    parser.add_argument("--check-report", type=Path)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    if args.check_report:
        passed = parity_passes(json.loads(args.check_report.read_text()), corpus)
        print(json.dumps({"passed": passed, "report": str(args.check_report)}))
        raise SystemExit(0 if passed else 1)
    if args.output.exists():
        raise ValueError("choose a new output directory")
    args.output.mkdir(parents=True)
    references = []
    for case in corpus["cases"]:
        reference = oracle(case["input_ids"], corpus["max_new_tokens"])
        if reference["tokens"] != case["expected_tokens"]:
            raise ValueError(f"Prism reference drift: {case['name']}")
        references.append(
            {"name": case["name"], "input_ids": case["input_ids"], "prism": reference}
        )
    (args.output / "reference.json").write_text(
        json.dumps(references, ensure_ascii=False, indent=2)
    )
    llm = LLM(
        model="converted/bonsai2-pq2",
        enforce_eager=True,
        dtype=args.dtype,
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.45,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        attention_backend="FLASH_ATTN",
        max_logprobs=-1,
    )
    greedy = llm.generate(
        [{"prompt_token_ids": r["input_ids"]} for r in references],
        SamplingParams(temperature=0, max_tokens=corpus["max_new_tokens"], logprobs=20),
    )
    report = {"dtype": args.dtype, "cases": []}
    for r, v in zip(references, greedy, strict=True):
        c = v.outputs[0]
        report["cases"].append(
            {
                "name": r["name"],
                "prism_tokens": r["prism"]["tokens"],
                "vllm_tokens": list(c.token_ids),
                "prism_text": r["prism"]["content"],
                "vllm_text": c.text,
                "exact_greedy": r["prism"]["tokens"] == list(c.token_ids),
            }
        )
    # Freeze the reference's first four decision prefixes, so divergence cannot
    # change the inputs being compared. Capture all vocabulary log probabilities.
    decisions = []
    for r in references:
        for step in range(min(corpus["teacher_forcing_steps"], len(r["prism"]["tokens"]))):
            ids = r["input_ids"] + r["prism"]["tokens"][:step]
            decisions.append({"name": r["name"], "step": step, "input_ids": ids})
    results = llm.generate(
        [{"prompt_token_ids": d["input_ids"]} for d in decisions],
        SamplingParams(temperature=0, max_tokens=1, logprobs=-1),
    )
    distribution = []
    for i, (d, result) in enumerate(zip(decisions, results, strict=True)):
        reference = oracle(d["input_ids"], 1, 248320)
        row = reference["completion_probabilities"][0]
        p = np.full(248320, -np.inf, dtype=np.float64)
        for item in row["top_logprobs"]:
            p[item["id"]] = item["logprob"]
        q = np.full(248320, -np.inf, dtype=np.float64)
        for k, item in result.outputs[0].logprobs[0].items():
            q[k] = item.logprob
        common = np.isfinite(p) & np.isfinite(q)
        pp, qq = np.exp(p), np.exp(q)
        pp /= pp.sum()
        qq /= qq.sum()
        top = int(np.argmax(p))
        topq = int(np.argmax(q))
        top20p = set(np.argsort(p)[-20:])
        top20q = set(np.argsort(q)[-20:])
        rec = {
            "name": d["name"],
            "step": d["step"],
            "top1_equal": top == topq,
            "prism_top1": top,
            "vllm_top1": topq,
            "finite_common": int(common.sum()),
            "top20_overlap": len(top20p & top20q) / 20,
            "total_variation": float(np.abs(pp - qq).sum() / 2),
            "prism_top1_logprob_delta": float(q[top] - p[top]),
            "reference_margin": float(np.sort(p)[-1] - np.sort(p)[-2]),
        }
        distribution.append(rec)
        np.savez_compressed(
            args.output / f"logprobs-{i:03d}.npz",
            prism=p,
            vllm=q,
            input_ids=np.array(d["input_ids"]),
        )
    report["decisions"] = distribution
    report["exact_greedy_count"] = sum(r["exact_greedy"] for r in report["cases"])
    report["top1_match_count"] = sum(r["top1_equal"] for r in distribution)
    report["max_total_variation"] = max(r["total_variation"] for r in distribution)
    report["passed"] = parity_passes(report, corpus)
    report["corpus"] = str(args.corpus)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps({k: v for k, v in report.items() if k not in ["cases", "decisions"]}, indent=2)
    )

    if not report["passed"]:
        raise SystemExit("M3 model parity gate failed; see report.json")


if __name__ == "__main__":
    main()
