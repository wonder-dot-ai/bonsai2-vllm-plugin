"""Matched HTTP completion workload; run each server on the otherwise idle GPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output file")
    session = requests.Session()
    session.trust_env = False
    models = session.get(args.base_url + "/v1/models", timeout=10)
    models.raise_for_status()
    model = models.json()["data"][0]["id"]
    corpus = json.loads((args.corpus or Path("tests/data/m3/corpus.json")).read_text())
    report = {
        "engine": args.engine,
        "model": model,
        "warmup_requests": 1,
        "timed_requests_per_case": 3,
        "timing": "non-streaming HTTP request wall time",
        "cases": [],
    }
    for case in corpus["cases"] if args.corpus else corpus["cases"][:2]:
        body = {
            "model": model,
            "prompt": case["input_ids"],
            "temperature": 0,
            "max_tokens": 128,
            "ignore_eos": True,
            "cache_prompt": False,
            "stream": False,
            "seed": 123,
        }
        times = []
        outputs = []
        token_ids = []
        if args.engine != "prism":
            body["return_token_ids"] = True
        for run in range(4):
            start = time.perf_counter()
            result = session.post(args.base_url + "/v1/completions", json=body, timeout=300)
            result.raise_for_status()
            seconds = time.perf_counter() - start
            value = result.json()
            assert value["usage"]["completion_tokens"] == 128, value
            assert value["usage"]["prompt_tokens"] == len(case["input_ids"]), value
            if run:
                times.append(seconds)
                outputs.append(value["choices"][0]["text"])
                token_ids.append(value["choices"][0].get("token_ids"))
        report["cases"].append(
            {
                "name": case["name"],
                "input_ids": case["input_ids"],
                "input_tokens": len(case["input_ids"]),
                "output_tokens": 128,
                "seconds": times,
                "median_seconds": statistics.median(times),
                "tokens_per_second": 128 / statistics.median(times),
                "outputs": outputs,
                "token_ids": token_ids,
            }
        )
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report["cases"][-1].items()
                    if k not in ("input_ids", "outputs", "token_ids")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
