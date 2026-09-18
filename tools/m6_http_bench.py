"""Fixed English streaming benchmark, with distinct decode and HTTP rates."""

import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--engine", choices=["integer", "triton"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output file")
    session = requests.Session()
    session.trust_env = False

    def metrics():
        response = session.get(args.base_url + "/metrics", timeout=10)
        response.raise_for_status()
        result = {}
        for line in response.text.splitlines():
            if line.startswith("vllm:spec_decode") and "_total{" in line:
                key, value = line.rsplit(" ", 1)
                result[key] = float(value)
        return result

    models = session.get(args.base_url + "/v1/models", timeout=10)
    models.raise_for_status()
    model = models.json()["data"][0]["id"]
    corpus = json.loads((args.corpus or Path("tests/data/m6/english_fixed.json")).read_text())
    report = {
        "engine": args.engine,
        "model": model,
        "warmup_requests": 1,
        "timed_requests_per_case": 3,
        "timing": "HTTP wall time; decode excludes time and tokens in first token chunk",
        "cases": [],
    }
    for case in corpus["cases"]:
        before = metrics()
        body = {
            "model": model,
            "prompt": case["input_ids"],
            "temperature": 0,
            "max_tokens": 128,
            "ignore_eos": True,
            "cache_prompt": False,
            "stream": True,
            "stream_options": {"include_usage": True},
            "seed": 123,
        }
        times = []
        outputs = []
        token_ids = []
        streaming = []
        if args.engine != "prism":
            body["return_token_ids"] = True
        for run in range(4):
            start = time.perf_counter()
            result = session.post(
                args.base_url + "/v1/completions", json=body, stream=True, timeout=300
            )
            result.raise_for_status()
            ids, chunks, text = [], [], ""
            usage = None
            for line in result.iter_lines(chunk_size=1):
                if not line.startswith(b"data: "):
                    continue
                if line == b"data: [DONE]":
                    break
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("token_ids") or []
                    text += choice.get("text", "")
                    if delta:
                        chunks.append(
                            {"seconds": time.perf_counter() - start, "tokens": len(delta)}
                        )
                        ids.extend(delta)
            result.close()
            seconds = time.perf_counter() - start
            assert usage and usage["completion_tokens"] == len(ids) == 128, usage
            assert usage["prompt_tokens"] == len(case["input_ids"]), usage
            assert len(chunks) > 1, chunks
            if run:
                times.append(seconds)
                outputs.append(text)
                token_ids.append(ids)
                streaming.append(
                    {
                        "ttft_seconds": chunks[0]["seconds"],
                        "decode_tokens": 128 - chunks[0]["tokens"],
                        "decode_seconds": chunks[-1]["seconds"] - chunks[0]["seconds"],
                        "chunks": chunks,
                    }
                )
        report["cases"].append(
            {
                "name": case["name"],
                "input_ids": case["input_ids"],
                "input_tokens": len(case["input_ids"]),
                "output_tokens": 128,
                "seconds": times,
                "median_seconds": statistics.median(times),
                "tokens_per_second": 128 / statistics.median(times),
                "median_decode_tokens_per_second": statistics.median(
                    r["decode_tokens"] / r["decode_seconds"] for r in streaming
                ),
                "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in streaming),
                "streaming": streaming,
                "outputs": outputs,
                "token_ids": token_ids,
                "speculative_counter_delta": {
                    key: value - before.get(key, 0) for key, value in metrics().items()
                },
            }
        )
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report["cases"][-1].items()
                    if k not in ("input_ids", "outputs", "token_ids", "streaming")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
