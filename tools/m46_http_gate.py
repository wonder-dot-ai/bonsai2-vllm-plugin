"""Frozen greedy token gate over HTTP, including a repeat after state reuse."""

import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    session = requests.Session()
    session.trust_env = False
    corpus = json.loads(Path("tests/data/m3/corpus.json").read_text())
    records = []
    for case in [*corpus["cases"], corpus["cases"][0]]:
        response = session.post(
            args.base_url + "/v1/completions",
            json={
                "model": "bonsai2",
                "prompt": case["input_ids"],
                "temperature": 0,
                "max_tokens": corpus["max_new_tokens"],
                "return_token_ids": True,
            },
            timeout=90,
        )
        response.raise_for_status()
        result = response.json()
        tokens = result["choices"][0]["token_ids"]
        records.append(
            {
                "name": case["name"],
                "tokens": tokens,
                "expected": case["expected_tokens"],
                "exact": tokens == case["expected_tokens"],
            }
        )
    report = {"passed": all(c["exact"] for c in records), "cases": records}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit("Frozen greedy HTTP gate failed")


if __name__ == "__main__":
    main()
