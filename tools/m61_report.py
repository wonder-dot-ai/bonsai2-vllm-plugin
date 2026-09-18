"""Compare fixed-corpus streaming runs without changing their acceptance inputs."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(run):
    report = json.loads((run / "benchmark.json").read_text())
    gate = json.loads((run / "gate.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    cases = report["cases"]
    return {
        "run": str(run),
        "corpus_sha256": provenance["benchmark_corpus_sha256"],
        "frozen_gate_passed": gate["passed"],
        "combined_http_tps": sum(c["output_tokens"] for c in cases)
        / sum(c["median_seconds"] for c in cases),
        "combined_decode_tps": sum(
            statistics.median(s["decode_tokens"] for s in c["streaming"]) for c in cases
        )
        / sum(statistics.median(s["decode_seconds"] for s in c["streaming"]) for c in cases),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output file")
    baseline, candidate = summarize(args.baseline), summarize(args.candidate)
    assert baseline["corpus_sha256"] == candidate["corpus_sha256"]
    assert len(baseline["cases"]) == len(candidate["cases"]) == 4
    comparisons = []
    for before, after in zip(baseline["cases"], candidate["cases"], strict=True):
        assert before["name"] == after["name"]
        assert before["input_ids"] == after["input_ids"]
        assert before["output_tokens"] == after["output_tokens"] == 128
        assert len(before["token_ids"]) == len(after["token_ids"]) == 3
        comparisons.append(
            {
                "name": before["name"],
                "http_tps_before": before["tokens_per_second"],
                "http_tps_after": after["tokens_per_second"],
                "http_speedup": after["tokens_per_second"] / before["tokens_per_second"],
                "decode_tps_before": before["median_decode_tokens_per_second"],
                "decode_tps_after": after["median_decode_tokens_per_second"],
                "ttft_ms_before": before["median_ttft_seconds"] * 1000,
                "ttft_ms_after": after["median_ttft_seconds"] * 1000,
                "all_output_ids_match_baseline": all(
                    ids == before["token_ids"][0]
                    for ids in before["token_ids"] + after["token_ids"]
                ),
                "candidate_repeats_exact": len({tuple(t) for t in after["token_ids"]}) == 1,
            }
        )
    result = {
        "baseline": {k: v for k, v in baseline.items() if k != "cases"},
        "candidate": {k: v for k, v in candidate.items() if k != "cases"},
        "cases": comparisons,
        "all_outputs_match": all(c["all_output_ids_match_baseline"] for c in comparisons),
        "all_cases_http_200_pass": all(c["http_tps_after"] >= 200 for c in comparisons),
        "all_cases_decode_200_pass": all(c["decode_tps_after"] >= 200 for c in comparisons),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
