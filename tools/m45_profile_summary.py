"""Summarize actual GPU kernel durations from a Chrome/PyTorch trace."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output file")
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as source:
        trace = json.load(source)
    durations = defaultdict(float)
    counts = defaultdict(int)
    for event in trace["traceEvents"]:
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            durations[event["name"]] += event["dur"] / 1000
            counts[event["name"]] += 1
    report = {
        "trace": str(args.trace),
        "note": "Sum of kernel durations, not request wall time; concurrent kernels may overlap.",
        "total_kernel_ms": sum(durations.values()),
        "kernels": [
            {"name": name, "milliseconds": ms, "count": counts[name]}
            for name, ms in sorted(durations.items(), key=lambda item: item[1], reverse=True)
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
