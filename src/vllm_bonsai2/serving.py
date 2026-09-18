"""Portable launcher for the validated single-A100 serving configuration."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path


def build_command(model, draft, host="127.0.0.1", port=8000, prefill_tokens=256):
    command = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        str(model),
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        "bonsai2",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        str(prefill_tokens),
        "--gpu-memory-utilization",
        "0.85",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--attention-backend",
        "FLASH_ATTN",
        "--compilation-config",
        json.dumps(
            {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [8 if draft else 1],
            }
        ),
    ]
    if draft:
        command += [
            "--speculative-config",
            json.dumps(
                {
                    "method": "dflash",
                    "model": str(draft),
                    "num_speculative_tokens": 7,
                }
            ),
        ]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("converted/bonsai2-pq2"))
    parser.add_argument("--draft", type=Path, default=Path("artifacts/m46-dflash2"))
    parser.add_argument("--no-draft", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    draft = None if args.no_draft else args.draft
    for path in [args.model, *([draft] if draft else [])]:
        if not (path / "config.json").is_file():
            parser.error(f"Missing {path}/config.json; run tools/download_models.py first")
    command = build_command(args.model, draft, args.host, args.port)
    env = os.environ.copy()
    env.update(BONSAI_BACKEND="integer", BONSAI_VERIFY_KERNEL="marlin")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if args.dry_run:
        print("BONSAI_BACKEND=integer BONSAI_VERIFY_KERNEL=marlin " + shlex.join(command))
        return
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
