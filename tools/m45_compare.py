"""Own a temporary local server, benchmark it, optionally profile, then shut down."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

from vllm_bonsai2.paths import llama_server_path, model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["prism", "integer", "triton"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--speculative-config", type=json.loads)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--benchmark-corpus", type=Path)
    parser.add_argument("--extra-corpus", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    port = 8091 if args.engine == "prism" else 8092
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", port))
    if args.engine == "prism":
        command = [
            str(llama_server_path()),
            "-m",
            str(model_path()),
            "-ngl",
            "99",
            "-c",
            "2048",
            "-np",
            "1",
            "-b",
            "128",
            "-ub",
            "128",
            "-fa",
            "on",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    else:
        query_length = 1 + (args.speculative_config or {}).get("num_speculative_tokens", 0)
        compilation = {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [query_length * i for i in [1, 2, 3, 4]],
        }
        command = [
            str(Path(sys.executable).parent / "vllm"),
            "serve",
            "converted/bonsai2-pq2",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            "bonsai2",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "4",
            "--max-num-batched-tokens",
            "128",
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--no-enable-prefix-caching",
            "--enable-chunked-prefill",
            "--attention-backend",
            "FLASH_ATTN",
            "--compilation-config",
            json.dumps(compilation),
        ]
        if args.speculative_config:
            command += ["--speculative-config", json.dumps(args.speculative_config)]
        if args.profile:
            profiler = {
                "profiler": "torch",
                "torch_profiler_dir": str(args.output.resolve() / "trace"),
                "torch_profiler_with_stack": False,
                "ignore_frontend": True,
                "delay_iterations": 2,
                "max_iterations": 8,
            }
            command += ["--profiler-config", json.dumps(profiler)]
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if args.engine != "prism":
        env["BONSAI_BACKEND"] = args.engine
    (args.output / "command.json").write_text(json.dumps(command, indent=2))
    provenance = {
        "engine": args.engine,
        "kernel_options": {key: env.get(key) for key in ["BONSAI_VERIFY_KERNEL"]},
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *sorted(Path("src/vllm_bonsai2").glob("*.py")),
                Path("tools/m45_compare.py"),
                Path("tools/m45_http_bench.py"),
            ]
        },
        "corpus_sha256": hashlib.sha256(Path("tests/data/m3/corpus.json").read_bytes()).hexdigest(),
        "benchmark_corpus_sha256": hashlib.sha256(args.benchmark_corpus.read_bytes()).hexdigest()
        if args.benchmark_corpus
        else None,
        "extra_corpus_sha256": hashlib.sha256(args.extra_corpus.read_bytes()).hexdigest()
        if args.extra_corpus
        else None,
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        session = requests.Session()
        session.trust_env = False
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited: see {args.output / 'server.log'}")
                try:
                    if session.get(base + "/health", timeout=2).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup timed out")
                time.sleep(0.5)
            subprocess.run(
                [
                    sys.executable,
                    "tools/m45_http_bench.py",
                    "--base-url",
                    base,
                    "--engine",
                    args.engine,
                    "--output",
                    str(args.output / "benchmark.json"),
                    *(["--corpus", str(args.benchmark_corpus)] if args.benchmark_corpus else []),
                ],
                check=True,
                timeout=300,
            )
            if args.engine != "prism":
                response = session.get(base + "/metrics", timeout=10)
                response.raise_for_status()
                (args.output / "metrics.txt").write_text(response.text)
            if args.extra_corpus:
                subprocess.run(
                    [
                        sys.executable,
                        "tools/m45_http_bench.py",
                        "--base-url",
                        base,
                        "--engine",
                        args.engine,
                        "--output",
                        str(args.output / "extra-benchmark.json"),
                        "--corpus",
                        str(args.extra_corpus),
                    ],
                    check=True,
                    timeout=300,
                )
                response = session.get(base + "/metrics", timeout=10)
                response.raise_for_status()
                (args.output / "extra-metrics.txt").write_text(response.text)
            if args.validate:
                subprocess.run(
                    [
                        sys.executable,
                        "tools/m46_http_gate.py",
                        "--base-url",
                        base,
                        "--output",
                        str(args.output / "gate.json"),
                    ],
                    check=True,
                    timeout=300,
                )
            if args.profile and args.engine != "prism":
                session.post(base + "/start_profile", timeout=30).raise_for_status()
                ids = json.loads(Path("tests/data/m3/corpus.json").read_text())["cases"][0][
                    "input_ids"
                ]
                session.post(
                    base + "/v1/completions",
                    json={
                        "model": "bonsai2",
                        "prompt": ids,
                        "max_tokens": 32,
                        "ignore_eos": True,
                        "temperature": 0,
                    },
                    timeout=180,
                ).raise_for_status()
                session.post(base + "/stop_profile", timeout=120).raise_for_status()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()
