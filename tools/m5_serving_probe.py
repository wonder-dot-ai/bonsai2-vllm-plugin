"""Owned local-server probe for prefix cache and hybrid-state serving lifecycle."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

from vllm_bonsai2.serving import build_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-draft", action="store_true")
    parser.add_argument("--prefix-entries", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = "http://127.0.0.1:8093"
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", 8093))
    command = build_command("converted/bonsai2-pq2", "artifacts/m46-dflash2", prefill_tokens=128)
    command[command.index("--port") + 1] = "8093"
    if not args.disable_prefix_caching:
        command[command.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
        command += ["--mamba-cache-mode", "align"]
    command[command.index("--max-num-seqs") + 1] = str(args.max_num_seqs)
    if args.no_draft:
        index = command.index("--speculative-config")
        del command[index : index + 2]
    query_length = 1 if args.no_draft else 8
    command[command.index("--compilation-config") + 1] = json.dumps(
        {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [query_length * i for i in range(1, args.max_num_seqs + 1)],
        }
    )
    env = os.environ.copy()
    env.update(
        BONSAI_BACKEND="integer", BONSAI_VERIFY_KERNEL="marlin", VLLM_SERVER_DEV_MODE="1"
    )
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    (args.output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (args.output / "probe-source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [*sorted(Path("src/vllm_bonsai2").glob("*.py")), Path(__file__)]
                },
                "environment": {
                    k: env[k]
                    for k in ["BONSAI_BACKEND", "BONSAI_VERIFY_KERNEL", "VLLM_SERVER_DEV_MODE"]
                },
            },
            indent=2,
        )
        + "\n"
    )
    tokenizer = AutoTokenizer.from_pretrained("converted/bonsai2-pq2")
    facts = "\n".join(
        f"Archive entry {i}: station Cedar stores {i + 11} blue notebooks and "
        f"station Maple stores {i + 23} green notebooks. "
        "The inventory is checked every Monday by the station manager."
        for i in range(args.prefix_entries)
    )
    cases = []
    for name, question in [
        ("cedar", "In entry 7, how many blue notebooks does Cedar store? Explain briefly."),
        ("maple", "In entry 12, how many green notebooks does Maple store? Explain briefly."),
        ("schedule", "When is the inventory checked, and by whom? Explain briefly."),
        ("difference", "In entry 3, what is the difference between the two counts? Explain."),
    ]:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": facts + "\n\n" + question}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert len(ids) + 128 < 2048
        cases.append({"name": name, "input_ids": ids})
    (args.output / "corpus.json").write_text(json.dumps(cases, indent=2) + "\n")
    report = {"passed": False, "checks": [], "requests": [], "draft": not args.no_draft}

    def save():
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def session():
        client = requests.Session()
        client.trust_env = False
        return client

    client = session()

    def metrics(label):
        response = client.get(base + "/metrics", timeout=15)
        response.raise_for_status()
        (args.output / f"metrics-{label}.txt").write_text(response.text)
        return {
            key: sum(
                float(line.rsplit(" ", 1)[1])
                for line in response.text.splitlines()
                if line.startswith(f"vllm:{key}{{")
            )
            for key in ["prefix_cache_hits_total", "prefix_cache_queries_total"]
        }

    def reset():
        response = client.post(base + "/reset_prefix_cache", timeout=30)
        response.raise_for_status()
        assert response.json()["success"], response.text

    def generate(case, label):
        payload = {
            "model": "bonsai2",
            "prompt": case["input_ids"],
            "temperature": 0,
            "max_tokens": 32,
            "ignore_eos": True,
            "return_token_ids": True,
            "seed": 123,
        }
        start = time.monotonic()
        with session() as local:
            response = local.post(base + "/v1/completions", json=payload, timeout=180)
        response.raise_for_status()
        body = response.json()
        tokens = body["choices"][0]["token_ids"]
        assert len(tokens) == body["usage"]["completion_tokens"] == 32
        return {
            "case": case["name"],
            "phase": label,
            "tokens": tokens,
            "seconds": time.monotonic() - start,
            "usage": body["usage"],
        }

    def check(name, value, **details):
        report["checks"].append({"name": name, "passed": bool(value), **details})
        save()

    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError("server exited during startup; see server.log")
                try:
                    if client.get(base + "/health", timeout=2).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup timed out")
                time.sleep(0.5)
            before = metrics("before")
            cold = {}
            for case in cases:
                reset()
                record = generate(case, "cold")
                cold[case["name"]] = record["tokens"]
                report["requests"].append(record)
                record = generate(case, "warm")
                report["requests"].append(record)
                check(f"warm-{case['name']}", record["tokens"] == cold[case["name"]])
            warm = metrics("warm")
            check(
                "cache-disabled" if args.disable_prefix_caching else "actual-prefix-hits",
                warm["prefix_cache_hits_total"] == 0
                if args.disable_prefix_caching
                else warm["prefix_cache_hits_total"] > before["prefix_cache_hits_total"],
                before=before,
                after=warm,
            )
            for case in [cases[0], cases[1], cases[0], cases[2], cases[3], cases[0]]:
                record = generate(case, "interleaved")
                report["requests"].append(record)
                check(f"interleaved-{case['name']}", record["tokens"] == cold[case["name"]])
            with ThreadPoolExecutor(max_workers=4) as pool:
                records = list(pool.map(lambda c: generate(c, "concurrent"), cases))
            for record in records:
                report["requests"].append(record)
                check(
                    f"concurrent-{record['case']}",
                    record["tokens"] == cold[record["case"]],
                )
            # Disconnect while a longer stream is active, then verify a reused slot.
            with session() as local:
                with local.post(
                    base + "/v1/completions",
                    json={
                        "model": "bonsai2",
                        "prompt": cases[0]["input_ids"],
                        "temperature": 0,
                        "max_tokens": 128,
                        "ignore_eos": True,
                        "stream": True,
                    },
                    stream=True,
                    timeout=180,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(chunk_size=1):
                        if line.startswith(b"data: {"):
                            report["cancelled_stream_after_first_event"] = True
                            break
            record = generate(cases[0], "after-disconnect")
            report["requests"].append(record)
            check("after-disconnect", record["tokens"] == cold[cases[0]["name"]])
            reset()
            record = generate(cases[0], "after-reset")
            report["requests"].append(record)
            check("after-reset", record["tokens"] == cold[cases[0]["name"]])
            check("healthy", client.get(base + "/health", timeout=10).status_code == 200)
            metrics("final")
            subprocess.run(
                [
                    sys.executable,
                    "tools/m46_http_gate.py",
                    "--base-url",
                    base,
                    "--output",
                    str(args.output / "frozen-gate.json"),
                ],
                check=True,
                timeout=300,
            )
            check("frozen-gate", True)
            if args.benchmark:
                subprocess.run(
                    [
                        sys.executable,
                        "tools/m45_http_bench.py",
                        "--base-url",
                        base,
                        "--engine",
                        "integer",
                        "--output",
                        str(args.output / "benchmark.json"),
                    ],
                    check=True,
                    timeout=300,
                )
            report["passed"] = all(c["passed"] for c in report["checks"])
        except Exception as exc:
            report["error"] = repr(exc)
            raise
        finally:
            save()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}))
    if not report["passed"]:
        raise SystemExit("Serving probe failed; see results.json")


if __name__ == "__main__":
    main()
