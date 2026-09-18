"""Start the packaged serving profile, check chat and frozen outputs, then stop it."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    command = [str(Path(sys.executable).with_name("bonsai2-serve")), "--port", str(args.port)]
    session = requests.Session()
    session.trust_env = False
    base = f"http://127.0.0.1:{args.port}"
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited; inspect {args.output / 'server.log'}")
                try:
                    if session.get(base + "/health", timeout=2).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup timed out")
                time.sleep(0.5)
            response = session.post(
                base + "/v1/chat/completions",
                json={
                    "model": "bonsai2",
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is 6 times 7? Answer only with the number.",
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 32,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=180,
            )
            response.raise_for_status()
            chat = response.json()
            (args.output / "chat.json").write_text(json.dumps(chat, indent=2) + "\n")
            assert chat["choices"][0]["message"]["content"].strip() == "42", chat
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
            (args.output / "result.json").write_text(
                json.dumps(
                    {
                        "chat_answer_42": True,
                        "frozen_short_gate_passed": True,
                        "launcher": "bonsai2-serve (M6.1 default profile)",
                    },
                    indent=2,
                )
                + "\n"
            )
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
