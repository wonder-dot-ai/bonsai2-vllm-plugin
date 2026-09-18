"""Build the small graph harness against the exact M1 Prism source and binaries."""

import argparse
import json
import subprocess
from pathlib import Path

from vllm_bonsai2.convert import PRISM_REVISION, sha256_file
from vllm_bonsai2.paths import llama_server_path

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=Path("artifacts/prism-llama.cpp"))
parser.add_argument("--lib-dir", type=Path, default=llama_server_path().parent)
parser.add_argument("--output", type=Path, default=Path("artifacts/libbonsai-prism-projection.so"))
args = parser.parse_args()
source, libs, output = args.source.resolve(), args.lib_dir.resolve(), args.output.resolve()
revision = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
).strip()
if revision != PRISM_REVISION:
    raise SystemExit(f"Expected Prism {PRISM_REVISION}, got {revision}")
if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
    raise SystemExit("Prism source must be clean for reproducible instrumentation")
output.parent.mkdir(parents=True, exist_ok=True)
harness = Path(__file__).with_name("prism_projection.cpp").resolve()
command = ["c++", "-std=c++17", "-O2", "-shared", "-fPIC", str(harness)]
for include in ["ggml/include", "ggml/src", "src"]:
    command += ["-I", str(source / include)]
command += [
    "-L",
    str(libs),
    f"-Wl,-rpath,{libs}",
    "-lggml-base",
    "-lggml-cpu",
    "-lggml-cuda",
    "-o",
    str(output),
]
subprocess.run(command, check=True)
record = {
    "revision": revision,
    "command": command,
    "harness_sha256": sha256_file(harness),
    "output_sha256": sha256_file(output),
    "libraries": {
        n: sha256_file(libs / n) for n in ["libggml-base.so", "libggml-cpu.so", "libggml-cuda.so"]
    },
}
output.with_suffix(".build.json").write_text(json.dumps(record, indent=2) + "\n")
print(output)
