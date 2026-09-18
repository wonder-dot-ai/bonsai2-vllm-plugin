"""Explicitly generate small synthetic Prism golden vectors (no model weights)."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vllm_bonsai2.convert import PRISM_REVISION, sha256_file
from vllm_bonsai2.pq2 import join_blocks
from vllm_bonsai2.prism_oracle import PrismOracle
from vllm_bonsai2.reference import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("--library", type=Path, default=Path("artifacts/libbonsai-prism-projection.so"))
parser.add_argument("--output", type=Path, default=Path("tests/data/m2"))
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("Choose a new output directory; existing golden vectors are immutable")
args.output.mkdir(parents=True)
oracle = PrismOracle(args.library)
rng = np.random.default_rng(20260918)
record = {
    "prism_revision": PRISM_REVISION,
    "prism_commit": oracle.commit,
    "harness_sha256": sha256_file(args.library),
    "seed": 20260918,
    "synthetic_weights": True,
    "fixtures": {},
}
for name, width, nk, rep in [("gate", 5120, 1, 1), ("down", 17408, 1, 1), ("ssm_out", 6144, 16, 3)]:
    codes = rng.integers(0, 256, (32, width // 128, 32), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.03, (32, width // 128)).astype(np.float16)
    signs = tuple(int(v) for v in rng.choice([-1, 1], width))
    rotation = Rotation(1024, signs, nk, rep)
    x = rng.standard_normal((4, width)).astype(np.float32)
    raw = join_blocks(codes, scales)
    arrays = {
        "qweight": codes,
        "scales": scales,
        "input": x,
        "signs": np.array(signs, dtype=np.int8),
    }
    for backend in ("cpu", "cuda"):
        arrays.update(
            {backend + "_" + k: v for k, v in oracle.project(raw, x, rotation, backend).items()}
        )
    path = args.output / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    rotation_metadata = asdict(rotation)
    del rotation_metadata["signs"]
    record["fixtures"][name] = {"sha256": sha256_file(path), "rotation": rotation_metadata}
(args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
print(args.output)
