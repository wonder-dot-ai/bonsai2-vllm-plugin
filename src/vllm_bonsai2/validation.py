"""M2 projection gates and reproducible activation cases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import load_file

from vllm_bonsai2.convert import sha256_file, verify_projection
from vllm_bonsai2.pq2 import join_blocks
from vllm_bonsai2.prism_oracle import PrismOracle
from vllm_bonsai2.reference import (
    linear_reference,
    rotation_from_manifest,
    transform_activation,
    unpack_pq2,
)

# Versioned acceptance gates. Each row must pass both limits, so batch averages
# cannot hide a broken sequence. Max error is normalized by that row's RMS.
LIMITS = {
    "transform": (2e-6, 1e-5),
    "torch_cuda": (1e-5, 1e-4),
    "cpu_dense": (1e-5, 1e-4),
    "cuda_dense": (1e-3, 1e-2),
    "packed": (1.5e-2, 8e-2),
}


def compare(actual: np.ndarray, expected: np.ndarray, gate: str) -> dict:
    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError("comparison requires equal two-dimensional shapes")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"passed": False, "reason": "non-finite value"}
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    rms = np.maximum(np.sqrt(np.mean(expected.astype(np.float64) ** 2, axis=1)), 1e-12)
    rel_l2 = np.sqrt(np.mean(delta**2, axis=1)) / rms
    scaled_max = np.max(np.abs(delta), axis=1) / rms
    l2_limit, max_limit = LIMITS[gate]
    return {
        "passed": bool(np.all(rel_l2 <= l2_limit) and np.all(scaled_max <= max_limit)),
        "max_abs": float(np.max(np.abs(delta))),
        "worst_row_relative_l2": float(np.max(rel_l2)),
        "worst_row_max_over_rms": float(np.max(scaled_max)),
        "relative_l2_limit": l2_limit,
        "max_over_rms_limit": max_limit,
    }


def activation_cases(width: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260918)
    cases = {
        f"normal_b{batch}": rng.standard_normal((batch, width)).astype(np.float32)
        for batch in (1, 4, 32)
    }
    cases["zero_b1"] = np.zeros((1, width), dtype=np.float32)
    cases["constant_b1"] = np.ones((1, width), dtype=np.float32)
    cases["alternating_b1"] = np.tile(np.array([-1, 1], dtype=np.float32), width // 2)[None]
    impulses = np.zeros((4, width), dtype=np.float32)
    for row, col in enumerate((0, 127, width // 2, width - 1)):
        impulses[row, col] = (-1) ** row
    cases["impulses_b4"] = impulses
    cases["small_b4"] = cases["normal_b4"] * 1e-3
    cases["large_b4"] = cases["normal_b4"] * 1e3
    return cases


def validate_operator(projection: Path, source: Path, library: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("output directory already exists; choose a new run directory")
    # Tie the M2 evidence to the verified original model bytes and rotation fields.
    verify_projection(projection, source)
    with safe_open(str(projection), framework="numpy") as f:
        manifest = json.loads(f.metadata()["manifest"])
    rotation = rotation_from_manifest(manifest)
    arrays = load_file(str(projection))
    raw = join_blocks(arrays["qweight"], arrays["scales"])
    oracle = PrismOracle(library)
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    codes, scales = torch.from_numpy(arrays["qweight"]), torch.from_numpy(arrays["scales"])
    weights = unpack_pq2(codes, scales)
    gpu_codes, gpu_scales = codes.cuda(), scales.cuda()
    report = {
        "schema": "bonsai2-m2-validation-v1",
        "tensor": manifest["tensor_name"],
        "source_sha256": manifest["source_sha256"],
        "projection_sha256": sha256_file(projection),
        "prism_commit": oracle.commit,
        "harness_sha256": sha256_file(library),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "limits": LIMITS,
        "seed": 20260918,
        "cases": {},
        "passed": True,
    }
    for name, x in activation_cases(weights.shape[1]).items():
        inputs = torch.from_numpy(x)
        rotated = transform_activation(inputs, rotation)
        expected = (rotated @ weights.T).numpy()
        actual_gpu = linear_reference(inputs.cuda(), gpu_codes, gpu_scales, rotation).cpu().numpy()
        record = {"torch_cuda": compare(actual_gpu, expected, "torch_cuda")}
        vectors = {
            "input": x,
            "torch_rotated": rotated.numpy(),
            "torch_output": expected,
            "torch_cuda_output": actual_gpu,
        }
        for backend in ("cpu", "cuda"):
            result = oracle.project(raw, x, rotation, backend)
            record[backend + "_transform"] = compare(
                result["rotated"], rotated.numpy(), "transform"
            )
            record[backend + "_dense"] = compare(result["dense"], expected, backend + "_dense")
            record[backend + "_packed"] = compare(result["packed"], expected, "packed")
            vectors.update({backend + "_" + key: val for key, val in result.items()})
        passed = all(value["passed"] for value in record.values())
        vector_path = output / f"{name}.npz"
        np.savez_compressed(vector_path, **vectors)
        report["cases"][name] = {
            "passed": passed,
            "comparisons": record,
            "vectors_sha256": sha256_file(vector_path),
        }
        report["passed"] &= passed
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError(f"projection parity failed; see {output / 'report.json'}")
    return {
        "passed": True,
        "tensor": report["tensor"],
        "cases": len(report["cases"]),
        "report": str((output / "report.json").resolve()),
    }
