"""Inventory and export one PQ2_0 tensor with a byte-exact round-trip gate."""

from __future__ import annotations

import ctypes
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFEndian, GGUFReader
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from vllm_bonsai2.pq2 import dequantize, join_blocks, split_blocks

PRISM_REVISION = "d8f26eec76da6d09bb708bcba51ef64b8cd868a3"
SCHEMA = "bonsai2-pq2-projection-v1"


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def open_source(path: Path) -> GGUFReader:
    reader = GGUFReader(str(path), mode="r")
    if reader.endianess != GGUFEndian.LITTLE:
        raise ValueError("only little-endian GGUF is supported by the v1 checkpoint contract")
    return reader


def inventory(source: Path, output: Path) -> dict:
    """Write all metadata and tensor descriptors; never expand model weights."""
    if source.resolve() == output.resolve():
        raise ValueError("inventory output must not overwrite the source checkpoint")
    reader = open_source(source)
    result = {
        "source": str(source.resolve()),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "reader_revision": PRISM_REVISION,
        "tensor_count": len(reader.tensors),
        "tensor_types": dict(Counter(t.tensor_type.name for t in reader.tensors)),
        "metadata": {name: field.contents() for name, field in reader.fields.items()},
        "tensors": [
            {
                "name": t.name,
                "type": t.tensor_type.name,
                "type_id": int(t.tensor_type),
                "ggml_shape": t.shape.tolist(),
                "logical_shape": t.shape[::-1].tolist(),
                "size_bytes": int(t.n_bytes),
                "data_offset": int(t.data_offset),
            }
            for t in reader.tensors
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {k: v for k, v in result.items() if k not in ("metadata", "tensors")}


def export_projection(source: Path, tensor_name: str, output: Path) -> dict:
    if output.suffix != ".safetensors" or output.resolve() == source.resolve():
        raise ValueError("projection output must be a separate .safetensors file")
    reader = open_source(source)
    tensor = next((t for t in reader.tensors if t.name == tensor_name), None)
    if tensor is None:
        raise ValueError(f"tensor not found: {tensor_name}")
    if tensor.tensor_type != GGMLQuantizationType.PQ2_0 or len(tensor.shape) != 2:
        raise ValueError("export requires a two-dimensional PQ2_0 tensor")
    shape = tuple(int(x) for x in tensor.shape[::-1])
    codes, scales = split_blocks(tensor.data, shape)
    # Preserve ALL Prism fields, including explicit signs, inverse-lookup names,
    # and GDN feature ordering. Architecture metadata describes the GDN geometry.
    context = {
        key: field.contents()
        for key, field in reader.fields.items()
        if key.startswith(("prism.", "qwen35.")) or key == "general.architecture"
    }
    manifest = {
        "schema": SCHEMA,
        "reader_revision": PRISM_REVISION,
        "source_path": str(source.resolve()),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "tensor_name": tensor_name,
        "ggml_shape": tensor.shape.tolist(),
        "logical_shape": list(shape),
        "tensor_type": "PQ2_0",
        "group_size": 128,
        "block_bytes": 34,
        "byte_order": "little",
        "raw_sha256": _digest(tensor.data),
        "qweight_sha256": _digest(codes),
        "scales_sha256": _digest(scales),
        "context": context,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"qweight": codes, "scales": scales},
        str(output),
        metadata={"manifest": json.dumps(manifest, sort_keys=True)},
    )
    loaded = load_file(str(output))
    restored = join_blocks(loaded["qweight"], loaded["scales"])
    with safe_open(str(output), framework="numpy") as saved:
        if json.loads(saved.metadata()["manifest"]) != manifest:
            raise ValueError("safetensors metadata round-trip mismatch")
    if not np.array_equal(restored, tensor.data.reshape(restored.shape)):
        raise ValueError("safetensors byte round-trip mismatch")
    report = {
        "output": str(output.resolve()),
        "tensor_name": tensor_name,
        "logical_shape": list(shape),
        "qweight_shape": list(codes.shape),
        "scales_shape": list(scales.shape),
        "raw_sha256": manifest["raw_sha256"],
        "source_sha256": manifest["source_sha256"],
        "byte_roundtrip_equal": True,
        "metadata_roundtrip_equal": True,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_projection(path: Path, source: Path, library: Path | None = None) -> dict:
    """Compare saved bytes/metadata to GGUF and optionally all values to Prism C."""
    with safe_open(str(path), framework="numpy") as saved:
        manifest = json.loads(saved.metadata()["manifest"])
    if manifest["schema"] != SCHEMA or manifest["reader_revision"] != PRISM_REVISION:
        raise ValueError("unsupported projection schema or reader revision")
    for key, expected in {
        "tensor_type": "PQ2_0",
        "group_size": 128,
        "block_bytes": 34,
        "byte_order": "little",
    }.items():
        if manifest.get(key) != expected:
            raise ValueError(f"projection layout mismatch: {key}")
    if sha256_file(source) != manifest["source_sha256"]:
        raise ValueError("source checkpoint checksum mismatch")
    reader = open_source(source)
    tensor = next((t for t in reader.tensors if t.name == manifest["tensor_name"]), None)
    if tensor is None:
        raise ValueError("source tensor name mismatch")
    if (
        tensor.tensor_type != GGMLQuantizationType.PQ2_0
        or tensor.shape.tolist() != manifest["ggml_shape"]
        or tensor.shape[::-1].tolist() != manifest["logical_shape"]
    ):
        raise ValueError("source tensor type or shape mismatch")
    context = {
        k: f.contents()
        for k, f in reader.fields.items()
        if k.startswith(("prism.", "qwen35.")) or k == "general.architecture"
    }
    if context != manifest["context"]:
        raise ValueError("rotation/architecture metadata mismatch")
    arrays = load_file(str(path))
    codes, scales = arrays["qweight"], arrays["scales"]
    raw = join_blocks(codes, scales)
    for key, array in (("raw", raw), ("qweight", codes), ("scales", scales)):
        if _digest(array) != manifest[f"{key}_sha256"]:
            raise ValueError(f"{key} checksum mismatch")
    expected_shape = tuple(manifest["logical_shape"])
    if raw.shape != (expected_shape[0], expected_shape[1] // 128 * 34) or not np.array_equal(
        raw, tensor.data.reshape(raw.shape)
    ):
        raise ValueError("source byte round-trip mismatch")
    if not np.isfinite(scales).all():
        raise ValueError("non-finite scales are not supported for numerical parity")
    result = {
        "tensor_name": tensor.name,
        "byte_roundtrip_equal": True,
        "metadata_roundtrip_equal": True,
        "value_count": int(tensor.n_elements),
        "prism_c_values_equal": None,
    }
    if library is not None:
        lib = ctypes.CDLL(str(library.resolve()))
        lib.ggml_commit.restype = ctypes.c_char_p
        commit = lib.ggml_commit().decode()
        if len(commit) < 7 or not PRISM_REVISION.startswith(commit):
            raise ValueError(f"Prism library commit mismatch: {commit}")
        decode = lib.dequantize_row_pq2_0
        decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        decode.restype = None
        # Bound peak memory even for embedding/LM-head tensors.
        for start in range(0, len(raw), 64):
            block = np.ascontiguousarray(raw[start : start + 64])
            expected = np.empty((len(block), expected_shape[1]), dtype=np.float32)
            decode(block.ctypes.data, expected.ctypes.data, expected.size)
            actual = dequantize(codes[start : start + 64], scales[start : start + 64])
            if not np.array_equal(actual, expected):
                raise ValueError(f"Prism C dequantization mismatch at row {start}")
        result.update(
            prism_c_values_equal=True,
            prism_commit=commit,
            prism_library_sha256=sha256_file(library),
        )
    return result
