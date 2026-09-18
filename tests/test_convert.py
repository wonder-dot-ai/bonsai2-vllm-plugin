import json

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFWriter
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from vllm_bonsai2.convert import export_projection, inventory, verify_projection


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.gguf"
    writer = GGUFWriter(str(path), "qwen35")
    writer.add_uint32("prism.hadamard.version", 1)
    writer.add_uint32("prism.hadamard.block_size", 128)
    writer.add_string("prism.hadamard.sign_mode", "explicit")
    writer.add_array("prism.hadamard.sign_widths", [128])
    writer.add_array("prism.hadamard.sign_values", [-1, 1] * 64)
    writer.add_array("prism.hadamard.weight_names", ["blk.0.ffn_gate.weight"])
    raw = np.full((2, 34), 0xE4, dtype=np.uint8)
    raw[:, :2] = np.array([0.5, -2], dtype="<f2").view(np.uint8).reshape(2, 2)
    writer.add_tensor("blk.0.ffn_gate.weight", raw, raw_dtype=GGMLQuantizationType.PQ2_0)
    writer.add_tensor("output_norm.weight", np.ones(128, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_gguf_safetensors_roundtrip(source, tmp_path):
    output = tmp_path / "projection.safetensors"
    report = export_projection(source, "blk.0.ffn_gate.weight", output)
    assert report["logical_shape"] == [2, 128]
    assert report["byte_roundtrip_equal"]
    assert verify_projection(output, source)["metadata_roundtrip_equal"]
    summary = inventory(source, tmp_path / "inventory.json")
    assert summary["tensor_types"] == {"PQ2_0": 1, "F32": 1}


@pytest.mark.parametrize("name", ["missing", "output_norm.weight"])
def test_wrong_tensor_rejected(source, tmp_path, name):
    with pytest.raises(ValueError):
        export_projection(source, name, tmp_path / "bad.safetensors")


@pytest.mark.parametrize("tamper", ["code", "sign", "source", "layout"])
def test_corruption_detected(source, tmp_path, tamper):
    output = tmp_path / "projection.safetensors"
    export_projection(source, "blk.0.ffn_gate.weight", output)
    with safe_open(str(output), framework="numpy") as saved:
        metadata = saved.metadata()
    tensors = load_file(str(output))
    if tamper == "code":
        tensors["qweight"][0, 0, 0] ^= 1
    elif tamper in ("sign", "layout"):
        manifest = json.loads(metadata["manifest"])
        if tamper == "sign":
            manifest["context"]["prism.hadamard.sign_values"][0] *= -1
        else:
            manifest["group_size"] = 64
        metadata["manifest"] = json.dumps(manifest)
    else:
        with source.open("ab") as stream:
            stream.write(b"changed")
    save_file(tensors, str(output), metadata=metadata)
    with pytest.raises(ValueError, match="mismatch"):
        verify_projection(output, source)


def test_output_cannot_overwrite_source(source):
    before = source.read_bytes()
    with pytest.raises(ValueError):
        inventory(source, source)
    with pytest.raises(ValueError):
        export_projection(source, "blk.0.ffn_gate.weight", source)
    assert source.read_bytes() == before
