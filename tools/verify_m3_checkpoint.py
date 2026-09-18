"""Independently reverse every converted parameter back to the GGUF representation."""

import argparse
import json
from pathlib import Path

import torch
from gguf import GGMLQuantizationType
from safetensors import safe_open

from vllm_bonsai2.convert import open_source, sha256_file
from vllm_bonsai2.paths import model_path
from vllm_bonsai2.pq2 import join_blocks

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", type=Path, default=Path("converted/bonsai2-pq2"))
parser.add_argument(
    "--output", type=Path, default=Path("artifacts/m3-checkpoint-verification.json")
)
args = parser.parse_args()
root = args.model_dir
report = json.loads((root / "conversion.json").read_text())
config = json.loads((root / "config.json").read_text())
index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
reader = open_source(model_path())
assert report["source_sha256"] == sha256_file(model_path())
for shard in report["shards"]:
    assert sha256_file(root / shard["file"]) == shard["sha256"]


def get(name):
    with safe_open(str(root / index[name]), framework="pt") as f:
        return f.get_tensor(name)


def tile(value, hd):
    # HF [nk,rep,hd] -> original GGUF [rep,nk,hd].
    return value.reshape(16, 3, hd, *value.shape[1:]).transpose(0, 1).reshape_as(value)


counts = {"packed_byte_exact": 0, "dense_exact": 0, "dense_numerical": 0}
max_errors = {}
for t in reader.tensors:
    record = report["source_tensors"][t.name]
    if t.tensor_type == GGMLQuantizationType.PQ2_0:
        prefix, slot = record["destination"].rsplit(".", 1)
        raw = join_blocks(
            get(f"{prefix}.qweight_{slot}").numpy(), get(f"{prefix}.scales_{slot}").numpy()
        )
        assert raw.tobytes() == t.data.tobytes(), t.name
        counts["packed_byte_exact"] += 1
        continue
    source = torch.from_numpy(t.data.copy())
    if t.tensor_type == GGMLQuantizationType.BF16:
        source = source.view(torch.bfloat16).reshape(tuple(t.shape[::-1])).float()
    value = get(record["destination"]).clone()
    mode = record["transform"]
    if mode == "norm_minus_one":
        restored = value + 1
    elif mode == "log_negative_untile":
        restored = -tile(value, 1).exp()
    elif mode == "untile":
        if ".ssm_beta." in t.name:
            value = value[:48]
        elif ".ssm_alpha." in t.name:
            value = value[48:]
        restored = tile(value, 1)
    elif mode == "conv":
        value = value.squeeze(1)
        restored = torch.cat((value[:4096], tile(value[4096:], 128)), dim=0)
    else:
        restored = value
    if torch.equal(restored, source):
        counts["dense_exact"] += 1
    else:
        delta = (restored - source).abs().max().item()
        max_errors[mode] = max(max_errors.get(mode, 0), delta)
        torch.testing.assert_close(restored, source, rtol=3e-7, atol=1e-7, msg=t.name)
        counts["dense_numerical"] += 1
assert sum(counts.values()) == 851
result = {"passed": True, "counts": counts, "max_inverse_transform_abs_errors": max_errors}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
