"""Compare identical real-checkpoint layers using M3 and M4, without model caching."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from vllm_bonsai2.quantization import Bonsai2Method


def measure(fn, repeats=7):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/m4-operators.json"))
    args = parser.parse_args()
    root = Path("converted/bonsai2-pq2")
    config = json.loads((root / "config.json").read_text())["quantization_config"]
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for suffix in ["mlp.gate_up_proj", "mlp.down_proj", "linear_attn.out_proj", "lm_head"]:
        prefix = next(p for p in config["layers"] if p.endswith(suffix))
        method = Bonsai2Method(config, prefix)
        layer = nn.Module()
        shapes = [s["shape"] for s in method.spec["segments"]]
        width, rows = shapes[0][1], sum(s[0] for s in shapes)
        method.create_weights(layer, width, [rows], width, rows, torch.bfloat16)
        for name, parameter in layer.named_parameters():
            key = prefix + "." + name
            with safe_open(str(root / index[key]), framework="pt") as f:
                parameter.data.copy_(f.get_tensor(key))
        layer.cuda()
        for batch in [1, 4, 16, 128]:
            torch.manual_seed(123)
            x = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
            record = {"prefix": prefix, "shape": [batch, rows, width]}
            values = {}
            for backend in ["reference", "triton"]:
                method.backend = backend
                values[backend] = method.apply(layer, x).float()
                record[backend] = measure(lambda: method.apply(layer, x))
            record["relative_l2_after_bf16_cast"] = (
                (values["triton"] - values["reference"]).norm() / values["reference"].norm()
            ).item()
            record["speedup"] = record["reference"]["median_ms"] / record["triton"]["median_ms"]
            records.append(record)
            args.output.write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
