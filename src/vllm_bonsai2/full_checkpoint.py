"""M3A: bounded-memory conversion to the text-only Bonsai vLLM checkpoint."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from gguf import GGMLQuantizationType
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from .convert import PRISM_REVISION, open_source, sha256_file
from .pq2 import join_blocks, split_blocks

UPSTREAM_MODEL = "Qwen/Qwen3.8-27B"
UPSTREAM_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
UPSTREAM_ASSET_SHA256 = {
    "config.json": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
}


def untile_rows(value: torch.Tensor, nk: int, rep: int, hd: int) -> torch.Tensor:
    """GGUF [rep,nk,hd,...] -> HF [nk,rep,hd,...]."""
    return value.reshape(rep, nk, hd, *value.shape[1:]).transpose(0, 1).reshape_as(value)


def convert_model(source: Path, hf_assets: Path, output: Path, shard_bytes: int = 1_000_000_000):
    if output.exists():
        raise ValueError("output must be a new directory")
    for name, digest in UPSTREAM_ASSET_SHA256.items():
        if sha256_file(hf_assets / name) != digest:
            raise ValueError(f"upstream asset does not match pinned revision: {name}")
    reader = open_source(source)
    metadata = {k: f.contents() for k, f in reader.fields.items()}
    context = {
        k: v
        for k, v in metadata.items()
        if k.startswith(("prism.", "qwen35.")) or k == "general.architecture"
    }
    if context["general.architecture"] != "qwen35" or not context.get(
        "prism.hadamard.gdn_v_grouped"
    ):
        raise ValueError("M3 supports only the validated qwen35 grouped-V Bonsai checkpoint")
    upstream = json.loads((hf_assets / "config.json").read_text())
    config = dict(upstream["text_config"])
    pairs = {
        "hidden_size": "embedding_length",
        "intermediate_size": "feed_forward_length",
        "num_hidden_layers": "block_count",
        "linear_num_key_heads": "ssm.group_count",
        "linear_num_value_heads": "ssm.time_step_rank",
        "head_dim": "attention.key_length",
    }
    for key, gguf_key in pairs.items():
        if config[key] != metadata["qwen35." + gguf_key]:
            raise ValueError(f"upstream configuration mismatch: {key}")
    tokenizer = AutoTokenizer.from_pretrained(hf_assets)
    vocab = metadata["tokenizer.ggml.tokens"]
    mismatches = [i for i, token in enumerate(vocab) if tokenizer.convert_ids_to_tokens(i) != token]
    # GGUF fills unused vocabulary slots with [PAD{id}] sentinels.
    if any(not vocab[i].startswith("[PAD") for i in mismatches):
        raise ValueError(f"tokenizer vocabulary mismatch: {mismatches[:10]}")
    output.mkdir(parents=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copyfile(hf_assets / name, output / name)
    tok_config = json.loads((output / "tokenizer_config.json").read_text())
    tok_config["chat_template"] = metadata["tokenizer.chat_template"]
    (output / "tokenizer_config.json").write_text(
        json.dumps(tok_config, ensure_ascii=False, indent=2)
    )
    config.update(
        architectures=["Bonsai2ForCausalLM"],
        tie_word_embeddings=False,
        eos_token_id=metadata["tokenizer.ggml.eos_token_id"],
        bos_token_id=metadata["tokenizer.ggml.bos_token_id"],
        pad_token_id=metadata["tokenizer.ggml.padding_token_id"],
    )
    quant = {
        "quant_method": "bonsai2",
        "schema": "bonsai2-vllm-v1",
        "context": context,
        "layers": {},
        "reference_chunk_rows": 2048,
    }
    tensors = {t.name: t for t in reader.tensors}
    source_records, weight_map, shard, sizes = {}, {}, {}, 0
    shards = []
    destination_hashes = {}
    nk = config["linear_num_key_heads"]
    rep = config["linear_num_value_heads"] // nk
    hd = config["linear_value_head_dim"]
    qk = 2 * config["linear_key_head_dim"] * nk

    def flush():
        nonlocal shard, sizes
        if not shard:
            return
        name = f"model-{len(shards) + 1:05d}.safetensors"
        save_file(shard, str(output / name), metadata={"format": "pt"})
        for key in shard:
            weight_map[key] = name
        shards.append({"file": name, "sha256": sha256_file(output / name), "bytes": sizes})
        shard, sizes = {}, 0

    def add(key, tensor):
        nonlocal sizes
        size = tensor.numel() * tensor.element_size()
        if sizes + size > shard_bytes:
            flush()
        if key in weight_map or key in shard:
            raise ValueError(f"duplicate destination {key}")
        shard[key] = tensor.contiguous()
        destination_hashes[key] = hashlib.sha256(
            shard[key].view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        sizes += size

    def source_record(name):
        if name in source_records:
            raise ValueError(f"duplicate source {name}")
        t = tensors[name]
        source_records[name] = {
            "type": t.tensor_type.name,
            "shape": t.shape[::-1].tolist(),
            "raw_sha256": hashlib.sha256(t.data.tobytes()).hexdigest(),
        }
        return t

    def packed(prefix, names, permutations=None, inverse=False):
        segments = []
        for i, name in enumerate(names):
            t = source_record(name)
            if t.tensor_type != GGMLQuantizationType.PQ2_0:
                raise ValueError(f"expected PQ2_0: {name}")
            shape = tuple(int(v) for v in t.shape[::-1])
            codes, scales = split_blocks(t.data, shape)
            add(f"{prefix}.qweight_{i}", torch.from_numpy(codes))
            add(f"{prefix}.scales_{i}", torch.from_numpy(scales))
            segments.append(
                {
                    "name": name,
                    "shape": shape,
                    "output_order": (permutations or ["none"] * len(names))[i],
                }
            )
            source_records[name]["destination"] = f"{prefix}.{i}"
        quant["layers"][prefix] = {"segments": segments, "inverse": inverse}

    def dense(name, destination, transform="identity"):
        t = source_record(name)
        value = torch.from_numpy(t.data.copy())
        if t.tensor_type == GGMLQuantizationType.BF16:
            value = value.view(torch.bfloat16).reshape(tuple(t.shape[::-1])).float()
        elif t.tensor_type != GGMLQuantizationType.F32:
            raise ValueError(f"unsupported dense type {name}")
        if transform == "norm_minus_one":
            value = value.float() - 1
        elif transform == "log_negative_untile":
            if not (value < 0).all():
                raise ValueError("SSM A must be negative")
            value = untile_rows((-value).log(), nk, rep, 1)
        elif transform == "untile":
            value = untile_rows(value, nk, rep, 1)
        elif transform == "conv":
            value = torch.cat((value[:qk], untile_rows(value[qk:], nk, rep, hd)), dim=0)
            value = value.unsqueeze(1)
        source_records[name].update(destination=destination, transform=transform)
        return value

    packed("model.embed_tokens", ["token_embd.weight"], inverse=True)
    packed("lm_head", ["output.weight"])
    add("model.norm.weight", dense("output_norm.weight", "model.norm.weight", "norm_minus_one"))
    for i in range(config["num_hidden_layers"]):
        src, dst = f"blk.{i}", f"model.layers.{i}"
        for gg, hf in [
            ("attn_norm", "input_layernorm"),
            ("post_attention_norm", "post_attention_layernorm"),
        ]:
            key = f"{dst}.{hf}.weight"
            add(key, dense(f"{src}.{gg}.weight", key, "norm_minus_one"))
        packed(f"{dst}.mlp.gate_up_proj", [f"{src}.ffn_gate.weight", f"{src}.ffn_up.weight"])
        packed(f"{dst}.mlp.down_proj", [f"{src}.ffn_down.weight"])
        if config["layer_types"][i] == "full_attention":
            attn = f"{dst}.self_attn"
            packed(f"{attn}.qkv_proj", [f"{src}.attn_{k}.weight" for k in ("q", "k", "v")])
            packed(f"{attn}.o_proj", [f"{src}.attn_output.weight"])
            for k in ("q", "k"):
                key = f"{attn}.{k}_norm.weight"
                add(key, dense(f"{src}.attn_{k}_norm.weight", key, "norm_minus_one"))
        else:
            attn = f"{dst}.linear_attn"
            packed(
                f"{attn}.in_proj_qkvz",
                [f"{src}.attn_qkv.weight", f"{src}.attn_gate.weight"],
                ["qkv_untile", "untile"],
            )
            packed(f"{attn}.out_proj", [f"{src}.ssm_out.weight"])
            key = f"{attn}.in_proj_ba.weight"
            add(
                key,
                torch.cat(
                    [dense(f"{src}.ssm_{k}.weight", key, "untile") for k in ("beta", "alpha")],
                    dim=0,
                ),
            )
            for name, target, transform in [
                ("ssm_a", "A_log", "log_negative_untile"),
                ("ssm_dt.bias", "dt_bias", "untile"),
                ("ssm_conv1d.weight", "conv1d.weight", "conv"),
                ("ssm_norm.weight", "norm.weight", "identity"),
            ]:
                key = f"{attn}.{target}"
                add(key, dense(f"{src}.{name}", key, transform))
        print(f"converted layer {i + 1}/{config['num_hidden_layers']}", flush=True)
    flush()
    if set(source_records) != set(tensors):
        raise ValueError(f"unmapped source tensors: {set(tensors) - set(source_records)}")
    # Verify every packed source against reloaded shard bytes, one shard at a time.
    checked_packed = 0
    for item in shards:
        saved = load_file(str(output / item["file"]))
        for key, value in saved.items():
            digest = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
            if digest != destination_hashes[key]:
                raise ValueError(f"saved parameter checksum mismatch: {key}")
        for prefix, spec in quant["layers"].items():
            for i, segment in enumerate(spec["segments"]):
                ckey, skey = f"{prefix}.qweight_{i}", f"{prefix}.scales_{i}"
                if ckey not in saved:
                    continue
                scale = saved.get(skey)
                if scale is None:
                    scale = load_file(str(output / weight_map[skey]))[skey]
                raw = join_blocks(saved[ckey].numpy(), scale.numpy())
                if (
                    hashlib.sha256(raw.tobytes()).hexdigest()
                    != source_records[segment["name"]]["raw_sha256"]
                ):
                    raise ValueError(f"packed round-trip failed: {segment['name']}")
                checked_packed += 1
    config["quantization_config"] = quant
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": sum(x["bytes"] for x in shards)}, "weight_map": weight_map},
            indent=2,
        )
    )
    report = {
        "schema": "bonsai2-vllm-v1",
        "source_sha256": sha256_file(source),
        "prism_revision": PRISM_REVISION,
        "source_tensors": source_records,
        "shards": shards,
        "packed_roundtrip_count": checked_packed,
        "destination_sha256": destination_hashes,
        "upstream_model": UPSTREAM_MODEL,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_assets_sha256": {
            name: sha256_file(hf_assets / name)
            for name in (
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "generation_config.json",
            )
        },
        "tokenizer_matching_slots": len(vocab) - len(mismatches),
        "unused_tokenizer_slots": mismatches,
        "upstream_config_sha256": sha256_file(hf_assets / "config.json"),
    }
    (output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"output": str(output.resolve()), "tensors": len(source_records), "shards": len(shards)}
