"""Out-of-tree Bonsai quantization with reference and fused A100 backends."""

from __future__ import annotations

import os
from dataclasses import replace

import torch
from torch import nn
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from .full_checkpoint import untile_rows
from .reference import hadamard, rotation_from_manifest, unpack_pq2


class Bonsai2Config(QuantizationConfig):
    def __init__(self, config):
        super().__init__()
        if config.get("schema") != "bonsai2-vllm-v1":
            raise ValueError("unsupported Bonsai checkpoint schema")
        self.config = config

    def get_name(self):
        return "bonsai2"

    def get_supported_act_dtypes(self):
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls):
        return 80

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def get_quant_method(self, layer, prefix):
        if prefix in self.config["layers"]:
            return Bonsai2Method(self.config, prefix)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, VocabParallelEmbedding):
            raise ValueError(f"missing packed embedding/head definition: {prefix}")
        return None


class Bonsai2Method(LinearMethodBase):
    def __init__(self, config, prefix):
        self.config = config
        self.spec = config["layers"][prefix]
        self.backend = os.environ.get("BONSAI_BACKEND", "triton")
        if self.backend not in ("reference", "triton", "integer"):
            raise ValueError("BONSAI_BACKEND must be reference, triton or integer")
        self.context = config["context"]
        self.chunk_rows = int(config.get("reference_chunk_rows", 2048))
        if self.chunk_rows <= 0:
            raise ValueError("chunk size must be positive")
        self.rotations = []
        for segment in self.spec["segments"]:
            context = dict(self.context)
            if self.spec["inverse"]:
                if segment["name"] not in context["prism.hadamard.inverse_weight_names"]:
                    raise ValueError("embedding missing from inverse rotation metadata")
                context["prism.hadamard.weight_names"] = [segment["name"]]
            rotation = rotation_from_manifest(
                {
                    "context": context,
                    "tensor_name": segment["name"],
                    "logical_shape": segment["shape"],
                }
            )
            # Unlike GGML, HF/vLLM GDN already emits grouped value heads.
            self.rotations.append(replace(rotation, perm_nk=1, perm_rep=1))

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **kwargs,
    ):
        if input_size_per_partition != input_size:
            raise ValueError("Bonsai reference supports TP=1 only")
        segments = self.spec["segments"]
        if sum(s["shape"][0] for s in segments) != sum(output_partition_sizes):
            raise ValueError("packed output shape does not match vLLM layer")
        layer.bonsai_output_dtype = params_dtype
        for i, segment in enumerate(segments):
            rows, width = segment["shape"]
            if width != input_size:
                raise ValueError("packed input shape does not match vLLM layer")
            layer.register_parameter(
                f"qweight_{i}",
                nn.Parameter(
                    torch.empty(rows, width // 128, 32, dtype=torch.uint8), requires_grad=False
                ),
            )
            layer.register_parameter(
                f"scales_{i}",
                nn.Parameter(
                    torch.empty(rows, width // 128, dtype=torch.float16), requires_grad=False
                ),
            )
            layer.register_buffer(
                f"bonsai_signs_{i}",
                torch.tensor(self.rotations[i].signs, dtype=torch.float32),
                persistent=False,
            )

    def _transform(self, layer, x, i):
        return hadamard(
            x.float() * getattr(layer, f"bonsai_signs_{i}"), self.rotations[i].block_size
        )

    def process_weights_after_loading(self, layer):
        if self.backend != "integer" or self.spec["inverse"] or not layer.qweight_0.is_cuda:
            return
        first = self.rotations[0]
        if any(
            r.block_size != first.block_size or tuple(r.signs) != tuple(first.signs)
            for r in self.rotations[1:]
        ):
            return  # Different rotations cannot share an activation transform.
        segments = self.spec["segments"]
        codes = [getattr(layer, f"qweight_{i}") for i in range(len(segments))]
        scales = [getattr(layer, f"scales_{i}") for i in range(len(segments))]
        merged_codes = torch.cat(codes).detach() if len(codes) > 1 else codes[0].detach()
        merged_scales = torch.cat(scales).detach() if len(scales) > 1 else scales[0].detach()
        offset = 0
        order = []
        for segment, code, scale in zip(segments, codes, scales, strict=True):
            size = code.shape[0]
            # Preserve checkpoint names and values without a second persistent copy.
            code.data = merged_codes[offset : offset + size]
            scale.data = merged_scales[offset : offset + size]
            order.append(
                self._output_order(
                    torch.arange(size).reshape(1, -1), segment["output_order"]
                ).reshape(-1)
                + offset
            )
            offset += size
        rows = torch.cat(order)
        rows = None if torch.equal(rows, torch.arange(offset)) else rows.to(merged_codes.device)
        layer.register_buffer("bonsai_merged_codes", merged_codes, persistent=False)
        layer.register_buffer("bonsai_merged_scales", merged_scales, persistent=False)
        layer.register_buffer("bonsai_output_rows", rows, persistent=False)
        if os.environ.get("BONSAI_VERIFY_KERNEL") == "marlin" and offset % 64 == 0:
            from .marlin_kernels import prepare_marlin

            weight, scale, workspace = prepare_marlin(merged_codes, merged_scales, rows)
            layer.register_buffer("bonsai_marlin_weight", weight, persistent=False)
            layer.register_buffer("bonsai_marlin_scale", scale, persistent=False)
            layer.register_buffer("bonsai_marlin_workspace", workspace, persistent=False)

    def _output_order(self, y, kind):
        if kind == "none":
            return y
        nk = self.context["qwen35.ssm.group_count"]
        rep = self.context["qwen35.ssm.time_step_rank"] // nk
        hd = self.context["qwen35.ssm.inner_size"] // (nk * rep)
        if kind == "untile":
            return untile_rows(y.T, nk, rep, hd).T
        if kind == "qkv_untile":
            qk = self.context["qwen35.ssm.state_size"] * nk * 2
            return torch.cat((y[:, :qk], untile_rows(y[:, qk:].T, nk, rep, hd).T), dim=-1)
        raise ValueError(f"unknown output permutation {kind}")

    def apply(self, layer, x, bias=None):
        original_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        if self.backend == "integer" and bias is None and hasattr(layer, "bonsai_merged_codes"):
            if x.shape[0] > 1 and hasattr(layer, "bonsai_marlin_weight"):
                from .marlin_kernels import marlin_project

                output = marlin_project(
                    x,
                    layer.bonsai_marlin_weight,
                    layer.bonsai_marlin_scale,
                    layer.bonsai_signs_0,
                    layer.bonsai_marlin_workspace,
                    self.rotations[0].block_size,
                    layer.bonsai_merged_codes.shape[0],
                )
                return output.reshape(*original_shape, output.shape[-1])
            from .integer_kernels import integer_project

            output = integer_project(
                x,
                layer.bonsai_merged_codes,
                layer.bonsai_merged_scales,
                layer.bonsai_signs_0,
                self.rotations[0].block_size,
                layer.bonsai_output_rows,
            )
            return output.reshape(*original_shape, output.shape[-1])
        outputs = []
        for i, segment in enumerate(self.spec["segments"]):
            codes, scales = getattr(layer, f"qweight_{i}"), getattr(layer, f"scales_{i}")
            if self.backend in ("triton", "integer") and x.is_cuda:
                from .kernels import packed_linear

                operation = packed_linear
                if self.backend == "integer":
                    from .integer_kernels import integer_linear

                    operation = integer_linear
                result = operation(
                    x,
                    codes,
                    scales,
                    getattr(layer, f"bonsai_signs_{i}"),
                    self.rotations[i].block_size,
                )
            else:
                transformed = self._transform(layer, x, i)
                result = torch.empty((len(x), len(codes)), dtype=torch.float32, device=x.device)
                for start in range(0, len(codes), self.chunk_rows):
                    stop = start + self.chunk_rows
                    weight = unpack_pq2(codes[start:stop], scales[start:stop])
                    result[:, start:stop] = transformed @ weight.T
            outputs.append(self._output_order(result, segment["output_order"]))
        output = torch.cat(outputs, dim=-1)
        if bias is not None:
            output = output + bias.float()
        return output.reshape(*original_shape, output.shape[-1]).to(x.dtype)

    def embedding(self, layer, input_ids):
        if not self.spec["inverse"] or len(self.spec["segments"]) != 1:
            raise ValueError("invalid packed embedding definition")
        if self.backend in ("triton", "integer") and input_ids.is_cuda:
            from .kernels import packed_embedding

            return packed_embedding(
                input_ids,
                layer.qweight_0,
                layer.scales_0,
                layer.bonsai_signs_0,
                self.rotations[0].block_size,
                layer.bonsai_output_dtype,
            )
        flat = input_ids.reshape(-1)
        gathered = unpack_pq2(layer.qweight_0[flat], layer.scales_0[flat])
        restored = hadamard(gathered, self.rotations[0].block_size) * layer.bonsai_signs_0
        return restored.reshape(*input_ids.shape, restored.shape[-1]).to(layer.bonsai_output_dtype)
