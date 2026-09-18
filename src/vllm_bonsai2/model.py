"""Thin text-only adapter over vLLM's Qwen3.5 transformer and GDN implementation."""

from __future__ import annotations

import os

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM


class Bonsai2ForCausalLM(Qwen3_5ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Bonsai supports TP=1, PP=1 only")
        if (
            os.environ.get("BONSAI_BACKEND", "triton") == "reference"
            and not vllm_config.model_config.enforce_eager
        ):
            raise ValueError("Bonsai reference backend requires enforce_eager=True")
        if vllm_config.lora_config is not None:
            raise ValueError("LoRA is not supported by the Bonsai adapter")
        torch.backends.cuda.matmul.allow_tf32 = False
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if os.environ.get("BONSAI_BACKEND", "triton") == "integer":
            from vllm.model_executor.layers.layernorm import GemmaRMSNorm

            from .normalization import FusedGemmaRMSNorm

            for parent in list(self.modules()):
                for name, child in list(parent.named_children()):
                    if isinstance(child, GemmaRMSNorm):
                        setattr(parent, name, FusedGemmaRMSNorm(child))
        # Upstream Qwen3_5Model creates an unquantized embedding unconditionally.
        self.model.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix="model.embed_tokens",
        )

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            if name not in parameters:
                raise ValueError(f"unexpected Bonsai checkpoint parameter: {name}")
            parameter = parameters[name]
            if parameter.shape != value.shape:
                raise ValueError(
                    f"Bonsai shape mismatch: {name}: {parameter.shape} != {value.shape}"
                )
            if name in loaded:
                raise ValueError(f"duplicate checkpoint parameter: {name}")
            parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(name)
        if missing := set(parameters) - loaded:
            raise ValueError(f"missing checkpoint parameters: {sorted(missing)}")
        return loaded
