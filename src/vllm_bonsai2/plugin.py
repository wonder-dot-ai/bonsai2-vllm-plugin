"""Entry point loaded by every vLLM worker."""


def register():
    from vllm import ModelRegistry
    from vllm.model_executor.layers.quantization import register_quantization_config

    from .quantization import Bonsai2Config

    register_quantization_config("bonsai2")(Bonsai2Config)
    ModelRegistry.register_model("Bonsai2ForCausalLM", "vllm_bonsai2.model:Bonsai2ForCausalLM")
