# M0 — Base environment

Completed in the preceding setup task and rechecked during M1/M2.

- Independent uv project: `/home/ubuntu/vllm-bonsai2`.
- Python 3.12.14, vLLM 0.29.0, PyTorch 2.13.0+cu130.
- NVIDIA A100-SXM4-40GB, compute capability 8.0.
- Source model and reference libraries are read in place from the sibling
  `/home/ubuntu/Bonsai-demo` checkout.
- Dependencies and the Prism GGUF reader revision are fixed in `uv.lock`.

Reproduce with `uv sync --locked`, `uv run bonsai2-vllm doctor`,
`uv run pytest`, and `uv run ruff check .`.

Subsequent stage records: `M1_VALIDATION.md`, `M2_VALIDATION.md`,
`M3A_CHECKPOINT.md`, `M3B_VLLM_RUNTIME.md`, `M3C_MODEL_PARITY.md`, and
`M4_A100_KERNEL.md`.
New stages must add a separate Markdown record covering scope, implementation,
commands, measured results, and remaining limitations.
