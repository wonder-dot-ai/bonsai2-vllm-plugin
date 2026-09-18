# M1 validation — 2026-09-18

## Inputs

- Source: `Ternary-Bonsai-2-27B-PQ2_0.gguf`
- Size: 7,206,168,928 bytes
- Source SHA-256: `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`
- Prism reader: `d8f26eec76da6d09bb708bcba51ef64b8cd868a3`
- Prism C library commit: `d8f26eec7`
- C library SHA-256: `79a6723d3679f1c8c82d3e393e626c7ced76390b459e5a018fd1146f7752a8f8`
- Host: NVIDIA A100-SXM4 40GB; M1 conversion/parity runs on CPU.

## Observations

Inventory: 851 tensors — 402 PQ2_0, 353 F32, 96 BF16.
All GGUF metadata and tensor descriptors are in `artifacts/inventory.json`.

Selected tensor: `blk.0.ffn_gate.weight`.

- GGML shape: `[5120, 17408]`
- Logical weight shape: `[17408, 5120]`
- Packed code shape: `[17408, 40, 32]` (uint8)
- Scale shape: `[17408, 40]` (FP16)
- Original packed payload: 23,674,880 bytes
- Payload SHA-256: `320487d38e518a8b459d70a8dc42d6080ce9850a14380f511fbc0eab9cc7c5e6`
- Every original tensor byte reconstructed exactly after safetensors reload.
- All saved rotation and architecture metadata matched the GGUF.
- All 89,128,960 decoded FP32 weights matched the pinned
  Prism `dequantize_row_pq2_0` C function exactly (NumPy array equality).

The model stores explicit Hadamard signs for widths 5120, 6144, and 17408.
Its metadata also marks the inverse embedding path and grouped GDN value order;
these are retained in the embedded manifest.

## Reproduction

```bash
uv sync --locked
uv run bonsai2-vllm doctor
uv run bonsai2-vllm inventory
uv run bonsai2-vllm export-projection
uv run bonsai2-vllm verify-projection artifacts/blk.0.ffn_gate.safetensors \
  --prism-library /home/ubuntu/Bonsai-demo/bin/cuda/libggml-base.so \
  --output artifacts/blk.0.ffn_gate.parity.json
uv run pytest
uv run ruff check .
```

Validation: 19 tests passed, Ruff passed, locked environment sync passed.
Tests include all 65,536 FP16 bit patterns, explicit codec slot order, generated
GGUF-to-safetensors round trips, and detection of changed codes, signs, source
bytes, and layout metadata. Tests need neither the full model nor a GPU.
Generated model files and the inspection-only Prism source checkout remain
ignored under `artifacts/`; existing Bonsai-demo files were not modified.

## Next gate: M2

Update: the forward single-projection M2 gate is now complete; see
`M2_VALIDATION.md`. The paragraph below records the handoff from M1.

Implement a PyTorch signed block Hadamard reference and projection operation,
then compare outputs with an instrumented Prism runtime. Freeze numerical
tolerances and golden activation vectors. The current C comparison checks
weight decoding only; it does not establish activation, matmul, model-logit,
text-generation, or vLLM serving parity. Full conversion and a vLLM quantization
plugin remain M3. The discussed tau2 trajectory corpus is not yet collected.
