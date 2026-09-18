# M3A — Full text checkpoint conversion

Status: completed on 2026-09-18 for the pinned text checkpoint.

The converter preserves PQ2 codes and scale bits in their original GGUF row
order. The runtime consumes separate segments for fused gate/up, attention QKV,
and GDN QKV/Z projections. This avoids silently permuting Hadamard-latent columns.

Dense GGUF transformations are explicitly reversed: RMSNorm weights subtract 1,
SSM A becomes log(-A), and tiled GDN value-head rows become HF grouped rows.
The convolution, dt bias and alpha/beta projections receive the same inverse
head permutation. Packed GDN output rows are reordered after matmul at runtime;
SSM out consumes HF's already-grouped activation before signs/Hadamard.

The text-only adapter subclasses vLLM's existing Qwen3.5 causal model. It uses
`model_type=qwen3_5_text` and an out-of-tree model registry identifier because the
upstream constructor does not pass quantization configuration to embeddings.
It does not implement a separate transformer architecture.

Every source tensor must be accounted for. Packed source payload SHA-256 hashes
are checked after shard reload. Conversion metadata records dense transforms,
source hashes, shard hashes, vocabulary slot matching and configuration hashes.
Tokenizer chat template and special IDs are taken from the GGUF. No vision
projector or MTP weights are added to the text-only checkpoint.

## Inputs and reproduction

- GGUF SHA-256: `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`.
- HF configuration/tokenizer: `Qwen/Qwen3.8-27B` revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Each of the four HF asset hashes is enforced by the converter; the complete
  hashes and transformation records are stored in `conversion.json`.

Run from the repository root, using a new output directory:

```bash
uv run hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --include config.json tokenizer.json tokenizer_config.json generation_config.json \
  --local-dir artifacts/qwen-config
uv run bonsai2-vllm convert-model \
  --source /home/ubuntu/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --hf-assets artifacts/qwen-config --output converted/bonsai2-pq2
uv run python tools/verify_m3_checkpoint.py
```

## Measured results

- All 851 source tensors accounted for: 402 PQ2, 353 FP32, 96 BF16.
- 1,205 destination parameters in eight shards, 7,242,233,856 tensor bytes.
- Independent reverse conversion: 402 packed tensors byte-exact; all 449 dense
  tensors exactly restored after undoing the recorded transforms.
- Second independent conversion produced the same eight shard hashes. Reload
  validation hashed all 1,205 saved parameters.
- Tokenizer vocabulary matching covered 248,077 slots; 243 unused padding slots
  complete the 248,320 model vocabulary. Nine tokenizer input cases also matched
  Prism token IDs; this is not an exhaustive tokenizer-equivalence proof.

Evidence: `converted/bonsai2-pq2/conversion.json`,
`artifacts/m3-checkpoint-verification.json`,
`artifacts/m3-conversion-reproducibility.json`, and
`artifacts/m3-tokenizer-parity.json`. Large generated artifacts are git-ignored.
The independent verification script targets this model's fixed head geometry.
