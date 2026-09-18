# Recreate the public safetensors checkpoint

Serving the public model does not require this conversion. These commands
reproduce the stored representation from the original packed PQ2_0 GGUF.
Run from the repository root after `uv sync --locked`.

```bash
uv run --locked hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf LICENSE NOTICE.txt \
  --revision 6ed5e12bf84b7a63069882c91dd9e9218647d17b \
  --local-dir artifacts/source-gguf
uv run --locked hf download Qwen/Qwen3.8-27B \
  config.json tokenizer.json tokenizer_config.json generation_config.json \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir artifacts/qwen-config
export BONSAI_GGUF_DIR="$PWD/artifacts/source-gguf"
uv run --locked bonsai2-vllm convert-model \
  --source "$BONSAI_GGUF_DIR/Ternary-Bonsai-2-27B-PQ2_0.gguf" \
  --hf-assets artifacts/qwen-config --output converted/bonsai2-pq2
uv run --locked python tools/verify_m3_checkpoint.py
```

The converter refuses an existing output directory. If the public model was
already downloaded, convert into a new directory instead; the independent
verification tool accepts `--model-dir` to select it. The expected source GGUF
SHA-256 is `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`.
The converter enforces the pinned upstream configuration/tokenizer asset hashes.
The independent reverse check verifies every source tensor and the shard hashes.

The release adds a model card, license, notices and checksum manifest to the
converted checkpoint. These additions do not modify the eight weight shards.
See [M3A](M3A_CHECKPOINT.md) and [checkpoint format](CHECKPOINT_FORMAT.md) for
transformations, row ordering and tokenizer provenance.
