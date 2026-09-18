# PQ2_0 projection checkpoint v1

The single-projection contract is implemented and round-trip tested. This is
not yet a full Hugging Face checkpoint or a loadable vLLM quantization format.

## Reference

- Prism llama.cpp release: `prism-b10683-d8f26ee`
- Reader/source revision: `d8f26eec76da6d09bb708bcba51ef64b8cd868a3`
- `gguf` is pinned to that Git revision under `[tool.uv.sources]` and in `uv.lock`.
  Use `uv sync --locked` to reproduce this environment.
- [Block layout](https://github.com/PrismML-Eng/llama.cpp/blob/d8f26eec76da6d09bb708bcba51ef64b8cd868a3/ggml/src/ggml-common.h)
- [Reference decoder](https://github.com/PrismML-Eng/llama.cpp/blob/d8f26eec76da6d09bb708bcba51ef64b8cd868a3/ggml/src/ggml-quants.c)
- [Rotation metadata loader](https://github.com/PrismML-Eng/llama.cpp/blob/d8f26eec76da6d09bb708bcba51ef64b8cd868a3/src/llama-model.cpp)
- [Activation transform order](https://github.com/PrismML-Eng/llama.cpp/blob/d8f26eec76da6d09bb708bcba51ef64b8cd868a3/src/llama-graph.cpp)

## Layout

Only little-endian, two-dimensional GGUF type `PQ2_0` (142) is supported.
GGML shape is `[in_features, out_features]`; logical NumPy/PyTorch shape is
`[out_features, in_features]`. Input width must be divisible by 128.

Each consecutive group of 128 weights occupies 34 bytes:

```text
offset 0..1:  FP16 scale, little endian
offset 2..33: 32 packed code bytes, four 2-bit codes per byte
```

Code slots are decoded in shifts `0, 2, 4, 6`, mapping to `(code - 1) * scale`.
The codec supports `00=-1, 01=0, 10=+1, 11=+2`; the converter preserves all
codes, including 3, rather than silently forcing ternary values.

One projection file contains:

| Key | dtype | Shape |
| --- | --- | --- |
| `qweight` | uint8 | `[out_features, in_features / 128, 32]` |
| `scales` | float16 | `[out_features, in_features / 128]` |

The split is a byte copy. It neither transposes the logical weight matrix nor
dequantizes/requantizes it. Interleaving scale bytes and code bytes reconstructs
the exact original GGUF tensor payload. A later GPU loader may interleave these
arrays again; this schema does not promise a zero-copy whole-model upload.

## Embedded manifest

Safetensors string metadata `manifest` contains JSON with schema identifier
`bonsai2-pq2-projection-v1`, source path/size/SHA-256, reader revision, original
tensor name, GGML and logical shapes, layout parameters, and SHA-256 hashes of
the source tensor, code bytes, and scale bytes.

The `context` object preserves all `prism.*` and `qwen35.*` metadata plus
`general.architecture`. In the current model this includes:

- Hadamard version 1, block size 1024, normalized Sylvester transform.
- Input-last-dimension axis, **explicit** signs, sign widths and sign values.
- Forward weight-name list and inverse-after-lookup weight-name list.
- `gdn_v_grouped=true` and the model's GDN head geometry.

These fields must not be replaced with only a block-size declaration. Forward
linear activation transforms multiply by signs before Hadamard. `ssm_out`
additionally permutes GDN features before signs; embedding lookup uses an
inverse path. Preserving this metadata is verified here; implementing these
forward operators and comparing projection outputs are covered by M2; see
`M2_VALIDATION.md`. The inverse embedding operator remains M3 work.

## Validation

`export-projection` reloads the saved file and checks every reconstructed byte
and all embedded metadata. `verify-projection` additionally rehashes the source,
checks the tensor shape/layout/context, and can compare every decoded FP32
weight with the original `dequantize_row_pq2_0` C function. The C library's
commit must match the pinned source. Decoding is chunked into 64 rows.

This proves weight representation parity. It does not prove Hadamard execution,
matrix multiplication, model logits, greedy output, or vLLM serving parity.
