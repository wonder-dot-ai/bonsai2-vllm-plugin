# vLLM Bonsai 2

Out-of-tree vLLM support for the packed ternary Bonsai 2 27B checkpoint derived
from Qwen3.8-27B.

The initial target is `PQ2_0` on NVIDIA A100 (`sm_80`). The project keeps the
existing `Bonsai-demo` checkout as the llama.cpp correctness oracle and references
its model file in place.

## Scope

1. Define a lossless safetensors representation for PQ2_0 codes, FP16 group
   scales, and Hadamard rotation metadata.
2. Validate a single projection against the Prism llama.cpp implementation.
3. Register an out-of-tree Bonsai 2 quantization method in vLLM.
4. Add an A100 CUDA kernel after the reference implementation reaches parity.
5. Add batching, vision, and speculative decoding only after text inference is
   correct.

Qwen3.8 uses the `Qwen3_5ForConditionalGeneration` implementation identifier in
its upstream Hugging Face configuration. The implementation should therefore
reuse vLLM's existing Qwen3.8/Qwen3.5 execution path and specialize the packed
weight operations rather than introduce a new transformer architecture.

## Setup

```bash
cd /home/ubuntu/vllm-bonsai2
uv sync
uv run bonsai2-vllm doctor
uv run pytest
```

The default source model is discovered at the sibling checkout:

```text
/home/ubuntu/Bonsai-demo/models/bonsai2-gguf/27B/
```

Override it without copying the model:

```bash
export BONSAI_GGUF_DIR=/path/to/model/directory
export BONSAI_GGUF_MODEL=Ternary-Bonsai-2-27B-PQ2_0.gguf
```

## Reference pins

- vLLM: `0.29.0`
- Python: `3.12`
- Reference model: `Ternary-Bonsai-2-27B-PQ2_0.gguf`
- Reference llama.cpp release: `prism-b10683-d8f26ee`

## Single-projection conversion (M1)

```bash
uv sync --locked
uv run bonsai2-vllm inventory
uv run bonsai2-vllm export-projection
uv run bonsai2-vllm verify-projection artifacts/blk.0.ffn_gate.safetensors \
  --prism-library /home/ubuntu/Bonsai-demo/bin/cuda/libggml-base.so \
  --output artifacts/blk.0.ffn_gate.parity.json
```

The defaults inventory the existing GGUF and export `blk.0.ffn_gate.weight`.
Use `--source`, `--tensor`, and `--output` to select another input/tensor/output.
The inventory includes all metadata (including tokenizer fields) and all tensor
descriptors. Exported packed bytes, FP16 scales, and rotation metadata are
round-trip checked; the optional Prism library validates every decoded weight
against the pinned C implementation. Generated files stay under ignored
`artifacts/`. The source model is read in place without copying it.

## Reference operator (M2)

The PyTorch reference now implements PQ2 unpack, explicit signs, normalized
block Hadamard, grouped GDN feature order, and FP32 projection. It has been
compared against the pinned Prism CPU and A100 CUDA graph implementations.

```bash
uv run python tools/build_prism_oracle.py
uv run bonsai2-vllm validate-operator artifacts/blk.0.ffn_gate.safetensors \
  --output artifacts/m2-gate-new-run
uv run pytest
```

Building the harness requires the pinned Prism source checkout and release
libraries; see [M2 validation](../M2_VALIDATION.md) for setup and results.
Ordinary tests use frozen synthetic Prism vectors and need neither the model
nor the Prism libraries. CUDA tests skip when a GPU is unavailable.

The real FFN gate passed all nine input cases. Hadamard outputs were exact;
native packed projection outputs differ slightly from the mathematical FP32
reference because of runtime arithmetic, with explicit regression thresholds.
This implementation expands one projection for correctness checking.

## Full text model (M3)

The complete checkpoint now converts to eight sharded safetensors files with
packed PQ2 weights. A registered vLLM plugin supports linear layers, embeddings
and the LM head through the existing Qwen3.5 text model execution path.

Follow [M3A conversion](../M3A_CHECKPOINT.md) to create
`converted/bonsai2-pq2`, then run:

```bash
uv sync --locked
uv run python tools/m3_smoke.py
```

The M3 validation configuration was BF16, eager execution, TP=PP=1 and
one sequence. All 851 source tensors passed independent reverse conversion.
Nine short greedy continuations matched Prism exactly (60 token IDs); 33/33
fixed-prefix decisions agreed, with maximum full-vocabulary TV of 1.18%.
M4 now supplies the optimized CUDA path described below. Long-form quality,
full τ² trajectories and production serving are not yet validated.

## A100 kernels and CUDA graphs (M4)

CUDA inference now defaults to fused Triton PQ2 kernels. Weights stay compressed;
the kernels perform unpacking and multiplication without a persistent dense
weight cache. Dedicated kernels handle the signed Hadamard and inverse embedding
operations. Set `BONSAI_BACKEND=reference` to reproduce the M3 implementation.

[M4 implementation and measurements](../M4_A100_KERNEL.md) includes the exact
configuration for BF16 text inference, CUDA decode graphs and up to four
concurrent requests on one A100. It retains TP=PP=1, a 2,048-token context limit
and disabled prefix caching. Use its explicit `compilation_config`; model-wide
torch.compile is not part of the validated configuration.

In the short warm benchmark, single-request throughput increased from 1.37 to
16.70 output tokens/s; four concurrent requests achieved 23.01 tokens/s in total.
The unchanged frozen parity gate passed (9/9 continuations, 33/33 next-token
decisions, maximum TV 1.32%), and the full test suite passed 73 tests.

These speedups are relative to our M3 reference, not native Prism. A follow-up
measured Prism TG128 at 73.46 tok/s versus M4 at 21.61 tok/s for a 128-token
end-to-end request. [The diagnosis](../M4_PERFORMANCE_DIAGNOSIS.md) documents
the harness differences and missing native integer-matvec optimizations.
The follow-up [M4.5 record](../M4_5_PRISM_PERFORMANCE.md) tracks the native
performance target using a matched HTTP workload and the opt-in integer backend.
The M4 backend remains available as `BONSAI_BACKEND=triton`.

## Native arithmetic backend (M4.5)

`BONSAI_BACKEND=integer` adds Prism-compatible Q8 activation quantization and
packed DP4A decode, shared transforms for merged projections, and fused Gemma
RMSNorm. The safetensors checkpoint is unchanged. Use the same BF16, TP=PP=1,
2,048-token context and decode-graph configuration as M4. This baseline disables
prefix caching; M5 below provides a separately validated serialized configuration.

```bash
source .venv/bin/activate
BONSAI_BACKEND=integer vllm serve converted/bonsai2-pq2 \
  --host 127.0.0.1 --port 8000 --served-model-name bonsai2 \
  --dtype bfloat16 --max-model-len 2048 --max-num-seqs 4 \
  --max-num-batched-tokens 128 --gpu-memory-utilization 0.6 \
  --no-enable-prefix-caching --enable-chunked-prefill \
  --attention-backend FLASH_ATTN \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4]}'
```

See the M4.5 record for the numerical gate, benchmark results and limitations.
On the matched 128-output-token HTTP workload, the integer backend measures
70.52–70.53 tok/s versus Prism's 65.89–65.93 tok/s on the same A100. The frozen
gate passes with 9/9 continuations, 33/33 next-token decisions and maximum TV
1.80%; 105 tests pass. Longer contexts and M5 features require separate validation.

## Speculative serving (M4.6)

The opt-in DFlash2 + Marlin configuration reaches **157.82 / 162.35 output
tok/s** on the original two-case, 128-output-token HTTP benchmark on A100.
Four additional code/Korean/reasoning/prose workloads combine to **111.60 tok/s**;
150 tok/s is not a general-workload guarantee. The frozen numerical gate passes,
but two longer outputs differ from the no-drafter control. The drafter and extra
INT4 runtime cache increase GPU memory use to about **34 GiB**.

See [M4.6](../M4_6_150_TPS.md) for the pinned draft checkpoint, exact serving
command, independent measurements and correctness limits. The stored model
remains PQ2 safetensors. [M5](../M5_SERVING.md) tracks subsequent serving checks.
Its initial prefix-cache/state gate passes 19/19 checks with `max-num-seqs=1`,
maintaining 156.03 / 160.42 tok/s on the original benchmark. Incoming requests
queue; multi-request GPU batching does not yet pass the exact-output gate.

The [M6 fixed English benchmark](../M6_REALWORLD_200_TPS.md) measures code,
reasoning, summary and prose separately. The selected configuration achieves
123.97 tok/s across the serial suite including prefill, or 158.15 tok/s during
streaming decode. Only the code case exceeds 200 decode tok/s; the suite's
200 tok/s target remains open. Inputs, timing rules and unsuccessful experiments
are recorded, with no experimental runtime change retained.

[M6.1](../M6_1_PREFILL.md) raises the prefill token budget from 128 to 256.
On the same fixed English inputs, this reduces the summary's first-token wait
from 356 to about 193 ms and improves its full HTTP rate from 127.55 to about
152 tok/s. All output token IDs match the baseline. This is a prefill latency
improvement; the 200 tok/s suite target remains open.

Each stage has a Markdown record: [M0](../M0_ENVIRONMENT.md),
[M1](../M1_VALIDATION.md), [M2](../M2_VALIDATION.md),
[M3A](../M3A_CHECKPOINT.md), [M3B](../M3B_VLLM_RUNTIME.md), and
[M3C](../M3C_MODEL_PARITY.md), [M4](../M4_A100_KERNEL.md), and
[M4.5](../M4_5_PRISM_PERFORMANCE.md), [M4.6](../M4_6_150_TPS.md), and
[M5](../M5_SERVING.md), [M6](../M6_REALWORLD_200_TPS.md), and
[M6.1](../M6_1_PREFILL.md).

See [docs/CHECKPOINT_FORMAT.md](../CHECKPOINT_FORMAT.md) for the implemented
projection contract and [docs/ROADMAP.md](../ROADMAP.md) for milestone gates.
