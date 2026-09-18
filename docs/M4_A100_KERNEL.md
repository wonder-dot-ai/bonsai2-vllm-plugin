# M4 — A100 packed kernels and execution

Status: initial implementation/validation completed on 2026-09-18;
performance optimization against native Prism was left open at M4. See the
[native-baseline diagnosis](M4_PERFORMANCE_DIAGNOSIS.md): Prism TG128 measured
73.46 tok/s; this implementation measured 21.61 tok/s for a 128-token end-to-end
request. The harnesses differ, but extending generation does not close the gap.
The subsequent [M4.5 integer backend](M4_5_PRISM_PERFORMANCE.md) reaches
70.52–70.53 tok/s against Prism's 65.89–65.93 tok/s in the matched HTTP workload.
The historical measurements below describe M4's FP32 backend.

## Scope

Keep the M3 safetensors representation and frozen text parity gate. Optimize
packed linear/LM-head operations and inverse embedding, verify CUDA graph replay,
and measure single-sequence latency and multi-request throughput on the A100.
No checkpoint requantization, persistent dense weight cache or quality threshold
relaxation is part of this stage.

## Implementation

`src/vllm_bonsai2/kernels.py` supplies Triton CUDA kernels for signed FP32 block
Hadamard, fused unpack-and-GEMV, fused unpack-and-GEMM, and inverse embedding.
Small decode batches use FP32 reductions; larger batches use TF32x3 tensor-core
products with FP32 accumulators. See the [Triton dot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
for the precision mode. The FWHT preserves the M2 scale-before-butterflies order.

PyTorch custom operators with fake implementations isolate batch dispatch for
`torch.compile`. CUDA graph tests change activation values and token IDs between
replays to detect stale inputs. The reference implementation remains available
with `BONSAI_BACKEND=reference`; CUDA uses `triton` by default, CPU keeps reference.

The GEMM batch size is a runtime argument, preventing a separate compilation for
every prefill length. Kernel geometry remains specialized for each matrix width.
Tile selection is fixed from the local A100 experiment, not autotuned during
serving. No model weights or tokenizer files change in M4.

## Validated execution configuration

Use the existing converted checkpoint with the installed plugin. For example,
from the repository root:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="converted/bonsai2-pq2",
    dtype="bfloat16",
    enforce_eager=False,
    compilation_config={
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 3, 4],
    },
    max_model_len=2048,
    max_num_seqs=4,
    max_num_batched_tokens=128,
    gpu_memory_utilization=0.6,
    enable_prefix_caching=False,
    enable_chunked_prefill=True,
    attention_backend="FLASH_ATTN",
)
outputs = llm.generate(
    ["The capital of France is"],
    SamplingParams(temperature=0, max_tokens=16),
)
print(outputs[0].outputs[0].text)
```

`mode=0` disables model-wide torch.compile; `FULL_DECODE_ONLY` still captures
the decode forward pass in CUDA graphs. Operator-level torch.compile tests are
separate from this tested model configuration. Eager execution remains available
with `enforce_eager=True`. TP/PP must remain one; LoRA is rejected.

## Reproduction and evidence

```bash
uv run pytest
uv run ruff check .
uv run python tools/m4_operator_bench.py --output artifacts/m4-operators-new.json
uv run python tools/m4_model_validation.py --output artifacts/m4-eager-new.json
uv run python tools/m4_model_validation.py --graph --output artifacts/m4-graph-new.json
uv run python tools/m4_model_validation.py --backend reference --skip-parity \
  --output artifacts/m4-reference-new.json
```

Model validation requires M3's saved Prism full-vocabulary vectors under
`artifacts/m3-parity-bf16/`; regenerate those using the M3C procedure if absent.
Timing runs must use an otherwise idle GPU, with no concurrent oracle or benchmark.
Model timing includes prefill and generation, uses fixed 16-token outputs,
ignores EOS for equal work, warms each configuration, and reports the median of
three runs. It reports a single request and compares the same four requests
sequentially versus concurrently. Prefix caching is off and each measurement
includes prefill. It is not a decode-only or production-load benchmark.

The 73-test suite includes 21 new CUDA cases: FP32/FP16/BF16 activations, GEMV
and GEMM, partial row tiles, widths 5,120/6,144/17,408, noncontiguous inputs,
repeated embedding IDs, empty batches, changed inputs on graph replay and dynamic
batch torch.compile. Projection relative L2 must stay below `2e-6` against the
mathematical FP32 reference; embedding values must match exactly. This is much
tighter than the separate M3 full-model comparison against Prism.

## Full-model correctness result

The final kernel source passed the unchanged M3 gate with the graph configuration
above and four concurrent requests:

| Check | Result |
| --- | --- |
| Frozen greedy outputs | 9 / 9 exact, 60 token IDs |
| Fixed-prefix top-1 decisions | 33 / 33 match Prism |
| Finite full-vocabulary entries | 248,320 at every decision |
| Maximum TV versus Prism | 0.01316886 (1.3169%) |
| Mean TV versus Prism | 0.00333337 (0.3333%) |
| Maximum TV versus the saved M3 vLLM run | 0.01012470 (1.0125%) |

Evidence: `artifacts/m4-graph-final.json` and `artifacts/m4-graph-final.log`.
The report records kernel/runtime source hashes and the frozen corpus hash.
The saved M3 run used one sequence, so the M4-to-M3 comparison also includes
changes in batching and accumulation order; it does not isolate kernel error.

vLLM reported 7.0 GiB for model loading, 0.19 GiB peak activation memory in its
profiling pass, and 0.09 GiB actual CUDA graph pool memory. The separate KV cache
was approximately 16.3 GiB at the chosen 60% GPU-memory utilization setting.

## Performance results

End-to-end measurements on an otherwise idle GPU, after warmup, median of three
runs. The single request has five input tokens and 16 output tokens. The four
requests have 40 total input tokens and 64 output tokens. Generation ignores
EOS only for timing; the separate frozen parity test uses normal EOS stopping.

| Execution | Single request seconds | Single output tok/s | Four sequential seconds | Four concurrent seconds | Concurrent output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| M3 reference, eager | 11.7209 | 1.365 | 46.9001 | 13.8261 | 4.629 |
| M4 Triton, eager | 2.2977 | 6.963 | 9.3367 | 2.8502 | 22.454 |
| M4 Triton, decode graphs | 0.9581 | 16.700 | 3.9017 | 2.7809 | 23.014 |

The graph configuration is **12.23× faster** than the reference for the single
request and **4.97× faster** for the concurrent workload. Graphs improve M4
single-request throughput by 2.40× over eager; at four concurrent requests,
the improvement is only 1.025×. Within the graph configuration, submitting the
same four requests together instead of sequentially raises throughput by 1.403×.
These short, warm engine runs do not establish production throughput or p99 latency.

Evidence: `artifacts/m4-reference-final.json`, `artifacts/m4-eager-final.json`,
`artifacts/m4-graph-final.json` and their `.log` files. Source hashes recorded in
all three reports were checked against the final files.

The real-checkpoint per-layer benchmark uses CUDA events, two warmups and seven
samples per configuration. It measures eager calls, including launch gaps,
not isolated hardware compute time. Representative median latencies:

| Layer | Token rows | Reference ms | M4 ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| FFN gate + up | 1 | 5.9873 | 0.4751 | 12.60× |
| FFN down | 1 | 2.8733 | 0.2632 | 10.92× |
| GDN output | 1 | 1.3988 | 0.2386 | 5.86× |
| LM head | 1 | 33.9835 | 1.7367 | 19.57× |
| FFN gate + up | 128 | 8.1234 | 3.5625 | 2.28× |
| FFN down | 128 | 4.0581 | 2.5477 | 1.59× |
| GDN output | 128 | 1.8995 | 0.9718 | 1.95× |
| LM head | 128 | 52.7708 | 19.5799 | 2.70× |

All 16 measured layer/batch combinations improved; the complete results, including
rows 4 and 16 and BF16 output differences, are in
`artifacts/m4-operators-final.json`. The largest relative L2 after the final BF16
cast was `8.294e-5` (0.008294%). This differs from the pre-cast FP32 unit-test
criterion because values near BF16 rounding boundaries can select adjacent bins.

Final checks: `73 passed` in `artifacts/m4-tests.log`; `uv run ruff check .`
passed. All temporary inference workers exited and released GPU memory.

## Limits carried forward

- The frozen corpus has nine short prompts and 60 generated token IDs. Matching
  it does not prove long-context, long-generation or full τ² task equivalence.
- The TV gate is unchanged at 2%; it measures distribution difference, not a
  percentage loss in task accuracy. BF16 outputs need not be bit-identical to
  the M3 reference, even though the underlying FP32 operator errors are small.
- The measured configuration is an A100-SXM4-40GB, PyTorch 2.13.0+cu130,
  Triton 3.7.1 and vLLM 0.29.0. Other GPUs and full-model FP16/FP32 were not tested.
- Serving benchmarks use the local vLLM engine, not HTTP load generation. Prefix
  caching, abort/retry state lifecycle, longer contexts, tensor parallelism,
  vision and speculative decoding remain M5 or later work.
- Decode graph capture sizes are explicitly 1–4. Other concurrency levels and
  vLLM's default model-wide compilation settings are outside this validation.
