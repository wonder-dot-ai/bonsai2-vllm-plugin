# M4 follow-up — Gap to native Prism

Date: 2026-09-18. Runtime code is unchanged in this investigation.

Follow-up: [M4.5](M4_5_PRISM_PERFORMANCE.md) implements the work identified here
and passes the matched HTTP target: 70.52–70.53 tok/s versus Prism 65.89–65.93.
This document preserves the original M4 diagnosis and measurements.

The prior 12.23× improvement used our slow M3 reference as the baseline. It did
not establish performance parity with native Prism. Initial correctness and
graph integration passed; performance optimization against Prism remains open.

## Reproduction and measurements

On the same A100-SXM4-40GB and same original PQ2 checkpoint, the pinned Prism
binary measured **73.459 ± 0.147 output tokens/s** in `llama-bench` TG128,
without a speculative drafter:

```bash
/home/ubuntu/Bonsai-demo/bin/cuda/llama-bench \
  -m /home/ubuntu/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 -fa on -p 0 -n 128 -r 3 -o json
uv run python tools/m4_diagnose.py kernels
uv run python tools/m4_diagnose.py model
```

Keep the GPU idle between these runs. Results:

| Path | Workload | Observed rate |
| --- | --- | ---: |
| Prism `llama-bench` | TG128, decode-only bare loop | 73.46 tok/s |
| vLLM M4 + graphs | Five input tokens, 16 output tokens, end-to-end | 16.70 tok/s |
| vLLM M4 + graphs | Same input, 128 output tokens, end-to-end | 21.61 tok/s |

The vLLM 128-token request took 5.9242 seconds. Subtracting the 16-token request
time gives approximately **44.34 ms per additional token** (22.55 tok/s).
That is an incremental timing estimate, not a directly instrumented decode-only
measurement. The two engines still use different harnesses, inputs/cache dtypes
and timing boundaries. These results nevertheless show that the short-output
measurement penalty does not explain the performance gap.

Artifacts: `artifacts/m4-prism-tg128.json` and `.log`,
`artifacts/m4-longer-generation.json` and `.log`.

## Main implementation difference

The pinned native source's `vec_dot_pq2_0_q8_1` in
`artifacts/prism-llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh` decodes ternary symbols
into packed integer lanes, uses Q8_1 activations, accumulates with
`ggml_cuda_dp4a`, and applies scales to grouped partial sums. NVIDIA documents
[`__dp4a`](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html)
as a four-way INT8 dot product with INT32 accumulation.

Our `kernels.py::_gemv`, used for batches 1–4, instead decodes to FP32 in
registers, applies scales to individual weights, multiplies FP32 activations,
and reduces FP32 products. It does not use the native Q8/DP4A algorithm or the
TF32x3 GEMM path (the latter starts at five token rows). FP contraction is also
disabled on this path. It avoids a dense weight cache, but still performs much
more elementwise conversion and floating-point work than the native algorithm.

Hadamard transforms and projection outputs remain separate launches/buffers;
fused projections process segments separately, followed by permutation,
concatenation and dtype conversion. Graphs remove much of the host launch cost
but do not eliminate this GPU work. This is an additional optimization target,
not a separately quantified fraction of total latency.

## Kernel timing evidence and its limits

`m4_diagnose.py kernels` times each distinct projection shape under CUDA graph
replay, flushes 128 MiB between samples, and weights medians by the actual
checkpoint's 401 linear segments. The summed isolated GEMV cost was **48.78 ms**;
the analogous transform sum was **3.66 ms**. FFN gate/up and down alone account
for about 31.52 ms of that isolated GEMV sum.

This is a synthetic, cache-flushed component estimate, not an additive profile
of a real decode step. Cache residency, graph submission overhead and GPU clock
state differ from continuous model execution; the sum even exceeds the observed
incremental decode time. Do not turn it into a precise percentage of total time.
It supports prioritizing packed matvec over file loading or further host-only
graph changes. All tested GEMV variants used 128 registers/thread with zero
spills; achieved occupancy/stall causes were not measured with hardware counters.
Evidence: `artifacts/m4-kernel-diagnosis.json` and `.log`.

## Next performance work

Keep M4 performance work open before adding more serving features. Establish a
matched native/vLLM decode benchmark, then port the native Q8 activation + packed
integer dot-product strategy, reduce repeated transforms/output handling and
profile the remaining model path. Activation quantization changes arithmetic,
so compare against the native packed operator and the unchanged full-model
Prism regression gate; do not silently relax the existing FP32 reference tests.
Reaching 70+ tok/s is a target to measure, not a promised result of any one change.
