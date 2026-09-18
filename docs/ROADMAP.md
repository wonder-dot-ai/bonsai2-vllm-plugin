# Roadmap and gates

Each stage maintains its own Markdown record of implementation, reproduction,
measured results and limitations. M3 is split into three independently recorded
steps. Completed records are linked below; later stages must add their records.

## M0 — Environment

Completed: [environment record](M0_ENVIRONMENT.md).

- uv environment installs cleanly on Python 3.12.
- `bonsai2-vllm doctor` detects the A100 and source PQ2_0 checkpoint.
- Unit tests pass without loading the 7.2 GB model.

## M1 — One projection

Implemented: `inventory`, `export-projection`, and `verify-projection` CLI
commands. See `M1_VALIDATION.md` for the real-checkpoint validation record.

- Pin the Prism GGUF reader at the reference llama.cpp release.
- Inventory GGUF metadata and tensors.
- Export one projection's packed codes, scales, and rotation metadata.
- Prove byte- and value-level round-trip parity.

## M2 — Reference operator

Completed for forward projection: PyTorch PQ2 unpack, signed block Hadamard,
GDN feature ordering, instrumented Prism CPU/CUDA comparison, fixed tolerances,
and immutable regression vectors. See `M2_VALIDATION.md` for scope and results.

- Implement PQ2_0 unpack and the matching Hadamard transform in PyTorch.
- Compare a projection against an instrumented Prism llama.cpp run.
- Freeze numerical tolerances and golden vectors.

## M3 — Full text model

Initial text gate completed:

- [M3A checkpoint](M3A_CHECKPOINT.md): all 851 source tensors converted and
  independently reversed, eight reproducible shards.
- [M3B runtime](M3B_VLLM_RUNTIME.md): quantized linear, embedding and LM head;
  eager BF16 text inference on A100 with TP=PP=1.
- [M3C parity](M3C_MODEL_PARITY.md): nine frozen short continuations match
  exactly; 33/33 next-token decisions agree, maximum full-vocabulary TV 1.18%.

Full τ² trajectory and long-context quality validation remains outstanding.
The recorded distributions are log probabilities, not raw logits.

## M4 — A100 kernel

Initial implementation validated for BF16 text inference with TP=PP=1, context
limit 2,048 and up to four concurrent requests. Performance optimization against
native Prism was left open by this stage: see [M4 implementation](M4_A100_KERNEL.md)
and the [native-baseline diagnosis](M4_PERFORMANCE_DIAGNOSIS.md). M4.5 below closes
that gap for the documented HTTP workload using the separate integer backend.

- Fused Triton packed GEMV/GEMM, signed FWHT and inverse embedding on `sm_80`.
- Operator capture/replay and torch.compile tests; actual vLLM decode graphs.
- Unchanged frozen parity gate: 9/9 greedy continuations and 33/33 top-1
  decisions, maximum TV versus Prism 1.32%.
- Single-request throughput 1.37 → 16.70 output tok/s (12.23×); four concurrent
  requests 4.63 → 23.01 tok/s (4.97×), in the documented short warm workload.
- Same four requests processed concurrently versus sequentially: 1.40× higher
  throughput with decode graphs. Full suite: 73 tests passed.

Model-wide torch.compile, longer contexts, more concurrency and HTTP load tests
remain outside this milestone's validated configuration.

## M4.5 — Match native Prism performance

Completed for the measured A100 text workload:
[implementation and measurement record](M4_5_PRISM_PERFORMANCE.md). The matched
128-token HTTP benchmark reaches 70.52–70.53 tok/s versus native Prism's
65.89–65.93 tok/s, about 7% faster. The unchanged gate passes: 9/9 continuations,
33/33 next-token decisions, maximum TV 1.80%; 105 tests pass. This satisfies the
performance prerequisite for the full M5 scope in the user's chosen sequence.

- Match Prism's Q8 activation quantization and packed DP4A arithmetic.
- Profile actual decode execution and fuse avoidable intermediate operations.
- Record matched native/vLLM throughput, numerical results and reproduction.

## M4.6 — 150 tok/s target

Original matched benchmark completed: [150 tok/s record](M4_6_150_TPS.md).
DFlash2 plus optional Marlin verification reaches 157.82 / 162.35 tok/s on an
independent restart. The unchanged numerical gate passes; 117 tests pass.
The four additional workloads combine to 111.60 tok/s, with long-output
divergence on two of four cases against the no-drafter control. General-workload
150 tok/s remains open. M5 proceeds from the original benchmark's bounded gate.

## M5 — Serving features

In progress: [M5 serving record](M5_SERVING.md). Hardware and data prerequisites
must be recorded for each part. A single A100 does not validate multi-GPU behavior.
Initial serialized prefix-cache/state gate: 19/19 checks pass, with the original
benchmark at 156.03 / 160.42 tok/s. Four incoming requests queue and execute
sequentially. Actual four-request GPU batching fails exact-token comparisons and
remains open, as do eviction, TP, vision and local drafter training.

- Prefix caching and hybrid GDN state lifecycle.
- Tensor parallelism.
- Vision projector.
- Train and integrate a Bonsai 2-specific DSpark drafter only after the target
  runtime is stable.

## M6 — 200 tok/s on fixed English inputs

In progress: [fixed English benchmark record](M6_REALWORLD_200_TPS.md).
The code/reasoning/summary/prose corpus is frozen and measured. Combined serial
throughput is 123.97 tok/s including prefill, or 158.15 tok/s during decode.
The code case alone reaches 228.54 decode tok/s. The suite does not meet 200
under either metric; this milestone is not complete. Tau² evaluation is deferred.

- Preserve fixed inputs, output length, correctness gates and serial execution.
- Record HTTP throughput, decode throughput and TTFT separately.
- Optimize verification and draft acceptance without training on the benchmark.
- Keep unsuccessful experiments and per-case regressions in the stage record.

## M6.1 — Prefill latency

[Prefill configuration and measurements](M6_1_PREFILL.md): a 256-token budget
processes the fixed 197-token summary input in one chunk. Its first-token wait
falls from 356 to about 193 ms and its full HTTP rate improves by about 19%.
All four 128-token outputs match the M6 baseline; the frozen short HTTP gate
passes. Decode speed and the outstanding 200 tok/s target are unchanged.
