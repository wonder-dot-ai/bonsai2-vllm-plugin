# M4.5 — Native Prism performance target

Status: **completed for the measured A100 text workload, 2026-09-18**.
The final integer backend passes the unchanged numerical gate and exceeds
native Prism in both matched HTTP cases. This satisfies the M4.5 prerequisite
for proceeding to the user's full M5 scope; it is not a claim about unmeasured
context lengths, batch sizes or model quality.

## Final results

Same A100-SXM4-40GB, explicit five-token prompts, 128 generated tokens, one
request at a time, greedy generation, no prefix cache or drafter. Numbers are
output tokens divided by median HTTP wall time over three warm measurements.

| Input | Prism, final repeat | vLLM integer, final | Relative speed |
|---|---:|---:|---:|
| raw_0 | 65.895 tok/s | 70.529 tok/s | 1.0703× |
| raw_1 | 65.925 tok/s | 70.518 tok/s | 1.0697× |

Prism was measured before and after optimization; its earlier results were
65.882/65.947 tok/s. The final baseline reproduces that range. The earlier
73.46 tok/s `llama-bench` result excludes the HTTP/prefill boundary and is not
used as the denominator here. The final native HTTP server reports roughly
71.4 decode tok/s internally, consistent with this distinction.

- Full model: 9/9 exact frozen greedy continuations, 33/33 matching next-token
  decisions, all 248,320 log probabilities finite at every checked prefix.
- Maximum full-vocabulary TV: **0.01800836**, below the unchanged **0.02** gate.
  This is a short fixed-corpus compatibility check, not a general quality bound.
- Tests: **105 passed**. Native Q8 goldens and the separate FP32 tests both pass.
- Final eight-decode-iteration trace: **97.97 ms** summed kernel duration,
  down from 139.42 ms before normalization/projection fusion. Integer GEMV is
  70.66 ms; shared FWHT/Q8 is 7.44 ms. GEMV and transform launches each fall
  from 3,208 to 2,056 across those eight iterations (401 → 257 per token).
- The older 16-token in-process workload now reaches 35.33 tok/s for one
  request and 75.96 aggregate tok/s for four concurrent requests. These figures
  use different workload/timing boundaries from the HTTP table above.
- Model loading reports 7.1 GiB; weights remain packed. The server's configured
  KV/GDN cache budget is additional memory, not a dense-weight expansion.

Evidence: `artifacts/m45-acceptance.json`, `m45-integer-fused.json`,
`m45-tests-fused.log`, `m45-integer-http-fused/{benchmark,provenance,profile-summary}.json`,
and `m45-prism-http-final/benchmark.json`. The acceptance audit verified the
current runtime source hashes against the parity and final HTTP records.

## Implementation and intermediate measurements

The M4 FP32 backend and its numerical tests are preserved. The new opt-in
`BONSAI_BACKEND=integer` uses Q8_1-style activation quantization plus packed DP4A
for decode batches 1–4; larger batches currently retain the M4 prefill path.
Hadamard and Q8 conversion are fused into one kernel. Weight files are unchanged.

The pinned Prism CUDA build uses `-use_fast_math`. Matching its approximate
division is necessary for activation values near rounding ties; an exact divide
initially selected a few different bins in the frozen down-projection case.
Rounding is nearest with ties away from zero, and scales are stored as FP16.
The immutable M2 native packed outputs are the oracle for this arithmetic;
the separate M4 mathematical FP32 tests are not relaxed.

The first integer implementation passed the unchanged full-model gate (9/9
continuations, 33/33 top-1, maximum TV 1.702%). The first matched 128-output-token
HTTP workload measured 52.20–52.23 tok/s against Prism's 65.88–65.95 tok/s.
Artifacts: `artifacts/m45-integer-first.json`, `artifacts/m45-integer-http-first/`,
and `artifacts/m45-prism-http/`. These intermediate results do not meet the target.

An actual eight-iteration GPU trace showed 70.40 ms in integer GEMV and 11.52 ms
in FWHT/Q8 conversion, out of 139.42 ms total summed kernel duration. The trace
also exposed numerous copies, reductions, powers, additions and reciprocal square
roots from Gemma RMSNorm: its FP32 effective weights selected vLLM's native
multi-operation fallback. `normalization.py` fuses this operation, preserving
FP32 `weight + 1`, FP32 variance, and normalization of the **unrounded** residual
sum. The next residual is separately stored in the activation dtype. Tests cover
BF16/FP16/FP32, residual/no-residual, a non-power-of-two hidden width, and graph
replay. This optimization is restricted to the integer backend.

Normalization fusion raised HTTP throughput to 63.10–63.11 tok/s, still below
Prism. Its eight-iteration trace fell to 111.23 ms of summed kernel duration;
integer GEMV remained 70.37 ms. The frozen model gate still passed, maximum TV
1.801%. See `artifacts/m45-integer-norm.json` and `artifacts/m45-integer-http-norm/`.

The next optimization merges compatible projection segments in device memory
after weight loading. Original parameters become views of the merged storage,
so there is no second persistent packed-weight copy and checkpoint names/values
are preserved. Segments are merged only when their signed rotations match.
Decode uses one FWHT/Q8 transform and one DP4A launch per combined projection,
loads rows in HF order, and writes BF16 output directly. This removes repeated
transforms, output concatenation, permutation copies and separate FP32-to-BF16
casts. Larger prefills retain the FP32 M4 GEMM with the same row mapping.
The embedding inverse transform and biased projections retain their existing
paths. Tests explicitly exercise FFN, ordinary QKV, GDN untile/QKV-untile,
tail rows, batches 1/4/7, graph replay, and parameter values/shared storage.

Acceptance requires the unchanged frozen full-model Prism gate and a measured
native/vLLM performance comparison with matching input token IDs, generation
length, no drafter, idle GPU, warmup and comparable timing boundaries. The
previous 73.46 tok/s native TG128 and 21.61 tok/s vLLM end-to-end results are
diagnostic baselines, not a fully matched pair.

M5 follow-on scope remains prefix caching/state lifecycle, HTTP serving, TP,
vision and a Bonsai-specific drafter. Hardware/data prerequisites and validation
status must be recorded for each part; none are marked complete by this stage.

Longer prompts still use the FP32 prefill GEMM; this stage does not establish
prefill throughput parity, long-context quality, τ² trajectories, multi-GPU
correctness, vision or speculative decoding. CPU and GPU/backend combinations
outside the documented configuration are not newly validated by this result.

## Reproduction

Run from the repository root on the otherwise idle A100. Output destinations
must be new paths. The comparison tool starts a temporary loopback-only server,
waits for readiness, runs one warmup and three timed requests per prompt, and
terminates its process group even on failure. HTTP timing excludes startup and
includes prefill, decode and response transfer. All requests use explicit token
IDs, greedy decoding, 128 output tokens, ignored EOS, no prefix reuse and no
speculative model. Both prompts contain five input tokens. The vLLM process has
four available sequence slots; each measured request runs alone. The native
process has one slot. Optional tracing occurs **after** timed requests.

```bash
source .venv/bin/activate
pytest
python tools/m4_model_validation.py --backend integer --graph \
  --output artifacts/m45-parity-reproduction.json
python tools/m45_compare.py --engine prism \
  --output artifacts/m45-prism-reproduction
python tools/m45_compare.py --engine integer \
  --output artifacts/m45-integer-reproduction --profile
python tools/m45_profile_summary.py \
  artifacts/m45-integer-reproduction/trace/*.pt.trace.json.gz \
  --output artifacts/m45-integer-reproduction/profile-summary.json
```

Server commands, logs, current source hashes and GPU identity are saved with
each new comparison. The selected serving command is also in `README.md`.
`BONSAI_BACKEND=integer` is explicit: the default remains M4's `triton` backend,
and `reference` remains available for eager execution.
