# M5 — Serving features

Status: initial serialized-serving gate passed, 2026-09-18; full M5 remains open.
M4.6 exceeds 150 tok/s on the original matched
benchmark; its additional workload and numerical limitations remain explicit in
[M4.6](M4_6_150_TPS.md). Starting M5 does not close those limitations.

## Stage gates

| Part | Required evidence | Status |
| --- | --- | --- |
| Prefix cache and hybrid GDN state | Actual cache hits; cold/warm output comparisons; request interleaving, reset and concurrent serving | Initial gate passed with one executing request; GPU batch > 1 exact-token gate fails |
| Serving lifecycle | Request cancellation, slot reuse, health after failures, bounded concurrency | Disconnect/reuse, reset and queued arrivals pass; overload, failure recovery and eviction remain open |
| Tensor parallelism | Correct weight sharding and real multi-GPU parity/performance | Blocked on additional GPU; current adapter explicitly rejects TP > 1 |
| Vision | Compatible projector conversion, multimodal inputs and image-conditioned reference checks | Pending; current adapter is text-only |
| Bonsai drafter | Target-specific training data, training, held-out quality and latency evaluation | External Bonsai-specific DFlash2 integrated in M4.6; local training remains pending |

One A100 validates only TP=PP=1. Merely removing the TP guard would not implement
weight sharding. Likewise, inherited upstream model capabilities do not establish
that this checkpoint adapter supports vision.

## Prefix cache validation

Use the selected M4.6 runtime without changing its source or the measured
benchmark. Enable prefix caching explicitly with hybrid Mamba/GDN `align` mode.
Use long shared input prefixes, inspect server cache-hit counters, compare
explicit output token IDs and preserve results even when a check fails.
Cold/warm equality is required before recommending the configuration.

The cache remains disabled in the documented M4.6 performance command. The
separate M5 command below enables it with serialized execution. No cache-enabled
speed is substituted into the M4.6 150 tok/s claim.

The initial probe found two harness requirements rather than silently treating
successful generation as proof of cache reuse:

- `/reset_prefix_cache` is available only with `VLLM_SERVER_DEV_MODE=1`. The
  first attempt recorded HTTP 404 and stopped. Development endpoints are enabled
  only on the owned loopback test server, never added to the serving example.
- The selected hybrid speculative runtime uses 448-token cache blocks and
  excludes the last matching block from reuse. Inputs of 735–739 tokens had
  **zero cache hits**. All 18 other checks passed (cold/warm output, interleaved
  and concurrent requests, post-disconnect reuse, reset, health and frozen gate),
  but the overall probe correctly failed its actual-hit requirement.

Those records are preserved in `artifacts/m5-prefix-dflash7/` and
`artifacts/m5-prefix-dflash7-dev/`. The follow-up uses 1,167–1,171 input tokens to
exercise real cache reuse. Each output comparison uses 32 greedy token IDs, with
EOS ignored; this is bounded state-regression coverage, not a quality benchmark.

Reproduce using an unused output directory:

```bash
.venv/bin/python tools/m5_serving_probe.py --max-num-seqs 1 --benchmark \
  --output artifacts/m5-reproduction
```

The probe derives the selected command from
`artifacts/m46-final-repeat/command.json`, records its final command and source
hashes, binds port 8093 locally, and always shuts down its owned server. It saves
the explicit input IDs, output IDs, counters and failed checks for inspection.

## Initial concurrency finding

`artifacts/m5-prefix-dflash7-long/` records actual cache reuse: 1,792 cached
tokens among 9,356 queried tokens in the cold/warm phase. All four warm outputs
exactly match their cold outputs. Six interleaved requests, post-disconnect reuse,
reset, health and the frozen HTTP gate also pass.

With four requests executing concurrently, two outputs differ from their cold
sequential references: Cedar at output token 20, Maple at token 26. Both preserve
the requested numeric answer (18 and 35); the explanation wording diverges.
This still **fails the exact-token serving gate**, and is not waived as success.

The control `artifacts/m5-nocache-concurrent-control/` disables prefix caching and
also fails the concurrent Cedar comparison. Therefore cache reuse alone does
not explain the observed batch dependence. This does not isolate every numerical
source, nor prove that the additional Maple difference is unrelated to caching.
Multi-request GPU batches do not pass this stage's exact-token gate.

The follow-up restricts `max_num_seqs` to one: four incoming requests may queue,
but GPU execution is sequential. Its cache/state gate and original 128-token
throughput benchmark are evaluated separately; serialized service must not be
described as validated four-request GPU batching.

## Serialized configuration: passed

`artifacts/m5-prefix-serial/results.json` passes **19/19** checks. It contains
20 requests with recorded 32-token outputs, plus the ten-case frozen HTTP gate
(nine cases and one repeated case). Four incoming concurrent HTTP requests queue
and execute one at a time. All match their sequential cold references. The
stream-disconnect exercise is followed by an identical fresh output and an idle,
healthy server; this does not establish every cancellation race is covered.

The original benchmark measures **156.03 / 160.42 tok/s**, still above 150 on
both cases. These are median HTTP times from one warmup and three timed requests,
128 generated tokens each. No prefix cache hits are possible for the original
five-token inputs, so this rate does not rely on replaying their prompts from
cache. The M4.6 additional-workload limit (111.60 tok/s combined in that run) is
not superseded by this two-case measurement.

The validated serving configuration is:

```bash
source .venv/bin/activate
BONSAI_BACKEND=integer BONSAI_VERIFY_KERNEL=marlin \
vllm serve converted/bonsai2-pq2 \
  --host 127.0.0.1 --port 8000 --served-model-name bonsai2 \
  --dtype bfloat16 --max-model-len 2048 --max-num-seqs 1 \
  --max-num-batched-tokens 128 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --mamba-cache-mode align --enable-chunked-prefill \
  --attention-backend FLASH_ATTN \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8]}' \
  --speculative-config '{"method":"dflash","model":"artifacts/m46-dflash2","num_speculative_tokens":7}'
```

Validation uses greedy requests (`temperature=0`) with explicit token IDs.
The serving example excludes the development endpoints used by the test probe.
Model/runtime sources are unchanged from the M4.6 full numerical gate; this
stage adds serving configuration and lifecycle coverage, not another weight or
kernel revision. All owned test servers have been stopped.

## Remaining work

- Isolate batch-dependent numerical differences and validate multi-request GPU
  execution without weakening the fixed correctness criteria.
- Exercise cache eviction, preemption, overload and interrupted-request races.
- Add real TP sharding and test on multiple GPUs; current hardware has one GPU.
- Convert and validate the vision projector and multimodal request path.
- Train/evaluate a Bonsai drafter on separate training and held-out workloads;
  integration of an external checkpoint is not completion of this training gate.
