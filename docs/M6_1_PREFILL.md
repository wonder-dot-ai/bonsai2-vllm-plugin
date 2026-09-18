# M6.1 — Reduce fixed-English prefill latency

Status: completed for the fixed English suite, 2026-09-18.
Selected prefill budget: 256 tokens. The overall 200 tok/s target remains open.

The selected M6 configuration caps each scheduled batch at 128 tokens. The
197-token fixed summary input therefore requires two prefill chunks. This stage
compares a 256-token cap, permitting that input to prefill in one chunk. Target
weights, drafter, seven proposals, decode graphs and numerical tolerances remain
unchanged. Prefix caching stays disabled and only one request executes at a time.

The fixed English corpus and 128-token output contract from
[M6](M6_REALWORLD_200_TPS.md) remain the acceptance workload. Compare all four
inputs, HTTP throughput, decode throughput, TTFT, all 128 output token IDs and
the frozen short HTTP gate. A larger token budget is a serving configuration
change; it must not be reported as faster model decode without measurement.

The harness now accepts `--max-num-batched-tokens`; its default remains 128 so
existing M6 reproduction commands keep their original behavior.

```bash
BONSAI_VERIFY_KERNEL=marlin .venv/bin/python tools/m6_compare.py \
  --engine integer --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 256 \
  --speculative-config '{"method":"dflash","model":"artifacts/m46-dflash2","num_speculative_tokens":7}' \
  --output artifacts/m61-prefill256-reproduce --validate
```

Use a fresh output directory. The owned server is stopped after measurement.

## Initial results

Source: [`m61-comparison.json`](../artifacts/m61-comparison.json).

| Input | HTTP tok/s, 128 cap | HTTP tok/s, 256 cap | TTFT ms, 128 cap | TTFT ms, 256 cap |
|---|---:|---:|---:|---:|
| Code | 171.47 | 171.33 | 190.1 | 191.9 |
| Reasoning | 134.27 | 134.35 | 194.5 | 194.8 |
| Summary | 127.55 | 152.19 | 356.0 | 193.0 |
| Prose | 89.73 | 89.72 | 174.2 | 175.2 |

The summary's full HTTP rate improves **19.32%** and its first-token latency
falls **45.79%**. Combined serial HTTP throughput improves from **123.97** to
**129.05 tok/s** (**4.09%**). Decode throughput is essentially unchanged
(158.15 → 158.27 tok/s combined). The reduction in prefill chunks explains the
benefit; it does not resolve the drafter acceptance/verification bottleneck.

All four 128-token outputs match the baseline in every timed repetition. The
frozen HTTP gate passes nine continuations plus its repeated first case.
Production source hashes match M6. No model or numerical kernel changed, and no
full-vocabulary distribution rerun is claimed. Larger inputs, prefix-cache
lifecycle and concurrent GPU batching are outside this configuration check.

Both per-case 200 tok/s gates remain false. M6's overall target stays open.

## Reproduce the comparison

After the reproduction benchmark finishes:

```bash
.venv/bin/python tools/m61_report.py \
  --baseline artifacts/m6-english-baseline7 \
  --candidate artifacts/m61-prefill256-reproduce \
  --output artifacts/m61-reproduction-comparison.json
```

The comparison checks corpus hashes, case order, input token IDs, output length,
repeat consistency and exact output token equality. It records per-case rates
and aggregate rates without dropping slow cases.

## Independent restart and decision

[`m61-comparison-repeat.json`](../artifacts/m61-comparison-repeat.json) records a
fresh server process with the same 256-token budget. Summary throughput was
**152.00 tok/s**, TTFT **194.6 ms**, and combined HTTP throughput **129.00 tok/s**.
Combined decode throughput was **158.15 tok/s**, matching the baseline.
All 12 timed responses again matched their baseline token IDs; the frozen short
HTTP gate passed again. The only server-command change relative to M6 is the
prefill budget, 128 → 256.

Use the **256-token budget for subsequent fixed-English measurements**. The
harness's 128 default remains for historical reproduction; pass the new option
explicitly. The M5 cache-enabled serving configuration has its own validation
record and is not changed by this experiment.

`ruff check src tests tools` passes. The two benchmark runs validate the serving
configuration directly; no new kernel or model tests are needed for this
configuration-only change. Both owned servers stopped after measurement.

Next performance work should address draft acceptance and verification cost:
removing one prefill chunk improves HTTP latency but cannot close the remaining
decode-rate gap on prose and reasoning.
