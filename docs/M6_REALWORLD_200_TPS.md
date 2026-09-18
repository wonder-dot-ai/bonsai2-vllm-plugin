# M6 — 200 tok/s target on fixed English inputs

Status, 2026-09-18: the fixed English benchmark is implemented and measured.
**The suite has not reached 200 tok/s.** Selected runtime remains M4.6 DFlash2
with seven draft tokens and Marlin verification. No experimental runtime change
was retained. Tau² evaluation is deferred; these are speed measurements, not
Tau² scores or evidence of general real-use 200 tok/s.

## Frozen measurement contract

- A100-SXM4-40GB, one request at a time, TP=PP=1, BF16, context limit 2,048.
- Corpus: [`tests/data/m6/english_fixed.json`](../tests/data/m6/english_fixed.json).
  SHA256: `2941ca80ec120264c041417dba2dce410e6cf7f341227308f87910e62d08075c`.
- Four English inputs fixed before measurement: code (merge intervals), reasoning
  (schedule constraints), summary (library reservation trial), prose (lighthouse
  narrative). Existing M4.6 English cases retain their exact input token IDs.
- Chat template uses `enable_thinking=False`; requests send the frozen token IDs.
- Exactly 128 output tokens, temperature 0, seed 123, `ignore_eos=True`.
  This measures a fixed output prefix, not complete-answer quality.
- One warmup, then three timed requests per case; report medians. Prefix caching
  is disabled; responses are generated each time. No concurrent aggregate rate.
- Primary metric remains full HTTP output tok/s, including prefill and transport,
  consistent with the earlier 150 tok/s record. Decode rate is a separate metric:
  tokens after the first token-bearing SSE chunk divided by elapsed time between
  the first and last token-bearing chunks. It is a client-observed streaming rate,
  not GPU kernel throughput. TTFT is time to the first token-bearing chunk.
- Keep all four cases when assessing the suite; do not select only the fast case
  or train on these inputs. The 200 target remains open under both metrics.

## Selected baseline

Source: [`m6-english-baseline7/benchmark.json`](../artifacts/m6-english-baseline7/benchmark.json).

| Input | Input tokens | Full HTTP tok/s | Decode tok/s | TTFT ms |
|---|---:|---:|---:|---:|
| Code | 42 | 171.47 | 228.54 | 190.1 |
| Reasoning | 68 | 134.27 | 167.59 | 194.5 |
| Summary | 197 | 127.55 | 196.29 | 356.0 |
| Prose | 51 | 89.73 | 101.46 | 174.2 |

Combined HTTP throughput is **123.97 tok/s**, calculated as 512 output tokens
across the sum of the four median request times. Combined decode throughput is
**158.15 tok/s**, using the sum of median decode token counts divided by the sum
of median decode times. These are serial suite measurements, not averages of
per-case rates. Code exceeds 200 only when excluding initial token latency.
The earlier two-case 157.82 / 162.35 figures use different inputs and cannot be
compared directly to this suite.

All three timed outputs have identical token IDs within each case. The frozen
short HTTP gate passes all nine continuations plus a repeated first case. This
does not establish full-answer quality or long-output equality to no-draft
inference. M4.6's previously documented numerical/long-output limits still apply.
The full-vocabulary distribution gate was not rerun: all 18 production source
files match the pre-experiment snapshot byte for byte.

Machine-readable outcome: [`m6-acceptance.json`](../artifacts/m6-acceptance.json).
Both `all_cases_http_200_pass` and `all_cases_decode_200_pass` are false.

Follow-up: [M6.1](M6_1_PREFILL.md) evaluates a larger prefill token budget on
these same frozen inputs. The figures above remain the original M6 baseline.

## Experiments and decisions

### More speculative tokens

[`m6-english-draft15`](../artifacts/m6-english-draft15/benchmark.json) uses 15
proposals instead of seven. Decode rates, in corpus order, were 219.94 / 173.75 /
162.00 / 83.57 tok/s; combined HTTP throughput was 113.69 tok/s. The short gate
passed, but the summary's 128-token output differed from the selected baseline.
Other cases matched. Reject this setting: overall speed regressed.

An earlier three-proposal diagnostic on the older M4.6 corpus also regressed.
Its Korean case is not part of this English acceptance suite; artifacts remain
in `artifacts/m6-draft3/` for the experiment record.

### Direct packed PQ2 tensor-core kernel

`tools/m6_pq2_mma.py` evaluates direct 2-bit unpacking with FP16 dot products and
FP32 accumulation. At eight rows, best measured projection times were:

| Projection N × K | Experimental ms | Marlin ms |
|---|---:|---:|
| 34816 × 5120 | 0.347 | 0.113 |
| 5120 × 17408 | 0.208 | 0.077 |
| 248320 × 5120 | 2.252 | 0.572 |

Too slow to integrate. Oversized shared-memory configurations are skipped after
an initial resource failure. Results: `artifacts/m6-pq2-mma.json`.

### GemLite 2-bit path

Experimental source pinned to `89d9bc705c5dfca9115d3a5620f97a17ba0111a7` in
`artifacts/m6-gemlite-src`; not installed into the serving environment.
`tools/m6_gemlite_bench.py` measured 0.178 / 0.100 / 0.942 ms for the same eight-row
shapes, versus Marlin 0.104 / 0.071 / 0.527 ms in that run. Four-row cases were
also slower. No production integration or full-model numerical validation was
warranted. Results: `artifacts/m6-gemlite-bench.json`.

### Drafter context CUDA graph

Capturing the drafter's small context-KV updates did not improve speed. Code /
reasoning / summary / prose decode rates were 228.45 / 167.51 / 196.16 / 101.45
tok/s. All output token IDs matched the baseline, and the short gate passed.
The first attempt failed startup due to the subclass constructor signature; the
corrected run is `artifacts/m6-english-context-graph-retry/`.

The runtime patch was reverted. Source snapshots remain in
`artifacts/m6-context-graph-experiment.py` and
`artifacts/m6-context-graph-plugin.py` solely as experiment records.

### Marlin reduction and output modes

`tools/m6_marlin_modes.py` found little benefit from lower-precision reduction.
Atomic mode improved the head microbenchmark from about 0.550 to 0.528 ms, with
little change for the other shapes; this is not an end-to-end improvement and
was not integrated. FP16 input with a BF16 output buffer was rejected by the
operator, so that route cannot directly remove the output cast. Results and
errors: `artifacts/m6-marlin-modes.json` and `.log`.

## Interpretation and remaining work

Speculative acceptance varies with input, so reducing verification cost alone
does not yield a uniform speedup. The earlier eight-step profile attributes
about 58% of summed GPU kernel time to Marlin, with additional drafter GEMM,
output casting, GDN and Hadamard-transform costs. This is diagnostic kernel time,
not a wall-clock speedup prediction. The tested direct 2-bit alternatives did
not beat the current verifier; longer draft blocks degraded the suite.

The fixed benchmark is ready for subsequent optimization. Reaching 200 across
it requires further improvements to verification cost and/or drafter acceptance.
No evidence currently supports marking that target complete or advancing past
it as if it had passed. M5's remaining serving features stay separately open.

## Reproduce

From the repository root, with converted target and pinned M4.6 drafter present:

```bash
BONSAI_VERIFY_KERNEL=marlin .venv/bin/python tools/m6_compare.py \
  --engine integer --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"dflash","model":"artifacts/m46-dflash2","num_speculative_tokens":7}' \
  --output artifacts/m6-english-reproduce --validate
```

Choose a fresh output directory. The wrapper records the server command, source
and corpus hashes, GPU identity, streaming chunks, token IDs and speculative
metrics, then stops its server. `m6_http_bench.py` supports vLLM token-ID streaming;
use the existing M4.5 tooling for native Prism comparisons.

Validation: baseline and both completed English experiments passed the frozen
short HTTP gate; baseline repetitions were token-identical; production runtime
matches its pre-experiment snapshot; `ruff check src tests tools` passes. No
benchmark server or GPU process remains running after these measurements.
