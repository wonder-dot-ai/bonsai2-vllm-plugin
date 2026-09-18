# M3C — Full model parity

Status: initial text parity gate completed on 2026-09-18.

Compare the same frozen token-ID prefixes with Prism and vLLM. Record next-token
log probabilities/top-k, top-1 decisions, and greedy continuations. A successful
model load or readable text is not sufficient to declare numerical parity.
Investigate divergence before claiming M3 complete. Full tau2 trajectory
regressions remain dependent on acquiring a frozen trajectory corpus.

## Method and fixed regression contract

`tests/data/m3/corpus.json` freezes nine token-ID prompts, expected greedy tokens,
eight-token generation limits, four teacher-forced decisions per case (or fewer
when generation ends), and a maximum total variation distance of 0.02.
The inputs cover three raw continuations and six chat cases including Korean,
arithmetic, JSON, code and translation. Expected outputs were frozen from this
initial successful Prism run. The 2% ceiling is a calibrated regression limit,
not a pre-established guarantee of model quality.

Prism uses release `prism-b10683-d8f26ee` and the original PQ2 GGUF. vLLM uses
the M3A checkpoint, BF16 activations, eager execution, TP=PP=1, one sequence,
FLASH_ATTN, context limit 2,048, and disabled prefix caching. Both consume the
same token IDs. Teacher forcing keeps later comparison inputs identical.

We save complete 248,320-entry log-probability vectors for both engines.
These are normalized logits, not raw pre-softmax logits. Distribution distance
is `TV(p,q) = 0.5 * sum(abs(p-q))` after normalization. The gate requires exact
frozen greedy tokens, identical top-1 decisions, all vocabulary entries finite,
and TV at or below 0.02 for every decision.

## Results

| Check | Observed |
| --- | --- |
| Exact greedy continuations | 9 / 9 |
| Generated token IDs compared, including EOS | 60 |
| Teacher-forced top-1 agreement | 33 / 33 |
| Maximum full-vocabulary TV | 0.01179128 (1.1791%) |
| Mean full-vocabulary TV | 0.00315500 (0.3155%) |
| Minimum top-20 overlap | 95% |
| Mean top-20 overlap | 98.6364% |
| Finite common vocabulary entries per decision | 248,320 / 248,320 |

Evidence: `artifacts/m3-parity-bf16/report.json`, `reference.json`, and
`logprobs-000.npz` through `logprobs-032.npz`. The saved report was also checked
against the final frozen-corpus gate without rerunning inference.

## Reproduction

Start the pinned oracle in a separate terminal:

```bash
/home/ubuntu/Bonsai-demo/bin/cuda/llama-server \
  -m /home/ubuntu/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 -c 2048 -np 1 -b 128 -ub 128 \
  --host 127.0.0.1 --port 8091 --no-warmup
```

Then, from this repository, choose a new output directory:

```bash
BONSAI_BACKEND=reference uv run python tools/m3_parity.py \
  --dtype bfloat16 --output artifacts/m3-parity-new-run
uv run python tools/m3_parity.py --check-report artifacts/m3-parity-bf16/report.json
```

The offline command needs the saved report but no running oracle. Stop the
temporary oracle after inference. Generated evidence is local and git-ignored;
the small frozen corpus is kept with the tests.

## Interpretation and remaining work

M2's approximately 0.56% single-projection relative L2 error did not change any
greedy token in these short cases. This supports proceeding to M4, but does not
prove long-form or task-quality equivalence. TV and relative L2 measure different
things; 1.18% TV is not a 1.18% loss of accuracy. The JSON and code cases may end
at the eight-token limit and do not assert complete, valid solutions.

Full τ² trajectories, longer generation/context, task scores and error growth
remain untested. Preserve this gate during M4 optimization and expand the corpus
before making broader quality or serving-readiness claims.
