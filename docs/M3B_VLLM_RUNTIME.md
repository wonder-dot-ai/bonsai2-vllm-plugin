# M3B — vLLM integration

Status: completed on 2026-09-18 for initial single-sequence text inference.

Target: vLLM 0.29.0, TP=1, PP=1, eager execution, text-only Qwen3.5 path.
A general plugin registers Bonsai quantization and a thin model adapter.
The adapter installs a packed embedding and validates all loaded parameters.

Packed layers decode only a bounded output-row chunk to FP32 at a time, apply
the M2 transform and matrix multiplication, then cast output back to the model
activation dtype. Embedding gathers packed rows and applies the inverse
Hadamard then signs. This is a correctness implementation, not the M4 optimized
A100 kernel; speed and production serving are not acceptance criteria here.

## Implementation and reproduction

- `plugin.py` registers `bonsai2` quantization and `Bonsai2ForCausalLM` through
  the `vllm.general_plugins` entry point.
- `quantization.py` implements packed linear, embedding and LM head operations.
- `model.py` reuses vLLM's transformer execution and strictly checks parameter
  names, duplicates, missing parameters and shapes during loading.
- TP/PP other than one, graph execution and LoRA are explicitly rejected.

After M3A conversion, run from the repository root:

```bash
uv sync --locked
BONSAI_BACKEND=reference uv run python tools/m3_smoke.py
uv run pytest
uv run ruff check .
```

## Measured results and limits

The A100 loaded all 1,205 parameters successfully; vLLM reported approximately
7.0 GiB of model memory and 3.9 seconds for weight loading. These figures exclude
the KV cache and transient execution workspace. BF16 eager text generation
completed; the three initial eight-token continuations matched Prism.
The expanded comparison is recorded in [M3C](M3C_MODEL_PARITY.md).

The final unit/regression suite passed 52 tests, including chunked linear,
inverse embedding, GDN head ordering, TP rejection and conversion input guards.
Evidence: `artifacts/m3-smoke.json` and `artifacts/m3-prism-smoke.json`.

This reference path repeatedly decodes chunks and is slow (roughly 1.3 output
tokens/second in the short sequential run, not a controlled benchmark).
Tests used BF16 activations, a 2,048-token context limit, one sequence, chunked
prefill and disabled prefix caching. Long contexts, concurrent requests,
prefix-cache lifecycle, vision, speculative decoding and other activation
dtypes are outside this acceptance result. M4 owns kernel optimization and
performance measurement.

After M4, CUDA defaults to the optimized backend. The environment override above
preserves this stage's original reference execution; see
[M4](M4_A100_KERNEL.md) for the expanded graph/batching configuration.
