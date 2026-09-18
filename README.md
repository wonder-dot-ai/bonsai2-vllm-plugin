# Bonsai 2 vLLM plugin

An out-of-tree vLLM plugin for **Ternary Bonsai 2 27B PQ2_0**. Serve the converted
safetensors checkpoint through vLLM's OpenAI-compatible API. Model weights stay
packed; this is a custom PQ2 format requiring this plugin, not a standard dense
Transformers checkpoint or an ordinary GPTQ/AWQ model.

- Code: [wonder-dot-ai/bonsai2-vllm-plugin](https://github.com/wonder-dot-ai/bonsai2-vllm-plugin)
- Public weights: [the-sweater-cat/bonsai-2-pq2](https://huggingface.co/the-sweater-cat/bonsai-2-pq2)
- Tested: Linux x86_64, Python 3.12, **vLLM 0.29.0**, one NVIDIA **A100 40GB**.
- Validated scope: BF16 text, context 2,048, TP=PP=1, one executing request.
  Multiple API requests queue. Vision and simultaneous GPU batching are not validated.

## Quick start

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/),
and use an NVIDIA driver compatible with the locked CUDA 13 PyTorch runtime.
This profile also requires a CUDA 13 toolkit (`nvcc`) and a C++ compiler for
FlashInfer's first-run kernel compilation. Set `CUDA_HOME` if the toolkit is
outside its standard installation location; first startup can take several minutes.
The tested driver is recorded in [release evidence](reports/release/README.md).
Allow about 11 GB for target + drafter downloads plus the Python environment.
The fast configuration uses about 34 GiB of GPU memory. Other GPU models and
smaller GPUs have not been validated.

```bash
git clone https://github.com/wonder-dot-ai/bonsai2-vllm-plugin.git
cd bonsai2-vllm-plugin
uv sync --locked
uv run --locked python tools/download_models.py
uv run --locked bonsai2-vllm doctor --runtime-only
uv run --locked bonsai2-serve
```

The download helper uses `hf download` with exact Hub revisions and checks every
file's SHA-256 against [configs/release.json](configs/release.json). Public model
downloads do not require a Hugging Face token. No source GGUF or sibling
`Bonsai-demo` checkout is required for this serving path. Run all commands from
the cloned repository root. The first startup includes kernel compilation.

The server listens on `127.0.0.1:8000`. Test it in another terminal:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai2","messages":[{"role":"user","content":"What is 6 times 7?"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

Use `--host 0.0.0.0 --port 8000` when external access is intended, with access
control provided by your deployment. `bonsai2-serve --dry-run` prints the exact
vLLM command. Stop the server with Ctrl-C.

The launcher selects the validated fast profile: `BONSAI_BACKEND=integer`,
`BONSAI_VERIFY_KERNEL=marlin`, seven DFlash2 proposals, decode CUDA graphs,
256-token prefill budget and prefix caching disabled. The safetensors weights
remain PQ2; Marlin creates an additional lossless INT4 runtime cache. The default
plugin backend when invoking `vllm serve` directly remains `triton`.

The drafter is the separately downloaded, pinned
[ProCreations DFlash2 checkpoint](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2).
To omit it, use `tools/download_models.py --no-draft` and `bonsai2-serve --no-draft`.
The rates below apply to the fast profile with the drafter.

## Reproduce the fixed English benchmark

Stop any running server to free the GPU, then run:

```bash
BONSAI_VERIFY_KERNEL=marlin uv run --locked python tools/m6_compare.py \
  --engine integer --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 256 \
  --speculative-config '{"method":"dflash","model":"artifacts/m46-dflash2","num_speculative_tokens":7}' \
  --output artifacts/english-reproduction --validate
```

Choose a new output directory for each run. This command owns and stops its
local server, saves token IDs and streaming timing, and runs the frozen short
continuation gate. Inputs are fixed in `tests/data/m6/english_fixed.json`:
one warmup plus three timed repetitions, 128 output tokens, greedy sampling,
EOS ignored, no prefix cache, one request at a time.

Measured on A100 40GB after an independent server restart:

| Fixed English input | Full HTTP tok/s | Streaming decode tok/s |
|---|---:|---:|
| Code | 171.15 | 228.62 |
| Reasoning | 134.41 | 167.56 |
| Summary | 152.00 | 196.25 |
| Prose | 89.71 | 101.48 |

The serial suite combines to **129.00 HTTP tok/s**, including prefill, or
**158.15 decode tok/s**. Only code exceeds 200 in the decode-only measurement;
**200 tok/s across the suite is not achieved**. These are fixed-prefix speed
measurements, not general task-quality or Tau² scores. Some longer outputs
previously differed from the no-drafter reference; short-gate equality does not
establish universal equivalence.

See [M6](docs/M6_REALWORLD_200_TPS.md), [M6.1](docs/M6_1_PREFILL.md), and the
[checked-in evidence](reports/release/README.md). Rates can vary across machines.

## Development and conversion

```bash
uv run --locked pytest
uv run --locked ruff check src tests tools
uv build
```

CUDA tests skip without a GPU. The locked environment is the reproducible path;
a wheel also carries the vLLM pin, plugin entry point and pinned Prism GGUF
reader dependency, but an unlocked pip install can resolve different transitive
versions. No vLLM source patch is required.

To recreate the checkpoint from the source GGUF, follow
[conversion instructions](docs/REPRODUCING_CONVERSION.md). Native Prism oracle
comparisons require its separate source/build; these are optional for serving.
`BONSAI_GGUF_DIR`, `BONSAI_GGUF_MODEL` and `BONSAI_LLAMA_SERVER` configure those
paths; see `.env.example`. Environment files are not loaded automatically.

## Status and records

[Roadmap](docs/ROADMAP.md) tracks incomplete performance and serving work.
[M5](docs/M5_SERVING.md) documents a separately validated serialized prefix-cache
profile; its remaining eviction, overload and concurrent GPU execution checks
are open. Multi-GPU, vision, longer contexts and local drafter training remain
unfinished. Do not interpret inherited vLLM features as validated adapter support.

Each stage has its own record:
[M0](docs/M0_ENVIRONMENT.md), [M1](docs/M1_VALIDATION.md),
[M2](docs/M2_VALIDATION.md), [M3A](docs/M3A_CHECKPOINT.md),
[M3B](docs/M3B_VLLM_RUNTIME.md), [M3C](docs/M3C_MODEL_PARITY.md),
[M4](docs/M4_A100_KERNEL.md), [M4.5](docs/M4_5_PRISM_PERFORMANCE.md),
[M4.6](docs/M4_6_150_TPS.md), [M5](docs/M5_SERVING.md),
[M6](docs/M6_REALWORLD_200_TPS.md), [M6.1](docs/M6_1_PREFILL.md),
[public release](docs/RELEASE.md).

Historical stage documents retain paths and artifact names from the development
machine. Large checkpoints, profiles and unsuccessful experiment outputs under
`artifacts/` are not distributed in Git. Use the quick start and checked-in
release reports for portable reproduction. The original development README is
preserved in [the archive](docs/archive/DEVELOPMENT_HISTORY.md).

Code is distributed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE).
The model repository preserves Prism ML's license and notices and records its
Qwen3.8 ancestry. The external drafter carries its own upstream notices.
