# Public release — portable installation and safetensors distribution

Status: preparing and validating publication, 2026-09-18.

Destinations:

- Code: https://github.com/wonder-dot-ai/bonsai2-vllm-plugin
- Public model: https://huggingface.co/the-sweater-cat/bonsai-2-pq2

The serving quick start no longer depends on the development machine's source
GGUF, Prism binary or sibling checkout. The release downloads the converted
checkpoint and optional external drafter using `hf`, exact Hub commits and file
SHA-256 checks. Only model assets are fetched; no upstream scripts are executed.

The plugin package remains `vllm-bonsai2`, with `vllm.general_plugins` registration.
`bonsai2-serve` launches the selected M6.1 single-A100 profile, and `doctor
--runtime-only` checks the runtime without requiring conversion/oracle inputs.
The pinned Prism GGUF dependency is now encoded in wheel metadata as well as
the uv lock, so pip does not silently substitute an unrelated GGUF reader.

Portable conversion instructions are separate from serving. Native comparison
tools use configurable paths. The M5 lifecycle probe constructs its command
without depending on an ignored previous benchmark artifact. Historical stage
documents remain unchanged except for links; their local artifacts are not all
shipped. Selected evidence is copied into `reports/release/`.

The model release includes the unchanged eight safetensors shards, config,
tokenizer, generation config, conversion provenance, SHA256SUMS, model card and
upstream license/notices. The card explicitly identifies the plugin requirement,
text-only scope, 2,048-token validated limit and the unmet 200 tok/s suite target.
The optional DFlash2 weights remain in their upstream repository.

Validation and published revisions are recorded below after release checks.

## Checks completed before publication

- `uv sync --locked` and `uv lock --check` succeed.
- 119 tests pass, including CUDA kernels and the release profile/integrity checks.
- Ruff passes; wheel and source distribution build successfully.
- A separate venv synced from the exported lock imports the built wheel from
  site-packages and registers the Bonsai model/quantization plugin.
- All 19 staged model files and six separately downloaded draft files pass SHA-256
  validation. The eight target shard hashes match the original conversion.
- The pinned source Hub file's LFS hash matches the conversion's source GGUF hash.
- The portable M5 base command matches the historical profile aside from fields
  that the probe explicitly overrides.

No model inference kernel or weight tensor changed during release preparation.

## Hugging Face publication

Public model commit: `585817989752ac49540871ccc1696b35234586b8`.
Anonymous Hub metadata access succeeds; all 19 manifest files are present and
all eight remote LFS shard hashes match the staged release. The initial weights
commit was `400abc838524b3a9139ce08ee7e77b5717c75f81`; the follow-up updates only
the first-start CUDA-toolkit prerequisite and its checksum manifest.

The source pin and every download hash are stored in `configs/release.json`.
