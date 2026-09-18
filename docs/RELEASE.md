# Public release — portable installation and safetensors distribution

Status: published and validated, 2026-09-18.

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

## Public code and clean-environment verification

Initial GitHub code commit: `0ca6956` on `main`. The repository and model are
public. Git commit identity is `shamuiscoding <toebee@snu.ac.kr>`.

- Cloned the public GitHub URL into a new `/tmp` directory and installed with
  `uv sync --locked` in its own venv. No development checkout was used as a
  package dependency.
- Runtime doctor succeeds with deliberately nonexistent GGUF/Prism paths.
- Re-downloaded target and drafter from public Hub repositories without implicit
  authentication; verified all 19 target and six draft files against the
  checked-in manifest.
- The clean clone's launcher accepts those public-download directories.
- A separately installed wheel served the staged, hash-identical checkpoint on
  the A100. Chat answered `42`; all nine frozen continuations plus the repeated
  case passed. This was an actual HTTP server, not only an import check.
- First startup in the separate venv compiled FlashInfer CUDA kernels and took
  several minutes. CUDA toolkit 13.0.88 (`nvcc`) and GCC 11.4.0 were present.
  This prerequisite is documented in both the README and model card.
- The owned validation server was stopped after the checks.

These are clean-path/environment checks on the original A100, not a claim of
validation on a second physical machine. The package and HTTP evidence are in
`reports/release/package-validation.json` and `reports/release/wheel-serving/`.

Use the `v0.1.0` Git tag for this code release and the Hub revision pinned in
`configs/release.json` for its model assets. Installation instructions are in
the root README.

## Model card simplification

The public model card was shortened to essential setup, plugin requirements,
validated scope and attribution. Technical details remain in the plugin docs.
Only `README.md` and `SHA256SUMS` changed on the Hub; weight hashes are unchanged.
The current main-branch download manifest pins `d65271ba89fe17583f2cba0e1c2fa21328e57f42` instead of
`585817989752ac49540871ccc1696b35234586b8`. The existing `v0.1.0` tag keeps its original immutable pin.
Anonymous re-download verified both changed files and remote shard hashes.
