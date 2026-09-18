# M2 validation — 2026-09-18

M2 is complete for the **forward single-projection reference operator**. This
establishes the mathematical operation and its numerical relationship to Prism's
native CPU/CUDA kernels. It does not establish end-to-end model or vLLM parity.

## Implementation

`src/vllm_bonsai2/reference.py` provides:

- `unpack_pq2`: bit-exact PQ2 code/FP16-scale interpretation into FP32 weights.
- `rotation_from_manifest`: validates the forward rotation, explicit signs, and
  grouped GDN geometry. Rejects the inverse embedding-lookup path.
- `hadamard`: normalized block Sylvester FWHT, scaling before butterflies.
- `transform_activation`: optional GDN feature permutation, signs, then FWHT.
- `linear_reference`: transformed activation times decoded weight transpose.

FP16/BF16 activations are promoted to FP32. Outputs are FP32. Leading dimensions
and non-contiguous activations are supported. The validation CLI disables
PyTorch TF32; other callers must set their desired matmul precision explicitly.
This is an eager correctness oracle, not an efficient packed-weight kernel: it
expands the selected projection into FP32 and is not suitable for full-model
serving on a 40GB device.

## Independent runtime oracle

`tools/prism_projection.cpp` builds a small instrumented graph against the
original Prism release libraries. It includes the original
`llama_mul_mat_hadamard` helper from the pinned source and uses the same graph
operations/order as `build_lora_mm`: feature permutation (when required), sign
multiply, hinted Hadamard, and `ggml_mul_mat` on PQ2_0 weights. It retrieves the
transformed activation and final projection output on both CPU and CUDA.

A second matmul uses weights decoded by the original Prism C decoder into F32.
This separates layout/rotation errors from native packed-kernel arithmetic. All
numerical operations in this harness execute in Prism libraries, not in a C++
copy of the Python reference. The harness isolates a layer graph; it does not
load the complete llama-server model or collect real prompt activations.

Reference revision: `d8f26eec76da6d09bb708bcba51ef64b8cd868a3`.
The build script requires a clean checkout at that revision, records the source,
shared-library and harness hashes, and the wrapper checks the runtime commit.

Source anchors:

- [Forward linear graph](https://github.com/PrismML-Eng/llama.cpp/blob/d8f26eec76da6d09bb708bcba51ef64b8cd868a3/src/llama-graph.cpp)
- [Hadamard helper](https://github.com/PrismML-Eng/llama.cpp/blob/d8f26eec76da6d09bb708bcba51ef64b8cd868a3/src/llama-impl.h)
- [CUDA dispatch](https://github.com/PrismML-Eng/llama.cpp/blob/d8f26eec76da6d09bb708bcba51ef64b8cd868a3/ggml/src/ggml-cuda/ggml-cuda.cu)

## Real checkpoint results

Model SHA-256:
`3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`.

Projection: `blk.0.ffn_gate.weight`, logical shape `[17408, 5120]`, Hadamard
block 1024, source explicit signs. M1 bytes and metadata were rechecked before
running M2. Hardware: NVIDIA A100-SXM4-40GB. PyTorch: `2.13.0+cu130`.

Nine deterministic input cases (seed 20260918): Gaussian batches 1/4/32, zero,
constant, alternating signs, four impulses at distinct feature positions,
and Gaussian inputs scaled by 0.001 and 1000. All nine passed every gate.
These are synthetic activations with actual model weights, not a quality corpus.

Worst **per-input-row** relative L2 error across all cases:

| Comparison to PyTorch CPU FP32 | Observed | Limit |
| --- | ---: | ---: |
| PyTorch CUDA projection | 5.69e-7 | 1e-5 |
| Prism CPU signed Hadamard | 0 (exact) | 2e-6 |
| Prism CUDA signed Hadamard | 0 (exact) | 2e-6 |
| Prism CPU dense projection | 5.52e-7 | 1e-5 |
| Prism CUDA native dense projection | 4.28e-4 | 1e-3 |
| Prism CPU packed projection | 5.56e-3 | 1.5e-2 |
| Prism CUDA packed projection | 5.56e-3 | 1.5e-2 |

The packed runtime quantizes activations for its dot products; the FP32 reference
does not. Native CUDA F32 tensor paths can also use reduced-mantissa arithmetic
(the pinned implementation includes TF32 MMA). F32 storage/precision requests
therefore do not imply bitwise full-FP32 arithmetic for every dispatch path.
The CPU dense control is the strict numerical gate; native CUDA dense and packed
outputs have separate fixed tolerances. The observed packed difference is at
most about **0.56% relative L2**, not zero and not a model-quality measurement.

`validation.LIMITS` freezes two gates for each row: relative L2 and max absolute
error divided by that row's reference RMS. Limits respectively are:

- Transform: `2e-6`, `1e-5`.
- PyTorch CUDA and Prism CPU dense: `1e-5`, `1e-4`.
- Native Prism CUDA dense: `1e-3`, `1e-2`.
- Native packed CPU/CUDA: `0.015`, `0.08`.

The RMS denominator has a `1e-12` floor for zero inputs. Non-finite outputs fail.
No batch-wide averaging can hide a broken sequence. These gates were calibrated
from a separate initial Gaussian smoke comparison, then applied to the fixed
nine-case suite. They are regression thresholds, not a theoretical guarantee for
all inputs or a tolerance for final model logits.

Full report and saved inputs/intermediates/outputs:
`artifacts/m2-gate/report.json` and the nine adjacent `.npz` files. Each vector
file is hashed in the report. This run directory is never silently overwritten.

## Frozen offline regressions

`tests/data/m2/` contains three immutable **synthetic-weight** fixtures generated
by the Prism harness, with hashes and rotation geometry in `manifest.json`:

- Gate input width 5120.
- Down projection input width 17408.
- Grouped `ssm_out` input width 6144 (`nk=16`, `rep=3`).

Each fixture contains packed weights, FP16 scales, explicit signs, inputs, and
Prism CPU/CUDA transformed and projected outputs. No original model weights are
redistributed. Tests run the Python implementation against the saved external
outputs on CPU and, when available, CUDA. These tests do not need the model,
Prism checkout, compiler, or shared libraries. Golden generation is an explicit
command and refuses to overwrite an existing directory.

Validation: **46 tests passed** on A100; Ruff passed. Coverage includes dense
Hadamard parity, inverse/norm properties, non-contiguous half input, codec parity,
width-specific sign lookup, GDN feature order, malformed metadata, missing-sign
negative controls, and per-row/non-finite error detection.

## Reproduction

From the project root, using the existing M1 export:

```bash
uv sync --locked
# Only if the ignored reference source checkout is missing:
git clone --depth 1 --branch prism-b10683-d8f26ee \
  https://github.com/PrismML-Eng/llama.cpp.git artifacts/prism-llama.cpp
uv run python tools/build_prism_oracle.py
uv run bonsai2-vllm validate-operator artifacts/blk.0.ffn_gate.safetensors \
  --output artifacts/m2-gate-new-run
uv run pytest
uv run ruff check .
```

The live integration command requires CUDA, the source GGUF and Prism release
libraries. `--source`, `--prism-library`, and `--output` override its paths. The
harness builder also accepts `--source` (Prism checkout) and `--lib-dir`.
To explicitly generate new synthetic goldens for inspection, use a new path:

```bash
uv run python tools/freeze_reference_goldens.py --output artifacts/new-goldens
```

## Next: M3

Full text-model conversion and vLLM quantization/loader integration. This still
requires the inverse embedding operator, tensor naming/layout mapping, other
non-PQ2 tensors, model-logit and greedy-output parity, and later trajectory
regressions. Only the real FFN gate weight was tested end to end in M2; the down
and GDN paths currently have synthetic graph-level coverage. No generation
speedup or complete-model quality equivalence is claimed by this milestone.
