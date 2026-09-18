# Release evidence

These small reports are checked into Git so they remain available in a clean
clone. Detailed historical profiles and intermediate experiment outputs under
`artifacts/` are excluded from Git.

- `m3-checkpoint-verification.json`: independent reverse conversion, all 851
  source tensors accounted for, packed bytes and dense inverse transforms exact.
- `m46-acceptance.json`: numerical gate and short/long output limitations.
- `m6-acceptance.json`: original fixed English performance contract and results.
- `m61-acceptance.json`, `m61-comparison*.json`: 256-token prefill budget,
  independent server restart, exact timed outputs and unmet 200 tok/s target.
- `m6-english-baseline7/` and `m61-prefill256-repeat/`: raw timing chunks, token
  IDs, outputs and frozen short-continuation gates supporting the comparison.

Reports retain their original `artifacts/` names as provenance. Corresponding
selected files are mirrored here; this is not a claim that all historical
artifacts are distributed. Historical source hashes predate the packaging-only
release changes. The inference kernels and converted weight shards are unchanged.

Validation hardware: NVIDIA A100-SXM4-40GB; driver 580.126.20, CUDA toolkit
13.0.88 and GCC 11.4.0. The driver version
must be compatible with the locked CUDA 13 PyTorch build on another machine.
No second machine has been used for verification; clean-path/packaging tests
on the original A100 are described in `docs/RELEASE.md`.
