# M3 frozen text regression corpus

`corpus.json` contains nine short raw/chat inputs as explicit token IDs and the
greedy outputs observed with pinned Prism and the original Bonsai PQ2 checkpoint.
It freezes the initial M3 baseline; it is not a held-out quality benchmark or a
τ² trajectory dataset. The 0.02 TV threshold was calibrated after the initial
comparison. The generation limit intentionally truncates some answers.

Run `uv run python tools/m3_parity.py` with the converted checkpoint and local
Prism oracle to compare both engines against this baseline. See
`docs/M3C_MODEL_PARITY.md` for setup, metrics and limitations. Full distributions
and reports are generated under ignored `artifacts/`, not embedded here.
