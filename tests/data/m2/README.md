# Frozen Prism operator vectors

These NPZ files are synthetic data, not extracted model weights. They were
produced once by `tools/freeze_reference_goldens.py` using the pinned Prism
CPU/CUDA graph harness. `manifest.json` records hashes and rotation geometry.

Normal tests only read these files. To investigate a runtime or schema change,
generate candidates into a new directory and compare them; do not automatically
refresh the test expectations. See `docs/M2_VALIDATION.md` for scope, numerical
gates, and the separate full real-checkpoint projection run.
