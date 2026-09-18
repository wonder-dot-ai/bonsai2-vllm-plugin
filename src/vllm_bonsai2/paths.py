"""Paths for the source checkpoint and reference runtime."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_ROOT = PROJECT_ROOT.parent / "Bonsai-demo"
DEFAULT_MODEL_NAME = "Ternary-Bonsai-2-27B-PQ2_0.gguf"


def model_directory() -> Path:
    """Return the directory containing the source GGUF checkpoint."""

    configured = os.environ.get("BONSAI_GGUF_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_DEMO_ROOT / "models" / "bonsai2-gguf" / "27B"


def model_path() -> Path:
    """Return the source PQ2_0 checkpoint path."""

    name = os.environ.get("BONSAI_GGUF_MODEL", DEFAULT_MODEL_NAME)
    return model_directory() / name


def llama_server_path() -> Path:
    """Return the Prism llama-server path used as the reference implementation."""

    configured = os.environ.get("BONSAI_LLAMA_SERVER")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_DEMO_ROOT / "bin" / "cuda" / "llama-server"
