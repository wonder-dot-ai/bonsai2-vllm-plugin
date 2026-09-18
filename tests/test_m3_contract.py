from pathlib import Path

import pytest

from vllm_bonsai2.full_checkpoint import convert_model


def test_conversion_refuses_unpinned_assets_before_opening_model(tmp_path):
    assets = tmp_path / "hf"
    assets.mkdir()
    (assets / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="pinned revision"):
        convert_model(tmp_path / "missing.gguf", assets, tmp_path / "converted")


def test_conversion_never_overwrites_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "important"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="new directory"):
        convert_model(Path("missing.gguf"), Path("missing-assets"), output)
    assert sentinel.read_text() == "keep"
