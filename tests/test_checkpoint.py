from pathlib import Path

from vllm_bonsai2.checkpoint import GGUF_MAGIC, inspect_source


def test_inspect_source_reads_magic_and_size(tmp_path: Path) -> None:
    model = tmp_path / "tiny.gguf"
    model.write_bytes(GGUF_MAGIC + b"payload")

    source = inspect_source(model)

    assert source.path == model.resolve()
    assert source.size_bytes == 11
    assert source.is_gguf


def test_non_gguf_is_rejected_by_magic_check(tmp_path: Path) -> None:
    model = tmp_path / "wrong.bin"
    model.write_bytes(b"NOPE")

    assert not inspect_source(model).is_gguf
