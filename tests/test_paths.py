from pathlib import Path

from vllm_bonsai2.paths import DEFAULT_MODEL_NAME, model_directory, model_path


def test_model_path_uses_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BONSAI_GGUF_DIR", str(tmp_path))
    monkeypatch.setenv("BONSAI_GGUF_MODEL", "custom.gguf")

    assert model_directory() == tmp_path.resolve()
    assert model_path() == tmp_path.resolve() / "custom.gguf"


def test_default_model_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BONSAI_GGUF_DIR", str(tmp_path))
    monkeypatch.delenv("BONSAI_GGUF_MODEL", raising=False)

    assert model_path().name == DEFAULT_MODEL_NAME
