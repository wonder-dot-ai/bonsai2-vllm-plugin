"""Release integrity and serving-profile contract checks."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from vllm_bonsai2.serving import build_command


def test_serving_profile_preserves_validated_limits():
    command = build_command(Path("model"), Path("draft"))
    assert command[command.index("--max-num-seqs") + 1] == "1"
    assert command[command.index("--max-model-len") + 1] == "2048"
    assert command[command.index("--max-num-batched-tokens") + 1] == "256"
    assert "--no-enable-prefix-caching" in command
    spec = json.loads(command[command.index("--speculative-config") + 1])
    graph = json.loads(command[command.index("--compilation-config") + 1])
    assert graph["cudagraph_capture_sizes"] == [spec["num_speculative_tokens"] + 1]
    no_draft = build_command(Path("model"), None)
    assert "--speculative-config" not in no_draft
    assert json.loads(no_draft[no_draft.index("--compilation-config") + 1])[
        "cudagraph_capture_sizes"
    ] == [1]


def test_download_verifier_rejects_corruption_and_path_escape(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "download_models", Path(__file__).parents[1] / "tools/download_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    file = tmp_path / "weights"
    file.write_bytes(b"valid")
    hashes = {"weights": hashlib.sha256(b"valid").hexdigest()}
    module.verify_files(tmp_path, hashes)
    file.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.verify_files(tmp_path, hashes)
    with pytest.raises(ValueError, match="Invalid manifest path"):
        module.verify_files(tmp_path, {"../weights": hashes["weights"]})
