"""Small checkpoint checks that do not depend on a GGUF parser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GGUF_MAGIC = b"GGUF"


@dataclass(frozen=True)
class SourceCheckpoint:
    path: Path
    size_bytes: int
    magic: bytes

    @property
    def is_gguf(self) -> bool:
        return self.magic == GGUF_MAGIC


def inspect_source(path: Path) -> SourceCheckpoint:
    """Read the minimum stable facts needed for an environment smoke test."""

    resolved = path.expanduser().resolve()
    with resolved.open("rb") as model_file:
        magic = model_file.read(4)
    return SourceCheckpoint(path=resolved, size_bytes=resolved.stat().st_size, magic=magic)
