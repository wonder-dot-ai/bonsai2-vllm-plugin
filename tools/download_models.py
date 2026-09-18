"""Download the release's pinned model assets with hf, then verify SHA-256."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_files(directory, files):
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid manifest path: {name}")
        path = directory / relative
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")
    print(f"Verified {len(files)} files in {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("converted/bonsai2-pq2"))
    parser.add_argument("--draft-dir", type=Path, default=Path("artifacts/m46-dflash2"))
    parser.add_argument("--no-draft", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/release.json").read_text())
    items = [(manifest["target"], args.model_dir)]
    if not args.no_draft:
        items.append((manifest["drafter"], args.draft_dir))
    for asset, directory in items:
        if not args.verify_only:
            if not asset["revision"] or len(asset["revision"]) != 40:
                raise ValueError("Release must pin a full Hub commit before downloading")
            subprocess.run(
                [
                    str(Path(sys.executable).with_name("hf")),
                    "download",
                    asset["repo_id"],
                    *asset["files"],
                    "--revision",
                    asset["revision"],
                    "--local-dir",
                    str(directory),
                ],
                check=True,
            )
        verify_files(directory, asset["files"])


if __name__ == "__main__":
    main()
