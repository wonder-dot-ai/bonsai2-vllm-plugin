"""Developer commands for the Bonsai 2 vLLM port."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from vllm_bonsai2.checkpoint import inspect_source
from vllm_bonsai2.paths import llama_server_path, model_path


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def doctor(runtime_only: bool = False) -> int:
    """Report whether the development environment has its required inputs."""

    failures: list[str] = []
    source = model_path()
    reference_server = llama_server_path()

    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"vLLM: {_package_version('vllm')}")
    print(f"safetensors: {_package_version('safetensors')}")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            print(f"GPU: {device.name}")
            print(f"Compute capability: {device.major}.{device.minor}")
            print(f"VRAM: {device.total_memory / 2**30:.1f} GiB")
            if (device.major, device.minor) != (8, 0):
                failures.append("initial CUDA target is NVIDIA sm_80")
        else:
            failures.append("CUDA is not available to PyTorch")
    except ImportError:
        failures.append("PyTorch is not installed")

    if runtime_only:
        print("Runtime-only check: source GGUF and Prism oracle are optional")
    elif source.is_file():
        checkpoint = inspect_source(source)
        print(f"Source model: {checkpoint.path}")
        print(f"Source size: {checkpoint.size_bytes / 10**9:.3f} GB")
        print(f"GGUF magic: {checkpoint.magic!r}")
        if not checkpoint.is_gguf:
            failures.append("source model does not start with GGUF magic")
    else:
        print(f"Source model: missing ({source})")
        failures.append("source PQ2_0 checkpoint is missing")

    print(f"Reference server: {reference_server}")
    if not runtime_only and not reference_server.is_file():
        failures.append("reference llama-server is missing")

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        "\nOK: runtime environment is ready"
        if runtime_only
        else "\nOK: development environment is ready"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="bonsai2-vllm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor", help="validate the local environment")
    doctor_parser.add_argument(
        "--runtime-only", action="store_true", help="do not require source GGUF or Prism oracle"
    )
    inspect_parser = subparsers.add_parser("inventory", help="dump GGUF metadata and tensors")
    inspect_parser.add_argument("--source", type=Path, default=model_path())
    inspect_parser.add_argument("--output", type=Path, default=Path("artifacts/inventory.json"))
    export_parser = subparsers.add_parser("export-projection", help="export one PQ2_0 tensor")
    export_parser.add_argument("--source", type=Path, default=model_path())
    export_parser.add_argument("--tensor", default="blk.0.ffn_gate.weight")
    export_parser.add_argument(
        "--output", type=Path, default=Path("artifacts/blk.0.ffn_gate.safetensors")
    )
    verify_parser = subparsers.add_parser(
        "verify-projection", help="verify bytes and Prism C values"
    )
    verify_parser.add_argument("projection", type=Path)
    verify_parser.add_argument("--source", type=Path, default=model_path())
    verify_parser.add_argument("--prism-library", type=Path)
    verify_parser.add_argument("--output", type=Path)
    operator_parser = subparsers.add_parser(
        "validate-operator", help="compare the PyTorch reference to Prism CPU and CUDA"
    )
    operator_parser.add_argument("projection", type=Path)
    operator_parser.add_argument("--source", type=Path, default=model_path())
    operator_parser.add_argument(
        "--prism-library", type=Path, default=Path("artifacts/libbonsai-prism-projection.so")
    )
    operator_parser.add_argument("--output", type=Path, default=Path("artifacts/m2-gate"))
    full_parser = subparsers.add_parser(
        "convert-model", help="convert the complete text checkpoint"
    )
    full_parser.add_argument("--source", type=Path, default=model_path())
    full_parser.add_argument("--hf-assets", type=Path, default=Path("artifacts/qwen-config"))
    full_parser.add_argument("--output", type=Path, default=Path("converted/bonsai2-pq2"))
    args = parser.parse_args()

    if args.command == "doctor":
        return doctor(args.runtime_only)
    from vllm_bonsai2.convert import export_projection, inventory, verify_projection

    try:
        if args.command == "inventory":
            result = inventory(args.source, args.output)
        elif args.command == "export-projection":
            result = export_projection(args.source, args.tensor, args.output)
        elif args.command == "verify-projection":
            if args.output and args.output.resolve() in {
                args.source.resolve(),
                args.projection.resolve(),
            }:
                raise ValueError("verification report must not overwrite an input file")
            result = verify_projection(args.projection, args.source, args.prism_library)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        elif args.command == "validate-operator":
            from vllm_bonsai2.validation import validate_operator

            result = validate_operator(
                args.projection, args.source, args.prism_library, args.output
            )
        elif args.command == "convert-model":
            from vllm_bonsai2.full_checkpoint import convert_model

            result = convert_model(args.source, args.hf_assets, args.output)
        else:
            parser.error(f"unknown command: {args.command}")
    except (ValueError, OSError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
