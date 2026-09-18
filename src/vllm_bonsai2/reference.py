"""Readable FP32 PyTorch oracle; intentionally not a production inference kernel."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Rotation:
    block_size: int
    signs: tuple[int, ...]
    perm_nk: int = 1
    perm_rep: int = 1


def rotation_from_manifest(manifest: dict) -> Rotation:
    """Validate and resolve the forward linear transform for this projection.

    Inverse embedding lookup is a separate operator and is deliberately rejected.
    """
    context = manifest["context"]
    prefix = "prism.hadamard."
    name = manifest["tensor_name"]
    width = manifest["logical_shape"][1]
    if name not in context.get(prefix + "weight_names", []):
        raise ValueError("tensor is not on the forward Hadamard linear path")
    if context.get(prefix + "version") != 1:
        raise ValueError("unsupported Hadamard version")
    if context.get(prefix + "transform") != "normalized-sylvester-walsh-hadamard":
        raise ValueError("unsupported Hadamard transform")
    if context.get(prefix + "axis") != "input-last-dimension":
        raise ValueError("unsupported Hadamard axis")
    block = context[prefix + "block_size"]
    if block <= 0 or block & (block - 1) or width % block:
        raise ValueError("invalid Hadamard block size")
    mode = context[prefix + "sign_mode"]
    if mode == "identity":
        signs = (1,) * width
    elif mode == "explicit":
        widths = context[prefix + "sign_widths"]
        values = context[prefix + "sign_values"]
        if (
            not widths
            or len(set(widths)) != len(widths)
            or any(w <= 0 or w % block for w in widths)
            or sum(widths) != len(values)
            or any(v not in (-1, 1) for v in values)
        ):
            raise ValueError("invalid explicit sign table")
        if width not in widths:
            raise ValueError("missing sign vector for input width")
        offset = sum(widths[: widths.index(width)])
        signs = tuple(values[offset : offset + width])
    else:
        raise ValueError("unsupported sign mode")
    nk, rep = 1, 1
    if context.get(prefix + "gdn_v_grouped", False) and ".ssm_out." in name:
        nk = context["qwen35.ssm.group_count"]
        nv = context["qwen35.ssm.time_step_rank"]
        if nk <= 0 or nv <= 0 or nv % nk or width % nv:
            raise ValueError("invalid grouped GDN geometry")
        rep = nv // nk
    return Rotation(block, signs, nk, rep)


def unpack_pq2(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode [out, in/128, 32] codes and FP16 scales to FP32 [out, in]."""
    if codes.dtype != torch.uint8 or codes.ndim != 3 or codes.shape[-1] != 32:
        raise ValueError("qweight must be uint8 [out, in/128, 32]")
    if scales.dtype != torch.float16 or tuple(scales.shape) != tuple(codes.shape[:2]):
        raise ValueError("scales must be float16 [out, in/128]")
    if codes.device != scales.device:
        raise ValueError("codes and scales must be on the same device")
    shifts = torch.arange(0, 8, 2, dtype=torch.uint8, device=codes.device)
    values = ((codes.unsqueeze(-1) >> shifts) & 3).to(torch.float32) - 1
    return (values.reshape(*scales.shape, 128) * scales.float().unsqueeze(-1)).flatten(1)


def hadamard(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Normalized Sylvester FWHT, scaled before butterflies as in Prism."""
    if (
        x.ndim == 0
        or block_size <= 0
        or block_size & (block_size - 1)
        or x.shape[-1] == 0
        or x.shape[-1] % block_size
    ):
        raise ValueError("last dimension must be a positive multiple of a power-of-two block")
    if not x.is_floating_point():
        raise ValueError("Hadamard input must be floating point")
    original_shape = x.shape
    y = x.float().reshape(-1, block_size) * (block_size**-0.5)
    stride = 1
    while stride < block_size:
        pairs = y.reshape(-1, block_size // (2 * stride), 2, stride)
        left, right = pairs[:, :, 0], pairs[:, :, 1]
        y = torch.stack((left + right, left - right), dim=2).reshape(-1, block_size)
        stride *= 2
    return y.reshape(original_shape)


def transform_activation(x: torch.Tensor, rotation: Rotation) -> torch.Tensor:
    width = len(rotation.signs)
    if x.ndim == 0 or x.shape[-1] != width:
        raise ValueError("activation width does not match rotation")
    if not x.is_floating_point():
        raise ValueError("activation must be floating point")
    nk, rep = rotation.perm_nk, rotation.perm_rep
    if nk <= 0 or rep <= 0 or width % (nk * rep):
        raise ValueError("invalid GDN permutation")
    if any(sign not in (-1, 1) for sign in rotation.signs):
        raise ValueError("signs must be +/-1")
    y = x.float()
    if rep > 1:
        # GGML dims [hd,nk,rep,batch] -> [hd,rep,nk,batch], with dim0 contiguous.
        y = y.reshape(*x.shape[:-1], rep, nk, width // (nk * rep))
        y = y.transpose(-3, -2).reshape(x.shape)
    signs = torch.tensor(rotation.signs, device=x.device, dtype=torch.float32)
    return hadamard(y * signs, rotation.block_size)


def linear_reference(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    rotation: Rotation,
) -> torch.Tensor:
    """FP32 signed-Hadamard projection. Caller controls TF32/global precision."""
    if x.device != codes.device:
        raise ValueError("activations and weights must be on the same device")
    weights = unpack_pq2(codes, scales)
    if x.shape[-1] != weights.shape[1]:
        raise ValueError("activation and weight widths differ")
    return transform_activation(x, rotation) @ weights.T
