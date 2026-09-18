"""Lossless PQ2->INT4 runtime repacking for multi-token Marlin verification.

Storage remains PQ2; this optional runtime cache consumes additional GPU memory.
FWHT uses FP32 arithmetic then FP16 activations for the tensor-core operation.
"""

import torch
import triton
import triton.language as tl
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.scalar_type import scalar_types

from .kernels import _transform


@triton.jit
def _pack_gptq(W, OUT, ROWS, N: tl.constexpr, K: tl.constexpr, REORDER: tl.constexpr):
    index = tl.program_id(0) * 256 + tl.arange(0, 256)
    n, k8 = index // (K // 8), index % (K // 8)
    src = tl.load(ROWS + n, n < N, other=0) if REORDER else n
    W16 = W.to(tl.pointer_type(tl.uint16))
    code = tl.load(W16 + src * (K // 8) + k8, n < N, other=0).to(tl.uint32)
    packed = tl.full((256,), 0, tl.uint32)
    for i in tl.static_range(8):
        # PQ2 code c represents c-1; uint4b8 code c+7 represents the same value.
        packed |= (((code >> (2 * i)) & 3) + 7) << (4 * i)
    tl.store(OUT + k8 * N + n, packed.to(tl.int32), n < N)


def prepare_marlin(codes, scales, rows=None):
    n, k = codes.shape[0], codes.shape[1] * 128
    if n % 64 or k % 128:
        raise ValueError("Marlin requires output rows divisible by 64 and group size 128")
    packed = torch.empty((k // 8, n), device=codes.device, dtype=torch.int32)
    _pack_gptq[(triton.cdiv(n * (k // 8), 256),)](codes, packed, rows, n, k, rows is not None)
    empty = torch.empty(0, device=codes.device, dtype=torch.int32)
    weight = ops.gptq_marlin_repack(packed, empty, k, n, 4)
    effective_scales = scales if rows is None else scales.index_select(0, rows)
    scale = marlin_permute_scales(effective_scales.T.contiguous(), k, n, 128)
    return weight, scale, marlin_make_workspace_new(codes.device)


@torch.library.custom_op("bonsai2::marlin_project", mutates_args=("workspace",))
def marlin_project(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    signs: torch.Tensor,
    workspace: torch.Tensor,
    block: int,
    n: int,
) -> torch.Tensor:
    k = x.shape[-1]
    x = x.reshape(-1, k).contiguous()
    m = x.shape[0]
    rotated = torch.empty((m, k), device=x.device, dtype=torch.float16)
    if not m:
        return x.new_empty((0, n))
    _transform[(m, k // block)](x, signs, rotated, k, block, enable_fp_fusion=False)
    output = ops.marlin_gemm(
        rotated,
        None,
        weight,
        None,
        scale,
        None,
        None,
        None,
        None,
        None,
        workspace,
        scalar_types.uint4b8,
        m,
        n,
        k,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
    )
    return output.to(x.dtype)


@marlin_project.register_fake
def _marlin_fake(x, weight, scale, signs, workspace, block, n):
    return x.new_empty((x.numel() // x.shape[-1], n))
