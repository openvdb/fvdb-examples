# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for using fVDB in Point Transformer V3
"""

from typing import Any, cast

import fvdb
import torch

# Add NVTX import for profiling
try:
    import torch.cuda.nvtx as nvtx

    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False

    class DummyNVTX:
        def range_push(self, msg):
            pass

        def range_pop(self):
            pass

    nvtx = DummyNVTX()


class NVTXRange:
    """
    Context manager for NVTX range push/pop.
    Enables usage:
        with NVTXRange("msg"):
            ...
    to automatically push and pop NVTX profiling ranges.
    If NVTX is unavailable, this is a no-op.
    """

    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        nvtx.range_push(self.msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        nvtx.range_pop()


def jagged_cumulative_argsort(unsorted_jt: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
    sorted_indices = torch.empty(unsorted_jt.rshape[0], dtype=torch.long, device=unsorted_jt.device)
    offset = 0
    for t_i in unsorted_jt:
        num_elements = t_i.rshape[0]
        if num_elements == 0:
            continue
        indices = torch.argsort(t_i.jdata)
        sorted_indices[offset : offset + num_elements] = indices + offset
        offset += num_elements
    return unsorted_jt.jagged_like(sorted_indices)


def morton_from_jagged_ijk(jagged_ijk: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
    ijk_j: torch.Tensor = jagged_ijk.jdata
    kji_j = ijk_j[:, [2, 1, 0]].contiguous()
    morton_j = fvdb.morton(kji_j)
    return jagged_ijk.jagged_like(morton_j)


def morton_flipped_from_jagged_ijk(jagged_ijk: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
    ijk_j: torch.Tensor = jagged_ijk.jdata
    kij_j = ijk_j[:, [2, 0, 1]].contiguous()
    morton_j = fvdb.morton(kij_j)
    return jagged_ijk.jagged_like(morton_j)


def hilbert_from_jagged_ijk(jagged_ijk: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
    ijk_j: torch.Tensor = jagged_ijk.jdata
    jki_j = ijk_j[:, [1, 2, 0]].contiguous()
    hilbert_j = fvdb.hilbert(jki_j)
    return jagged_ijk.jagged_like(hilbert_j)


def hilbert_flipped_from_jagged_ijk(jagged_ijk: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
    ijk_j: torch.Tensor = jagged_ijk.jdata
    ikj_j = ijk_j[:, [0, 2, 1]].contiguous()
    hilbert_j = fvdb.hilbert(ikj_j)
    return jagged_ijk.jagged_like(hilbert_j)


def identity_from_jagged_ijk(jagged_ijk: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
    # We don't need to individually separate out the jaggedness.
    ijk_j = jagged_ijk.jdata
    count = ijk_j.shape[0]
    identity_j = torch.arange(count, device=ijk_j.device, dtype=torch.int64)
    return jagged_ijk.jagged_like(identity_j)


def space_filling_curve_from_jagged_ijk(jagged_ijk: fvdb.JaggedTensor, curve_type: str) -> fvdb.JaggedTensor:
    match curve_type:
        case "morton" | "z":
            return morton_from_jagged_ijk(jagged_ijk)
        case "morton_zyx" | "z-trans":
            return morton_flipped_from_jagged_ijk(jagged_ijk)
        case "hilbert":
            return hilbert_from_jagged_ijk(jagged_ijk)
        case "hilbert-trans":
            return hilbert_flipped_from_jagged_ijk(jagged_ijk)
        case "vdb" | "identity":
            return identity_from_jagged_ijk(jagged_ijk)
        case _:
            raise ValueError(f"Unsupported curve type: {curve_type}")


# Source: perm: torch.Tensor | None = None
# if order_type != "vdb":
#     perm_jagged = space_filling_curve_from_jagged_ijk(grid.ijk, order_type)
#     perm = perm_jagged.jdata.squeeze(-1)  # [num_voxels]
#     # Use torch.gather for permutation: expand perm to match feats_j dimensions
#     perm_expanded = perm.unsqueeze(-1).expand(-1, feats_j.shape[-1])  # [num_voxels, hidden_size]
#     feats_j = torch.gather(feats_j, 0, perm_expanded)
#     feats = feats.jagged_like(feats_j)


def order_features_from_jagged_ijk(
    jagged_feats: fvdb.JaggedTensor, jagged_ijk: fvdb.JaggedTensor, order_type: str
) -> tuple[fvdb.JaggedTensor, fvdb.JaggedTensor]:
    curve_values = space_filling_curve_from_jagged_ijk(jagged_ijk, order_type)
    # Argsort the curve values to get permutation indices (sort within each batch)
    permutation = jagged_cumulative_argsort(curve_values)
    # Expand permutation to match feature dimensions for torch.gather
    perm_expanded = permutation.jdata.unsqueeze(-1).expand(-1, jagged_feats.jdata.shape[-1])
    feats_reodered = jagged_feats.jagged_like(torch.gather(jagged_feats.jdata, 0, perm_expanded))
    return feats_reodered, permutation


# Restore jagged features to their original ordering using the provided permutation.
# orderd_jagged_feats: fvdb.JaggedTensor with permuted features
# permutation: fvdb.JaggedTensor with indices such that orderd_jagged_feats.jdata = feats_jdata[permutation.jdata]
# We need to invert this permutation so that output[permutation[j]] = orderd_jagged_feats[j]
def inverse_order_features_from_perm(
    orderd_jagged_feats: fvdb.JaggedTensor, permutation: fvdb.JaggedTensor
) -> fvdb.JaggedTensor:
    # permutation.jdata: shape [N], dtype torch.long
    perm = permutation.jdata
    inverse_perm = torch.empty_like(perm)
    inverse_perm[perm] = torch.arange(perm.shape[0], device=perm.device)
    # Now restore the original order using gather:
    feats_jdata = orderd_jagged_feats.jdata
    restored_feats = torch.gather(feats_jdata, 0, inverse_perm.unsqueeze(-1).expand_as(feats_jdata))
    return orderd_jagged_feats.jagged_like(restored_feats)


class FVDBGridModule(torch.nn.Module):
    """
    Base class for modules that operate on (JaggedTensor, GridBatch) -> JaggedTensor.

    Provides a typed __call__ override so static type checkers can verify
    argument types and order when calling the module.
    """

    def __call__(self, feats: fvdb.JaggedTensor, grid: fvdb.GridBatch) -> fvdb.JaggedTensor:
        """Override __call__ to preserve type hints from forward."""
        assert isinstance(feats, fvdb.JaggedTensor), "Input feats must be a JaggedTensor"
        assert isinstance(grid, fvdb.GridBatch), "Input grid must be a GridBatch"
        return super().__call__(feats, grid)

    def forward(self, feats: fvdb.JaggedTensor, grid: fvdb.GridBatch) -> fvdb.JaggedTensor:
        assert isinstance(feats, fvdb.JaggedTensor), "Input feats must be a JaggedTensor"
        assert isinstance(grid, fvdb.GridBatch), "Input grid must be a GridBatch"
        raise NotImplementedError("Subclasses must implement forward()")


class FVDBJaggedModule(torch.nn.Module):
    """
    Base class for modules that operate on JaggedTensor -> JaggedTensor.

    Provides a typed __call__ override so static type checkers can verify
    argument types and order when calling the module.
    """

    def __call__(self, feats: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        """Override __call__ to preserve type hints from forward."""
        assert isinstance(feats, fvdb.JaggedTensor), "Input jagged_tensor must be a JaggedTensor"
        return super().__call__(feats)

    def forward(self, feats: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        assert isinstance(feats, fvdb.JaggedTensor), "Input jagged_tensor must be a JaggedTensor"
        raise NotImplementedError("Subclasses must implement forward()")


class FVDBJaggedWrapper(FVDBJaggedModule):
    """
    Wrap a standard `torch.nn.Module` so it operates on `fvdb.JaggedTensor`s.

    The wrapped module is stored as `self.module` and is always called on
    the `.jdata` of the incoming `JaggedTensor`. The output is wrapped back
    into a `JaggedTensor` with the same jagged structure.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def __call__(self, jagged_tensor: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        """Override __call__ to preserve type hints from forward."""
        assert isinstance(jagged_tensor, fvdb.JaggedTensor), "Input jagged_tensor must be a JaggedTensor"
        return super().__call__(jagged_tensor)

    def forward(self, jagged_tensor: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        assert isinstance(jagged_tensor, fvdb.JaggedTensor), "Input jagged_tensor must be a JaggedTensor"
        new_jdata = self.module(jagged_tensor.jdata)
        return jagged_tensor.jagged_like(new_jdata)


# Alias to keep call sites concise (e.g., `self.fc1 = FJTM(nn.Linear(...))`)
FJTM = FVDBJaggedWrapper


def jagged_attention(
    feats: fvdb.JaggedTensor,
    qkv: fvdb.JaggedTensor,
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    patch_size: int,
    sliding_window_attention: bool,
    scale: float,
) -> fvdb.JaggedTensor:

    if (sliding_window_attention and patch_size > 0) or (patch_size > 0):
        try:
            import flash_attn
        except ImportError:
            raise RuntimeError(
                "flash_attn is required for jagged_attention with "
                "sliding_window_attention or patch_size > 0. "
                "Install with: pip install flash-attn"
            )

    qkv_j = qkv.jdata
    feats_j = feats.jdata

    if sliding_window_attention and patch_size > 0:
        # Perform sliding window attention per-grid using flash attention
        num_voxels = feats_j.shape[0]
        H = num_heads
        D = head_dim
        offsets = feats.joffsets.to(device=qkv.device, dtype=torch.int64)
        outputs = []
        for b in range(offsets.numel() - 1):
            start = int(offsets[b].item())
            end = int(offsets[b + 1].item())
            Li = end - start
            if Li <= 0:
                continue
            qkv_b = qkv_j[start:end].view(1, Li, 3, H, D)
            window_size = (patch_size // 2, patch_size // 2)
            out_b = cast(
                Any,
                flash_attn.flash_attn_qkvpacked_func(
                    qkv_b.half(),
                    dropout_p=0.0,
                    softmax_scale=scale,
                    window_size=window_size,
                ),
            ).reshape(
                Li, hidden_size
            )  # dtype: float16
            outputs.append(out_b)
        if len(outputs) == 0:
            feats_out_j = torch.empty_like(qkv_j[:, :hidden_size])
        else:
            feats_out_j = torch.cat(outputs, dim=0)

        feats_out_j = feats_out_j.to(feats_j.dtype)

    elif patch_size > 0:
        # Perform attention within each patch_size window per-grid using varlen API
        num_voxels = feats_j.shape[0]
        H = num_heads
        D = head_dim
        qkv_j = qkv_j.view(-1, 3, H, D)  # (num_voxels, 3, num_heads, head_dim)

        # Build cu_seqlens as concatenation of per-grid patches so we never cross grid boundaries
        offsets = feats.joffsets.to(device=qkv_j.device, dtype=torch.int64)
        lengths = []
        for b in range(offsets.numel() - 1):
            start = int(offsets[b].item())
            end = int(offsets[b + 1].item())
            Li = end - start
            if Li <= 0:
                continue
            full = Li // patch_size
            rem = Li % patch_size
            if full > 0:
                lengths.extend([patch_size] * full)
            if rem > 0:
                lengths.append(rem)
        if len(lengths) == 0:
            feats_out_j = torch.empty((0, hidden_size), device=qkv_j.device, dtype=feats_j.dtype)
        else:
            cu_seqlens = torch.zeros(len(lengths) + 1, device=qkv.device, dtype=torch.int32)
            cu_seqlens[1:] = torch.as_tensor(lengths, device=qkv.device, dtype=torch.int32).cumsum(dim=0)

            feats_out_j = cast(
                Any,
                flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv_j.to(dtype=torch.bfloat16),
                    cu_seqlens,
                    max_seqlen=patch_size,
                    dropout_p=0.0,  # TODO: implement attention dropout in the future. By default, it is 0.
                    softmax_scale=scale,
                ),
            ).reshape(
                num_voxels, hidden_size
            )  # dtype: bfloat16

            feats_out_j = feats_out_j.to(feats_j.dtype)
    else:
        feats_out_j = qkv_j[:, :hidden_size].contiguous()
    return feats.jagged_like(feats_out_j)
