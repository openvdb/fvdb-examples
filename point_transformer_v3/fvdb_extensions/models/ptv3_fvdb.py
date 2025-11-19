# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
PTV3 FVDB Implementation

This module contains the core Point Transformer V3 implementation using FVDB.
It works directly with FVDB GridBatch and JaggedTensor types.

For pointcept framework integration, see point_transformer_v3m1_fvdb.py
"""

from typing import Dict, Tuple, Union, List

# Add NVTX import for profiling
try:
    import flash_attn
except ImportError:
    flash_attn = None

import torch
import torch.nn
import torch.nn.functional as F
from timm.layers import DropPath
from functools import partial

import fvdb

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


class PTV3_Embedding(torch.nn.Module):
    """
    PTV3_Embedding for 3D point cloud embedding.
    """

    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer_module: torch.nn.Module = torch.nn.LayerNorm,
        embedding_mode: str = "linear",
        shared_plan_cache: Dict = None,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input features.
            embed_channels (int): Number of channels in the output features.
            norm_layer_module (torch.nn.Module): Normalization layer module.
            embedding_mode (str): The type of embedding layer, "linear" or "conv3x3", "conv5x5".
            shared_plan_cache (Dict): Shared cache for ConvolutionPlans across all layers.
        """
        super().__init__()
        self.embedding_mode = embedding_mode
        self.shared_plan_cache = shared_plan_cache

        if embedding_mode == "linear":
            self.embed = torch.nn.Linear(in_channels, embed_channels)
        elif embedding_mode == "conv3x3":
            # Initialize embedding using FVDB's sparse 3D convolution
            self.embed_conv3x3_1 = fvdb.nn.SparseConv3d(
                in_channels, embed_channels, kernel_size=3, stride=1, bias=False
            )
        elif embedding_mode == "conv5x5":
            ## Implementation Option 1: Cascaded 3x3 convolutions
            # This approach uses two 3x3 convs to achieve a 5x5 receptive field with fewer parameters
            # Parameters: (27 × in_channels × embed_channels) + (27 × embed_channels²)
            self.embed_conv3x3_1 = fvdb.nn.SparseConv3d(
                in_channels, embed_channels, kernel_size=3, stride=1, bias=False
            )
            self.embed_conv3x3_2 = fvdb.nn.SparseConv3d(
                embed_channels, embed_channels, kernel_size=3, stride=1, bias=False
            )

            ## Implementation Option 2: Direct 5x5 convolution
            # TODO: Implementation pending - requires additional sparse convolution support from fVDB-core.
            # Expected parameters: 125 × in_channels × embed_channels
            # self.embed_conv5x5_1 = fvdb.nn.SparseConv3d(in_channels, embed_channels, kernel_size=5, stride=1)
        else:
            raise ValueError(f"Unsupported embedding mode: {embedding_mode}")

        self.norm_layer = norm_layer_module(embed_channels)
        self.act_layer = torch.nn.GELU()

    def _get_plan(self, grid, kernel_size, stride):
        """Get or create a ConvolutionPlan from shared cache."""
        cache_key = (grid.address, kernel_size, stride)
        if cache_key not in self.shared_plan_cache:
            self.shared_plan_cache[cache_key] = fvdb.ConvolutionPlan.from_grid_batch(
                kernel_size=kernel_size, stride=stride, source_grid=grid, target_grid=grid
            )
        return self.shared_plan_cache[cache_key]

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_Embedding")

        if self.embedding_mode == "linear":
            jfeats = feats.jdata
            jfeats = self.embed(jfeats)
        elif self.embedding_mode == "conv3x3":
            # Apply 3x3 sparse convolution using shared ConvolutionPlan cache
            plan = self._get_plan(grid, kernel_size=3, stride=1)
            feats = self.embed_conv3x3_1(feats, plan)
            jfeats = feats.jdata
        elif self.embedding_mode == "conv5x5":
            # First 3x3 convolution
            plan1 = self._get_plan(grid, kernel_size=3, stride=1)
            feats = self.embed_conv3x3_1(feats, plan1)

            # Second 3x3 convolution (same grid since stride=1, in-place)
            plan2 = self._get_plan(grid, kernel_size=3, stride=1)
            feats = self.embed_conv3x3_2(feats, plan2)
            jfeats = feats.jdata

        jfeats = self.norm_layer(jfeats)
        jfeats = self.act_layer(jfeats)

        feats = grid.jagged_like(jfeats)
        nvtx.range_pop()
        return grid, feats


class PTV3_Pooling(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 64,
        out_channels: int = 64,
        norm_layer_module: torch.nn.Module = torch.nn.LayerNorm,
    ):
        """
        Args:
            kernel_size (int): Kernel size for the pooling operation.
            in_channels (int): Number of channels in the input features.
            out_channels (int): Number of channels in the output features.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.proj = torch.nn.Linear(in_channels, out_channels)
        self.norm_layer = norm_layer_module(out_channels)
        self.act_layer = torch.nn.GELU()

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_Pooling")
        feats_j = self.proj(feats.jdata)
        feats = grid.jagged_like(feats_j)

        ds_feature, ds_grid = grid.max_pool(self.kernel_size, feats, stride=self.kernel_size, coarse_grid=None)
        ds_feature_j = ds_feature.jdata
        ds_feature_j = self.norm_layer(ds_feature_j)
        ds_feature_j = self.act_layer(ds_feature_j)
        ds_feature = ds_grid.jagged_like(ds_feature_j)
        nvtx.range_pop()
        return ds_grid, ds_feature


class PTV3_Unpooling(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 64,
        out_channels: int = 64,
        skip_channels: int = 64,
        norm_layer_module: torch.nn.Module = torch.nn.LayerNorm,
    ):
        """
        Args:
            kernel_size (int): Kernel size for the pooling operation.
            in_channels (int): Number of channels in the input features.
            out_channels (int): Number of channels in the output features.
            skip_channels (int): Number of channels in the skip connection.
        """
        super().__init__()
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.proj = torch.nn.Linear(in_channels, out_channels)
        self.norm = norm_layer_module(out_channels)
        self.act_layer = torch.nn.GELU()
        self.proj_skip = torch.nn.Linear(skip_channels, out_channels)
        self.norm_skip = norm_layer_module(out_channels)
        self.act_layer_skip = torch.nn.GELU()

    def forward(self, grid, feats, last_grid, last_feats):

        feats_j = self.proj(
            feats.jdata
        )  # BUG: When enabled AMP, despite both feats.jdata and linear.weights are float32, the output becomes float16 which causes the subsequent convolution operation to fail.
        feats_j = self.norm(feats_j)
        feats_j = self.act_layer(feats_j)

        last_feats_j = self.proj_skip(last_feats.jdata)
        last_feats_j = self.norm_skip(last_feats_j)
        last_feats_j = self.act_layer_skip(last_feats_j)

        feats, _ = grid.refine(self.kernel_size, grid.jagged_like(feats_j), fine_grid=last_grid)
        feats_j = feats.jdata

        new_feats_j = last_feats_j + feats_j
        return last_grid, last_grid.jagged_like(new_feats_j)


class PTV3_MLP(torch.nn.Module):
    def __init__(self, hidden_size: int, proj_drop: float = 0.0):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            proj_drop (float): Dropout rate for MLP layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size * 4)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_size * 4, hidden_size)
        self.drop = torch.nn.Dropout(proj_drop)

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_MLP")
        feats_j = feats.jdata  # TODO: deprecate the .jdata usage.

        feats_j = self.fc1(feats_j)
        feats_j = self.act(feats_j)
        feats_j = self.drop(feats_j)
        feats_j = self.fc2(feats_j)
        feats_j = self.drop(feats_j)
        feats = grid.jagged_like(feats_j)
        nvtx.range_pop()
        return grid, feats


class PTV3_Attention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        proj_drop: float = 0.0,
        patch_size: int = 0,
        qk_scale: float = None,
        sliding_window_attention: bool = False,
        order_index: int = 0,
        order_types: tuple = ("vdb",),
    ):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            num_heads (int): Number of attention heads in each block.
            proj_drop (float): Dropout rate for MLP layers.
            patch_size (int): Patch size for patch attention.
            qk_scale (float): Scale factor for query-key dot product. If None, uses 1/sqrt(head_dim).
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
            order_index (int): Index into order_types to select which order to use for this block.
            order_types (tuple): Tuple of order type strings (e.g., ("z", "z-trans")).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.scale = qk_scale or (self.head_dim) ** -0.5
        self.qkv = torch.nn.Linear(hidden_size, hidden_size * 3)  # Combined QKV projection
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.drop = torch.nn.Dropout(proj_drop)
        self.patch_size = patch_size
        self.order_index = order_index
        self.order_types = order_types

        # TODO: Add attention dropout

        # Sliding window attention parameter
        self.sliding_window_attention = sliding_window_attention

    def _compute_permutation(self, grid, curve_codes):
        """
        Get permutation indices to sort voxels by space-filling curve order.

        Takes pre-computed space-filling curve codes (e.g., from morton(), hilbert(), etc.)
        and returns the permutation indices that would sort voxels according to those codes.
        This is useful for spatially coherent data access patterns and cache optimization.

        Args:
            grid: The grid batch containing voxel information.
            curve_codes (JaggedTensor): Space-filling curve codes for each voxel.
                Shape: `[num_grids, -1, 1]`. Typically obtained from morton(), morton_zyx(),
                hilbert(), or hilbert_zyx() methods.

        Returns:
            JaggedTensor: A JaggedTensor of shape `[num_grids, -1, 1]` containing
                the permutation indices. Use these indices to reorder voxel data for spatial coherence.
        """
        # Get the curve codes as a flat tensor
        curve_data = curve_codes.jdata.squeeze(-1)  # Shape: [total_voxels]

        # Create output tensor for permutation indices
        permutation_indices = torch.empty_like(curve_data, dtype=torch.long)

        # Sort curve codes and get permutation indices for each grid
        offset = 0
        for grid_idx in range(grid.grid_count):
            num_voxels = grid.num_voxels_at(grid_idx)
            if num_voxels == 0:
                continue

            # Extract curve codes for this grid
            grid_curve_codes = curve_data[offset : offset + num_voxels]

            # Sort and get indices
            _, indices = torch.sort(grid_curve_codes, dim=0)

            # Store indices with offset
            permutation_indices[offset : offset + num_voxels] = indices + offset

            offset += num_voxels

        # Return as JaggedTensor with the same structure as the input
        return grid.jagged_like(permutation_indices.unsqueeze(-1))

    def _permutation_morton(self, grid):
        """
        Return permutation indices to sort voxels by Morton curve order.
        """
        return self._compute_permutation(grid, grid.morton())

    def _permutation_morton_zyx(self, grid):
        """
        Return permutation indices to sort voxels by transposed Morton curve order.
        """
        return self._compute_permutation(grid, grid.morton_zyx())

    def _permutation_hilbert(self, grid):
        """
        Return permutation indices to sort voxels by Hilbert curve order.
        """
        return self._compute_permutation(grid, grid.hilbert())

    def _permutation_hilbert_zyx(self, grid):
        """
        Return permutation indices to sort voxels by transposed Hilbert curve order.
        """
        return self._compute_permutation(grid, grid.hilbert_zyx())

    def _permute(self, grid, order_type):
        if order_type == "z":
            return self._permutation_morton(grid)
        elif order_type == "z-trans":
            return self._permutation_morton_zyx(grid)
        elif order_type == "hilbert":
            return self._permutation_hilbert(grid)
        elif order_type == "hilbert-trans":
            return self._permutation_hilbert_zyx(grid)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_Attention")
        feats_j = feats.jdata

        # Get the shuffled order from grid metadata if available, otherwise use default order_types
        # This allows for order shuffling per forward pass (matching reference implementation)
        active_order_types = grid._shuffled_order

        # Get the order type for this block using the order index
        order_type = active_order_types[self.order_index % len(active_order_types)]

        if order_type != "vdb":
            perm = self._permute(grid, order_type).jdata.squeeze(-1)  # [num_voxels]
            # Use torch.gather for permutation: expand perm to match feats_j dimensions
            perm_expanded = perm.unsqueeze(-1).expand(-1, feats_j.shape[-1])  # [num_voxels, hidden_size]
            feats_j = torch.gather(feats_j, 0, perm_expanded)

        # import pdb; pdb.set_trace()

        qkv = self.qkv(feats_j)  # (num_voxels, 3 * hidden_size)

        if self.sliding_window_attention and self.patch_size > 0:
            # Perform sliding window attention per-grid using flash attention
            assert flash_attn is not None, "flash_attn is required for sliding_window_attention. Install with: pip install flash-attn"
            num_voxels = feats_j.shape[0]
            H = self.num_heads
            D = self.head_dim
            offsets = feats.joffsets.to(device=qkv.device, dtype=torch.int64)
            outputs = []
            for b in range(offsets.numel() - 1):
                start = int(offsets[b].item())
                end = int(offsets[b + 1].item())
                Li = end - start
                if Li <= 0:
                    continue
                qkv_b = qkv[start:end].view(1, Li, 3, H, D)
                window_size = (self.patch_size // 2, self.patch_size // 2)
                out_b = flash_attn.flash_attn_qkvpacked_func(
                    qkv_b.half(), dropout_p=0.0, softmax_scale=self.scale, window_size=window_size
                ).reshape(
                    Li, self.hidden_size
                )  # dtype: float16
                outputs.append(out_b)
            if len(outputs) == 0:
                feats_out_j = torch.empty_like(qkv[:, : self.hidden_size])
            else:
                feats_out_j = torch.cat(outputs, dim=0)

            feats_out_j = feats_out_j.to(feats_j.dtype)

        elif self.patch_size > 0:
            # Perform attention within each patch_size window per-grid using varlen API
            assert flash_attn is not None, "flash_attn is required when patch_size > 0. Install with: pip install flash-attn"
            num_voxels = feats_j.shape[0]
            H = self.num_heads
            D = self.head_dim
            qkv = qkv.view(-1, 3, H, D)  # (num_voxels, 3, num_heads, head_dim)

            # Build cu_seqlens as concatenation of per-grid patches so we never cross grid boundaries
            offsets = feats.joffsets.to(device=qkv.device, dtype=torch.int64)
            lengths = []
            for b in range(offsets.numel() - 1):
                start = int(offsets[b].item())
                end = int(offsets[b + 1].item())
                Li = end - start
                if Li <= 0:
                    continue
                full = Li // self.patch_size
                rem = Li % self.patch_size
                if full > 0:
                    lengths.extend([self.patch_size] * full)
                if rem > 0:
                    lengths.append(rem)
            if len(lengths) == 0:
                feats_out_j = torch.empty((0, self.hidden_size), device=qkv.device, dtype=feats_j.dtype)
            else:
                cu_seqlens = torch.zeros(len(lengths) + 1, device=qkv.device, dtype=torch.int32)
                cu_seqlens[1:] = torch.as_tensor(lengths, device=qkv.device, dtype=torch.int32).cumsum(dim=0)

                feats_out_j = flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv.half(),
                    cu_seqlens,
                    max_seqlen=self.patch_size,
                    dropout_p=0.0,  # TODO: implement attention dropout in the future. By default, it is 0.
                    softmax_scale=self.scale,
                ).reshape(
                    num_voxels, self.hidden_size
                )  # dtype: float16

                feats_out_j = feats_out_j.to(feats_j.dtype)
        else:
            feats_out_j = qkv[:, : self.hidden_size].contiguous()

        if order_type != "vdb":
            perm_reverse = torch.empty_like(perm)
            perm_reverse[perm] = torch.arange(perm.shape[0], device=perm.device)  # [num_voxels]
            perm_reverse_expanded = perm_reverse.unsqueeze(-1).expand(
                -1, feats_out_j.shape[-1]
            )  # [num_voxels, hidden_size]
            feats_out_j = torch.gather(feats_out_j, 0, perm_reverse_expanded)

        feats_out_j = self.proj(feats_out_j)
        feats_out_j = self.drop(feats_out_j)
        feats_out = grid.jagged_like(feats_out_j)
        nvtx.range_pop()
        return grid, feats_out


class PTV3_CPE(torch.nn.Module):
    def __init__(self, hidden_size: int, no_conv_in_cpe: bool = False, shared_plan_cache: Dict = None):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            shared_plan_cache (Dict): Shared cache for ConvolutionPlans across all layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.no_conv_in_cpe = no_conv_in_cpe
        self.shared_plan_cache = shared_plan_cache
        self.cpe = torch.nn.ModuleList(
            [
                (
                    fvdb.nn.SparseConv3d(hidden_size, hidden_size, kernel_size=3, stride=1)  # by default, bias is True.
                    if not no_conv_in_cpe
                    else torch.nn.Identity()
                ),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.LayerNorm(hidden_size),
            ]
        )

    def _get_plan(self, grid, kernel_size, stride):
        """Get or create a ConvolutionPlan from shared cache."""
        cache_key = (grid.address, kernel_size, stride)
        if cache_key not in self.shared_plan_cache:
            self.shared_plan_cache[cache_key] = fvdb.ConvolutionPlan.from_grid_batch(
                kernel_size=kernel_size, stride=stride, source_grid=grid, target_grid=grid
            )
        return self.shared_plan_cache[cache_key]

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_CPE")

        if not self.no_conv_in_cpe:
            # Apply 3x3 sparse convolution using shared ConvolutionPlan cache
            plan = self._get_plan(grid, kernel_size=3, stride=1)
            out_feature = self.cpe[0](feats, plan)
            # Note: bias is already handled inside SparseConv3d.forward()
        else:
            out_feature = feats

        out_feature_j = out_feature.jdata
        out_feature_j = self.cpe[1](out_feature_j)
        out_feature_j = self.cpe[2](out_feature_j)
        out_feature = grid.jagged_like(out_feature_j)

        nvtx.range_pop()
        return grid, out_feature


class PTV3_Block(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        drop_path: float,
        proj_drop: float = 0.0,
        patch_size: int = 0,
        qk_scale: float = None,
        no_conv_in_cpe: bool = False,
        sliding_window_attention: bool = False,
        order_index: int = 0,
        order_types: tuple = ("vdb",),
        shared_plan_cache: Dict = None,
    ):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            num_heads (int): Number of attention heads in each block.
            drop_path (float): Drop path rate for regularization.
            proj_drop (float): Dropout rate for MLP layers.
            patch_size (int): Patch size for patch attention.
            qk_scale (float): Scale factor for query-key dot product. If None, uses 1/sqrt(head_dim).
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
            order_index (int): Index into order_types to select which order to use for this block.
            order_types (tuple): Tuple of order type strings (e.g., ("z", "z-trans")).
            shared_plan_cache (Dict): Shared cache for ConvolutionPlans across all layers.
        """
        super().__init__()

        self.cpe = PTV3_CPE(hidden_size, no_conv_in_cpe, shared_plan_cache)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.attn = PTV3_Attention(
            hidden_size,
            num_heads,
            proj_drop,
            patch_size,
            qk_scale,
            sliding_window_attention,
            order_index,
            order_types,
        )
        self.norm2 = torch.nn.LayerNorm(hidden_size)
        self.order_index = order_index
        self.mlp = PTV3_MLP(hidden_size, proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_Block")
        grid, feats_out = self.cpe(grid, feats)
        feats = grid.jagged_like(feats.jdata + feats_out.jdata)  # Is this a potential issue?
        short_cut = feats.jdata

        feats = grid.jagged_like(self.norm1(feats.jdata))

        grid, feats_out = self.attn(grid, feats)
        feats_out = grid.jagged_like(
            self.drop_path(feats_out.jdata)
        )  # This drop_path is applied to each point independently.

        feats = grid.jagged_like(short_cut + feats_out.jdata)
        short_cut = feats.jdata

        feats = grid.jagged_like(self.norm2(feats.jdata))

        grid, feats_out = self.mlp(grid, feats)
        feats_out = grid.jagged_like(
            self.drop_path(feats_out.jdata)
        )  # This drop_path is applied to each point independently.

        feats = grid.jagged_like(short_cut + feats_out.jdata)

        nvtx.range_pop()
        return grid, feats


class PTV3_Encoder(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        depth: int,
        num_heads: int,
        drop_path,  # drop_path is a list of drop path rates for each block.
        proj_drop: float = 0.0,
        patch_size: int = 0,
        qk_scale: float = None,
        no_conv_in_cpe: bool = False,
        sliding_window_attention: bool = False,
        order_types: tuple = ("vdb",),
        shared_plan_cache: Dict = None,
    ):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            depth (int): Number of blocks in the encoder.
            num_heads (int): Number of attention heads in each block.
            drop_path (list): Drop path rates for each block.
            proj_drop (float): Dropout rate for MLP layers.
            patch_size (int): Patch size for patch attention.
            qk_scale (float): Scale factor for query-key dot product. If None, uses 1/sqrt(head_dim).
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
            order_types (tuple): Tuple of order type strings (e.g., ("z", "z-trans")).
            shared_plan_cache (Dict): Shared cache for ConvolutionPlans across all layers.
        """
        super().__init__()
        self.depth = depth
        self.blocks = torch.nn.ModuleList(
            [
                PTV3_Block(
                    hidden_size,
                    num_heads,
                    drop_path[i],
                    proj_drop,
                    patch_size,
                    qk_scale,
                    no_conv_in_cpe,
                    sliding_window_attention,
                    i % len(order_types),  # order_index cycles through available order types
                    order_types,
                    shared_plan_cache,
                )
                for i in range(depth)
            ]
        )
        self.order_types = order_types

    def forward(self, grid, feats):
        for block in self.blocks:
            grid, feats = block(grid, feats)
        return grid, feats


class PTV3(torch.nn.Module):

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 6,  # xyz + intensity/reflectance + additional features
        enc_depths: Tuple[int, ...] = (
            2,
            2,
            2,
            2,
        ),  # default hyper-parameters to align with sonata ptv3's default hyper-parameters.
        enc_channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        enc_num_heads: Tuple[int, ...] = (2, 4, 8, 16, 32),
        # enc_patch_size: Tuple[int, ...] = (4096),
        dec_depths: Tuple[int, ...] = (),  # by default, no decoder.
        dec_channels: Tuple[int, ...] = (),
        dec_num_heads: Tuple[int, ...] = (),
        patch_size: int = 0,
        drop_path: float = 0.3,
        proj_drop: float = 0.0,
        qk_scale: float = None,
        enable_batch_norm: bool = False,
        embedding_mode: str = "linear",
        no_conv_in_cpe: bool = False,
        sliding_window_attention: bool = False,
        order_type: Union[str, tuple] = ("z", "z-trans"),
        shuffle_orders: bool = True,
    ) -> None:
        """
        ptv3 for 3D point cloud segmentation.

        Args:
            num_classes (int): Number of classes for segmentation.
            input_dim (int): Input feature dimension (default: 4 for xyz + intensity).
            hidden_dims (Tuple[int, ...]): Hidden layer dimensions (not used in simplified version).
            enc_depths (Tuple[int, ...]): Number of encoder blocks for each stage.
            enc_channels (Tuple[int, ...]): Number of channels for each stage.
            enc_num_heads (Tuple[int, ...]): Number of attention heads for each stage.
            dec_depths (Tuple[int, ...]): Number of decoder blocks for each stage.
            dec_channels (Tuple[int, ...]): Number of channels for each stage.
            dec_num_heads (Tuple[int, ...]): Number of attention heads for each stage.
            patch_size (int): Patch size for patch attention.
            drop_path (float): Drop path rate for regularization.
            proj_drop (float): Dropout rate for MLP layers.
            qk_scale (float): Scale factor for query-key dot product. If None, uses 1/sqrt(head_dim).
            enable_batch_norm (bool): Whether to use batch normalization for the embedding, down pooling, and up pooling.
            embedding_mode (bool): the mode for the embedding layer, "linear" or "conv3x3", "conv5x5".
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
            order (Union[str, tuple]): The type(s) of point ordering. Can be a single string ("vdb", "z", "z-trans", "hilbert", "hilbert-trans")
                or a tuple of strings (e.g., ("z", "z-trans")). Each block within a stage cycles through the order types.
            shuffle_orders (bool): Whether to shuffle the order of order types at the beginning of each forward pass and after each pooling.
        """
        super().__init__()
        self.num_classes = num_classes
        self.drop_path = drop_path
        self.proj_drop = proj_drop
        self.qk_scale = qk_scale
        self.no_conv_in_cpe = no_conv_in_cpe
        self.sliding_window_attention = sliding_window_attention
        self.shuffle_orders = shuffle_orders

        # Handle order: convert to tuple for uniform processing (matching reference implementation)
        self.order_type = tuple([order_type]) if isinstance(order_type, str) else tuple(order_type)

        if not enable_batch_norm:
            self.norm_layer = torch.nn.LayerNorm
        else:
            self.norm_layer = partial(torch.nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # Shared ConvolutionPlan cache across all layers to avoid redundant computation.
        # Cache is cleared at the end of each forward pass to prevent OOM.
        self.shared_plan_cache = {}

        self.embedding = PTV3_Embedding(
            input_dim,
            enc_channels[0],
            norm_layer_module=self.norm_layer,
            embedding_mode=embedding_mode,
            shared_plan_cache=self.shared_plan_cache,
        )

        self.num_stages = len(enc_depths)
        if self.num_stages > 0:
            self.enc = torch.nn.ModuleList()
            enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
            for i in range(self.num_stages):
                if i > 0:
                    self.enc.append(
                        PTV3_Pooling(
                            kernel_size=2,
                            in_channels=enc_channels[i - 1],
                            out_channels=enc_channels[i],
                            norm_layer_module=self.norm_layer,
                        )
                    )
                # All encoder stages share the same order types; blocks within each stage cycle through them
                self.enc.append(
                    PTV3_Encoder(
                        enc_channels[i],
                        enc_depths[i],
                        enc_num_heads[i],
                        enc_drop_path[sum(enc_depths[:i]) : sum(enc_depths[: i + 1])],
                        proj_drop,
                        patch_size,
                        qk_scale,
                        no_conv_in_cpe,
                        sliding_window_attention,
                        self.order_type,
                        self.shared_plan_cache,
                    )
                )

        # create decoder
        self.num_dec_stages = len(dec_depths)
        if self.num_dec_stages > 0:
            assert (
                self.num_dec_stages == self.num_stages - 1
            ), "The number of decoder stages must be one less than the number of encoder stages."
            self.dec = torch.nn.ModuleList()
            dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
            dec_drop_path = dec_drop_path[::-1]

            for i in range(self.num_dec_stages):
                dec_drop_path_ = dec_drop_path[sum(dec_depths[:i]) : sum(dec_depths[: i + 1])]
                if i == 0:
                    last_channels = enc_channels[-1]
                else:
                    last_channels = dec_channels[i - 1]
                self.dec.append(
                    PTV3_Unpooling(
                        kernel_size=2,
                        in_channels=last_channels,
                        out_channels=dec_channels[i],
                        skip_channels=enc_channels[self.num_stages - 2 - i],
                        norm_layer_module=self.norm_layer,
                    )
                )
                # All decoder stages share the same order types; blocks within each stage cycle through them
                self.dec.append(
                    PTV3_Encoder(
                        dec_channels[i],
                        dec_depths[i],
                        dec_num_heads[i],
                        dec_drop_path_,
                        proj_drop,
                        patch_size,
                        qk_scale,
                        no_conv_in_cpe,
                        sliding_window_attention,
                        self.order_type,
                        self.shared_plan_cache,
                    )
                )

    def _shuffle_order(self):
        """
        Randomly shuffle the order tuple to create variation across forward passes.
        Returns a new shuffled tuple of order types.
        """
        if self.shuffle_orders:
            indices = torch.randperm(len(self.order_type))
            return tuple(self.order_type[i] for i in indices)
        else:
            return self.order_type

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_Forward")

        # Shuffle order at the beginning of forward pass (matching reference implementation)
        shuffled_order = self._shuffle_order()

        # Store shuffled order in grid metadata so all blocks can access it
        grid._shuffled_order = shuffled_order

        grid, feats = self.embedding(grid, feats)

        layer_id = 0
        stack = []  # Stack stores (grid, feats, shuffled_order) tuples
        for i in range(self.num_stages):
            if i > 0:
                nvtx.range_push(f"PTV3_Pooling_{layer_id}")
                # Push grid, feats, AND the current shuffled_order to stack
                # The decoder will reuse this exact shuffled order for the corresponding stage
                stack.append((grid, feats, shuffled_order))
                grid, feats = self.enc[layer_id](grid, feats)

                # Shuffle order after pooling for the next (downsampled) stage
                shuffled_order = self._shuffle_order()
                grid._shuffled_order = shuffled_order

                nvtx.range_pop()
                layer_id += 1
            nvtx.range_push(f"PTV3_Encoder_{layer_id}")
            grid, feats = self.enc[layer_id](grid, feats)
            nvtx.range_pop()
            layer_id += 1

        if self.num_dec_stages > 0:
            layer_id = 0
            for i in range(self.num_dec_stages):
                nvtx.range_push(f"PTV3_Unpooling_{layer_id}")
                # Pop grid, feats, AND the shuffled_order from the corresponding encoder stage
                last_grid, last_feats, last_shuffled_order = stack.pop()

                # Restore the shuffled order from the encoder stage to the grids
                # This ensures decoder blocks use the SAME order as the corresponding encoder blocks
                last_grid._shuffled_order = last_shuffled_order

                grid, feats = self.dec[layer_id](grid, feats, last_grid, last_feats)
                # After unpooling, grid becomes last_grid with the restored shuffled order
                nvtx.range_pop()
                layer_id += 1

                nvtx.range_push(f"PTV3_Decoder_{layer_id}")
                # Decoder blocks use grid with the restored shuffled order from encoder
                grid, feats = self.dec[layer_id](grid, feats)
                nvtx.range_pop()
                layer_id += 1

        # Clear cache after forward pass to prevent OOM between batches
        # Plans are shared across layers during this forward pass, but won't be needed for next batch
        self.shared_plan_cache.clear()

        nvtx.range_pop()
        return grid, feats
