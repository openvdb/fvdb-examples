# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
PTV3 FVDB Implementation

This module contains the core Point Transformer V3 implementation using FVDB.
It works directly with FVDB GridBatch and JaggedTensor types.

For pointcept framework integration, see point_transformer_v3m1_fvdb.py
"""

from functools import partial
from typing import Callable

import fvdb
import torch
import torch.nn
import torch.nn.functional as F
from timm.layers import DropPath

from .fvdb_utils import (
    FJTM,
    FVDBGridModule,
    NVTXRange,
    inverse_order_features_from_perm,
    jagged_attention,
    order_features_from_jagged_ijk,
)


class PTV3_Embedding(FVDBGridModule):
    """
    PTV3_Embedding for 3D point cloud embedding.
    """

    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer_module: type[torch.nn.Module] | Callable = torch.nn.LayerNorm,
        embedding_mode: str = "linear",
        shared_plan_cache: dict | None = None,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input features.
            embed_channels (int): Number of channels in the output features.
            norm_layer_module (type[torch.nn.Module] | Callable): Normalization layer module.
            embedding_mode (str): The type of embedding layer, "linear" or "conv3x3", "conv5x5".
            #shared_plan_cache (dict | None): Shared cache for ConvolutionPlans across all layers.
        """
        super().__init__()
        self.embedding_mode = embedding_mode
        self.shared_plan_cache = shared_plan_cache if shared_plan_cache is not None else {}

        if embedding_mode == "linear":
            self.embed = FJTM(torch.nn.Linear(in_channels, embed_channels))
        elif embedding_mode == "conv3x3":
            # Initialize embedding using FVDB's sparse 3D convolution
            self.embed_conv3x3_1 = fvdb.nn.SparseConv3d(
                in_channels, embed_channels, kernel_size=3, stride=1, bias=False
            )
        elif embedding_mode == "conv5x5":
            # Initialize embedding using FVDB's sparse 3D convolution
            self.embed_conv5x5_1 = fvdb.nn.SparseConv3d(in_channels, embed_channels, kernel_size=5, stride=1)
        else:
            raise ValueError(f"Unsupported embedding mode: {embedding_mode}")

        self.norm_layer = FJTM(norm_layer_module(embed_channels))
        self.act_layer = FJTM(torch.nn.GELU())

    def _get_plan(self, grid: fvdb.GridBatch, kernel_size, stride):
        """Get or create a ConvolutionPlan from shared cache."""
        # target_grid = grid.conv_grid(kernel_size=kernel_size, stride=stride)

        # We definitely want the target grid to be the same as the source grid,
        # because we need the topology to remain the same.
        return fvdb.ConvolutionPlan.from_grid_batch(
            kernel_size=kernel_size, stride=stride, source_grid=grid, target_grid=grid
        )
        # cache_key = (grid.address, kernel_size, stride)
        # if cache_key not in self.shared_plan_cache:
        #     self.shared_plan_cache[cache_key] = fvdb.ConvolutionPlan.from_grid_batch(
        #         kernel_size=kernel_size, stride=stride, source_grid=grid, target_grid=grid
        #     )
        # return self.shared_plan_cache[cache_key]

    # We use the same output grid as the input grid to maintain topology, so only the
    # features are updated.
    def forward(self, feats: fvdb.JaggedTensor, grid: fvdb.GridBatch) -> fvdb.JaggedTensor:
        with NVTXRange("PTV3_Embedding"):
            if self.embedding_mode == "linear":
                feats = self.embed(feats)
            elif self.embedding_mode == "conv3x3":
                plan = self._get_plan(grid, kernel_size=3, stride=1)
                feats = self.embed_conv3x3_1(feats, plan)
            elif self.embedding_mode == "conv5x5":
                plan = self._get_plan(grid, kernel_size=5, stride=1)
                feats = self.embed_conv5x5_1(feats, plan)

            feats = self.norm_layer(feats)
            feats = self.act_layer(feats)
        return feats


class PTV3_Pooling(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 64,
        out_channels: int = 64,
        norm_layer_module: type[torch.nn.Module] | Callable = torch.nn.LayerNorm,
    ):
        """
        Args:
            kernel_size (int): Kernel size for the pooling operation.
            in_channels (int): Number of channels in the input features.
            out_channels (int): Number of channels in the output features.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.proj = FJTM(torch.nn.Linear(in_channels, out_channels))
        self.norm_layer = FJTM(norm_layer_module(out_channels))
        self.act_layer = FJTM(torch.nn.GELU())

    def __call__(self, feats: fvdb.JaggedTensor, grid: fvdb.GridBatch) -> tuple[fvdb.JaggedTensor, fvdb.GridBatch]:
        """Override __call__ to preserve type hints from forward."""
        return super().__call__(feats, grid)

    def forward(self, feats: fvdb.JaggedTensor, grid: fvdb.GridBatch) -> tuple[fvdb.JaggedTensor, fvdb.GridBatch]:
        with NVTXRange("PTV3_Pooling"):
            feats = self.proj(feats)

            ds_feature, ds_grid = grid.max_pool(self.kernel_size, feats, stride=self.kernel_size, coarse_grid=None)
            ds_feature = self.norm_layer(ds_feature)
            ds_feature = self.act_layer(ds_feature)
        return ds_feature, ds_grid


class PTV3_Unpooling(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 64,
        out_channels: int = 64,
        skip_channels: int = 64,
        norm_layer_module: type[torch.nn.Module] | Callable = torch.nn.LayerNorm,
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

        self.proj = FJTM(torch.nn.Linear(in_channels, out_channels))
        self.proj_skip = FJTM(torch.nn.Linear(skip_channels, out_channels))
        self.norm = FJTM(norm_layer_module(out_channels))
        self.norm_skip = FJTM(norm_layer_module(out_channels))
        self.act_layer = FJTM(torch.nn.GELU())
        self.act_layer_skip = FJTM(torch.nn.GELU())

    def __call__(
        self, feats: fvdb.JaggedTensor, grid: fvdb.GridBatch, last_feats: fvdb.JaggedTensor, last_grid: fvdb.GridBatch
    ) -> tuple[fvdb.JaggedTensor, fvdb.GridBatch]:
        """Override __call__ to preserve type hints from forward."""
        return super().__call__(feats, grid, last_feats, last_grid)

    def forward(
        self, feats: fvdb.JaggedTensor, grid: fvdb.GridBatch, last_feats: fvdb.JaggedTensor, last_grid: fvdb.GridBatch
    ) -> tuple[fvdb.JaggedTensor, fvdb.GridBatch]:
        with NVTXRange("PTV3_Unpooling"):
            # The conversion is to avoid the bug when enabled AMP,
            # despite both feats.jdata and linear.weights are float32,
            # the output becomes float16 which causes the subsequent convolution operation to fail.
            feats = self.proj(feats) #.to(torch.float32)
            feats = self.norm(feats)
            feats = self.act_layer(feats)

            last_feats = self.proj_skip(last_feats)
            last_feats = self.norm_skip(last_feats)
            last_feats = self.act_layer_skip(last_feats)

            feats, _match_last_grid = grid.refine(self.kernel_size, feats, fine_grid=last_grid)
            assert last_grid.is_same(_match_last_grid), "The last grid and the matched grid are not the same."

            feats = fvdb.add(feats, last_feats)
        return feats, last_grid


class PTV3_MLP(torch.nn.Module):
    def __init__(self, hidden_size: int, proj_drop: float = 0.0):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            proj_drop (float): Dropout rate for MLP layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = FJTM(torch.nn.Linear(hidden_size, hidden_size * 4))
        self.act = FJTM(torch.nn.GELU())
        self.fc2 = FJTM(torch.nn.Linear(hidden_size * 4, hidden_size))
        self.drop = FJTM(torch.nn.Dropout(proj_drop))

    def __call__(self, feats: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        """Override __call__ to preserve type hints from forward."""
        return super().__call__(feats)

    def forward(self, feats: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        with NVTXRange("PTV3_MLP"):
            feats = self.fc1(feats)
            feats = self.act(feats)
            feats = self.drop(feats)
            feats = self.fc2(feats)
            feats = self.drop(feats)
        return feats


class PTV3_Attention(FVDBGridModule):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        proj_drop: float = 0.0,
        patch_size: int = 0,
        qk_scale: float | None = None,
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
            qk_scale (float | None): Scale factor for query-key dot product. If None, uses 1/sqrt(head_dim).
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
        self.qkv = FJTM(torch.nn.Linear(hidden_size, hidden_size * 3))  # Combined QKV projection
        self.proj = FJTM(torch.nn.Linear(hidden_size, hidden_size))
        self.drop = FJTM(torch.nn.Dropout(proj_drop))
        self.patch_size = patch_size
        self.order_index = order_index
        self.order_types = order_types

        # TODO: Add attention dropout

        # Sliding window attention parameter
        self.sliding_window_attention = sliding_window_attention

    def forward(self, feats: fvdb.JaggedTensor, grid: fvdb.GridBatch) -> fvdb.JaggedTensor:
        with NVTXRange("PTV3_Attention"):
            # Get the shuffled order from grid metadata if available, otherwise use default order_types
            # This allows for order shuffling per forward pass (matching reference implementation)
            active_order_types = grid._shuffled_order  # type: ignore

            # Get the order type for this block using the order index
            order_type = active_order_types[self.order_index % len(active_order_types)]

            feats_ordered, perm = order_features_from_jagged_ijk(feats, grid.ijk, order_type)

            qkv = self.qkv(feats_ordered)

            feats_ordered_out = jagged_attention(
                feats_ordered,
                qkv,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                patch_size=self.patch_size,
                sliding_window_attention=self.sliding_window_attention,
                scale=self.scale,
            )

            feats_out = inverse_order_features_from_perm(feats_ordered_out, perm)
            feats_out = self.proj(feats_out)
            feats_out = self.drop(feats_out)
        return feats_out


class PTV3_CPE(FVDBGridModule):
    def __init__(self, hidden_size: int, no_conv_in_cpe: bool = False, shared_plan_cache: dict | None = None):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            shared_plan_cache (dict | None): Shared cache for ConvolutionPlans across all layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.no_conv_in_cpe = no_conv_in_cpe
        self.shared_plan_cache = shared_plan_cache if shared_plan_cache is not None else {}

        self.maybe_conv: fvdb.nn.SparseConv3d | None = (
            None
            if no_conv_in_cpe
            else fvdb.nn.SparseConv3d(hidden_size, hidden_size, kernel_size=3, stride=1, bias=True)
        )
        self.linear = FJTM(torch.nn.Linear(hidden_size, hidden_size))
        self.norm = FJTM(torch.nn.LayerNorm(hidden_size))

    def _get_plan(self, grid, kernel_size, stride):
        """Get or create a ConvolutionPlan from shared cache."""
        # target_grid = grid.conv_grid(kernel_size=kernel_size, stride=stride)
        # We need target grid to be the same as the source grid to maintain topology.
        return fvdb.ConvolutionPlan.from_grid_batch(
            kernel_size=kernel_size, stride=stride, source_grid=grid, target_grid=grid
        )
        # cache_key = (grid.address, kernel_size, stride)
        # if cache_key not in self.shared_plan_cache:
        #     self.shared_plan_cache[cache_key] = fvdb.ConvolutionPlan.from_grid_batch(
        #         kernel_size=kernel_size, stride=stride, source_grid=grid, target_grid=grid
        #     )
        # return self.shared_plan_cache[cache_key]

    # Target grid is same as source grid to maintain topology.
    def forward(self, feats: fvdb.JaggedTensor, grid: fvdb.GridBatch) -> fvdb.JaggedTensor:
        with NVTXRange("PTV3_CPE"):
            if not self.no_conv_in_cpe:
                # Apply 3x3 sparse convolution using shared ConvolutionPlan cache
                plan = self._get_plan(grid, kernel_size=3, stride=1)
                assert self.maybe_conv is not None, "maybe_conv is not initialized"
                feats = self.maybe_conv(feats, plan)

            feats = self.linear(feats)
            feats = self.norm(feats)

        return feats


class PTV3_Block(FVDBGridModule):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        drop_path: float,
        proj_drop: float = 0.0,
        patch_size: int = 0,
        qk_scale: float | None = None,
        no_conv_in_cpe: bool = False,
        sliding_window_attention: bool = False,
        order_index: int = 0,
        order_types: tuple = ("vdb",),
        shared_plan_cache: dict | None = None,
    ):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            num_heads (int): Number of attention heads in each block.
            drop_path (float): Drop path rate for regularization.
            proj_drop (float): Dropout rate for MLP layers.
            patch_size (int): Patch size for patch attention.
            qk_scale (float | None): Scale factor for query-key dot product. If None, uses 1/sqrt(head_dim).
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
            order_index (int): Index into order_types to select which order to use for this block.
            order_types (tuple): Tuple of order type strings (e.g., ("z", "z-trans")).
            shared_plan_cache (dict | None): Shared cache for ConvolutionPlans across all layers.
        """
        super().__init__()

        self.cpe = PTV3_CPE(hidden_size, no_conv_in_cpe, shared_plan_cache)
        self.norm1 = FJTM(torch.nn.LayerNorm(hidden_size))
        self.attn = PTV3_Attention(
            hidden_size,
            num_heads,
            proj_drop,
            patch_size,
            qk_scale,
            sliding_window_attention,
            order_index,
            order_types,
        ) # temporary disable attention
        self.norm2 = FJTM(torch.nn.LayerNorm(hidden_size))
        self.order_index = order_index
        self.mlp = PTV3_MLP(hidden_size, proj_drop)
        self.drop_path = FJTM(DropPath(drop_path)) if drop_path > 0.0 else FJTM(torch.nn.Identity())

    def forward(self, feats: fvdb.JaggedTensor, grid: fvdb.GridBatch) -> fvdb.JaggedTensor:
        assert isinstance(feats, fvdb.JaggedTensor), "Input feats must be a JaggedTensor"
        assert isinstance(grid, fvdb.GridBatch), "Input grid must be a GridBatch"
        with NVTXRange("PTV3_Block"):
            short_cut = feats
            feats = self.cpe(feats, grid)
            feats = fvdb.add(short_cut, feats)
            short_cut = feats

            feats = self.norm1(feats)
            feats = self.attn(feats, grid) # temporary disable attention
            feats = self.drop_path(feats) # temporary disable attention
            # The drop_path is applied to each point independently.
            feats = fvdb.add(short_cut, feats)

            short_cut = feats
            feats = self.norm2(feats)
            feats = self.mlp(feats)
            feats = self.drop_path(feats)
            feats = fvdb.add(short_cut, feats)

        return feats


class PTV3_Encoder(FVDBGridModule):
    def __init__(
        self,
        hidden_size: int,
        depth: int,
        num_heads: int,
        drop_path,  # drop_path is a list of drop path rates for each block.
        proj_drop: float = 0.0,
        patch_size: int = 0,
        qk_scale: float | None = None,
        no_conv_in_cpe: bool = False,
        sliding_window_attention: bool = False,
        order_types: tuple = ("vdb",),
        shared_plan_cache: dict | None = None,
    ):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            depth (int): Number of blocks in the encoder.
            num_heads (int): Number of attention heads in each block.
            drop_path (list): Drop path rates for each block.
            proj_drop (float): Dropout rate for MLP layers.
            patch_size (int): Patch size for patch attention.
            qk_scale (float | None): Scale factor for query-key dot product. If None, uses 1/sqrt(head_dim).
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
            order_types (tuple): Tuple of order type strings (e.g., ("z", "z-trans")).
            shared_plan_cache (dict | None): Shared cache for ConvolutionPlans across all layers.
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

    def forward(self, feats: fvdb.JaggedTensor, grid: fvdb.GridBatch) -> fvdb.JaggedTensor:
        for block in self.blocks:
            assert isinstance(block, PTV3_Block), "All blocks must be of type PTV3_Block"
            feats = block(feats, grid)
        return feats


class PTV3(FVDBGridModule):

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 6,  # xyz + intensity/reflectance + additional features
        enc_depths: tuple[int, ...] = (
            2,
            2,
            2,
            2,
        ),  # default hyper-parameters to align with sonata ptv3's default hyper-parameters.
        enc_channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        enc_num_heads: tuple[int, ...] = (2, 4, 8, 16, 32),
        # enc_patch_size: tuple[int, ...] = (4096),
        dec_depths: tuple[int, ...] = (),  # by default, no decoder.
        dec_channels: tuple[int, ...] = (),
        dec_num_heads: tuple[int, ...] = (),
        patch_size: int = 0,
        drop_path: float = 0.3,
        proj_drop: float = 0.0,
        qk_scale: float | None = None,
        enable_batch_norm: bool = False,
        embedding_mode: str = "linear",
        no_conv_in_cpe: bool = False,
        sliding_window_attention: bool = False,
        order_type: str | tuple = ("z", "z-trans"),
        shuffle_orders: bool = True,
    ) -> None:
        """
        ptv3 for 3D point cloud segmentation.

        Args:
            num_classes (int): Number of classes for segmentation.
            input_dim (int): Input feature dimension (default: 4 for xyz + intensity).
            hidden_dims (tuple[int, ...]): Hidden layer dimensions (not used in simplified version).
            enc_depths (tuple[int, ...]): Number of encoder blocks for each stage.
            enc_channels (tuple[int, ...]): Number of channels for each stage.
            enc_num_heads (tuple[int, ...]): Number of attention heads for each stage.
            dec_depths (tuple[int, ...]): Number of decoder blocks for each stage.
            dec_channels (tuple[int, ...]): Number of channels for each stage.
            dec_num_heads (tuple[int, ...]): Number of attention heads for each stage.
            patch_size (int): Patch size for patch attention.
            drop_path (float): Drop path rate for regularization.
            proj_drop (float): Dropout rate for MLP layers.
            qk_scale (float | None): Scale factor for query-key dot product. If None, uses 1/sqrt(head_dim).
            enable_batch_norm (bool): Whether to use batch normalization for the embedding, down pooling, and up pooling.
            embedding_mode (bool): the mode for the embedding layer, "linear" or "conv3x3", "conv5x5".
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
            order (str | tuple): The type(s) of point ordering. Can be a single string ("vdb", "z", "z-trans", "hilbert", "hilbert-trans")
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

        assert (
            len(enc_depths) == len(enc_channels) == len(enc_num_heads)
        ), "The number of encoder depths, channels, and heads must be the same."

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
                # All decoder stages share the same order types;
                # blocks within each stage cycle through them
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

    def _shuffle_order(self, shuffled_order):
        """
        Randomly shuffle the order tuple to create variation across forward passes.
        Returns a new shuffled tuple of order types.
        """
        if self.shuffle_orders:
            indices = torch.randperm(len(shuffled_order))
            return tuple(shuffled_order[i] for i in indices)
        else:
            return shuffled_order

    def forward(self, feats: fvdb.JaggedTensor, grid: fvdb.GridBatch) -> fvdb.JaggedTensor:
        original_grid = grid
        with NVTXRange("PTV3_Forward"):

            # Shuffle order at the beginning of forward pass (matching reference implementation)
            shuffled_order = self._shuffle_order(self.order_type)

            # Store shuffled order in grid metadata so all blocks can access it
            grid._shuffled_order = shuffled_order  # type: ignore

            feats = self.embedding(feats, grid)

            layer_id = 0
            stack = []  # Stack stores (grid, feats, shuffled_order) tuples
            for i in range(self.num_stages):
                if i > 0:
                    with NVTXRange(f"PTV3_Pooling_{layer_id}"):
                        # Push grid, feats, AND the current shuffled_order to stack
                        # The decoder will reuse this exact shuffled order for the corresponding stage
                        stack.append((grid, feats, shuffled_order))
                        pooler = self.enc[layer_id]
                        assert isinstance(pooler, PTV3_Pooling), "All encoder poolers must be of type PTV3_Pooling"
                        feats, grid = pooler(feats, grid)

                        # Shuffle order after pooling for the next (downsampled) stage
                        shuffled_order = self._shuffle_order(shuffled_order)
                        grid._shuffled_order = shuffled_order  # type: ignore
                        layer_id += 1
                with NVTXRange(f"PTV3_Encoder_{layer_id}"):
                    encoder = self.enc[layer_id]
                    assert isinstance(encoder, PTV3_Encoder), "All encoder stages must be of type PTV3_Encoder"
                    feats = encoder(feats, grid)
                    layer_id += 1

            if self.num_dec_stages > 0:
                layer_id = 0
                for i in range(self.num_dec_stages):
                    with NVTXRange(f"PTV3_Unpooling_{layer_id}"):
                        # Pop grid, feats, AND the shuffled_order from the corresponding encoder stage
                        last_grid, last_feats, last_shuffled_order = stack.pop()

                        # Restore the shuffled order from the encoder stage to the grids
                        # This ensures decoder blocks use the SAME order as the corresponding encoder blocks
                        last_grid._shuffled_order = last_shuffled_order

                        unpooler = self.dec[layer_id]
                        assert isinstance(
                            unpooler, PTV3_Unpooling
                        ), "All decoder unpoolers must be of type PTV3_Unpooling"
                        feats, grid = unpooler(feats, grid, last_feats, last_grid)
                        # After unpooling, grid becomes last_grid with the restored shuffled order
                        layer_id += 1

                    with NVTXRange(f"PTV3_Decoder_{layer_id}"):
                        # Decoder blocks use grid with the restored shuffled order from encoder
                        decoder = self.dec[layer_id]
                        assert isinstance(decoder, PTV3_Encoder), "All decoder stages must be of type PTV3_Encoder"
                        feats = decoder(feats, grid)
                        layer_id += 1

            # Clear cache after forward pass to prevent OOM between batches
            # Plans are shared across layers during this forward pass, but won't be needed for next batch
            self.shared_plan_cache.clear()
            assert original_grid.is_same(grid), "The original grid and the final grid are not the same."

        return feats
