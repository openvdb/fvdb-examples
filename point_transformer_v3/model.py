# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple, Union, List

# Add NVTX import for profiling
import flash_attn
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
    ):
        """
        Args:
            in_channels (int): Number of channels in the input features.
            embed_channels (int): Number of channels in the output features.
            norm_layer_module (torch.nn.Module): Normalization layer module.
            embedding_mode (str): The type of embedding layer, "linear" or "conv3x3", "conv5x5".
        """
        super().__init__()
        self.embedding_mode = embedding_mode

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

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_Embedding")

        # Initialize kernel map (kmap) for sparse convolution operations
        # kmap tracks the mapping between input and output features during sparse convolutions
        if not hasattr(grid, "kmap"):
            grid.kmap = None

        if self.embedding_mode == "linear":
            jfeats = feats.jdata
            jfeats = self.embed(jfeats)
        elif self.embedding_mode == "conv3x3":
            # Apply 3x3 sparse convolution while maintaining kernel mapping
            # Note: Bias is intentionally disabled to maintain consistency with standard transformer architectures
            grid, feats, out_kmap = self.embed_conv3x3_1._dispatch_conv(feats, grid, grid.kmap, grid)
            grid.kmap = out_kmap
            jfeats = feats.jdata
        elif self.embedding_mode == "conv5x5":
            grid, feats, out_kmap = self.embed_conv3x3_1._dispatch_conv(feats, grid, grid.kmap, grid)
            grid.kmap = out_kmap
            grid, feats, out_kmap = self.embed_conv3x3_2._dispatch_conv(feats, grid, grid.kmap, grid)
            grid.kmap = out_kmap
            jfeats = feats.jdata

        jfeats = self.norm_layer(jfeats)
        jfeats = self.act_layer(jfeats)

        feats = feats.jagged_like(jfeats)
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
        feats = feats.jagged_like(feats_j)

        ds_feature, ds_grid = grid.max_pool(self.kernel_size, feats, stride=self.kernel_size, coarse_grid=None)
        ds_feature_j = ds_feature.jdata
        ds_feature_j = self.norm_layer(ds_feature_j)
        ds_feature_j = self.act_layer(ds_feature_j)
        ds_feature = ds_feature.jagged_like(ds_feature_j)
        nvtx.range_pop()
        ds_grid.kmap = None
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

        feats, _ = grid.subdivide(self.kernel_size, grid.jagged_like(feats_j), fine_grid=last_grid)
        feats_j = feats.jdata

        new_feats_j = last_feats_j + feats_j
        last_grid.kmap = (
            None  # Because of the pooling operation, the previous kmap for convolution is not valid anymore.
        )
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
        feats_j = feats.jdata

        feats_j = self.fc1(feats_j)
        feats_j = self.act(feats_j)
        feats_j = self.drop(feats_j)
        feats_j = self.fc2(feats_j)
        feats_j = self.drop(feats_j)
        feats = feats.jagged_like(feats_j)
        nvtx.range_pop()
        return grid, feats


class PTV3_Attention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        proj_drop: float = 0.0,
        patch_size: int = 0,
        sliding_window_attention: bool = False,
        order_type: str = "vdb",
    ):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            num_heads (int): Number of attention heads in each block.
            proj_drop (float): Dropout rate for MLP layers.
            patch_size (int): Patch size for patch attention.
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
            order_type (str): The type of order of the points, "vdb" or "z".
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.qkv = torch.nn.Linear(hidden_size, hidden_size * 3)  # Combined QKV projection
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.drop = torch.nn.Dropout(proj_drop)
        self.patch_size = patch_size
        self.order_type = order_type

        # Sliding window attention parameter
        self.sliding_window_attention = sliding_window_attention

    def _permute(self, grid, order_type):
        if order_type == "z":
            return grid.permutation_morton()
        elif order_type == "z-trans":
            return grid.permutation_morton_zyx()
        elif order_type == "hilbert":
            return grid.permutation_hilbert()
        elif order_type == "hilbert-trans":
            return grid.permutation_hilbert_zyx()
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_Attention")
        feats_j = feats.jdata

        if self.order_type != "vdb":
            perm = self._permute(grid, self.order_type).jdata.squeeze(-1)  # [num_voxels]
            # Use torch.gather for permutation: expand perm to match feats_j dimensions
            perm_expanded = perm.unsqueeze(-1).expand(-1, feats_j.shape[-1])  # [num_voxels, hidden_size]
            feats_j = torch.gather(feats_j, 0, perm_expanded)

        qkv = self.qkv(feats_j)  # (num_voxels, 3 * hidden_size)

        if self.sliding_window_attention and self.patch_size > 0:
            # Perform sliding window attention per-grid using flash attention
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
                    qkv_b.half(), dropout_p=0.0, softmax_scale=1.0, window_size=window_size
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
                    qkv.half(), cu_seqlens, max_seqlen=self.patch_size, dropout_p=0.0, softmax_scale=1.0
                ).reshape(
                    num_voxels, self.hidden_size
                )  # dtype: float16

                feats_out_j = feats_out_j.to(feats_j.dtype)
        else:
            feats_out_j = qkv[:, : self.hidden_size].contiguous()

        if self.order_type != "vdb":
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
    def __init__(self, hidden_size: int, no_conv_in_cpe: bool = False):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.no_conv_in_cpe = no_conv_in_cpe
        self.cpe = torch.nn.ModuleList(
            [
                (
                    fvdb.nn.SparseConv3d(hidden_size, hidden_size, kernel_size=3, stride=1)
                    if not no_conv_in_cpe
                    else torch.nn.Identity()
                ),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.LayerNorm(hidden_size),
            ]
        )

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_CPE")

        if not hasattr(grid, "kmap"):
            grid.kmap = None

        if not self.no_conv_in_cpe:
            grid, out_feature, out_kmap = self.cpe[0]._dispatch_conv(feats, grid, grid.kmap, grid)
            grid.kmap = out_kmap  # update the kmap
            if self.cpe[0].bias is not None:
                out_feature.jdata = out_feature.jdata + self.cpe[0].bias
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
        no_conv_in_cpe: bool = False,
        sliding_window_attention: bool = False,
        order_type: str = "vdb",
    ):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            num_heads (int): Number of attention heads in each block.
            drop_path (float): Drop path rate for regularization.
            proj_drop (float): Dropout rate for MLP layers.
            patch_size (int): Patch size for patch attention.
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
            order_type (str): The type of order of the points: "vdb", "z", "z-trans", "hilbert", "hilbert-trans".
        """
        super().__init__()

        self.cpe = PTV3_CPE(hidden_size, no_conv_in_cpe)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.attn = PTV3_Attention(
            hidden_size,
            num_heads,
            proj_drop,
            patch_size,
            sliding_window_attention,
            order_type,
        )
        self.norm2 = torch.nn.LayerNorm(hidden_size)
        self.order_type = order_type
        self.mlp = PTV3_MLP(hidden_size, proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_Block")
        grid, feats_out = self.cpe(grid, feats)
        feats = feats.jagged_like(feats.jdata + feats_out.jdata)
        short_cut = feats.jdata

        feats = feats.jagged_like(self.norm1(feats.jdata))

        grid, feats_out = self.attn(grid, feats)
        feats_out = feats.jagged_like(
            self.drop_path(feats_out.jdata)
        )  # This drop_path is applied to each point independently.

        feats = feats.jagged_like(short_cut + feats_out.jdata)
        short_cut = feats.jdata

        feats = feats.jagged_like(self.norm2(feats.jdata))

        grid, feats_out = self.mlp(grid, feats)
        feats_out = feats.jagged_like(
            self.drop_path(feats_out.jdata)
        )  # This drop_path is applied to each point independently.

        feats = feats.jagged_like(short_cut + feats_out.jdata)

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
        no_conv_in_cpe: bool = False,
        sliding_window_attention: bool = False,
        order_type: str = "vdb",
    ):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            depth (int): Number of blocks in the encoder.
            num_heads (int): Number of attention heads in each block.
            drop_path (list): Drop path rates for each block.
            proj_drop (float): Dropout rate for MLP layers.
            patch_size (int): Patch size for patch attention.
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
            order_type (str): The type of order of the points, "vdb" or "z".
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
                    no_conv_in_cpe,
                    sliding_window_attention,
                    order_type,
                )
                for i in range(depth)
            ]
        )
        self.order_type = order_type

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
        enable_batch_norm: bool = False,
        embedding_mode: str = "linear",
        no_conv_in_cpe: bool = False,
        sliding_window_attention: bool = False,
        order_type: Union[str, List[str]] = "vdb",
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
            enable_batch_norm (bool): Whether to use batch normalization for the embedding, down pooling, and up pooling.
            embedding_mode (bool): the mode for the embedding layer, "linear" or "conv3x3", "conv5x5".
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
            order_type (Union[str, List[str]]): The type of order of the points. Can be a single string ("vdb", "z", "z-trans", "hilbert", "hilbert-trans")
                for all layers, or a list of strings for different layers. Each encoder and decoder stage will use
                order_type[i % len(order_type)] where i is the stage index.
        """
        super().__init__()
        self.num_classes = num_classes
        self.drop_path = drop_path
        self.proj_drop = proj_drop
        self.no_conv_in_cpe = no_conv_in_cpe
        self.sliding_window_attention = sliding_window_attention

        # Handle order_type: convert to list for uniform processing
        if isinstance(order_type, str):
            self.order_type_list = [order_type]
        else:
            self.order_type_list = order_type
        self.order_type = order_type  # Keep original for backward compatibility

        if not enable_batch_norm:
            self.norm_layer = torch.nn.LayerNorm
        else:
            self.norm_layer = partial(torch.nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.embedding = PTV3_Embedding(
            input_dim, enc_channels[0], norm_layer_module=self.norm_layer, embedding_mode=embedding_mode
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
                # Select order_type for this encoder stage using modulo
                stage_order_type = self.order_type_list[i % len(self.order_type_list)]
                self.enc.append(
                    PTV3_Encoder(
                        enc_channels[i],
                        enc_depths[i],
                        enc_num_heads[i],
                        enc_drop_path[sum(enc_depths[:i]) : sum(enc_depths[: i + 1])],
                        proj_drop,
                        patch_size,
                        no_conv_in_cpe,
                        sliding_window_attention,
                        stage_order_type,
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
                # Select order_type for this decoder stage using modulo
                # Use reverse order for decoder (from last encoder stage backwards)
                dec_stage_idx = self.num_stages - 1 - i
                stage_order_type = self.order_type_list[dec_stage_idx % len(self.order_type_list)]
                self.dec.append(
                    PTV3_Encoder(
                        dec_channels[i],
                        dec_depths[i],
                        dec_num_heads[i],
                        dec_drop_path_,
                        proj_drop,
                        patch_size,
                        no_conv_in_cpe,
                        sliding_window_attention,
                        stage_order_type,
                    )
                )

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_Forward")

        grid, feats = self.embedding(grid, feats)

        layer_id = 0
        stack = []
        for i in range(self.num_stages):
            if i > 0:
                nvtx.range_push(f"PTV3_Pooling_{layer_id}")
                stack.append((grid, feats))
                grid, feats = self.enc[layer_id](grid, feats)
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
                last_grid, last_feats = stack.pop()
                grid, feats = self.dec[layer_id](grid, feats, last_grid, last_feats)
                nvtx.range_pop()
                layer_id += 1

                nvtx.range_push(f"PTV3_Decoder_{layer_id}")
                grid, feats = self.dec[layer_id](grid, feats)
                nvtx.range_pop()
                layer_id += 1

        nvtx.range_pop()
        return grid, feats
