# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple

# Add NVTX import for profiling
import flash_attn
import torch
import torch.nn
import torch.nn.functional as F
from timm.layers import DropPath

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

    def __init__(self, in_channels, embed_channels):
        """
        Args:
            in_channels (int): Number of channels in the input features.
            embed_channels (int): Number of channels in the output features.
        """
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, embed_channels)
        self.norm = torch.nn.LayerNorm(embed_channels)
        self.act_layer = torch.nn.GELU()

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_Embedding")
        jfeats = feats.jdata
        jfeats = self.linear(jfeats)
        jfeats = self.norm(jfeats)
        jfeats = self.act_layer(jfeats)

        feats = feats.jagged_like(jfeats)
        nvtx.range_pop()
        return grid, feats


class PTV3_Pooling(torch.nn.Module):
    def __init__(self, kernel_size: int = 2, in_channels: int = 64, out_channels: int = 64):
        """
        Args:
            kernel_size (int): Kernel size for the pooling operation.
            in_channels (int): Number of channels in the input features.
            out_channels (int): Number of channels in the output features.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.proj = torch.nn.Linear(in_channels, out_channels)
        self.ln_layer = torch.nn.LayerNorm(out_channels)
        self.act_layer = torch.nn.GELU()

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_Pooling")
        feats_j = self.proj(feats.jdata)
        feats = feats.jagged_like(feats_j)

        ds_feature, ds_grid = grid.max_pool(self.kernel_size, feats, stride=self.kernel_size, coarse_grid=None)
        ds_feature_j = ds_feature.jdata
        ds_feature_j = self.ln_layer(ds_feature_j)
        ds_feature_j = self.act_layer(ds_feature_j)
        ds_feature = ds_feature.jagged_like(ds_feature_j)
        nvtx.range_pop()
        ds_grid.kmap = None
        return ds_grid, ds_feature


class PTV3_Unpooling(torch.nn.Module):
    def __init__(self, kernel_size: int = 2, in_channels: int = 64, out_channels: int = 64, skip_channels: int = 64):
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
        self.norm = torch.nn.LayerNorm(out_channels)
        self.act_layer = torch.nn.GELU()
        self.proj_skip = torch.nn.Linear(skip_channels, out_channels)
        self.norm_skip = torch.nn.LayerNorm(out_channels)
        self.act_layer_skip = torch.nn.GELU()

    def forward(self, grid, feats, last_grid, last_feats):

        feats_j = self.proj(feats.jdata)
        # BUG: When enabled AMP within Pointcept training pipeline, despite both the input and weights are float32, the output becomes float16.
        feats_j = self.norm(feats_j)
        feats_j = self.act_layer(feats_j)

        last_feats_j = self.proj_skip(last_feats.jdata)
        last_feats_j = self.norm_skip(last_feats_j)
        last_feats_j = self.act_layer_skip(last_feats_j)

        feats, _ = grid.subdivide(self.kernel_size, grid.jagged_like(feats_j), fine_grid=last_grid)
        feats_j = feats.jdata

        new_feats_j = last_feats_j + feats_j
        last_grid.kmap = None  # the topology of the last grid is not valid anymore.
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
        self.drop = torch.nn.Dropout(proj_drop)  # simplified setting: no dropout now.

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
        cross_patch_attention: bool = False,
        cross_patch_pooling: str = "mean",
        sliding_window_attention: bool = False,
    ):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            num_heads (int): Number of attention heads in each block.
            proj_drop (float): Dropout rate for MLP layers.
            patch_size (int): Patch size for patch attention.
            cross_patch_attention (bool): Whether to use cross-patch attention.
            cross_patch_pooling (str): Pooling method for cross-patch attention ("mean" or "max").
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
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

        self.cross_patch_attention = cross_patch_attention
        self.cross_patch_pooling = cross_patch_pooling  # "mean" or "max"

        # Sliding window attention parameter
        self.sliding_window_attention = sliding_window_attention

    def forward(self, grid, feats):
        nvtx.range_push("PTV3_Attention")
        feats_j = feats.jdata

        qkv = self.qkv(feats_j)  # (num_voxels, 3 * hidden_size)

        if self.sliding_window_attention and self.patch_size > 0:
            # Perform sliding window attention using flash attention
            num_voxels = feats_j.shape[0]
            qkv = qkv.view(1, num_voxels, 3, self.num_heads, self.head_dim)  # (1, num_voxels, 3, num_heads, head_dim)

            window_size = (self.patch_size // 2, self.patch_size // 2)

            feats_out_j = flash_attn.flash_attn_qkvpacked_func(
                qkv.half(), dropout_p=0.0, softmax_scale=1.0, window_size=window_size
            ).reshape(num_voxels, self.hidden_size)

            feats_out_j = feats_out_j.to(feats_j.dtype)

        elif self.patch_size > 0:
            # Perform attention within each patch_size window.
            num_voxels = feats_j.shape[0]
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)  # (num_voxels, 3, num_heads, head_dim)
            cu_seqlens = torch.cat(
                [
                    torch.arange(0, num_voxels, self.patch_size, device=qkv.device, dtype=torch.int32),
                    torch.tensor([num_voxels], device=qkv.device, dtype=torch.int32),
                ]
            )

            feats_out_j = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half(), cu_seqlens, max_seqlen=self.patch_size, dropout_p=0.0, softmax_scale=1.0
            ).reshape(num_voxels, self.hidden_size)

            if self.cross_patch_attention:
                num_complete_patches = num_voxels // self.patch_size
                remaining_voxels = num_voxels % self.patch_size

                if num_complete_patches > 1:  # Only do cross-patch if we have multiple patches
                    complete_voxels = num_complete_patches * self.patch_size
                    qkv_complete = qkv[:complete_voxels]  # (complete_voxels, 3, num_heads, head_dim)

                    qkv_patches = qkv_complete.view(
                        num_complete_patches, self.patch_size, 3, self.num_heads, self.head_dim
                    )

                    if self.cross_patch_pooling == "mean":
                        qkv_pooled = qkv_patches.mean(dim=1)  # (num_complete_patches, 3, num_heads, head_dim)
                    elif self.cross_patch_pooling == "max":
                        qkv_pooled = qkv_patches.max(dim=1)[0]  # (num_complete_patches, 3, num_heads, head_dim)
                    else:
                        raise ValueError(f"Unsupported pooling method: {self.cross_patch_pooling}")

                    if remaining_voxels > 0:
                        qkv_remaining = qkv[complete_voxels:]  # (remaining_voxels, 3, num_heads, head_dim)
                        if self.cross_patch_pooling == "mean":
                            qkv_remaining_pooled = qkv_remaining.mean(
                                dim=0, keepdim=True
                            )  # (1, 3, num_heads, head_dim)
                        else:  # max pooling
                            qkv_remaining_pooled = qkv_remaining.max(dim=0, keepdim=True)[
                                0
                            ]  # (1, 3, num_heads, head_dim)
                        qkv_pooled = torch.cat(
                            [qkv_pooled, qkv_remaining_pooled], dim=0
                        )  # (num_complete_patches + 1, 3, num_heads, head_dim)
                        num_total_patches = num_complete_patches + 1
                    else:
                        num_total_patches = num_complete_patches

                    qkv_pooled_unsqueezed = qkv_pooled.unsqueeze(0)
                    cross_attn_out = flash_attn.flash_attn_qkvpacked_func(
                        qkv_pooled_unsqueezed.half(), dropout_p=0.0, softmax_scale=1.0
                    ).reshape(num_total_patches, self.hidden_size)

                    cross_attn_complete = cross_attn_out[:num_complete_patches]
                    cross_attn_expanded = cross_attn_complete.unsqueeze(1).expand(
                        -1, self.patch_size, -1
                    )  # (num_complete_patches, patch_size, hidden_size)
                    cross_attn_flat = cross_attn_expanded.reshape(
                        complete_voxels, self.hidden_size
                    )  # (complete_voxels, hidden_size)

                    cross_attn_all = torch.zeros_like(feats_out_j)
                    cross_attn_all[:complete_voxels] = cross_attn_flat.to(feats_out_j.dtype)
                    if remaining_voxels > 0:
                        cross_attn_all[complete_voxels:] = cross_attn_out[-1].unsqueeze(0).expand(remaining_voxels, -1)

                    feats_out_j = feats_out_j + cross_attn_all

            feats_out_j = feats_out_j.to(feats_j.dtype)
        else:
            assert False, "Only sliding window attention and patch attention are supported now. "

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
        # Wrap components in Sequential to match parameter naming convention
        self.cpe = torch.nn.ModuleList(
            [
                (
                    fvdb.nn.SparseConv3d(hidden_size, hidden_size, kernel_size=3, stride=1)
                    if not no_conv_in_cpe
                    else torch.nn.Identity()
                ),  # cpe.0
                torch.nn.Linear(hidden_size, hidden_size),  # cpe.1
                torch.nn.LayerNorm(hidden_size),  # cpe.2
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
        cross_patch_attention: bool = False,
        cross_patch_pooling: str = "mean",
        sliding_window_attention: bool = False,
    ):
        """
        Args:
            hidden_size (int): Number of channels in the input features.
            num_heads (int): Number of attention heads in each block.
            drop_path (float): Drop path rate for regularization.
            proj_drop (float): Dropout rate for MLP layers.
            patch_size (int): Patch size for patch attention.
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            cross_patch_attention (bool): Whether to use cross-patch attention.
            cross_patch_pooling (str): Pooling method for cross-patch attention ("mean" or "max").
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
        """
        super().__init__()
        # one attention and one mlp
        self.cpe = PTV3_CPE(hidden_size, no_conv_in_cpe)
        self.norm1 = torch.nn.Sequential(torch.nn.LayerNorm(hidden_size))  # norm1.0
        self.attn = PTV3_Attention(
            hidden_size,
            num_heads,
            proj_drop,
            patch_size,
            cross_patch_attention,
            cross_patch_pooling,
            sliding_window_attention,
        )
        self.norm2 = torch.nn.Sequential(torch.nn.LayerNorm(hidden_size))  # norm2.0
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
        cross_patch_attention: bool = False,
        cross_patch_pooling: str = "mean",
        sliding_window_attention: bool = False,
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
            cross_patch_attention (bool): Whether to use cross-patch attention.
            cross_patch_pooling (str): Pooling method for cross-patch attention ("mean" or "max").
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
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
                    cross_patch_attention,
                    cross_patch_pooling,
                    sliding_window_attention,
                )
                for i in range(depth)
            ]
        )

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
        no_conv_in_cpe: bool = False,
        cross_patch_attention: bool = False,
        cross_patch_pooling: str = "mean",
        sliding_window_attention: bool = False,
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
            no_conv_in_cpe (bool): Whether to disable convolution in CPE.
            cross_patch_attention (bool): Whether to use cross-patch attention.
            cross_patch_pooling (str): Pooling method for cross-patch attention ("mean" or "max").
            sliding_window_attention (bool): Whether to use sliding window attention (uses patch_size as window size).
        """
        super().__init__()
        self.num_classes = num_classes
        self.drop_path = drop_path
        self.proj_drop = proj_drop
        self.no_conv_in_cpe = no_conv_in_cpe
        self.cross_patch_attention = cross_patch_attention
        self.cross_patch_pooling = cross_patch_pooling
        self.sliding_window_attention = sliding_window_attention

        # sliding_window_attention and cross_patch_attention should not be used together.
        assert not (
            sliding_window_attention and cross_patch_attention
        ), "sliding_window_attention and cross_patch_attention should not be used together."

        self.embedding = PTV3_Embedding(input_dim, enc_channels[0])

        self.num_stages = len(enc_depths)
        self.enc = torch.nn.ModuleList()
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        for i in range(self.num_stages):
            if i > 0:
                self.enc.append(
                    PTV3_Pooling(kernel_size=2, in_channels=enc_channels[i - 1], out_channels=enc_channels[i])
                )
            self.enc.append(
                PTV3_Encoder(
                    enc_channels[i],
                    enc_depths[i],
                    enc_num_heads[i],
                    enc_drop_path[sum(enc_depths[:i]) : sum(enc_depths[: i + 1])],
                    proj_drop,
                    patch_size,
                    no_conv_in_cpe,
                    cross_patch_attention,
                    cross_patch_pooling,
                    sliding_window_attention,
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
                    )
                )
                self.dec.append(
                    PTV3_Encoder(
                        dec_channels[i],
                        dec_depths[i],
                        dec_num_heads[i],
                        dec_drop_path_,
                        proj_drop,
                        patch_size,
                        no_conv_in_cpe,
                        cross_patch_attention,
                        cross_patch_pooling,
                        sliding_window_attention,
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
