# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import List, Tuple

import torch

import fvdb
import fvdb.nn
from fvdb import ConvolutionPlan, GridBatch, JaggedTensor

from .blocks import BasicConvolutionBlock, BasicDeconvolutionBlock, ResidualBlock


class MaskPLSEncoderDecoder(torch.nn.Module):

    channels = [32, 32, 64, 128, 256, 256, 128, 96, 96]

    def __init__(
        self,
        input_dim: int = 4,
        stem_blocks: int = 1,
        output_feature_levels: List[int] = [3],
        bn_momentum: float = 0.02,
    ):
        super().__init__()
        self.output_feature_levels = output_feature_levels
        down_res_blocks = [2, 3, 4, 6]

        # Stem: stride=1, ks=3 convolutions
        self.stem_convs = torch.nn.ModuleList()
        self.stem_bns = torch.nn.ModuleList()
        self.stem_convs.append(fvdb.nn.SparseConv3d(input_dim, self.channels[0], kernel_size=3))
        self.stem_bns.append(fvdb.nn.BatchNorm(self.channels[0], momentum=bn_momentum))
        for _ in range(1, stem_blocks):
            self.stem_convs.append(fvdb.nn.SparseConv3d(self.channels[0], self.channels[0], kernel_size=3))
            self.stem_bns.append(fvdb.nn.BatchNorm(self.channels[0], momentum=bn_momentum))

        # Encoder stages: each starts with a stride-2 downsample then residual blocks
        self.stage1 = torch.nn.ModuleList([
            BasicConvolutionBlock(self.channels[0], self.channels[0], ks=2, stride=2, bn_mom=bn_momentum),
            ResidualBlock(self.channels[0], self.channels[1], ks=3, bn_mom=bn_momentum),
        ] + [
            ResidualBlock(self.channels[1], self.channels[1], ks=3, bn_mom=bn_momentum)
            for _ in range(1, down_res_blocks[0])
        ])

        self.stage2 = torch.nn.ModuleList([
            BasicConvolutionBlock(self.channels[1], self.channels[1], ks=2, stride=2, bn_mom=bn_momentum),
            ResidualBlock(self.channels[1], self.channels[2], ks=3, bn_mom=bn_momentum),
        ] + [
            ResidualBlock(self.channels[2], self.channels[2], ks=3, bn_mom=bn_momentum)
            for _ in range(1, down_res_blocks[1])
        ])

        self.stage3 = torch.nn.ModuleList([
            BasicConvolutionBlock(self.channels[2], self.channels[2], ks=2, stride=2, bn_mom=bn_momentum),
            ResidualBlock(self.channels[2], self.channels[3], ks=3, bn_mom=bn_momentum),
        ] + [
            ResidualBlock(self.channels[3], self.channels[3], ks=3, bn_mom=bn_momentum)
            for _ in range(1, down_res_blocks[2])
        ])

        self.stage4 = torch.nn.ModuleList([
            BasicConvolutionBlock(self.channels[3], self.channels[3], ks=2, stride=2, bn_mom=bn_momentum),
            ResidualBlock(self.channels[3], self.channels[4], ks=3, bn_mom=bn_momentum),
        ] + [
            ResidualBlock(self.channels[4], self.channels[4], ks=3, bn_mom=bn_momentum)
            for _ in range(1, down_res_blocks[3])
        ])

        # Decoder: each level has a deconv block + residual blocks after skip concatenation
        self.up1_deconv = BasicDeconvolutionBlock(self.channels[4], self.channels[5], ks=2, stride=2, bn_mom=bn_momentum)
        self.up1_res = torch.nn.ModuleList([
            ResidualBlock(self.channels[5] + self.channels[3], self.channels[5], ks=3, bn_mom=bn_momentum),
            ResidualBlock(self.channels[5], self.channels[5], ks=3, bn_mom=bn_momentum),
        ])

        self.up2_deconv = BasicDeconvolutionBlock(self.channels[5], self.channels[6], ks=2, stride=2, bn_mom=bn_momentum)
        self.up2_res = torch.nn.ModuleList([
            ResidualBlock(self.channels[6] + self.channels[2], self.channels[6], ks=3, bn_mom=bn_momentum),
            ResidualBlock(self.channels[6], self.channels[6], ks=3, bn_mom=bn_momentum),
        ])

        self.up3_deconv = BasicDeconvolutionBlock(self.channels[6], self.channels[7], ks=2, stride=2, bn_mom=bn_momentum)
        self.up3_res = torch.nn.ModuleList([
            ResidualBlock(self.channels[7] + self.channels[1], self.channels[7], ks=3, bn_mom=bn_momentum),
            ResidualBlock(self.channels[7], self.channels[7], ks=3, bn_mom=bn_momentum),
        ])

        self.up4_deconv = BasicDeconvolutionBlock(self.channels[7], self.channels[8], ks=2, stride=2, bn_mom=bn_momentum)
        self.up4_res = torch.nn.ModuleList([
            ResidualBlock(self.channels[8] + self.channels[0], self.channels[8], ks=3, bn_mom=bn_momentum),
            ResidualBlock(self.channels[8], self.channels[8], ks=3, bn_mom=bn_momentum),
        ])

        self.mask_feat = fvdb.nn.SparseConv3d(
            self.channels[-1],
            self.channels[-1],
            kernel_size=3,
            stride=1,
        )

    def _run_stage(
        self, stage: torch.nn.ModuleList, data: JaggedTensor, grid: GridBatch
    ) -> Tuple[JaggedTensor, GridBatch]:
        for block in stage:
            data, grid = block(data, grid)
        return data, grid

    def _run_decoder_level(
        self,
        deconv: BasicDeconvolutionBlock,
        res_blocks: torch.nn.ModuleList,
        data: JaggedTensor,
        source_grid: GridBatch,
        skip_data: JaggedTensor,
        skip_grid: GridBatch,
    ) -> Tuple[JaggedTensor, GridBatch]:
        data = deconv(data, source_grid, skip_grid)
        data = fvdb.jcat([data, skip_data], dim=1)
        grid = skip_grid
        for block in res_blocks:
            data, grid = block(data, grid)
        return data, grid

    def forward(self, x) -> List[Tuple[JaggedTensor, GridBatch]]:
        data: JaggedTensor = x["features"]
        grid: GridBatch = x["grid"]

        # Stem: stride=1, ks=3 convolutions (grid topology unchanged)
        stem_plan = ConvolutionPlan.from_grid_batch(kernel_size=3, stride=1, source_grid=grid, target_grid=grid)
        for conv, bn in zip(self.stem_convs, self.stem_bns):
            data = conv(data, stem_plan)
            data = bn(data, grid)
            data = fvdb.relu(data)
        x0_data, x0_grid = data, grid

        # Encoder
        x1_data, x1_grid = self._run_stage(self.stage1, x0_data, x0_grid)
        x2_data, x2_grid = self._run_stage(self.stage2, x1_data, x1_grid)
        x3_data, x3_grid = self._run_stage(self.stage3, x2_data, x2_grid)
        x4_data, x4_grid = self._run_stage(self.stage4, x3_data, x3_grid)

        # Decoder
        y1_data, y1_grid = self._run_decoder_level(
            self.up1_deconv, self.up1_res, x4_data, x4_grid, x3_data, x3_grid
        )
        y2_data, y2_grid = self._run_decoder_level(
            self.up2_deconv, self.up2_res, y1_data, y1_grid, x2_data, x2_grid
        )
        y3_data, y3_grid = self._run_decoder_level(
            self.up3_deconv, self.up3_res, y2_data, y2_grid, x1_data, x1_grid
        )
        y4_data, y4_grid = self._run_decoder_level(
            self.up4_deconv, self.up4_res, y3_data, y3_grid, x0_data, x0_grid
        )

        out_feats = [
            (y1_data, y1_grid),
            (y2_data, y2_grid),
            (y3_data, y3_grid),
            (y4_data, y4_grid),
        ]

        feat_levels = self.output_feature_levels + [3]
        out_feats = [out_feats[i] for i in feat_levels]

        # Apply mask projection conv to the last feature level
        last_data, last_grid = out_feats[-1]
        mask_plan = ConvolutionPlan.from_grid_batch(
            kernel_size=3, stride=1, source_grid=last_grid, target_grid=last_grid
        )
        out_feats[-1] = (self.mask_feat(last_data, mask_plan), last_grid)

        return out_feats
