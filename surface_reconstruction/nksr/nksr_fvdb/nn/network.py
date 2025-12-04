# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn

from nksr_fvdb.nn.encdec import MultiscalePointDecoder, PointEncoder
from nksr_fvdb.nn.unet import SparseStructureNet


class NKSRNetwork(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = PointEncoder(dim=6)

        self.sdf_decoder = MultiscalePointDecoder(
            c_each_dim=self.hparams.kernel_dim, multiscale_depths=self.hparams.tree_depth, coords_depths=[2, 3]
        )
        normal_channels = 3

        # self.unet = SparseStructureNet(
        #     in_channels=32,
        #     num_blocks=self.hparams.tree_depth,
        #     basis_channels=self.hparams.kernel_dim,
        #     normal_channels=normal_channels,
        #     f_maps=self.hparams.unet.f_maps,
        #     udf_branch_dim=16 if self.hparams.udf.enabled else 0,
        # )

        if self.hparams.udf.enabled:
            self.udf_decoder = MultiscalePointDecoder(
                c_each_dim=16,
                multiscale_depths=self.hparams.tree_depth,
                out_init=5 * self.hparams.voxel_size,
                coords_depths=[2, 3],
            )
