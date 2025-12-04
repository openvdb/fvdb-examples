# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import logging
import math

import torch
import torch.nn as nn
from fvdb import JaggedTensor
from fvdb.nn.simple_unet import SimpleUNet

from nksr_fvdb.fields import BaseField, LayerField, NeuralField
from nksr_fvdb.nn.network import NKSRNetwork
from nksr_fvdb.presets import all_presets
from nksr_fvdb.svh import SparseFeatureHierarchy

logger = logging.getLogger(__name__)


class Reconstructor:
    """
    Main Reconstructor class that reconstructs an implicit field from a point cloud.
    """

    def __init__(self, device: torch.device | str, preset: str = "ks"):
        """
        Args:
            device (torch.device): device to run the reconstructor on.
            preset (str): name of the reconstructor preset.
        """
        self.device = device
        self.chunk_tmp_device = self.device
        self.hparams = all_presets[preset]
        self.network = NKSRNetwork(self.hparams).to(self.device).eval().requires_grad_(False)
        # ckpt_data = torch.hub.load_state_dict_from_url(self.hparams.url)
        # self.network.load_state_dict(ckpt_data["state_dict"])

        self.simple_unet = (
            SimpleUNet(
                in_channels=32,
                base_channels=32,
                out_channels=32,
                channel_growth_rate=2,
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )

    def reconstruct(
        self,
        xyz: torch.Tensor | JaggedTensor,
        normal: torch.Tensor | JaggedTensor,
        voxel_size: float,
    ) -> BaseField:
        """

        Args:
            xyz (torch.Tensor): (N, 3) input point positions
            normal (torch.Tensor): (N, 3) input point normals
            voxel_size (float): the voxel size of the input point cloud to use

        Returns:
            field (Field): the implicit field to extract mesh from.
        """
        if isinstance(xyz, torch.Tensor):
            xyz = JaggedTensor.from_tensor(xyz)

        if isinstance(normal, torch.Tensor):
            normal = JaggedTensor.from_tensor(normal)

        if (global_scale := voxel_size / self.hparams.voxel_size) != 1.0:
            xyz = xyz / global_scale

        svh = SparseFeatureHierarchy(
            voxel_size=self.hparams.voxel_size, depth=self.hparams.tree_depth, device=self.device
        )
        svh.build_point_splatting(xyz)
        feat = self.network.encoder(xyz, normal, svh, 0)

        # Choose between our customized implementation and simple unet:
        # feat, svh, udf_svh = self.network.unet(feat, svh, adaptive_depth=self.hparams.adaptive_depth)
        feat = self.simple_unet(feat, grid=svh.grids[0])

        output_field = NeuralField(svh=svh, decoder=self.network.sdf_decoder, features=feat.basis_features)

        if self.hparams.udf.enabled:
            mask_field = NeuralField(svh=udf_svh, decoder=self.network.udf_decoder, features=feat.udf_features)
            mask_field.set_level_set(2 * self.hparams.voxel_size)
        else:
            mask_field = LayerField(svh, self.hparams.adaptive_depth)

        output_field.mask_field = mask_field
        output_field.set_scale(global_scale)

        return output_field
