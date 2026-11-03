# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Pointcept backbone that runs fvdb's Point Transformer V3.

Registers ``PT-v3fvdb`` with Pointcept's model registry. The backbone converts Pointcept's batched
points into an :class:`fvdb.GridBatch`, runs :class:`fvdb.nn.PointTransformerV3`, and returns
per-point features. Its arguments are those of ``fvdb.nn.PointTransformerV3``, which follow
Pointcept's ``PT-v3m1``, so a Pointcept config switches models by changing ``type``.
"""

from __future__ import annotations

import fvdb
import fvdb.nn as fvnn
import torch
from pointcept.models.builder import MODELS
from pointcept.models.modules import PointModule


def create_grid_from_points(
    grid_coord: torch.Tensor, feat: torch.Tensor, offset: torch.Tensor
) -> tuple[fvdb.GridBatch, fvdb.JaggedTensor, torch.Tensor]:
    """Build a grid batch from Pointcept's batched points.

    Args:
        grid_coord (torch.Tensor): Integer voxel coordinates of every point. Shape: ``(N, 3)``.
        feat (torch.Tensor): Point features. Shape: ``(N, C)``.
        offset (torch.Tensor): Cumulative point counts per sample, as Pointcept stores them. Shape: ``(B,)``.

    Returns:
        grid (fvdb.GridBatch): One grid per sample.
        voxel_feat (fvdb.JaggedTensor): Features in the grid's voxel order. Points that share a
            voxel keep one of their features.
        point_to_voxel (torch.Tensor): Voxel index of every input point. Shape: ``(N,)``.
    """
    offsets = torch.cat([offset.new_zeros(1), offset]).to(device=grid_coord.device, dtype=torch.int64)
    coords = fvdb.JaggedTensor.from_data_and_offsets(grid_coord.to(torch.int32), offsets)
    feats = fvdb.JaggedTensor.from_data_and_offsets(feat.to(torch.float32), offsets)
    grid = fvdb.GridBatch.from_ijk(coords)
    voxel_feat = grid.inject_from_ijk(coords, feats)
    point_to_voxel = grid.ijk_to_index(coords, cumulative=True).jdata
    return grid, voxel_feat, point_to_voxel


@MODELS.register_module("PT-v3fvdb")
class PointTransformerV3(PointModule):
    """Pointcept backbone wrapping :class:`fvdb.nn.PointTransformerV3`.

    Args:
        run_in_fp32 (bool): Disable autocast around the fvdb model. Attention still runs in
            bfloat16 inside fvdb. Default: ``True``.
        **kwargs: Arguments of :class:`fvdb.nn.PointTransformerV3`. To serialize the raw
            ``grid_coord`` as Pointcept does, pass ``serialization_offset="none"``. To reproduce
            a checkpoint trained with Pointcept, also pass ``pointcept_unpool_cpe=True``.
    """

    def __init__(self, run_in_fp32: bool = True, **kwargs) -> None:
        super().__init__()
        if kwargs.get("cls_mode", False):
            raise ValueError("PT-v3fvdb returns per-point features; cls_mode is not supported")
        self.run_in_fp32 = run_in_fp32
        self.model = fvnn.PointTransformerV3(**kwargs)

    def forward(self, data_dict: dict) -> torch.Tensor:
        """Run the backbone on a Pointcept batch.

        Args:
            data_dict (dict): Pointcept batch with ``grid_coord``, ``feat`` and ``offset``.

        Returns:
            torch.Tensor: Features for every input point. Shape: ``(N, dec_channels[0])``.
        """
        grid, voxel_feat, point_to_voxel = create_grid_from_points(
            data_dict["grid_coord"], data_dict["feat"], data_dict["offset"]
        )
        if self.run_in_fp32 and torch.is_autocast_enabled():
            with torch.autocast(device_type="cuda", enabled=False):
                out, _ = self.model(voxel_feat, grid)
        else:
            out, _ = self.model(voxel_feat, grid)
        return out.jdata[point_to_voxel]
