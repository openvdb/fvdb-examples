# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from enum import Enum

import torch
from fvdb import GridBatch, JaggedTensor


class VoxelStatus(Enum):
    """
    Status of a voxel in the structure pruning stage.
    """

    # The voxel should not exist (pruned)
    VS_NON_EXIST = 0

    # The voxel exists, but would not have children
    VS_EXIST_STOP = 1

    # The voxel exists, and would have children
    VS_EXIST_CONTINUE = 2


class SparseFeatureHierarchy:
    """A hierarchy of grid batch, where voxel corners align with the origin"""

    def __init__(self, voxel_size: float, depth: int, device):
        self.device = device
        self.voxel_size = voxel_size
        self.depth = depth
        self.grids: list[GridBatch] = [
            GridBatch.from_zero_voxels(
                self.device,
                self.get_voxel_size(d),
                self.get_origin(d),
            )
            for d in range(self.depth)
        ]

    def __repr__(self):
        text = f"SparseFeatureHierarchy - {self.depth} layers, Voxel size = {self.voxel_size}"
        text += "\n"
        for d, d_grid in enumerate(self.grids):
            text += f"\t[{d} {d_grid.num_voxels} voxels]"
        return text + "\n"

    def get_voxel_size(self, depth: int) -> float:
        return self.voxel_size * (2**depth)

    def get_origin(self, depth: int) -> float:
        return 0.5 * self.voxel_size * (2**depth)

    def get_voxel_centers(self, depth: int) -> JaggedTensor:
        grid = self.grids[depth]
        return grid.voxel_to_world(grid.ijk.float())

    def get_f_bound(self) -> tuple[JaggedTensor, JaggedTensor]:
        grid = self.grids[self.depth - 1]
        grid_coords = grid.ijk.float()
        min_extent = grid.voxel_to_world(grid_coords.jmin(dim=0)[0] - 1.5)
        max_extent = grid.voxel_to_world(grid_coords.jmax(dim=0)[0] + 1.5)
        return min_extent, max_extent

    def evaluate_voxel_status(self, grid: GridBatch, depth: int):
        """
        Evaluate the voxel status of given coordinates
        :param grid: Featuregrid Grid
        :param depth: int
        :return: (N, ) byte tensor, with value 0,1,2
        """
        status = torch.full((grid.num_voxels,), VoxelStatus.VS_NON_EXIST.value, dtype=torch.uint8, device=self.device)

        if self.grids[depth] is not None:
            exist_idx = grid.ijk_to_index(self.grids[depth].ijk)
            status[exist_idx[exist_idx != -1]] = VoxelStatus.VS_EXIST_STOP.value

            if depth > 0 and self.grids[depth - 1] is not None:
                child_coords = torch.div(self.grids[depth - 1].ijk, 2, rounding_mode="floor")
                child_idx = grid.ijk_to_index(child_coords)
                status[child_idx[child_idx != -1]] = VoxelStatus.VS_EXIST_CONTINUE.value

        return status

    def get_test_grid(self, depth: int = 0, resolution: int = 2):
        grid = self.grids[depth]
        assert grid is not None
        primal_coords = grid.ijk.float()
        box_coords = torch.linspace(-0.5, 0.5, resolution, device=self.device)
        box_coords = torch.stack(torch.meshgrid(box_coords, box_coords, box_coords, indexing="ij"), dim=3)
        box_coords = box_coords.view(-1, 3)
        query_pos = primal_coords.unsqueeze(1) + box_coords.unsqueeze(0)
        query_pos = grid.voxel_to_world(query_pos.view(-1, 3))
        return query_pos, primal_coords

    def to_(self, device: torch.device | str):
        device = torch.device(device)
        if device == self.device:
            return
        self.device = device
        self.grids = [v.to(device) if v is not None else None for v in self.grids]

    def build_iterative_coarsening(self, pts: JaggedTensor):
        assert pts.device == self.device, f"Device not match {pts.device} vs {self.device}."
        self.grids[0] = GridBatch.from_points(
            pts, voxel_sizes=self.get_voxel_size(0), origins=self.get_origin(0), device=self.device
        )
        for d in range(1, self.depth):
            self.grids[d] = self.grids[d - 1].coarsened_grid(2)

    def build_point_splatting(self, pts: JaggedTensor):
        assert pts.device == self.device, f"Device not match {pts.device} vs {self.device}."
        for d in range(self.depth):
            self.grids[d] = GridBatch.from_nearest_voxels_to_points(
                pts, voxel_sizes=self.get_voxel_size(d), origins=self.get_origin(d), device=self.device
            )
