# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch_scatter
from fvdb import GridBatch, JaggedTensor


class VoxelStatus(Enum):
    # Voxel Status: 0-NE, 1-E&-, 2-E&v
    VS_NON_EXIST = 0
    VS_EXIST_STOP = 1
    VS_EXIST_CONTINUE = 2


class SparseFeatureHierarchy:
    """A hierarchy of grid batch, where voxel corners align with the origin"""

    def __init__(self, voxel_size: float, depth: int, device):
        self.device = device
        self.voxel_size = voxel_size
        self.depth = depth
        self.grids: List[Optional[GridBatch]] = [None for _ in range(self.depth)]

    def __repr__(self):
        text = f"SparseFeatureHierarchy - {self.depth} layers, Voxel size = {self.voxel_size}"
        text += "\n"
        for d, d_grid in enumerate(self.grids):
            if d_grid is None:
                text += f"\t[{d} empty]"
            else:
                text += f"\t[{d} {d_grid.num_voxels} voxels]"
        return text + "\n"

    def get_voxel_size(self, depth: int) -> dict[str, float]:
        return self.voxel_size * (2**depth)

    def get_origin(self, depth: int) -> float:
        return 0.5 * self.voxel_size * (2**depth)

    def get_voxel_centers(self, depth: int):
        grid = self.grids[depth]
        if grid is None:
            return torch.zeros((0, 3), device=self.device)
        return grid.grid_to_world(grid.active_grid_coords().float())

    def get_f_bound(self):
        grid = self.grids[self.depth - 1]
        grid_coords = grid.active_grid_coords()
        min_extent = grid.grid_to_world(torch.min(grid_coords, dim=0).values.unsqueeze(0) - 1.5)[0]
        max_extent = grid.grid_to_world(torch.max(grid_coords, dim=0).values.unsqueeze(0) + 1.5)[0]
        return min_extent, max_extent

    def evaluate_voxel_status(self, grid, depth: int):
        """
        Evaluate the voxel status of given coordinates
        :param grid: Featuregrid Grid
        :param depth: int
        :return: (N, ) byte tensor, with value 0,1,2
        """
        status = torch.full((grid.num_voxels,), VoxelStatus.VS_NON_EXIST.value, dtype=torch.uint8, device=self.device)

        if self.grids[depth] is not None:
            exist_idx = grid.ijk_to_index(self.grids[depth].active_grid_coords())
            status[exist_idx[exist_idx != -1]] = VoxelStatus.VS_EXIST_STOP.value

            if depth > 0 and self.grids[depth - 1] is not None:
                child_coords = torch.div(self.grids[depth - 1].active_grid_coords(), 2, rounding_mode="floor")
                child_idx = grid.ijk_to_index(child_coords)
                status[child_idx[child_idx != -1]] = VoxelStatus.VS_EXIST_CONTINUE.value

        return status

    def get_test_grid(self, depth: int = 0, resolution: int = 2):
        grid = self.grids[depth]
        assert grid is not None
        primal_coords = grid.active_grid_coords()
        box_coords = torch.linspace(-0.5, 0.5, resolution, device=self.device)
        box_coords = torch.stack(torch.meshgrid(box_coords, box_coords, box_coords, indexing="ij"), dim=3)
        box_coords = box_coords.view(-1, 3)
        query_pos = primal_coords.unsqueeze(1) + box_coords.unsqueeze(0)
        query_pos = grid.grid_to_world(query_pos.view(-1, 3))
        return query_pos, primal_coords

    def get_visualization(self):
        wire_blocks = []
        for d in range(self.depth):
            if self.grids[d] is None:
                continue
            primal_coords = self.grids[d].active_grid_coords()
            is_lowest = len(wire_blocks) == 0
            wire_blocks.append(
                vis.wireframe_bbox(
                    self.grids[d].grid_to_world(primal_coords - (0.45 if is_lowest else 0.5)),
                    self.grids[d].grid_to_world(primal_coords + (0.45 if is_lowest else 0.5)),
                    ucid=d,
                    solid=is_lowest,
                )
            )
        return wire_blocks

    def to_(self, device: torch.device | str):
        device = torch.device(device)
        if device == self.device:
            return
        self.device = device
        self.kernel_maps = {k: v.to(device) for k, v in self.kernel_maps.items()}
        self.grids = [v.to(device) if v is not None else None for v in self.grids]

    def build_iterative_coarsening(self, pts: torch.Tensor):
        assert pts.device == self.device, f"Device not match {pts.device} vs {self.device}."
        self.grids[0] = SparseIndexGrid(*self.get_grid_voxel_size_origin(0), device=self.device)
        self.grids[0].build_from_pointcloud(pts, [0, 0, 0], [0, 0, 0])
        for d in range(1, self.depth):
            self.grids[d] = self.grids[d - 1].coarsened_grid(2)

    def build_point_splatting(self, pts: JaggedTensor):
        assert pts.device == self.device, f"Device not match {pts.device} vs {self.device}."
        for d in range(self.depth):
            self.grids[d] = GridBatch.from_nearest_voxels_to_points(
                pts, voxel_sizes=self.get_voxel_size(d), origins=self.get_origin(d), device=self.device
            )

    def build_adaptive_normal_variation(
        self, pts: torch.Tensor, normal: torch.Tensor, tau: float = 0.2, adaptive_depth: int = 100
    ):
        assert pts.device == normal.device == self.device, "Device not match"
        inv_mapping = None
        for d in range(self.depth - 1, -1, -1):
            # Obtain points & normals for this level
            if inv_mapping is not None:
                nx, ny, nz = torch.abs(normal[:, 0]), torch.abs(normal[:, 1]), torch.abs(normal[:, 2])
                vnx = torch_scatter.scatter_std(nx, inv_mapping)
                vny = torch_scatter.scatter_std(ny, inv_mapping)
                vnz = torch_scatter.scatter_std(nz, inv_mapping)
                pts_mask = ((vnx + vny + vnz) > tau)[inv_mapping]
                pts, normal = pts[pts_mask], normal[pts_mask]

            if pts.size(0) == 0:
                return

            self.grids[d] = SparseIndexGrid(*self.get_grid_voxel_size_origin(d), device=self.device)
            self.grids[d].build_from_pointcloud_nearest_voxels(pts)

            if 0 < d < adaptive_depth:
                inv_mapping = self.grids[d].ijk_to_index(self.grids[d].world_to_grid(pts).round().int())

    def build_from_grid_coords(self, depth: int, grid_coords: torch.Tensor, pad_min: list = None, pad_max: list = None):
        if pad_min is None:
            pad_min = [0, 0, 0]

        if pad_max is None:
            pad_max = [0, 0, 0]

        assert grid_coords.device == self.device, "Device not match"
        assert self.grids[depth] is None, "Grid is not empty"
        self.grids[depth] = SparseIndexGrid(*self.get_grid_voxel_size_origin(depth), device=self.device)
        self.grids[depth].build_from_ijk_coords(grid_coords, pad_min, pad_max)

    # def build_from_grid(self, depth: int, grid: SparseIndexGrid):
    #     assert self.grids[depth] is None, "Grid is not empty"
    #     grid_size, grid_origin = self.get_grid_voxel_size_origin(depth)
    #     assert grid.voxel_size == grid_size, f"Voxel size does not match: {grid.voxel_size} vs {grid_size}!"
    #     assert grid.origin == grid_origin, f"Origin does not match: {grid.origin} vs {grid_origin}!"
    #     self.grids[depth] = grid
