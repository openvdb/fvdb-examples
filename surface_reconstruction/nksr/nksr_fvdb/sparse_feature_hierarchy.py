# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import Enum

import torch
from fvdb import GridBatch, JaggedTensor

from .coord_xform import CoordXform, voxel_center_aligned_coarsening_xform


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


def evaluate_voxel_status(target_grid: GridBatch, coarse_grid: GridBatch, fine_grid: GridBatch | None) -> torch.Tensor:
    device = target_grid.device
    if coarse_grid.device != device:
        raise ValueError(f"Device not match {device} vs {coarse_grid.device}.")

    if fine_grid is not None and fine_grid.device != device:
        raise ValueError(f"Device not match {device} vs {fine_grid.device}.")

    # Start with a flat tensor of all voxels in the target grid, initialized to NON_EXIST.
    status = torch.full((target_grid.total_voxels,), VoxelStatus.VS_NON_EXIST.value, dtype=torch.uint8, device=device)

    # Set the intersection of the coarse grid and the target grid to EXIST_STOP.
    coarse_exist_idx = target_grid.ijk_to_index(coarse_grid.ijk, cumulative=True)
    coarse_exist_mask = coarse_exist_idx.jdata != -1
    status[coarse_exist_mask] = VoxelStatus.VS_EXIST_STOP.value

    # If there is a fine grid, set the intersection of the fine grid and the target grid to EXIST_CONTINUE.
    # This can overwrite the coarse grid's EXIST_STOP status.
    if fine_grid is not None:
        fine_coarsened = fine_grid.coarsened_grid(coarsening_factor=2)
        fine_exist_idx = target_grid.ijk_to_index(fine_coarsened.ijk, cumulative=True)
        fine_exist_mask = fine_exist_idx.jdata != -1
        status[fine_exist_mask] = VoxelStatus.VS_EXIST_CONTINUE.value

    return status


@dataclass(frozen=True)
class SparseFeatureLevel:
    grid: GridBatch
    world_T_voxel: CoordXform
    pseudo_voxel_size: float = field(init=False)

    def __post_init__(self) -> None:
        """Cache the pseudo voxel size from the transform's pseudo_scaling_factor.

        This value is cached because the computation requires tensor operations
        and we expect it to be accessed frequently (e.g., for display, for
        choosing algorithm parameters).
        """
        # Use object.__setattr__ because this is a frozen dataclass
        object.__setattr__(self, "pseudo_voxel_size", self.world_T_voxel.pseudo_scaling_factor)

    def get_test_grid(self, resolution: int = 2) -> tuple[JaggedTensor, JaggedTensor]:
        # The primal coords are the voxel centers of the grid batch.
        primal_coords: JaggedTensor = self.grid.ijk.float()

        # The box coords are the coordinates of the box centers, a pattern of sampling around
        # each primal coord.
        box_coords = torch.linspace(-0.5, 0.5, resolution, device=self.device)  # [R]
        box_coords = torch.stack(
            torch.meshgrid(box_coords, box_coords, box_coords, indexing="ij"), dim=3
        )  # [R, R, R, 3]
        box_coords = box_coords.view(-1, 3)  # [R^3, 3]
        queries_per_primal = box_coords.shape[0]
        assert queries_per_primal == resolution**3

        # Get the query positions in voxel space
        # primal_coords jdata unsqueeze(1): [N, 1,   3]
        # box coords unsqueeze(0):          [1, R^3, 3]
        # addition and .view(-1, 3):       [N * R^3, 3]
        query_voxel_flat = (primal_coords.jdata.unsqueeze(1) + box_coords.unsqueeze(0)).view(-1, 3)

        # make them jagged.
        query_voxel = JaggedTensor.from_data_and_offsets(
            query_voxel_flat, offsets=primal_coords.joffsets * queries_per_primal
        )

        # Transform the points to world space
        query_world: JaggedTensor = self.world_T_voxel @ query_voxel

        return query_world, primal_coords

    def __str__(self) -> str:
        return f"SparseFeatureLevel: {self.grid.total_voxels} voxels, pseudo_voxel_size={self.pseudo_voxel_size:.4g}"

    @property
    def bounds_voxel(self) -> JaggedTensor:
        """
        The bboxes returned by gridbatch normally are a regular tensor, but because
        of how the coord xform works, it might want to have a different transform per
        batch index. Therefore, we need to make sure that "per batch" things are always
        presented as jagged tensors. In this case, we can't just wrap the tensor, it
        will interpret that as a batch size of 1. Instead, we create an offsets tensor
        that's just arange.
        """
        offsets = torch.arange(0, self.grid.grid_count, dtype=torch.int32, device=self.grid.device)
        return JaggedTensor.from_data_and_offsets(self.grid.bboxes, offsets)

    @property
    def bounds_world(self) -> JaggedTensor:
        return self.world_T_voxel.apply_bounds(self.bounds_voxel)

    @property
    def device(self) -> torch.device:
        return self.grid.device


@dataclass(frozen=True)
class SparseFeatureHierarchy:
    levels: list[SparseFeatureLevel]

    @classmethod
    def from_refinement(
        cls, finest_level: SparseFeatureLevel, base: "SparseFeatureHierarchy"
    ) -> "SparseFeatureHierarchy":
        return cls(levels=[finest_level] + base.levels)

    @classmethod
    def from_iterative_coarsening(
        cls, world_points: JaggedTensor, world_T_voxel: CoordXform, depth: int, coarsening_factor: int = 2
    ) -> "SparseFeatureHierarchy":

        voxel_T_world = world_T_voxel.inverse()
        voxel_points = voxel_T_world @ world_points
        levels: list[SparseFeatureLevel] = [SparseFeatureLevel(GridBatch.from_points(voxel_points), world_T_voxel)]
        fine_T_coarse = voxel_center_aligned_coarsening_xform(coarsening_factor)
        for d in range(1, depth):
            coarsened_grid = levels[-1].grid.coarsened_grid(coarsening_factor)
            coarsened_xform = levels[-1].world_T_voxel @ fine_T_coarse
            levels.append(SparseFeatureLevel(coarsened_grid, coarsened_xform))

        return cls(levels=levels)

    @classmethod
    def from_point_splatting(
        cls, world_points: JaggedTensor, world_T_voxel: CoordXform, depth: int, coarsening_factor: int = 2
    ) -> "SparseFeatureHierarchy":

        voxel_T_world = world_T_voxel.inverse()
        voxel_points = voxel_T_world @ world_points
        levels: list[SparseFeatureLevel] = [
            SparseFeatureLevel(GridBatch.from_nearest_voxels_to_points(voxel_points), world_T_voxel)
        ]

        fine_T_coarse = voxel_center_aligned_coarsening_xform(coarsening_factor)
        for d in range(1, depth):
            world_T_voxel_d = levels[-1].world_T_voxel @ fine_T_coarse
            voxel_T_world_d = world_T_voxel_d.inverse()
            voxel_points_d = voxel_T_world_d @ world_points
            levels.append(SparseFeatureLevel(GridBatch.from_nearest_voxels_to_points(voxel_points_d), world_T_voxel_d))

        return cls(levels=levels)

    def evaluate_voxel_status(self, target_grid: GridBatch, coarse_depth: int = 0) -> torch.Tensor:
        if coarse_depth < 0 or coarse_depth >= self.depth:
            raise ValueError(f"Invalid coarse depth: {coarse_depth}. Must be between 0 and {self.depth - 1}.")

        if coarse_depth == 0:
            return evaluate_voxel_status(target_grid, self.levels[0].grid, None)
        else:
            return evaluate_voxel_status(
                target_grid, self.levels[coarse_depth].grid, self.levels[coarse_depth - 1].grid
            )

    @property
    def pseudo_voxel_size(self) -> float:
        """Pseudo voxel size at the finest level (level 0)."""
        return self.levels[0].pseudo_voxel_size

    @property
    def grids(self) -> list[GridBatch]:
        """Convenience property to access grids from all levels."""
        return [level.grid for level in self.levels]

    @property
    def depth(self) -> int:
        return len(self.levels)

    @property
    def bounds_voxel(self) -> JaggedTensor:
        return self.levels[0].bounds_voxel

    @property
    def bounds_world(self) -> JaggedTensor:
        return self.levels[0].bounds_world

    @property
    def device(self) -> torch.device:
        return self.levels[0].device

    def __str__(self) -> str:
        """Pretty-print representation with detailed per-level information."""
        lines = [f"SparseFeatureHierarchy - {self.depth} levels, pseudo_voxel_size={self.pseudo_voxel_size:.4g}"]
        for d, level in enumerate(self.levels):
            lines.append(f"\t[{d}] {level.grid.total_voxels} voxels")
        return "\n".join(lines)
