# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from abc import ABC
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from nksr_fvdb.svh import SparseFeatureHierarchy


@dataclass
class EvaluationResult:
    value: torch.Tensor
    gradient: torch.Tensor | None = None

    @classmethod
    def zero(cls, grad: bool = False) -> "EvaluationResult":
        return EvaluationResult(torch.tensor(0), torch.tensor(0) if grad else None)

    def __add__(self, other):
        assert isinstance(other, EvaluationResult)
        return EvaluationResult(
            self.value + other.value, (self.gradient + other.gradient) if self.gradient is not None else None
        )

    def __sub__(self, other):
        assert not isinstance(other, EvaluationResult)
        return EvaluationResult(self.value - other, self.gradient)


@dataclass(kw_only=True, slots=True)
class MeshingResult:
    vertices: torch.Tensor
    faces: torch.Tensor
    colors: torch.Tensor | None = None


class BaseField(ABC):
    """
    Base class for the 3D continuous field:
        f_bar = f - level_set
    """

    def __init__(self, svh: Optional[SparseFeatureHierarchy]):
        self.svh = svh
        self.scale = 1.0
        self.mask_field = None
        self.texture_field = None
        self.set_level_set(0.0)

    def set_level_set(self, level_set: float):
        self.level_set = level_set

    def set_scale(self, scale: float):
        self.scale = scale
        if self.mask_field is not None:
            self.mask_field.set_scale(scale)
        if self.texture_field is not None:
            self.texture_field.set_scale(scale)

    def to_(self, device: Union[torch.device, str]):
        if self.svh is not None:
            self.svh.to_(device)
        if self.mask_field is not None:
            self.mask_field.to_(device)
        if self.texture_field is not None:
            self.texture_field.to_(device)

    @property
    def device(self):
        return self.svh.device

    def evaluate_f(self, xyz: torch.Tensor, grad: bool = False):
        pass

    def evaluate_f_bar(self, xyz: torch.Tensor, max_points: int = -1, verbose: bool = True):
        n_chunks = int(np.ceil(xyz.size(0) / max_points)) if max_points > 0 else 1
        xyz_chunks = torch.chunk(xyz, n_chunks)
        f_bar_chunks = []

        if verbose and len(xyz_chunks) > 10:
            from tqdm import tqdm

            xyz_chunks = tqdm(xyz_chunks)

        for xyz_chunk in xyz_chunks:
            if self.scale != 1.0:
                xyz_chunk = xyz_chunk / self.scale
            f_chunk = self.evaluate_f(xyz_chunk, grad=False).value
            f_bar_chunks.append(f_chunk - self.level_set)

        return torch.cat(f_bar_chunks)

    def extract_primal_mesh(self, depth: int, resolution: int = 2, trim: bool = True, max_points: int = -1):
        primal_grid = self.svh.grids[depth]
        primal_grid_dense = primal_grid.subdivided_grid(
            resolution, torch.ones(primal_grid.num_voxels, dtype=bool, device=self.svh.device)
        )
        dual_grid_dense = primal_grid_dense.dual_grid()

        dual_graph = meshing.primal_cube_graph(primal_grid_dense, dual_grid_dense)
        dual_corner_pos = dual_grid_dense.grid_to_world(dual_grid_dense.active_grid_coords().float())
        if self.scale != 1.0:
            dual_corner_pos = dual_corner_pos * self.scale
        dual_corner_value = self.evaluate_f_bar(dual_corner_pos, max_points=max_points)

        primal_v, primal_f = MarchingCubes().apply(dual_graph, dual_corner_pos, dual_corner_value)

        if self.mask_field is not None and trim:
            vert_mask = self.mask_field.evaluate_f_bar(primal_v, max_points=max_points) < 0.0
            primal_v, primal_f = utils.apply_vertex_mask(primal_v, primal_f, vert_mask)

        if self.texture_field is not None:
            primal_c = self.texture_field.evaluate_f_bar(primal_v, max_points=max_points)
        else:
            primal_c = None

        return MeshingResult(primal_v, primal_f, primal_c)
