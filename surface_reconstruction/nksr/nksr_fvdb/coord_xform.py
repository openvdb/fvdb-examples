# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
This module contains the base class for coordinate frame transformations.

The naming convention we strongly advise using for transformations is a little bit
different from that used in OpenGL and computer graphics applications, but instead is inspired
by the notation described in:

Reference:
  Craig, J. J. (2018). Introduction to Robotics: Mechanics and Control
  (4th ed.). Pearson. ISBN: 978-0133489798.

In this notation, the transformation of a point p from frame "input" to frame "output" is written as:

.. code-block:: python
    p_output = output_T_input * p_input

When transformations are composed, the notation allows for easy validation that the composition
is correct and valid. Imagine two intermediate frames, "lower" and "upper", such that a valid
transformation path is input->lower->upper->output. Because we operate right to left in a
transformation chain, the composition of the transformations is:

.. code-block:: python
    p_output = output_T_upper * upper_T_lower * lower_T_input * p_input

A typical scene graph transformation chain might look like:

.. code-block:: python
    p_world = world_T_assembly * assembly_T_part * part_T_object * p_object

and a typical screen projection transformation chain might look like:

.. code-block:: python
    p_raster = raster_T_screen * screen_T_camera * camera_T_world * p_world

These transformations need not necessarily be 4x4 matrices - it is actually rarely the case that
4x4 matrices are used, especially for machine assemblies involving fast rotations and twists. We
leave room for additional transformation variations in the future, and assume that we can perform
optimizations where composed transformations become fused into a single transformation when
appropriate.

Many robotics applications use a pose xform that has a translation and a quaternion for rotation,
though this representation does not capture twists. When representing propellers or fans, an
explicit axis-angle or other form might be used, particularly as it may relate to differentiation.

"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import overload

import fvdb
import torch


class CoordXform(ABC):
    """Abstract base class for coordinate frame transformations.

    Transforms coordinates represented as fvdb.JaggedTensors from one
    coordinate frame to another. Transformations are composable and
    may have inverses.
    """

    @abstractmethod
    def apply_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """Transform coordinates from source frame to target frame.

        Args:
            coords: Tensor of shape [N, 3] containing 3D coordinates.

        Returns:
            Transformed coordinates as a Tensor with the same shape.
        """
        ...

    def apply_jagged(self, coords: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        """Transform coordinates from source frame to target frame.

        Args:
            coords: JaggedTensor of shape [B, Njagged, 3] containing 3D coordinates.

        Returns:
            Transformed coordinates as a JaggedTensor with the same structure.
        """
        return coords.jagged_like(self.apply_tensor(coords.jdata))

    @overload
    def apply(self, coords: torch.Tensor) -> torch.Tensor: ...
    @overload
    def apply(self, coords: fvdb.JaggedTensor) -> fvdb.JaggedTensor: ...

    def apply(self, coords: torch.Tensor | fvdb.JaggedTensor) -> torch.Tensor | fvdb.JaggedTensor:
        """Transform 3D coordinates from source frame to target frame.

        This is the primary entry point for coordinate transformation. It accepts
        either a torch.Tensor or an fvdb.JaggedTensor and returns the same type.

        Args:
            coords: 3D coordinates to transform.
                - torch.Tensor: shape [N, 3] where N is the number of points.
                - fvdb.JaggedTensor: shape [B, Njagged, 3] where B is batch size and Njagged
                  varies per batch element.

        Returns:
            Transformed coordinates with the same type and shape as input.
                - torch.Tensor input -> torch.Tensor output, shape [N, 3]
                - fvdb.JaggedTensor input -> fvdb.JaggedTensor output, shape [B, Njagged, 3]
        """
        if isinstance(coords, torch.Tensor):
            return self.apply_tensor(coords)
        elif isinstance(coords, fvdb.JaggedTensor):
            return self.apply_jagged(coords)
        else:
            raise ValueError(f"Unsupported type: {type(coords)}")

    @overload
    def __call__(self, coords: torch.Tensor) -> torch.Tensor: ...
    @overload
    def __call__(self, coords: fvdb.JaggedTensor) -> fvdb.JaggedTensor: ...

    def __call__(self, coords: torch.Tensor | fvdb.JaggedTensor) -> torch.Tensor | fvdb.JaggedTensor:
        """Enable functor-style usage: xform(coords) is equivalent to xform.apply(coords)."""
        return self.apply(coords)

    def apply_bounds_tensor(self, bounds: torch.Tensor) -> torch.Tensor:
        """Transform bounds from source frame to target frame.

        This is an inclusion function in the sense of interval analysis: given
        axis-aligned bounding boxes, it produces axis-aligned bounding boxes
        that are guaranteed to contain all transformed points from the original
        boxes.

        The default implementation expands each box to its 8 corners, transforms
        them via apply_tensor, and takes the axis-aligned bounding box of the
        result. This is conservative and correct for affine transformations.
        Subclasses with specialized structure (e.g., pure scaling) may override
        for tighter or more efficient bounds.

        Args:
            bounds: Tensor of shape [N, 2, 3] where bounds[:, 0, :] are the
                min corners and bounds[:, 1, :] are the max corners.

        Returns:
            Transformed bounds as a Tensor of shape [N, 2, 3].
        """
        # bounds: [N, 2, 3] where dim 1 is (min=0, max=1)
        N = bounds.shape[0]

        # Generate corner selection indices: 8 corners x 3 dimensions
        # Each row is a binary pattern indicating min (0) or max (1) for x, y, z
        # fmt: off
        corner_idx = torch.tensor([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=torch.long, device=bounds.device)  # [8, 3]
        # fmt: on

        # Build index tensors for advanced indexing: corners[n, c, d] = bounds[n, corner_idx[c, d], d]
        n_idx = torch.arange(N, device=bounds.device).view(N, 1, 1).expand(N, 8, 3)
        c_idx = corner_idx.view(1, 8, 3).expand(N, 8, 3)
        d_idx = torch.arange(3, device=bounds.device).view(1, 1, 3).expand(N, 8, 3)

        corners = bounds[n_idx, c_idx, d_idx]  # [N, 8, 3]

        # Transform all corners at once
        corners_flat = corners.reshape(N * 8, 3)
        transformed_flat = self.apply_tensor(corners_flat)
        transformed = transformed_flat.reshape(N, 8, 3)

        # Compute axis-aligned bounding box of transformed corners
        new_min = transformed.min(dim=1).values  # [N, 3]
        new_max = transformed.max(dim=1).values  # [N, 3]

        return torch.stack([new_min, new_max], dim=1)  # [N, 2, 3]

    def apply_bounds_jagged(self, bounds: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        """Transform axis-aligned bounding boxes stored in a JaggedTensor.

        Args:
            bounds: JaggedTensor of shape [N, 2, 3] containing bounding boxes.

        Returns:
            Transformed bounds as a JaggedTensor with the same structure.
        """
        return bounds.jagged_like(self.apply_bounds_tensor(bounds.jdata))

    @overload
    def apply_bounds(self, bounds: torch.Tensor) -> torch.Tensor: ...
    @overload
    def apply_bounds(self, bounds: fvdb.JaggedTensor) -> fvdb.JaggedTensor: ...

    def apply_bounds(self, bounds: torch.Tensor | fvdb.JaggedTensor) -> torch.Tensor | fvdb.JaggedTensor:
        """Transform axis-aligned bounding boxes from source frame to target frame.

        This is the primary entry point for bounds transformation. It accepts
        either a torch.Tensor or an fvdb.JaggedTensor and returns the same type.
        The output bounds are conservative (guaranteed to contain all transformed
        points from the original boxes).

        Args:
            bounds: Axis-aligned bounding boxes to transform.
                - torch.Tensor: shape [N, 2, 3] where N is the number of boxes,
                  dim 1 indexes min (0) and max (1) corners, dim 2 is xyz.
                - fvdb.JaggedTensor: shape [B][N, 2, 3] where B is batch size
                  and N varies per batch element.

        Returns:
            Transformed bounds with the same type and shape as input.
                - torch.Tensor input -> torch.Tensor output, shape [N, 2, 3]
                - fvdb.JaggedTensor input -> fvdb.JaggedTensor output, shape [B][N, 2, 3]
        """
        if isinstance(bounds, torch.Tensor):
            return self.apply_bounds_tensor(bounds)
        elif isinstance(bounds, fvdb.JaggedTensor):
            return self.apply_bounds_jagged(bounds)
        else:
            raise ValueError(f"Unsupported type: {type(bounds)}")

    def inverse(self) -> "CoordXform":
        """Return the inverse transformation.

        Returns:
            A CoordXform that undoes this transformation.

        Raises:
            NotImplementedError: If the transformation is not invertible.
        """
        raise NotImplementedError(f"{type(self).__name__} does not have an inverse.")

    @property
    def invertible(self) -> bool:
        """Whether this transformation has an inverse."""
        return False

    @property
    def pseudo_scaling_factor(self) -> float:
        """Compute the approximate uniform scaling factor of this transform.

        The concept of "scaling factor" is well-defined for uniform-scale transforms
        (e.g., UniformScaleThenTranslate), but becomes ambiguous when:
          - The transform has different scales per batch element
          - The transform is non-linear (e.g., frustum/perspective transforms)
          - The transform has non-uniform scaling along different axes

        However, in typical usage, there is a single meaningful scale that's
        approximately consistent across all batch elements and all points in space.

        To estimate this, we compute the "pseudo scaling factor" as follows:
          1. Take two reference points in source space: the origin (0,0,0) and the
             unit diagonal corner (1,1,1).
          2. Transform both points to target space.
          3. Compute the Euclidean distance between the transformed points.
          4. Divide by the reference distance (sqrt(3), the distance from origin
             to (1,1,1) in source space).

        This gives us the approximate scale factor, which equals the exact scale
        for uniform-scale transforms. For non-uniform or spatially-varying transforms,
        this is a representative "average" scale measured along the body diagonal.
        """
        # Create the two reference points in source space
        ref_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        # Transform to target space
        target_points = self.apply_tensor(ref_points)

        # Compute the distance in target space
        diff = target_points[1] - target_points[0]
        target_distance = torch.linalg.norm(diff).item()

        # Reference distance is sqrt(3) (distance from origin to (1,1,1))
        ref_distance = math.sqrt(3.0)

        # Pseudo scaling factor is the ratio
        return target_distance / ref_distance

    def compose(self, other: "CoordXform") -> "CoordXform":
        """Compose this transformation with another.

        The resulting transformation applies other first, then self.
        i.e., (self.compose(other))(x) == self(other(x))

        Args:
            other: The transformation to apply before this one.

        Returns:
            A new CoordXform representing the composition.
        """
        return ComposedXform([other, self])

    @overload
    def __matmul__(self, other: "CoordXform") -> "CoordXform": ...

    @overload
    def __matmul__(self, other: torch.Tensor) -> torch.Tensor: ...
    @overload
    def __matmul__(self, other: fvdb.JaggedTensor) -> fvdb.JaggedTensor: ...

    def __matmul__(
        self, other: "CoordXform | torch.Tensor | fvdb.JaggedTensor"
    ) -> "CoordXform | torch.Tensor | fvdb.JaggedTensor":
        """Matrix-multiply style operator supporting both composition and application.

        This operator is overloaded to support two distinct use cases:

        1. **Composition** (when other is a CoordXform):
           Returns a new transform representing self(other(x)).
           Uses mathematical convention where (A @ B)(x) = A(B(x)),
           so B (other) is applied first, then A (self).

           Example:
               world_T_object = world_T_camera @ camera_T_object
               # Equivalent to: world_T_camera.compose(camera_T_object)

        2. **Application** (when other is a Tensor or JaggedTensor):
           Applies this transform to the coordinates, returning transformed coords.
           This provides a convenient notation: xform @ coords instead of xform(coords).

           Example:
               world_coords = world_T_object @ object_coords
               # Equivalent to: world_T_object.apply(object_coords)

        Args:
            other: Either a CoordXform (for composition) or coordinates to transform
                   (torch.Tensor of shape [N, 3] or fvdb.JaggedTensor).

        Returns:
            - If other is CoordXform: A new CoordXform representing the composition.
            - If other is torch.Tensor: Transformed coordinates as torch.Tensor.
            - If other is fvdb.JaggedTensor: Transformed coordinates as fvdb.JaggedTensor.

        Raises:
            ValueError: If other is not a supported type.
        """
        if isinstance(other, torch.Tensor):
            return self.apply_tensor(other)
        elif isinstance(other, fvdb.JaggedTensor):
            return self.apply_jagged(other)
        elif isinstance(other, CoordXform):
            return self.compose(other)
        else:
            raise ValueError(f"Unsupported type: {type(other)}")


@dataclass(frozen=True)
class ComposedXform(CoordXform):
    """Composition of multiple transformations applied in sequence.
    The 0th transformation is applied first, followed by the 1st, etc.

    """

    xforms: list[CoordXform]

    def apply_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply all transformations in to tensor coords insequence."""
        for xform in self.xforms:
            coords = xform.apply_tensor(coords)
        return coords

    def apply_jagged(self, coords: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        """Apply all transformations to jagged tensor coords in sequence."""
        for xform in self.xforms:
            coords = xform.apply_jagged(coords)
        return coords

    def apply_bounds_tensor(self, bounds: torch.Tensor) -> torch.Tensor:
        """Apply all transformations to bounds tensor in sequence."""
        for xform in self.xforms:
            bounds = xform.apply_bounds_tensor(bounds)
        return bounds

    def apply_bounds_jagged(self, bounds: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        """Apply all transformations to bounds jagged tensor in sequence."""
        for xform in self.xforms:
            bounds = xform.apply_bounds_jagged(bounds)
        return bounds

    def inverse(self) -> CoordXform:
        """Return inverse: (all xforms)^-1 = inverse(all xforms) in reverse order."""
        if not self.invertible:
            raise NotImplementedError("Cannot invert: one or both transforms not invertible.")
        return ComposedXform([xform.inverse() for xform in reversed(self.xforms)])

    def compose(self, other: "CoordXform") -> "CoordXform":
        """Compose this transformation with another. Other is applied first, then self."""
        if isinstance(other, ComposedXform):
            return ComposedXform(other.xforms + self.xforms)
        else:
            return ComposedXform([other] + self.xforms)

    @property
    def invertible(self) -> bool:
        """True if both first and second are invertible."""
        return all(xform.invertible for xform in self.xforms)


@dataclass(frozen=True)
class IdentityXform(CoordXform):
    """Identity transformation: returns inputs unchanged.

    This is the neutral element for transformation composition.
    """

    def apply_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """Return coords unchanged."""
        return coords

    def apply_jagged(self, coords: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        """Return coords unchanged."""
        return coords

    def apply_bounds_tensor(self, bounds: torch.Tensor) -> torch.Tensor:
        """Return bounds unchanged."""
        return bounds

    def apply_bounds_jagged(self, bounds: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        """Return bounds unchanged."""
        return bounds

    def inverse(self) -> "IdentityXform":
        """Return self (identity is its own inverse)."""
        return self

    def compose(self, other: "CoordXform") -> "CoordXform":
        """Return other (identity composed with anything yields that thing)."""
        return other

    def __matmul__(self, other: "CoordXform") -> "CoordXform":
        """Return other (identity composed with anything yields that thing)."""
        return other

    @property
    def invertible(self) -> bool:
        """Always True."""
        return True

    @property
    def pseudo_scaling_factor(self) -> float:
        """Always 1.0 for identity."""
        return 1.0


@dataclass(frozen=True)
class UniformScaleThenTranslate(CoordXform):
    """Uniform scale and translate: coords_out = coords_in * scale + translation.

    This applies the same scalar scale and translation to all three coordinate axes.
    Invertible when scale != 0.
    """

    scale: float = 1.0
    translation: float = 0.0

    def __post_init__(self):
        if self.scale == 0:
            raise ValueError("Scale cannot be zero.")

    def apply_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply coords * scale + translation."""
        return coords * self.scale + self.translation

    def apply_bounds_tensor(self, bounds: torch.Tensor) -> torch.Tensor:
        """Apply bounds * scale + translation, swapping min/max if scale < 0."""
        transformed = bounds * self.scale + self.translation
        if self.scale < 0:
            # Negative scale swaps min and max corners
            transformed = torch.stack([transformed[:, 1, :], transformed[:, 0, :]], dim=1)
        return transformed

    def inverse(self) -> "UniformScaleThenTranslate":
        """Return inverse: scale'=1/scale, translation'=-translation/scale."""
        return UniformScaleThenTranslate(scale=1.0 / self.scale, translation=-self.translation / self.scale)

    def compose(self, other: "CoordXform") -> "CoordXform":
        """Compose this transformation with another.

        The resulting transformation applies other first, then self.
        i.e., (self.compose(other))(x) == self(other(x))

        If other is also a UniformScaleThenTranslate, fuse into a single transform.
        Given:
            other: y = x * s_other + t_other  (applied first)
            self:  z = y * s_self + t_self    (applied second)
        The fused transform is: z = x * (s_other * s_self) + (t_other * s_self + t_self)
        """
        if isinstance(other, UniformScaleThenTranslate):
            return UniformScaleThenTranslate(
                scale=other.scale * self.scale,
                translation=other.translation * self.scale + self.translation,
            )
        else:
            return super().compose(other)

    @property
    def invertible(self) -> bool:
        """True if scale != 0."""
        return self.scale != 0.0

    @property
    def pseudo_scaling_factor(self) -> float:
        """Return the absolute scale factor."""
        return abs(self.scale)


def world_T_voxcen_from_voxel_size(voxel_size: float) -> CoordXform:
    """Create a world_T_voxcen transform for a given finest voxel size.

    Uses center-aligned voxels where ijk coordinates reference voxel centers.
    The half-voxel offset ensures proper alignment.

    For a voxel_size of 0.05m:
    - Voxcen ijk (0,0,0) maps to world (0.025, 0.025, 0.025)
    - Voxcen ijk (1,0,0) maps to world (0.075, 0.025, 0.025)

    Args:
        voxel_size: The size of the voxels in world units.

    Returns:
        A CoordXform that maps voxcen coordinates to world coordinates.
    """
    return UniformScaleThenTranslate(scale=voxel_size, translation=0.5 * voxel_size)


def voxcen_coarsening_xform(factor: int) -> CoordXform:
    """Create a transform from coarse ijk to fine ijk for center-aligned grids.

    When grids use voxel-center tracking (ijk=0 maps to the center of voxel 0),
    coarsening requires a half-voxel offset to ensure coarse ijk=0 maps to the
    center of the coarse voxel (which is the midpoint of the fine voxels it contains).

    For factor=2:
    - Fine voxels 0 and 1 combine into coarse voxel 0
    - Coarse voxel 0's center is at fine ijk=0.5 (midpoint of 0 and 1)
    - So: fine_ijk = coarse_ijk * 2 + 0.5

    General formula: fine_ijk = coarse_ijk * factor + (factor - 1) / 2

    Usage:
        fine_T_coarse = voxcen_coarsening_xform(factor)
        world_T_coarse = world_T_fine.compose(fine_T_coarse)

    Args:
        factor: The coarsening factor (typically 2).

    Returns:
        A CoordXform that maps coarse ijk to fine ijk with center alignment.
    """
    return UniformScaleThenTranslate(scale=float(factor), translation=(factor - 1) / 2)
