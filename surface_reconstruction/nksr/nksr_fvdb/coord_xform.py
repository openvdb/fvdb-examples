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
            coords: JaggedTensor of shape [N, 3] containing 3D coordinates.

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
                - fvdb.JaggedTensor: shape [B][N, 3] where B is batch size and N
                  varies per batch element.

        Returns:
            Transformed coordinates with the same type and shape as input.
                - torch.Tensor input -> torch.Tensor output, shape [N, 3]
                - fvdb.JaggedTensor input -> fvdb.JaggedTensor output, shape [B][N, 3]
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

    def compose(self, other: "CoordXform") -> "CoordXform":
        """Compose this transformation with another.

        The resulting transformation applies self first, then other.
        i.e., (self.compose(other))(x) == other(self(x))

        Args:
            other: The transformation to apply after this one.

        Returns:
            A new CoordXform representing the composition.
        """
        return ComposedXform(self, other)

    def __matmul__(self, other: "CoordXform") -> "CoordXform":
        """Compose transformations using @ operator.

        Note: Uses mathematical convention where (A @ B)(x) = A(B(x)),
        so other is applied first, then self.
        """
        return ComposedXform(other, self)

    def coarsened(self, factor: int | float) -> "CoordXform":
        """Return a transform for a coarsened grid.

        When a grid is coarsened by a factor, ijk coordinates are divided
        by that factor. This composes the scaling into the transform.

        Args:
            factor: The coarsening factor (typically 2).

        Returns:
            A new CoordXform that maps to the coarsened grid's ijk space.
        """
        if factor <= 0:
            raise ValueError("Coarsening factor cannot be negative or zero.")
        if factor == 1 or factor == 1.0:
            return self
        return self.compose(ScalarGainBiasXform(gain=1.0 / factor))


@dataclass(frozen=True)
class ComposedXform(CoordXform):
    """Composition of two transformations applied in sequence.

    Represents the transformation: coords_out = second(first(coords_in))
    """

    first: CoordXform
    second: CoordXform

    def apply_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply first, then second transformation."""
        return self.second.apply_tensor(self.first.apply_tensor(coords))

    def apply_jagged(self, coords: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        """Apply first, then second transformation."""
        return self.second.apply_jagged(self.first.apply_jagged(coords))

    def apply_bounds_tensor(self, bounds: torch.Tensor) -> torch.Tensor:
        """Apply first, then second bounds transformation."""
        return self.second.apply_bounds_tensor(self.first.apply_bounds_tensor(bounds))

    def apply_bounds_jagged(self, bounds: fvdb.JaggedTensor) -> fvdb.JaggedTensor:
        """Apply first, then second bounds transformation."""
        return self.second.apply_bounds_jagged(self.first.apply_bounds_jagged(bounds))

    def inverse(self) -> CoordXform:
        """Return inverse: (second o first)^-1 = first^-1 o second^-1."""
        if not self.invertible:
            raise NotImplementedError("Cannot invert: one or both transforms not invertible.")
        return ComposedXform(self.second.inverse(), self.first.inverse())

    @property
    def invertible(self) -> bool:
        """True if both first and second are invertible."""
        return self.first.invertible and self.second.invertible


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


@dataclass(frozen=True)
class ScalarGainBiasXform(CoordXform):
    """Uniform scale and translate: coords_out = coords_in * gain + bias.

    This applies the same scalar gain and bias to all three coordinate axes.
    Invertible when gain != 0.
    """

    gain: float = 1.0
    bias: float = 0.0

    def apply_tensor(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply coords * gain + bias."""
        return coords * self.gain + self.bias

    def apply_bounds_tensor(self, bounds: torch.Tensor) -> torch.Tensor:
        """Apply bounds * gain + bias, swapping min/max if gain < 0."""
        transformed = bounds * self.gain + self.bias
        if self.gain < 0:
            # Negative gain swaps min and max corners
            transformed = torch.stack([transformed[:, 1, :], transformed[:, 0, :]], dim=1)
        return transformed

    def inverse(self) -> "ScalarGainBiasXform":
        """Return inverse: gain'=1/gain, bias'=-bias/gain."""
        return ScalarGainBiasXform(gain=1.0 / self.gain, bias=-self.bias / self.gain)

    @property
    def invertible(self) -> bool:
        """True if gain != 0."""
        return self.gain != 0.0
