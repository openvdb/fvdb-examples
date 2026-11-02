"""Purely geometric ray tracing against a voxelized scene.

Nothing here knows about spectra, materials, or units of radiance. A trace
answers two questions: what surface a ray met, and which point sources that
surface can see. Turning those answers into radiance is the caller's job.

The scene topology is an :class:`fvdb.GridBatch`, one grid per group of the
source geometry, each free to carry its own voxel size. A ray is traced against
every grid and the nearest hit wins, so the batch is a single scene rather than
a batch of independent scenes.
"""

from __future__ import annotations

from typing import NamedTuple

import fvdb
import torch

# Secondary rays start this many voxels off the surface so they do not
# immediately re-hit the voxel they were spawned from.
SURFACE_OFFSET_VOXELS = 1.5


class Hit(NamedTuple):
    """Where a bundle of rays first met the scene.

    Attributes:
        distance: Distance along each ray to the surface, in world units,
            infinite where the ray escaped, shape: (N,). N: number of rays.
        point: World-space intersection point, equal to the ray origin where
            the ray escaped, shape: (N, 3).
        normal: Unit surface normal at the intersection, zero where the ray
            escaped, shape: (N, 3).
        voxel: Index of the surface voxel into the batch's concatenated voxel
            ordering, the ordering every per-voxel attribute uses, so any of
            them can be looked up with it. Zero where the ray escaped,
            shape: (N,).
        hit: True where the ray met the scene, shape: (N,), dtype bool.
    """

    distance: torch.Tensor
    point: torch.Tensor
    normal: torch.Tensor
    voxel: torch.Tensor
    hit: torch.Tensor


def rays_from_pinhole(
    focal_length: float,
    x_count: int,
    y_count: int,
    x_spacing: float,
    y_spacing: float,
    origin: torch.Tensor,
    rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate scene-space rays for every pixel of a pinhole camera.

    The boresight points along the camera's ``-z`` axis. A pixel at column
    ``i`` and row ``j`` maps to focal-plane position
    ``((i - 0.5*(x_count-1))*x_spacing, (j - 0.5*(y_count-1))*y_spacing)``,
    giving unnormalized direction ``(x, y, -focal_length)`` before the rotation
    is applied. Rays come back row-major, so a per-ray result reshapes straight
    to an image of shape ``(y_count, x_count)``.

    Args:
        focal_length: Distance from the entrance pupil to the focal plane, in
            the same units as ``x_spacing`` and ``y_spacing``.
        x_count: Number of pixels along the camera's x axis.
        y_count: Number of pixels along its y axis.
        x_spacing: Center-to-center pixel pitch along x, in focal-plane units.
            Pass a negative value to flip the x axis.
        y_spacing: Center-to-center pixel pitch along y, in focal-plane units.
            Pass a negative value to flip the y axis.
        origin: Camera position in world coordinates, shape: (3,).
        rotation: Rotation matrix taking camera-frame directions into the world
            frame, shape: (3, 3).

    Returns:
        A pair of:

        - Ray origins in world coordinates, one per pixel, shape: (N, 3).
          N: number of pixels, ``x_count * y_count``.
        - Unit ray directions in world coordinates, shape: (N, 3).
    """
    device = origin.device
    dtype = origin.dtype
    y_index, x_index = torch.meshgrid(
        torch.arange(y_count, device=device, dtype=dtype),
        torch.arange(x_count, device=device, dtype=dtype),
        indexing="ij",
    )
    x_plane = (x_index - 0.5 * (x_count - 1)) * x_spacing
    y_plane = (y_index - 0.5 * (y_count - 1)) * y_spacing
    directions = torch.nn.functional.normalize(
        torch.stack(
            [x_plane, y_plane, torch.full_like(x_plane, -focal_length)], dim=-1
        ).reshape(-1, 3),
        dim=-1,
    )
    directions = directions @ rotation.to(device=device, dtype=dtype).T
    return origin.unsqueeze(0).expand(len(directions), -1), directions


def _march(
    topology: fvdb.GridBatch,
    origins: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """March one ray bundle against every grid in the batch, in a single call.

    :meth:`fvdb.GridBatch.voxels_along_rays` pairs the g-th ray set with the
    g-th grid, so the bundle is broadcast across the batch by handing it the
    same rays G times: one DDA launch covers the whole scene rather than one
    launch per grid. Results come back jagged over G * N rays in grid-major
    order, which reshapes straight to (G, N).

    Args:
        topology: Scene topology, one grid per geometry group. G: number of
            grids.
        origins: Ray origins in world units, shape: (N, 3). N: number of rays.
        directions: Unit ray directions, shape: (N, 3).

    Returns:
        A 2-tuple of:

        - Distance along each ray to where it entered each grid, infinite where
          the ray missed that grid, shape: (G, N).
        - Index of the entered voxel into the batch's concatenated voxel
          ordering, zero where the ray missed, shape: (G, N).
    """
    N = len(origins)
    G = topology.grid_count
    device = origins.device

    offsets = torch.arange(G + 1, device=device) * N
    rays = fvdb.JaggedTensor.from_data_and_offsets(
        origins.expand(G, N, 3).reshape(G * N, 3), offsets
    )
    steps = fvdb.JaggedTensor.from_data_and_offsets(
        directions.expand(G, N, 3).reshape(G * N, 3), offsets
    )
    # cumulative=True numbers voxels across the whole batch, so the returned
    # indices address a per-voxel JaggedTensor's jdata directly.
    voxels, times = topology.voxels_along_rays(
        rays, steps, max_voxels=1, return_ijk=False, cumulative=True
    )

    # Each ray contributes zero or one entry, in ray order, so the jagged
    # results scatter straight back into dense per-ray arrays.
    found = (voxels.joffsets[1:] - voxels.joffsets[:-1]) > 0
    entry = torch.full((G * N,), float("inf"), device=device)
    entry[found] = times.jdata[:, 0]
    index = torch.zeros(G * N, dtype=torch.long, device=device)
    index[found] = voxels.jdata.long()
    return entry.reshape(G, N), index.reshape(G, N)


def first_hit(
    topology: fvdb.GridBatch,
    normals: fvdb.JaggedTensor,
    origins: torch.Tensor,
    directions: torch.Tensor,
) -> Hit:
    """March a ray bundle through every grid and keep the nearest surface.

    Args:
        topology: Scene topology, one grid per geometry group.
        normals: Unit surface normal per voxel, ordered to match each grid's own
            voxel ordering, jagged shape: (G, M_g, 3). G: number of grids;
            M_g: voxels in grid g.
        origins: Ray origins in world units, shape: (N, 3). N: number of rays.
        directions: Unit ray directions, shape: (N, 3).

    Returns:
        The nearest surface each ray met, as a :class:`Hit`.
    """
    entry, voxel = _march(topology, origins, directions)

    nearest = entry.argmin(0)
    rays = torch.arange(len(origins), device=origins.device)
    distance = entry[nearest, rays]
    index = voxel[nearest, rays]
    hit = torch.isfinite(distance)
    normal = torch.where(hit[:, None], normals.jdata[index], 0.0)
    point = origins + torch.where(hit, distance, 0.0)[:, None] * directions
    return Hit(distance=distance, point=point, normal=normal, voxel=index, hit=hit)


def visible(
    topology: fvdb.GridBatch,
    origins: torch.Tensor,
    directions: torch.Tensor,
    limit: torch.Tensor,
) -> torch.Tensor:
    """Test whether each ray reaches its endpoint without meeting the scene.

    Args:
        topology: Scene topology, one grid per geometry group.
        origins: Ray origins in world units, shape: (N, 3). N: number of rays.
        directions: Unit ray directions, shape: (N, 3).
        limit: Distance along each ray to its endpoint, in world units. Surfaces
            beyond it do not block, shape: (N,).

    Returns:
        True where nothing stands between the origin and the endpoint,
        shape: (N,), dtype bool.
    """
    entry, _ = _march(topology, origins, directions)
    return ~(entry < limit[None, :]).any(0)


def shadowed(
    topology: fvdb.GridBatch,
    points: torch.Tensor,
    normals: torch.Tensor,
    directions: torch.Tensor,
    limit: torch.Tensor | None = None,
    offset_voxels: float = SURFACE_OFFSET_VOXELS,
) -> torch.Tensor:
    """Test whether scene geometry stands between each surface point and a source.

    A shadow ray is spawned from each point toward the source and lifted off the
    surface along the normal, so that it does not immediately re-hit the voxel it
    came from. That lift is the whole reason this exists rather than a bare
    :func:`visible` call.

    Args:
        topology: Scene topology, one grid per geometry group.
        points: World-space surface points, shape: (N, 3). N: number of points.
        normals: Unit surface normal at each point, along which the shadow ray
            is lifted, shape: (N, 3).
        directions: Unit direction from each point toward the source, shape:
            (N, 3). A single direction of shape (3,) broadcasts to every point,
            which is what a source at infinity looks like.
        limit: Distance from each point to the source, in world units, so that
            geometry behind the source does not shadow it, shape: (N,). Pass
            ``None`` for a source at infinity, where nothing is behind it.
        offset_voxels: How far to lift the shadow ray off the surface, in
            multiples of the largest voxel edge length in the batch.

    Returns:
        True where the source is occluded, shape: (N,), dtype bool.
    """
    offset = offset_voxels * float(topology.voxel_sizes.max())
    directions = directions.expand_as(points)
    if limit is None:
        limit = torch.full((len(points),), float("inf"), device=points.device)
    return ~visible(topology, points + offset * normals, directions, limit - offset)
