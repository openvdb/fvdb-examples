"""Turning meshes into voxel grids, and carrying attributes onto the voxels.

Every function here is jagged over the source geometry's groups: a group is the
unit of authoring, so it is the unit of voxelization, and each group becomes one
grid of an :class:`fvdb.GridBatch` free to carry its own voxel size.
"""

import math

import fvdb
import torch


def voxelize_mesh(
    vertices: fvdb.JaggedTensor,
    faces: fvdb.JaggedTensor,
    values: fvdb.JaggedTensor,
    voxel_size: float | torch.Tensor,
) -> tuple[fvdb.GridBatch, fvdb.JaggedTensor]:
    """Voxelize a mesh and interpolate per-corner values onto the resulting voxels.

    Every input is jagged over the mesh's groups, and each group becomes one
    grid in the returned :class:`fvdb.GridBatch`. Values live on face corners,
    three independent values per triangle, so an attribute may be discontinuous
    across an edge. They are interpolated to voxel centers by replaying the
    stratified barycentric lattice that :meth:`fvdb.GridBatch.from_mesh` uses
    internally, so every voxel is reached exactly once with barycentric
    coordinates intact. Each voxel takes the value from the sample nearest its
    center.

    Args:
        vertices: Group-local world-space vertex positions, jagged shape:
            (G, V_g, 3). G: number of groups; V_g: vertices in group g.
        faces: Triangle corner indices into the group's own vertices, jagged
            shape: (G, F_g, 3). F_g: triangles in group g.
        values: Per-corner attribute values to interpolate, jagged shape:
            (G, F_g, 3, D), aligned corner for corner with ``faces``.
            D: attribute dimensionality.
        voxel_size: Voxel edge length in world units. Either a scalar applied
            to every group, or a 1-D tensor of shape (G,) for per-group sizing.

    Returns:
        A 2-tuple of:

        - :class:`fvdb.GridBatch` with one grid per group built from the mesh
          surface.
        - :class:`fvdb.JaggedTensor` of interpolated values, jagged shape
          (G, M_g, D). M_g: number of voxels in group g's grid.
    """
    G = vertices.num_tensors
    device = vertices.device

    if torch.is_tensor(voxel_size):
        sizes = [[float(voxel_size[g])] * 3 for g in range(G)]
    else:
        sizes = [[float(voxel_size)] * 3] * G

    batch = fvdb.GridBatch.from_mesh(vertices, faces, voxel_sizes=sizes)

    # Sample spacing in voxel units, from fVDB's IjkForMesh.cu.
    spacing = math.sqrt(3.0) / 3.0

    voxel_values: list[torch.Tensor] = []
    for g in range(G):
        grid = fvdb.Grid.from_grid_batch(batch, g)
        group_vertices = vertices[g].jdata  # (V_g, 3)
        group_faces = faces[g].jdata.long()  # (F_g, 3)
        group_values = values[g].jdata  # (F_g, 3, D)

        # Replay the stratified lattice fVDB used to decide which voxels exist,
        # recovering barycentric coordinates for every sample.
        corners = grid.world_to_voxel(group_vertices)[group_faces]  # (F_g, 3, 3)
        origin = corners[:, 0]
        edge_u = corners[:, 1] - origin
        edge_v = corners[:, 2] - origin
        num_u = torch.ceil(
            ((edge_u * edge_u).sum(-1).sqrt() + spacing) / spacing
        ).long()
        num_v = torch.ceil(
            ((edge_v * edge_v).sum(-1).sqrt() + spacing) / spacing
        ).long()

        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), (num_u * num_v).cumsum(0)]
        )
        sample = torch.arange(int(offsets[-1]), device=device)
        triangle = torch.searchsorted(offsets, sample, right=True) - 1
        within = sample - offsets[triangle]
        rows = num_v[triangle]
        i = within // rows
        j = within - i * rows

        u = i.float() / torch.clamp(num_u[triangle] - 1, min=1).float()
        v = j.float() / torch.clamp(rows - 1, min=1).float()
        folded = (u + v) >= 1.0
        u = torch.where(folded, 1.0 - u, u)
        v = torch.where(folded, 1.0 - v, v)

        voxel_space = (
            origin[triangle]
            + edge_u[triangle] * u[:, None]
            + edge_v[triangle] * v[:, None]
        )
        index = grid.ijk_to_index(torch.floor(voxel_space + 0.5).int())
        inside = index >= 0

        barycentric = torch.stack([1.0 - u - v, u, v], dim=-1)  # (S, 3)
        interpolated = (barycentric[..., None] * group_values[triangle]).sum(
            1
        )  # (S, D)

        # Per voxel, keep the sample nearest the voxel center.
        centers = grid.voxel_to_world(grid.ijk.float())
        diff = grid.voxel_to_world(voxel_space[inside]) - centers[index[inside]]
        distance = (diff * diff).sum(-1).sqrt()
        nearest = torch.full((grid.num_voxels,), float("inf"), device=device)
        _ = nearest.scatter_reduce_(
            0, index[inside], distance, reduce="amin", include_self=True
        )

        winner = torch.zeros(grid.num_voxels, dtype=torch.long, device=device)
        candidates = torch.nonzero(inside).squeeze(1)
        won = distance <= nearest[index[inside]] + 1e-9
        _ = winner.scatter_(0, index[inside][won], candidates[won])
        voxel_values.append(interpolated[winner])

    return batch, fvdb.JaggedTensor(voxel_values)


def estimate_voxel_size(
    vertices: fvdb.JaggedTensor,
    faces: fvdb.JaggedTensor,
    detail_scale: float = 1.0,
    percentile: float = 10,
    smallest: float | None = None,
    largest: float | None = None,
) -> torch.Tensor:
    """Estimate a voxel size per group from triangle edge lengths.

    The estimate is driven by the ``percentile``-th triangle edge length scaled
    by ``detail_scale``. A modeller subdivides where there is detail, so the
    short edges track the finest tessellation present.

    Args:
        vertices: Group-local world-space vertex positions, jagged shape:
            (G, V_g, 3). G: number of groups; V_g: vertices in group g.
        faces: Triangle corner indices into the group's own vertices, jagged
            shape: (G, F_g, 3). F_g: triangles in group g.
        detail_scale: Voxel size as a multiple of the ``percentile``-th edge
            length. Smaller oversamples; larger blurs.
        percentile: Which edge-length percentile drives the base estimate.
        smallest: Absolute floor on the voxel size in world units. No floor
            applied when ``None``.
        largest: Absolute ceiling on the voxel size in world units. No ceiling
            applied when ``None``.

    Returns:
        Estimated voxel size per group in world units, shape: (G,).
    """

    def _size_for(g: int) -> torch.Tensor:
        c = vertices[g].jdata[faces[g].jdata.long()]  # (F_g, 3, 3)
        edges = torch.cat([c[:, 1] - c[:, 0], c[:, 2] - c[:, 1], c[:, 0] - c[:, 2]])
        lengths = (edges * edges).sum(dim=-1).sqrt()  # (3 * F_g,)
        s = detail_scale * torch.quantile(lengths, percentile / 100.0)
        if smallest is not None:
            s = s.clamp(min=smallest)
        if largest is not None:
            s = s.clamp(max=largest)
        return s

    return torch.stack([_size_for(g) for g in range(vertices.num_tensors)])


def voxel_centers(topology: fvdb.GridBatch) -> fvdb.JaggedTensor:
    """World-space center of every voxel in the batch.

    Args:
        topology: Voxelized geometry, one grid per group. G: number of grids.

    Returns:
        Voxel centers in world units, jagged shape: (G, M_g, 3). M_g: voxels in
        grid g.
    """
    return fvdb.JaggedTensor(
        [
            grid.voxel_to_world(grid.ijk.float())
            for grid in (
                fvdb.Grid.from_grid_batch(topology, g)
                for g in range(topology.grid_count)
            )
        ]
    )


def broadcast_to_voxels(
    topology: fvdb.GridBatch, values: torch.Tensor
) -> fvdb.JaggedTensor:
    """Give every voxel of a grid the value belonging to that grid.

    The trace reports which voxel a ray hit, so anything the caller knows per
    group -- a material id, a part label -- has to be resolved per voxel to be
    looked up with it.

    Args:
        topology: Voxelized geometry, one grid per group. G: number of grids.
        values: One value per grid, shape: (G,) or (G, D). D: value
            dimensionality.

    Returns:
        The grid's value repeated over its voxels, jagged shape: (G, M_g) or
        (G, M_g, D). M_g: voxels in grid g.
    """
    return fvdb.JaggedTensor(
        [
            values[g].expand(topology.num_voxels_at(g), *values.shape[1:]).contiguous()
            for g in range(topology.grid_count)
        ]
    )
