"""3D visualization utilities built on top of viser.

Wraps a :class:`viser.ViserServer` with helpers for displaying fVDB
:class:`~fvdb.GridBatch` objects, colored by arbitrary per-voxel tensors through
a pluggable color mapping function.
"""

from __future__ import annotations

from collections.abc import Callable

import fvdb
import numpy as np
import torch
import viser  # pyright: ignore[reportMissingImports]

from modsim_toolbox import voxelize

# A color function maps a per-voxel tensor of shape (N, ...) to an (N, 3)
# uint8 RGB array. The exact inner shape depends on the colormap.
ColorFn = Callable[[torch.Tensor], np.ndarray]


def normal_colormap() -> ColorFn:
    """Return a color function that encodes unit normals as RGB.

    Each axis is linearly mapped from ``[-1, 1]`` to ``[0, 255]``, so +x reads
    red, +y green, and +z blue. Opposing faces of a thin panel appear as
    complementary colors, which makes a mis-attributed normal easy to spot.

    Returns:
        A function that accepts normals of shape ``(N, 3)`` and returns
        ``uint8`` colors of shape ``(N, 3)``.
    """

    def _fn(normals: torch.Tensor) -> np.ndarray:
        return (127.5 * (normals + 1.0)).clamp(0, 255).to(torch.uint8).cpu().numpy()

    return _fn


class Visualizer:
    """A viser-backed 3D visualizer with fVDB grid helpers.

    Wraps :class:`viser.ViserServer` so that callers can display
    :class:`~fvdb.GridBatch` objects without touching the viser API directly.
    The server is started on construction and stays running until the process
    exits.

    Args:
        host: Network interface to bind the HTTP server to.
        port: TCP port for the viser web interface.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._server = viser.ViserServer(host=host, port=port)
        self._server.scene.set_up_direction("+z")

    @property
    def server(self) -> viser.ViserServer:
        """The underlying viser server, for callers that need direct access."""
        return self._server

    def add_grid_batch(
        self,
        path: str,
        grids: fvdb.GridBatch,
        values: fvdb.JaggedTensor,
        color_fn: ColorFn,
        point_size: float | torch.Tensor = 0.05,
        point_shape: str = "rounded",
        names: list[str] | None = None,
    ) -> None:
        """Display every grid in a :class:`~fvdb.GridBatch` as a colored point cloud.

        One viser point cloud is created per grid, placed under ``path/<name>``
        (or ``path/<g>`` when ``names`` is not supplied). Each voxel's world
        position is computed from the grid's IJK coordinates; its color comes
        from applying ``color_fn`` to the matching row of ``values``.

        Args:
            path: Base scene path. Each grid is added as a child, e.g.
                ``/voxels/body``.
            grids: Voxelized scene, one grid per group.
            values: Per-voxel attribute fed to ``color_fn``, jagged shape:
                ``(G, N_g, ...)``. G: number of grids; N_g: voxels in grid g.
            color_fn: Maps a per-voxel tensor of shape ``(N_g, ...)`` to
                ``uint8`` colors of shape ``(N_g, 3)``.
            point_size: Rendered radius of each point in world units. Either a
                single float applied to every grid, or a 1-D tensor of shape
                ``(G,)`` with one radius per grid.
            point_shape: Viser point shape, e.g. ``"rounded"`` or ``"circle"``.
            names: Labels for each grid, used as the last component of the
                scene path. Defaults to the grid index as a string.
        """
        sizes: torch.Tensor | float
        if isinstance(point_size, torch.Tensor):
            sizes = point_size.float().cpu()
        else:
            sizes = point_size

        centers = voxelize.voxel_centers(grids)
        for g in range(grids.grid_count):
            label = names[g] if names is not None else str(g)
            colors = color_fn(values[g].jdata)
            size = float(sizes[g]) if isinstance(sizes, torch.Tensor) else sizes
            self._server.scene.add_point_cloud(
                f"{path}/{label}",
                points=centers[g].jdata.cpu().numpy(),
                colors=colors,
                point_size=size,
                point_shape=point_shape,
            )
