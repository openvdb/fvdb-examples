"""Pure geometric utilities: rotations."""

from __future__ import annotations

import torch


def euler_to_mat(angles: torch.Tensor) -> torch.Tensor:
    """Build a 3×3 rotation matrix from Euler xyz angles.

    The x rotation is applied first, then y, then z:
    ``R = Rz @ Ry @ Rx``.

    Args:
        angles: Rotations about x, y and z axes in radians, shape: (3,).

    Returns:
        Rotation matrix, shape: (3, 3).
    """
    cx, cy, cz = torch.cos(angles).unbind()
    sx, sy, sz = torch.sin(angles).unbind()
    z, o = torch.zeros_like(cx), torch.ones_like(cx)
    rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx]).reshape(3, 3)
    ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy]).reshape(3, 3)
    rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o]).reshape(3, 3)
    return rz @ ry @ rx
