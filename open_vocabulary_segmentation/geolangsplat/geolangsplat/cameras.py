# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Camera helpers: pinhole intrinsics, orbit-camera generation, and projection.

Conventions
-----------
* World frame is ENU-style with +Z up (the satellite/aerial splats are stored
  this way: x=East, y=North, z=Up).
* ``elev`` is the elevation angle above the horizontal plane in degrees
  (90 = straight down / nadir, 0 = on the horizon).
* ``azimuth`` is measured in the world XY plane in degrees, 0 along +X (East),
  increasing toward +Y (North).
* A returned ``w2c`` is an OpenCV world->camera extrinsic (camera +X right,
  +Y down, +Z forward), which is what ``project`` and
  ``GaussianSplat3d.render_*`` expect.
"""
from __future__ import annotations

import numpy as np

_WORLD_UP = np.array([0.0, 0.0, 1.0])
_ALT_UP = np.array([0.0, 1.0, 0.0])

# Named up-axis specs -> unit vectors.
_AXES = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def up_vector(up=None) -> np.ndarray:
    """Resolve an up-axis spec (``"+z"`` ... ``"-x"``) to a unit vector.

    Geospatial splats are ``+z`` up; object/COLMAP plys carry no convention, so the
    caller can pick the axis that renders the scene upright (SAM3 needs upright
    images, and the viewer needs the right up to be navigable). A 3-vector is
    accepted and normalized as-is; ``""``/``"auto"``/``"none"`` map to ``+z``.
    """
    if isinstance(up, (list, tuple, np.ndarray)):
        v = np.asarray(up, dtype=np.float64).reshape(3)
        return v / max(np.linalg.norm(v), 1e-9)
    s = str(up).lower().replace(" ", "")
    if s in ("", "auto", "none"):
        return np.array([0.0, 0.0, 1.0])
    v = _AXES.get(s)
    if v is None:
        raise ValueError(f"unknown up axis {up!r}; use one of {sorted(_AXES)} or 'auto'")
    return np.array(v, dtype=np.float64)


def _perp_basis(up_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to ``up_hat``."""
    ref = _ALT_UP if abs(float(np.dot(up_hat, _WORLD_UP))) > 0.9 else _WORLD_UP
    e1 = np.cross(ref, up_hat)
    e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(up_hat, e1)
    e2 /= max(np.linalg.norm(e2), 1e-9)
    return e1, e2


def intrinsics(fov_deg: float, width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics (3x3) with square pixels and a centered principal point."""
    f = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_deg))
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)


def _look_at_w2c(eye: np.ndarray, target: np.ndarray, up=None) -> np.ndarray:
    """OpenCV world->camera matrix for a camera at ``eye`` looking at ``target``.

    ``up`` is the world-space up direction used to roll the camera (defaults to +z).
    """
    up = _WORLD_UP if up is None else np.asarray(up, dtype=np.float64).reshape(3)
    z_c = target - eye
    n = np.linalg.norm(z_c)
    z_c = np.array([0.0, 0.0, -1.0]) if n < 1e-9 else z_c / n
    if abs(float(np.dot(z_c, up))) > 0.999:
        # view direction parallel to up: fall back to an axis that isn't.
        up = _ALT_UP if abs(float(np.dot(z_c, _ALT_UP))) <= 0.999 else _WORLD_UP
    x_c = np.cross(z_c, up)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    R = np.stack([x_c, y_c, z_c], axis=0)
    t = -R @ eye
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def orbit_cameras(
    center,
    elev: float,
    radius: float,
    num_azimuth: int = 1,
    azimuth_offset_deg: float = 0.0,
    up=None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate ``num_azimuth`` cameras orbiting ``center`` at a fixed elevation.

    ``elev`` is measured from the plane perpendicular to ``up`` (90 = along +up,
    looking back down it; 0 = on that plane). ``up`` defaults to +z. Returns a list
    (length ``num_azimuth``) of ``(w2c, c2w)`` 4x4 float64 arrays.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    up_hat = up_vector("+z" if up is None else up)
    e1, e2 = _perp_basis(up_hat)
    e = np.deg2rad(elev)
    cos_e, sin_e = np.cos(e), np.sin(e)
    out = []
    n = max(1, num_azimuth)
    for i in range(n):
        az = azimuth_offset_deg + 360.0 * i / n
        a = np.deg2rad(az)
        offset = radius * (cos_e * (np.cos(a) * e1 + np.sin(a) * e2) + sin_e * up_hat)
        eye = center + offset
        w2c = _look_at_w2c(eye, center, up=up_hat)
        c2w = np.linalg.inv(w2c)
        out.append((w2c, c2w))
    return out


def inside_out_cameras(
    center,
    elevations,
    num_azimuth: int,
    up=None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cameras placed AT ``center`` looking OUTWARD (for interior / unbounded scenes).

    The outside-in orbit fails on a room: from outside you see the backs of walls.
    Here the eye sits at the scene core and sweeps a full azimuth ring at a few
    pitches, so the surrounding content (walls, counters, furniture) is actually
    framed. Returns ``(w2c, c2w)`` pairs. ``up`` defaults to +z.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    up_hat = up_vector("+z" if up is None else up)
    e1, e2 = _perp_basis(up_hat)
    n = max(1, int(num_azimuth))
    out = []
    for elev in elevations:
        e = np.deg2rad(elev)
        cos_e, sin_e = np.cos(e), np.sin(e)
        for i in range(n):
            a = np.deg2rad(360.0 * i / n)
            direction = cos_e * (np.cos(a) * e1 + np.sin(a) * e2) + sin_e * up_hat
            w2c = _look_at_w2c(center, center + direction, up=up_hat)
            c2w = np.linalg.inv(w2c)
            out.append((w2c, c2w))
    return out


def project(means, w2c, f: float, cx: float, cy: float):
    """Project world-space means into a view (OpenCV: x right, y down, z fwd).

    Returns ``(u, v, z)`` pixel coordinates and camera-space depth. ``means`` is
    ``[N, 3]`` and ``w2c`` is ``[4, 4]``; both are ``torch.Tensor`` on the same
    device.
    """
    import torch

    n = means.shape[0]
    homog = torch.cat([means, torch.ones(n, 1, device=means.device)], dim=1)
    pc = (w2c @ homog.T).T
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    u = f * x / z + cx
    v = f * y / z + cy
    return u, v, z
