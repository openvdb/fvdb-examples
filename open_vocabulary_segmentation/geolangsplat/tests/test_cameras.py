# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import torch

from geolangsplat.cameras import intrinsics, orbit_cameras, project


def test_intrinsics_principal_point_and_focal():
    K = intrinsics(60.0, 640, 480)
    assert K.shape == (3, 3)
    assert K[0, 2] == 320 and K[1, 2] == 240
    f = 0.5 * 640 / np.tan(np.deg2rad(30.0))
    assert np.isclose(K[0, 0], f) and np.isclose(K[1, 1], f)


def test_orbit_camera_extrinsic_orthonormal():
    ((w2c, c2w),) = orbit_cameras([0, 0, 0], elev=90.0, radius=10.0, num_azimuth=1)
    R = w2c[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.allclose(c2w @ w2c, np.eye(4), atol=1e-6)


def test_orbit_camera_looks_at_center():
    center = np.array([1.0, 2.0, 0.0])
    ((w2c, _),) = orbit_cameras(center, elev=70.0, radius=20.0, num_azimuth=1)
    f, cx, cy = 100.0, 64.0, 64.0
    u, v, z = project(
        torch.tensor(center, dtype=torch.float32).view(1, 3),
        torch.tensor(w2c, dtype=torch.float32),
        f,
        cx,
        cy,
    )
    assert float(z) > 0  # center is in front of the camera
    assert abs(float(u) - cx) < 1e-2 and abs(float(v) - cy) < 1e-2  # projects to image center


def test_orbit_num_azimuth_count():
    cams = orbit_cameras([0, 0, 0], elev=45.0, radius=5.0, num_azimuth=4)
    assert len(cams) == 4
