# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Geometry-driven view planning + up-axis estimation (CPU-only, no SAM3/GPU)."""
from __future__ import annotations

import numpy as np
import torch

from geolangsplat import autoview, cameras


def _ground_scene(seed: int = 0) -> torch.Tensor:
    """A flat z=0 ground slab with some structure above it (gravity = +z)."""
    rng = np.random.default_rng(seed)
    ground = np.c_[rng.uniform(-10, 10, (8000, 2)), np.zeros(8000)]
    above = np.c_[rng.uniform(-10, 10, (2000, 2)), rng.uniform(0.1, 3.0, 2000)]
    return torch.tensor(np.r_[ground, above], dtype=torch.float32)


def test_estimate_up_finds_plus_z_on_ground_scene():
    vec, conf = autoview.estimate_up(_ground_scene())
    assert conf > 0.3
    assert abs(float(vec[2])) > 0.9 and float(vec[2]) > 0  # points up, away from the slab


def _tile_scene(flip: bool = False, seed: int = 0) -> torch.Tensor:
    """A satellite-tile-like scene: a big dense ground slab, buildings rising above
    it, and some sub-ground floaters/terrain below (the case that flipped JAX_264).
    With ``flip`` the whole thing is mirrored in z, so a correct estimator must still
    return up pointing AWAY from the dense ground (toward the structures)."""
    rng = np.random.default_rng(seed)
    ground = np.c_[rng.uniform(-400, 400, (20000, 2)), rng.normal(0, 0.5, 20000)]
    bldg = np.c_[rng.uniform(-400, 400, (6000, 2)), rng.uniform(2, 100, 6000)]
    sub = np.c_[rng.uniform(-400, 400, (4000, 2)), rng.uniform(-60, -5, 4000)]  # below-ground mass
    pts = np.r_[ground, bldg, sub].astype(np.float32)
    if flip:
        pts[:, 2] = -pts[:, 2]
    return torch.tensor(pts)


def test_estimate_up_sign_robust_to_subground_mass():
    """Ground at the bottom -> up = +z; mirrored -> up = -z. The estimate must follow
    the structures (away from the dense ground), not be tipped by below-ground mass."""
    vec, conf = autoview.estimate_up(_tile_scene(flip=False))
    assert conf > 0.3 and float(vec[2]) > 0.9
    vecf, conff = autoview.estimate_up(_tile_scene(flip=True))
    assert conff > 0.3 and float(vecf[2]) < -0.9


def test_estimate_up_sign_stable_across_similar_tiles():
    """Two near-identical tiles must resolve to the SAME up sign (the JAX_264 vs
    JAX_175 disagreement that put one orbit underneath the scene)."""
    z = [float(autoview.estimate_up(_tile_scene(seed=s))[0][2]) for s in (1, 2, 3)]
    assert all(v > 0.9 for v in z), z


def test_resolve_up_honours_explicit_axis():
    assert np.allclose(autoview.resolve_up("+y", _ground_scene()), [0, 1, 0])


def test_resolve_up_falls_back_to_z_on_blob():
    rng = np.random.default_rng(1)
    blob = torch.tensor(rng.normal(0, 1, (5000, 3)), dtype=torch.float32)
    assert np.allclose(autoview.resolve_up("auto", blob), [0, 0, 1])


def test_recommend_aerial_for_flat_scene():
    stats = autoview.measure_geometry(_ground_scene())
    assert autoview.detect_capture(stats) == "aerial"
    rec = autoview.recommend_view_config(stats, budget=200)
    assert rec["capture"] == "aerial"
    assert rec["n_views_planned"] > 0
    assert len(rec["tiers"]) == 2  # overview + oblique


def test_recommend_object_for_isotropic_scene():
    rng = np.random.default_rng(2)
    sphere = torch.tensor(rng.normal(0, 1, (5000, 3)), dtype=torch.float32)
    stats = autoview.measure_geometry(sphere)
    assert autoview.detect_capture(stats) == "object"
    rec = autoview.recommend_view_config(stats, budget=120)
    assert rec["capture"] == "object"
    assert rec["view_source"] == "globe"  # objects use the auto-framed globe orbit


def test_globe_rings_dome_looks_down_and_hits_budget():
    elevs, az = autoview.globe_rings(200)
    assert min(elevs) > 0 and max(elevs) > 60  # upper dome: all rings look down
    assert len(elevs) == len(az)
    assert all(c >= 3 for c in az)
    assert 0.6 * 200 <= sum(az) <= 1.4 * 200  # roughly hits the requested budget
    # lower (more oblique) rings see more scene -> get more azimuths than the cap
    assert az[0] >= az[-1]


def test_inside_out_eye_at_center_and_looks_outward():
    center = np.array([1.0, 2.0, 3.0])
    cams = cameras.inside_out_cameras(center, (-25, 0, 25), 6, up="+z")
    assert len(cams) == 18
    w2c, _c2w = cams[0]
    eye = -np.linalg.inv(w2c[:3, :3]) @ w2c[:3, 3]
    assert np.allclose(eye, center, atol=1e-6)


def test_up_vector_named_axes():
    assert np.allclose(cameras.up_vector("-x"), [-1, 0, 0])
    assert np.allclose(cameras.up_vector("auto"), [0, 0, 1])
