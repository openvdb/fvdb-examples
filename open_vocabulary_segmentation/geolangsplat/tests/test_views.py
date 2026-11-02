# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the image-view loader's metadata<->COLMAP frame handling.

The critical correctness property: a splat saved during training is recentered by
a ``normalization_transform``, so its cameras live in a normalized frame while the
COLMAP poses live in the original (e.g. UTM) frame. We must match each metadata
camera to its source photo *through* that transform.
"""
import numpy as np
import pytest
import torch
from PIL import Image

import geolangsplat.views as views_mod
from geolangsplat.config import GeoLangSplatConfig
from geolangsplat.errors import GeoLangSplatError
from geolangsplat.views import (
    dome_radius_scale,
    dome_view_metrics,
    match_metadata_to_colmap,
    view_sharpness,
)


def _c2w(center: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = center
    return m


def test_match_recovers_permutation_through_normalization():
    rng = np.random.default_rng(0)
    # Original-frame COLMAP camera centers (large geospatial offset + scatter).
    base = np.array([6.3e6, 1.9e4, 0.0])
    col_centers = base[None, :] + rng.normal(scale=50.0, size=(8, 3))
    col_c2w = np.stack([_c2w(c) for c in col_centers], axis=0)

    # A normalization transform that recenters to the origin (rotation + translation).
    theta = 0.5
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    nt = np.eye(4)
    nt[:3, :3] = R
    nt[:3, 3] = -R @ base

    # Metadata cameras = the same cameras expressed in the normalized frame, shuffled.
    norm_c2w = nt[None] @ col_c2w
    perm = rng.permutation(8)
    meta_c2w = norm_c2w[perm]

    match = match_metadata_to_colmap(meta_c2w, col_c2w, nt)
    assert np.array_equal(match, perm)


def test_identity_transform_is_nearest_position():
    # With no normalization, matching is plain nearest-neighbor by position.
    centers = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0]])
    col_c2w = np.stack([_c2w(c) for c in centers], axis=0)
    meta_c2w = np.stack([_c2w(c + 0.1) for c in centers[::-1]], axis=0)
    match = match_metadata_to_colmap(meta_c2w, col_c2w, np.eye(4))
    assert np.array_equal(match, np.array([2, 1, 0]))


def test_dome_radius_scale_pulls_low_rings_closer():
    # Lowest ring sits at `low` (closer); top ring at 1.0; monotone in between.
    lo, hi, low = 18.0, 88.0, 0.72
    assert dome_radius_scale(lo, lo, hi, low) == low
    assert dome_radius_scale(hi, lo, hi, low) == 1.0
    mid = dome_radius_scale((lo + hi) / 2, lo, hi, low)
    assert low < mid < 1.0
    # Degenerate single-ring set: no scaling.
    assert dome_radius_scale(45.0, 45.0, 45.0, low) == 1.0


def test_view_sharpness_ranks_detail_above_blur():
    rng = np.random.default_rng(0)
    # High-frequency noise = lots of detail; flat gray = none.
    noisy = Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    flat = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    assert view_sharpness(noisy) > view_sharpness(flat)
    assert view_sharpness(flat) == 0.0


def test_dome_view_metrics_coverage_and_occlusion():
    H = W = 64
    radius = 10.0
    pil = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))

    # Good view: center fully covered by the splat, all at the subject distance.
    alpha = torch.ones(H, W)
    depth = torch.full((H, W), radius)
    cov, occ, _ = dome_view_metrics(pil, depth, alpha, radius)
    assert cov > 0.9 and occ == 0.0

    # See-through view: nothing covered in the center -> low coverage.
    cov, occ, _ = dome_view_metrics(pil, depth, torch.zeros(H, W), radius)
    assert cov == 0.0

    # Blocked view: covered, but the whole center sits right in front of the camera
    # (a near floater) -> high occlusion.
    near = torch.full((H, W), 0.2 * radius)
    cov, occ, _ = dome_view_metrics(pil, near, torch.ones(H, W), radius)
    assert cov > 0.9 and occ > 0.9


# -- reality-capture metadata cameras ------------------------------------------


def _rc_metadata(n: int, *, image_ids=None, camera_models=None, dist=None) -> dict:
    meta = {
        "camera_to_world_matrices": torch.eye(4).repeat(n, 1, 1),
        "projection_matrices": torch.tensor([[10.0, 0, 4], [0, 10.0, 3], [0, 0, 1]]).repeat(n, 1, 1),
        "image_sizes": torch.tensor([[6, 8]]).repeat(n, 1),
    }
    if image_ids is not None:
        meta["image_ids"] = torch.tensor(image_ids, dtype=torch.uint32)
    if camera_models is not None:
        meta["camera_models"] = torch.tensor(camera_models, dtype=torch.int32)
    if dist is not None:
        meta["distortion_coeffs"] = dist
    return meta


def _photo(path, value: int):
    Image.new("RGB", (8, 6), (value, value, value)).save(path)
    return path


def test_metadata_views_match_photos_by_image_id(monkeypatch, tmp_path):
    photos = {7: _photo(tmp_path / "a.png", 10), 3: _photo(tmp_path / "b.png", 200)}
    monkeypatch.setattr(views_mod, "_photos_by_image_id", lambda sfm: photos)
    cfg = GeoLangSplatConfig(device="cpu", sfm=str(tmp_path), n_views=0)

    out = views_mod._load_metadata_image_views(cfg, torch.device("cpu"), _rc_metadata(2, image_ids=[3, 7]))

    assert [v.pil.getpixel((0, 0))[0] for v in out] == [200, 10]


def test_metadata_views_reject_unknown_image_id(monkeypatch, tmp_path):
    monkeypatch.setattr(views_mod, "_photos_by_image_id", lambda sfm: {0: _photo(tmp_path / "a.png", 0)})
    cfg = GeoLangSplatConfig(device="cpu", sfm=str(tmp_path), n_views=0)

    with pytest.raises(GeoLangSplatError, match="image ids"):
        views_mod._load_metadata_image_views(cfg, torch.device("cpu"), _rc_metadata(1, image_ids=[5]))


def test_metadata_views_carry_distortion(monkeypatch, tmp_path):
    photos = {0: _photo(tmp_path / "a.png", 0), 1: _photo(tmp_path / "b.png", 0)}
    monkeypatch.setattr(views_mod, "_photos_by_image_id", lambda sfm: photos)
    dist = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    meta = _rc_metadata(2, image_ids=[0, 1], camera_models=[0, 1], dist=dist)
    cfg = GeoLangSplatConfig(device="cpu", sfm=str(tmp_path), n_views=0)

    pinhole, opencv = views_mod._load_metadata_image_views(cfg, torch.device("cpu"), meta)

    assert pinhole.camera_kwargs() == {}
    kw = opencv.camera_kwargs()
    assert int(kw["camera_model"]) == 1
    assert torch.equal(kw["distortion_coeffs"], dist[1:2])
