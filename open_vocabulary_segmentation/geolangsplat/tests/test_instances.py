# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""CPU tests for the instance split (connected components + result API)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import geolangsplat.outputs as out
from geolangsplat.config import GeoLangSplatConfig
from geolangsplat.instances import build_instances, connected_components


class _Model:
    """Minimal GaussianSplat3d stand-in: the tensors the writers/CC touch."""

    def __init__(self, means: torch.Tensor):
        n = means.shape[0]
        g = torch.Generator().manual_seed(0)
        self.means = means
        self.quats = torch.rand(n, 4, generator=g)
        self.log_scales = torch.rand(n, 3, generator=g)
        self.logit_opacities = torch.rand(n, 1, generator=g)
        self.sh0 = torch.rand(n, 1, 3, generator=g)
        self.shN = torch.rand(n, 15, 3, generator=g)


class _FakeGS:
    """Captures the tensors passed to from_tensors (mirrors test_outputs)."""

    last = None

    def __init__(self, **kw):
        self.kw = kw

    @classmethod
    def from_tensors(cls, **kw):
        inst = cls(**kw)
        _FakeGS.last = inst
        return inst

    def save_ply(self, path):
        with open(path, "w") as f:
            f.write(f"n={self.kw['means'].shape[0]}\n")


def _two_clusters(n_a: int = 100, n_b: int = 60, n_speckle: int = 3, seed: int = 0):
    """means with a big cluster near origin, a smaller far cluster, and a speckle.

    The clusters are separated by a gap far larger than the voxel link size, so
    connected components must split them; the speckle is below any sane min_size.
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n_a, 3, generator=g)  # ~[0, 1]^3
    b = torch.rand(n_b, 3, generator=g) + 50.0  # ~[50, 51]^3
    speck = torch.full((n_speckle, 3), 100.0)  # isolated, tiny
    return torch.cat([a, b, speck], dim=0), n_a, n_b, n_speckle


def test_connected_components_splits_and_ranks():
    means, n_a, n_b, n_sp = _two_clusters()
    sel = torch.ones(means.shape[0], dtype=torch.bool)
    cfg = GeoLangSplatConfig(device="cpu", inst_link_frac=0.02, inst_min_size=25)

    comp_ids, infos = connected_components(means, sel, cfg)

    assert len(infos) == 2  # speckle dropped by min_size
    assert [d["size"] for d in infos] == [n_a, n_b]  # largest first
    assert infos[0]["idx"] == 0 and infos[1]["idx"] == 1
    # membership: cluster A -> 0, cluster B -> 1, speckle -> -1
    assert int((comp_ids[:n_a] == 0).sum()) == n_a
    assert int((comp_ids[n_a : n_a + n_b] == 1).sum()) == n_b
    assert int((comp_ids[-n_sp:] == -1).sum()) == n_sp
    # centroid of the big cluster sits near the origin box
    assert np.all(infos[0]["centroid"] < 2.0)


def test_min_size_keeps_speckle_when_low():
    means, n_a, n_b, n_sp = _two_clusters()
    sel = torch.ones(means.shape[0], dtype=torch.bool)
    cfg = GeoLangSplatConfig(device="cpu", inst_link_frac=0.02, inst_min_size=1)
    _comp, infos = connected_components(means, sel, cfg)
    assert len(infos) == 3  # speckle now survives


def test_link_frac_merges_when_coarse():
    means, n_a, n_b, _ = _two_clusters()
    sel = torch.ones(means.shape[0], dtype=torch.bool)
    cfg = GeoLangSplatConfig(device="cpu")
    # A voxel large enough to bridge the 50-unit gap collapses both into one blob.
    _comp, infos = connected_components(means, sel, cfg, link_frac=1.0, min_size=1)
    assert len(infos) == 1
    assert infos[0]["size"] == means.shape[0]


def test_explicit_span_overrides_naive_range():
    # Clusters at 0 and 50; a far floater blows up the naive span and merges them,
    # but passing a robust span keeps the split.
    means, n_a, n_b, _ = _two_clusters()
    means = torch.cat([means, torch.full((1, 3), 1e5)], dim=0)
    sel = torch.zeros(means.shape[0], dtype=torch.bool)
    sel[: n_a + n_b] = True  # select the two clusters, not the floater
    cfg = GeoLangSplatConfig(device="cpu", inst_link_frac=0.02, inst_min_size=25)

    _c, merged = connected_components(means, sel, cfg)  # naive span ~1e5 -> one blob
    assert len(merged) == 1
    _c, split = connected_components(means, sel, cfg, span=51.0)  # real extent
    assert len(split) == 2


def test_empty_selection_yields_no_instances():
    means, *_ = _two_clusters()
    sel = torch.zeros(means.shape[0], dtype=torch.bool)
    cfg = GeoLangSplatConfig(device="cpu")
    comp_ids, infos = connected_components(means, sel, cfg)
    assert infos == []
    assert int((comp_ids == -1).sum()) == means.shape[0]


def test_build_instances_result_and_extract(monkeypatch, tmp_path):
    means, n_a, n_b, n_sp = _two_clusters()
    model = _Model(means)
    sel = torch.ones(means.shape[0], dtype=torch.bool)
    cfg = GeoLangSplatConfig(device="cpu", inst_link_frac=0.02, inst_min_size=10)

    res = build_instances(model, sel, cfg, "house")
    assert len(res) == 2
    assert res[0].size == n_a and res[1].size == n_b
    assert "house" in repr(res)
    assert int(res.mask(0).sum()) == n_a
    assert int(res.all_mask.sum()) == n_a + n_b  # speckle excluded

    with pytest.raises(IndexError):
        res.mask(5)

    # extract one instance -> only its Gaussians are written
    monkeypatch.setattr(out, "_gs", lambda: _FakeGS)
    written = res.extract(0, tmp_path / "house0.ply")
    assert written == n_a
    assert _FakeGS.last.kw["means"].shape[0] == n_a
