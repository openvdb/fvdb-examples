# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""CPU tests for the multi-prompt segment catalog (clustering + dedup + table)."""
from __future__ import annotations

import importlib.util

import pytest
import torch

import geolangsplat.outputs as out
from geolangsplat.catalog import COLUMNS, SegmentCatalog, catalog_from_scores
from geolangsplat.config import GeoLangSplatConfig


class _Model:
    """Minimal GaussianSplat3d stand-in: the tensors the writers / catalog touch."""

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
    """Captures the tensors passed to from_tensors (mirrors test_outputs/instances)."""

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


def _scene(n_a: int = 100, n_b: int = 60, seed: int = 0):
    """Two well-separated spatial clusters: A near the origin, B ~50 units away."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n_a, 3, generator=g)
    b = torch.rand(n_b, 3, generator=g) + 50.0
    return torch.cat([a, b], dim=0), n_a, n_b


def _scores_car_vehicle_tree(n_a: int, n_b: int) -> torch.Tensor:
    """[N, 3] columns: car & vehicle both fire on A (overlap), tree fires on B."""
    n = n_a + n_b
    s = torch.zeros(n, 3)
    s[:n_a, 0] = 0.9  # car -> A
    s[:n_a, 1] = 0.9  # vehicle -> A (should merge with car)
    s[n_a:, 2] = 0.9  # tree -> B
    return s


def _cfg():
    # clean3d off so selection is a pure threshold (keeps the test deterministic).
    return GeoLangSplatConfig(device="cpu", clean3d=False, inst_link_frac=0.02, inst_min_size=25)


def _build(cfg=None, iou=None):
    means, n_a, n_b = _scene()
    model = _Model(means)
    scores = _scores_car_vehicle_tree(n_a, n_b)
    cat = catalog_from_scores(
        model, ["car", "vehicle", "tree"], cfg or _cfg(), scores, seen=torch.ones(means.shape[0]), iou=iou
    )
    return cat, n_a, n_b


def test_catalog_merges_overlapping_prompts():
    cat, n_a, n_b = _build()
    assert len(cat) == 2  # {car, vehicle} -> one object; tree -> one object
    big, small = cat[0], cat[1]
    assert big.n_gaussians == n_a and small.n_gaussians == n_b  # largest first
    assert big.prompts == ["car", "vehicle"]  # both prompts recorded, sorted
    assert big.label in ("car", "vehicle")
    assert small.label == "tree" and small.prompts == ["tree"]


def test_catalog_no_merge_when_iou_high():
    # An impossible IoU floor keeps car and vehicle as separate objects.
    cat, n_a, n_b = _build(iou=1.1)
    assert len(cat) == 3
    assert sorted(o.label for o in cat) == ["car", "tree", "vehicle"]


def test_labels_and_masks_partition_the_scene():
    cat, n_a, n_b = _build()
    assert int(cat.all_mask.sum()) == n_a + n_b
    assert int(cat.mask(0).sum()) == n_a
    assert int(cat.mask(1).sum()) == n_b
    # labels are in input order: first n_a -> object 0, rest -> object 1
    assert int((cat.labels[:n_a] == 0).sum()) == n_a
    assert int((cat.labels[n_a:] == 1).sum()) == n_b
    with pytest.raises(KeyError):
        cat.mask(99)


def test_table_columns_and_rows():
    cat, _, _ = _build()
    rows = cat.rows()
    assert len(rows) == 2
    assert list(rows[0].keys()) == list(COLUMNS)
    tab = cat.table
    if importlib.util.find_spec("pandas") is not None:
        assert list(tab.columns) == list(COLUMNS)
        assert len(tab) == 2
    else:
        assert tab == rows


def test_extract_writes_only_that_object(monkeypatch, tmp_path):
    cat, n_a, _ = _build()
    monkeypatch.setattr(out, "_gs", lambda: _FakeGS)
    written = cat.extract(0, tmp_path / "obj0.ply")
    assert written == n_a
    assert _FakeGS.last.kw["means"].shape[0] == n_a


def test_export_all_writes_folder(monkeypatch, tmp_path):
    cat, _, _ = _build()
    monkeypatch.setattr(out, "_gs", lambda: _FakeGS)
    out_dir = cat.export_all(tmp_path / "cat", labeled_ply=True)
    assert (out_dir / "catalog.csv").exists()
    assert (out_dir / "catalog_labeled.ply").exists()
    plys = list((out_dir / "objects").glob("*.ply"))
    assert len(plys) == 2


def test_save_load_round_trip(tmp_path):
    cat, _, _ = _build()
    cat.save(tmp_path / "saved")
    model = cat._model
    reloaded = SegmentCatalog.load(tmp_path / "saved", model)
    assert len(reloaded) == len(cat)
    assert [o.label for o in reloaded] == [o.label for o in cat]
    assert [o.n_gaussians for o in reloaded] == [o.n_gaussians for o in cat]
    assert torch.equal(reloaded.labels, cat.labels)


def test_ids_mask_unions_selected_objects():
    cat, n_a, n_b = _build()
    assert int(cat._ids_mask([0]).sum()) == n_a
    assert int(cat._ids_mask([0, 1]).sum()) == n_a + n_b
    assert int(cat._ids_mask([]).sum()) == 0


def test_browse_without_ipywidgets_is_clear(monkeypatch):
    import builtins

    cat, _, _ = _build()
    real_import = builtins.__import__

    def _no_ipywidgets(name, *a, **k):
        if name == "ipywidgets":
            raise ModuleNotFoundError("No module named 'ipywidgets'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_ipywidgets)
    with pytest.raises(RuntimeError, match="ipywidgets"):
        cat.browse()


def test_robust_span_splits_objects_despite_floaters():
    # Two clusters 4 units apart + one far floater. The naive min/max span would be
    # ~1e4 -> a huge voxel that merges both; the quantile span tracks the real scene.
    g = torch.Generator().manual_seed(0)
    a = torch.rand(100, 3, generator=g)
    b = torch.rand(100, 3, generator=g) + 5.0
    floater = torch.full((1, 3), 1e4)
    means = torch.cat([a, b, floater], dim=0)

    class M:
        def __init__(s):
            n = means.shape[0]
            gg = torch.Generator().manual_seed(1)
            s.means = means
            s.quats = torch.rand(n, 4, generator=gg)
            s.log_scales = torch.rand(n, 3, generator=gg)
            s.logit_opacities = torch.rand(n, 1, generator=gg)
            s.sh0 = torch.rand(n, 1, 3, generator=gg)
            s.shN = torch.rand(n, 15, 3, generator=gg)

    scores = torch.zeros(means.shape[0], 1)
    scores[:200, 0] = 0.9  # select the two clusters, not the floater
    cat = catalog_from_scores(M(), ["building"], _cfg(), scores, seen=torch.ones(means.shape[0]))
    assert len(cat) == 2  # robust span keeps the two buildings apart


def test_empty_vocab_selection_yields_empty_catalog():
    means, _, _ = _scene()
    model = _Model(means)
    scores = torch.zeros(means.shape[0], 2)  # nothing clears the threshold
    cat = catalog_from_scores(model, ["car", "tree"], _cfg(), scores, seen=torch.ones(means.shape[0]))
    assert len(cat) == 0
    assert int((cat.labels == -1).sum()) == means.shape[0]
