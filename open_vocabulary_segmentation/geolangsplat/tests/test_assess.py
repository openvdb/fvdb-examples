# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch

from geolangsplat.config import GeoLangSplatConfig
from geolangsplat.engine import GeoLangSplatEngine
from geolangsplat.cli._common import read_vocab_file


def _engine_with(seen, denom, n_views=10):
    cfg = GeoLangSplatConfig(device="cpu", min_weight=0.03)

    class _M:
        means = torch.zeros(seen.shape[0], 3)

    eng = GeoLangSplatEngine(_M(), cfg, build=False)
    eng.seen = seen
    eng.denom = denom
    eng.views = list(range(n_views))
    return eng


def test_assess_good_scene():
    n = 100
    seen = torch.full((n,), 5.0)
    denom = torch.full((n,), 1.0)
    rep = _engine_with(seen, denom).assess()
    assert rep["verdict"] == "good"
    assert rep["coverage"] == 1.0
    assert rep["gaussians"] == n


def test_assess_poor_scene():
    n = 100
    seen = torch.zeros(n)
    seen[:20] = 1.0  # only 20% observed, once each
    denom = torch.zeros(n)
    denom[:20] = 0.001
    rep = _engine_with(seen, denom).assess()
    assert rep["verdict"] == "poor"
    assert abs(rep["coverage"] - 0.2) < 1e-5


def test_read_vocab_file(tmp_path):
    p = tmp_path / "classes.txt"
    p.write_text("house\n# a comment\ntree  \n\nfire hydrant # inline\n")
    assert read_vocab_file(p) == ["house", "tree", "fire hydrant"]
