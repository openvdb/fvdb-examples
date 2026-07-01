# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""End-to-end API tests using an injected fake engine (no GPU/SAM3/fVDB)."""
from __future__ import annotations

import torch

from geolangsplat.api import segment


def test_single_prompt_returns_mask(fake_engine):
    res = segment(None, "house", engine=fake_engine)
    assert res.label_ids is None
    assert res.scores.shape == (fake_engine.N,)
    assert res.selected.dtype == torch.bool
    assert res.num_selected == int(res.selected.sum())
    assert "query_seconds" in res.stats


def test_multi_prompt_returns_labels(fake_engine):
    res = segment(None, ["house", "tree", "grass"], engine=fake_engine)
    assert res.label_ids is not None
    assert res.scores.shape == (fake_engine.N, 3)
    assert bool((res.selected == (res.label_ids >= 0)).all())
    assert "labeled" in res.stats


def test_determinism(fake_engine):
    a = segment(None, "house", engine=fake_engine)
    b = segment(None, "house", engine=fake_engine)
    assert torch.equal(a.scores, b.scores)
    assert torch.equal(a.selected, b.selected)


def test_report_output_writes_file(fake_engine, tmp_path):
    res = segment(None, "house", engine=fake_engine)
    res.to_report(tmp_path / "out.csv")
    lines = (tmp_path / "out.csv").read_text().strip().splitlines()
    assert len(lines) == fake_engine.N + 1  # header + one row per gaussian, in order


def test_empty_prompt_raises(fake_engine):
    import pytest

    with pytest.raises(ValueError):
        segment(None, [], engine=fake_engine)


def test_output_requires_out_path(fake_engine):
    import pytest

    with pytest.raises(ValueError):
        segment(None, "house", engine=fake_engine, output="ply_overlay")
