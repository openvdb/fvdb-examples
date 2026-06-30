# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import json

import numpy as np
import torch

import geolangsplat.outputs as out


class _FakeGS:
    """Captures the tensors passed to from_tensors and 'saves' the count."""

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


def _patch_gs(monkeypatch):
    monkeypatch.setattr(out, "_gs", lambda: _FakeGS)


def test_write_ply_segmented_subsets(monkeypatch, fake_model, tmp_path):
    _patch_gs(monkeypatch)
    sel = torch.zeros(12, dtype=torch.bool)
    sel[:5] = True
    n = out.write_ply_segmented(fake_model, sel, tmp_path / "seg.ply")
    assert n == 5
    assert _FakeGS.last.kw["means"].shape[0] == 5  # only selected written


def test_write_ply_overlay_preserves_count(monkeypatch, fake_model, tmp_path):
    _patch_gs(monkeypatch)
    sel = torch.zeros(12, dtype=torch.bool)
    sel[:3] = True
    out.write_ply_overlay(fake_model, sel, tmp_path / "ov.ply", color=(1, 0, 0), blend=0.5)
    assert _FakeGS.last.kw["means"].shape[0] == 12  # full splat preserved
    # selected sh0 changed, unselected unchanged
    new_sh0 = _FakeGS.last.kw["sh0"]
    assert not torch.allclose(new_sh0[:3], fake_model.sh0[:3])
    assert torch.allclose(new_sh0[3:], fake_model.sh0[3:])


def test_report_csv_order_and_legend(tmp_path):
    means = np.arange(12 * 3).reshape(12, 3).astype(np.float32)
    labels = np.array([0, 1, -1, 0, 1, 2, -1, 0, 1, 2, 0, 1])
    scores = np.random.RandomState(0).rand(12)
    out.write_report_csv(means, labels, scores, ["a", "b", "c"], tmp_path / "r.csv")
    lines = (tmp_path / "r.csv").read_text().strip().splitlines()
    assert lines[0] == "index,x,y,z,label_id,label,score"
    assert len(lines) == 13  # header + 12 rows, in order
    assert lines[1].startswith("0,") and lines[12].startswith("11,")
    legend = json.loads((tmp_path / "legend.json").read_text())
    assert legend["0"] == "a" and legend["-1"] == "unlabeled"


def test_report_npz_roundtrip(tmp_path):
    means = np.zeros((4, 3), dtype=np.float32)
    labels = np.array([0, -1, 1, 0])
    scores = np.zeros((4, 2), dtype=np.float32)
    out.write_report_npz(means, labels, scores, ["a", "b"], tmp_path / "r.npz")
    d = np.load(tmp_path / "r.npz", allow_pickle=True)
    assert d["label_ids"].tolist() == [0, -1, 1, 0]
    assert list(d["vocab"]) == ["a", "b"]
