# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch

from geolangsplat.config import GeoLangSplatConfig
from geolangsplat.select import (
    assign_labels,
    competitor_idx,
    select_query,
    smooth_scores,
    spatial_cleanup,
)


def test_competitor_idx_drops_hypernym_and_self():
    names = ["vehicle", "building", "grass", "car"]
    keep = competitor_idx("car", names)
    # "vehicle" (hypernym of car) and "car" (self) dropped; building/grass kept.
    assert set(keep) == {1, 2}


def test_assign_labels_gates():
    # 3 gaussians, 2 classes
    scores = torch.tensor([[0.9, 0.1], [0.3, 0.28], [0.05, 0.02]])
    denom = torch.tensor([1.0, 1.0, 1.0])
    labels, stats = assign_labels(scores, denom, min_weight=0.03, tau=0.15, delta=0.05)
    assert labels[0].item() == 0  # confident + unambiguous
    assert labels[1].item() == -1  # ambiguous (0.3-0.28 < delta)
    assert labels[2].item() == -1  # below tau
    assert stats["labeled"] == 1


def test_assign_labels_min_weight_gate():
    scores = torch.tensor([[0.9, 0.1]])
    denom = torch.tensor([0.001])  # below min_weight
    labels, _ = assign_labels(scores, denom, min_weight=0.03, tau=0.15, delta=0.05)
    assert labels[0].item() == -1


def test_select_query_competition_off_does_not_suppress():
    cfg = GeoLangSplatConfig(device="cpu")  # compete defaults False
    qscore = torch.tensor([0.8, 0.5, 0.2])
    seen = torch.ones(3)
    denom = torch.ones(3)
    dist = torch.tensor([[0.9], [0.9], [0.9]])  # strong distractor everywhere
    sel = select_query(
        qscore,
        seen,
        cfg,
        query="house",
        select=0.4,
        denom=denom,
        dist_scores=dist,
        dist_names=["grass"],
    )
    # competition is off -> distractor is ignored; only the threshold applies
    assert sel.tolist() == [True, True, False]


def test_select_query_threshold_and_competition():
    cfg = GeoLangSplatConfig(device="cpu")
    qscore = torch.tensor([0.8, 0.5, 0.2])
    seen = torch.ones(3)
    denom = torch.ones(3)
    dist = torch.tensor([[0.1], [0.6], [0.0]])  # one distractor column
    sel = select_query(
        qscore,
        seen,
        cfg,
        query="house",
        select=0.4,
        margin=0.08,
        compete=True,
        denom=denom,
        dist_scores=dist,
        dist_names=["grass"],
    )
    # g0: 0.8>=0.4 and 0.8>=0.1+0.08 -> True
    # g1: 0.5>=0.4 but 0.5 < 0.6+0.08 -> False (loses to distractor)
    # g2: 0.2<0.4 -> False
    assert sel.tolist() == [True, False, False]


def test_select_query_relative_rescues_weak_prompt():
    # All scores far below the absolute select (0.30) -> fixed mode returns empty,
    # but relative mode keeps the peak region (the recall fix).
    cfg = GeoLangSplatConfig(device="cpu", select=0.30)
    qscore = torch.tensor([0.10, 0.06, 0.02])
    seen = torch.ones(3)
    denom = torch.ones(3)
    fixed = select_query(qscore, seen, cfg, denom=denom)
    assert fixed.sum().item() == 0  # weak prompt -> nothing (the 0.0-IoU failure)
    rel = select_query(qscore, seen, cfg, denom=denom, select_mode="relative", select_rel=0.5)
    # thr = 0.5 * 0.10 = 0.05 -> keep g0 (0.10) and g1 (0.06), drop g2 (0.02)
    assert rel.tolist() == [True, True, False]


def test_select_query_min_keep_guard():
    # Fixed mode, everything below threshold, but min_keep guarantees the top-1.
    cfg = GeoLangSplatConfig(device="cpu", select=0.30)
    qscore = torch.tensor([0.10, 0.06, 0.02])
    seen = torch.ones(3)
    denom = torch.ones(3)
    sel = select_query(qscore, seen, cfg, denom=denom, min_keep=1)
    assert sel.tolist() == [True, False, False]


def test_smooth_scores_fills_and_denoises():
    # Two voxels: a tight cluster near origin (mixed scores) and a lone far point.
    cfg = GeoLangSplatConfig(device="cpu", smooth=True, smooth_beta=1.0, smooth_vox_frac=0.2, smooth_weighted=False)
    means = torch.tensor([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [5.0, 5.0, 5.0]], dtype=torch.float32)
    q = torch.tensor([0.9, 0.1, 0.2, 0.8])
    out = smooth_scores(means, q, None, cfg)
    # beta=1 -> each score becomes its voxel mean. Cluster mean = (0.9+0.1+0.2)/3 = 0.4.
    assert torch.allclose(out[:3], torch.full((3,), 0.4), atol=1e-5)
    assert torch.allclose(out[3], torch.tensor(0.8), atol=1e-5)  # lone point unchanged


def test_smooth_scores_off_is_identity():
    cfg = GeoLangSplatConfig(device="cpu")  # smooth defaults False
    means = torch.randn(10, 3)
    q = torch.rand(10)
    assert torch.equal(smooth_scores(means, q, None, cfg), q)


def test_select_query_consensus_gate():
    # g0 strong + well-supported; g1 strong score but only 1 supporting view -> gated out.
    cfg = GeoLangSplatConfig(device="cpu", select=0.30, consensus=True, consensus_frac=0.5, consensus_min=2)
    qscore = torch.tensor([0.8, 0.8, 0.1])
    seen = torch.tensor([10.0, 10.0, 10.0])
    denom = torch.ones(3)
    support = torch.tensor([8.0, 1.0, 0.0])  # need = max(2, 0.5*10) = 5
    sel = select_query(qscore, seen, cfg, denom=denom, support=support)
    assert sel.tolist() == [True, False, False]


def test_spatial_cleanup_removes_isolated():
    cfg = GeoLangSplatConfig(device="cpu", clean3d=True, voxel_frac=0.2, min_pts=3)
    # cluster of 4 near origin + 1 far floater
    means = torch.tensor(
        [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0.01, 0.01, 0], [5.0, 5.0, 5.0]],
        dtype=torch.float32,
    )
    sel = torch.ones(5, dtype=torch.bool)
    out = spatial_cleanup(means, sel, cfg)
    assert out[:4].all() and not out[4]
