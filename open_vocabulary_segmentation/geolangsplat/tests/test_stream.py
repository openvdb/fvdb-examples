# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the low-VRAM streaming lift.

Two properties matter: (1) streaming the whole view set produces the SAME per-Gaussian
scores as the cached path (it's the same math, just evicted per view), and (2) the
early-stop fires once enough angularly-diverse views agree.
"""
import numpy as np
import torch
from PIL import Image

from geolangsplat.config import GeoLangSplatConfig
from geolangsplat.lift import (
    _azimuth_coverage,
    aggregate_alpha,
    aggregate_alpha_views,
    build_alpha_cache,
    stream_scores,
)
from geolangsplat.views import View


class _Jag:
    """Stand-in for fVDB's JaggedTensor (only the fields view_contrib reads)."""

    def __init__(self, jdata, joffsets=None):
        self.jdata = jdata
        self.joffsets = joffsets


class StreamFakeModel:
    """Fake splat with deterministic per-view contributions (keyed by w2c[0,3])."""

    def __init__(self, n=10, nviews=6, npix=16, top_k=2, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.means = torch.rand(n, 3, generator=g)
        self.n = n
        self.npix = npix
        self.top_k = top_k
        self._store = {
            i: (torch.randint(0, n, (npix * top_k,), generator=g), torch.rand(npix * top_k, generator=g))
            for i in range(nviews)
        }

    def render_contributing_gaussian_ids(
        self, *, world_to_camera_matrices, projection_matrices, image_width, image_height, near, far, top_k_contributors
    ):
        i = int(round(float(world_to_camera_matrices[0, 0, 3])))
        ids, w = self._store[i]
        joff = torch.arange(0, ids.numel() + 1, top_k_contributors)
        return _Jag(ids.clone(), joff), _Jag(w.clone())


class StreamFakeScorer:
    amp_dtype = torch.float32

    def __init__(self, score=0.8):
        self.score = score

    def encode(self, pils, height, width, batch_encode=False, batch_size=8):
        return [{"i": k} for k in range(len(pils))]

    def forward_text(self, prompt):
        return None

    def scoremap(self, prompt, state, text_outputs=None, out_hw=None):
        rh, rw = out_hw
        return torch.full((rh, rw), self.score)


class PromptStreamScorer(StreamFakeScorer):
    """Scorer whose per-pixel score depends on the prompt (so competition discriminates)."""

    def __init__(self, scores: dict):
        self.scores = scores

    def scoremap(self, prompt, state, text_outputs=None, out_hw=None):
        rh, rw = out_hw
        return torch.full((rh, rw), float(self.scores.get(prompt, 0.0)))


def _views(nviews=6, side=4):
    out = []
    for i in range(nviews):
        w2c = torch.eye(4)
        w2c[0, 3] = float(i)  # view-index tag the fake model reads
        w2c[1, 3] = float(i % 2)  # vary position so azimuths differ
        K = torch.tensor([[2.0, 0, side / 2], [0, 2.0, side / 2], [0, 0, 1]])
        pil = Image.fromarray(np.zeros((side, side, 3), dtype=np.uint8))
        out.append(View(pil=pil, w2c=w2c, K=K, height=side, width=side))
    return out


def _cfg():
    c = GeoLangSplatConfig(device="cpu")
    c.top_k = 2
    c.peak = 0.0
    return c


def _early_stop(view_source, capture, mode="auto"):
    """Exercise GeoLangSplatEngine._stream_early_stop without building an engine."""
    from geolangsplat.engine import GeoLangSplatEngine

    eng = GeoLangSplatEngine.__new__(GeoLangSplatEngine)
    eng.cfg = GeoLangSplatConfig(view_source=view_source, stream_early_stop=mode)
    eng.auto_report = {"capture": capture} if capture else None
    return eng._stream_early_stop()


def test_early_stop_is_capture_gated():
    # Compact subject (object dome) -> early-stop ON; spread-out captures -> OFF so
    # streaming sees every view and recall matches the full build.
    assert _early_stop("globe", None) is True
    assert _early_stop("render", "object") is True
    assert _early_stop("render", "aerial") is False
    assert _early_stop("images", "aerial") is False
    # Explicit override wins over the auto gate either way.
    assert _early_stop("render", "aerial", mode="on") is True
    assert _early_stop("globe", "object", mode="off") is False


def test_azimuth_coverage_spans_the_ring():
    assert _azimuth_coverage([0.0, 180.0]) == 180.0  # opposite sides
    assert _azimuth_coverage([0.0, 10.0, 20.0]) == 20.0  # clustered -> small
    assert _azimuth_coverage([5.0]) == 0.0  # one view -> none


def test_streaming_matches_cached_lift():
    cfg = _cfg()
    model = StreamFakeModel()
    scorer = StreamFakeScorer(score=0.8)
    views = _views()
    dev = torch.device("cpu")

    cache = build_alpha_cache(model, views, cfg, dev)
    states = scorer.encode([v.pil for v in views], views[0].height, views[0].width)
    cached = aggregate_alpha(scorer, states, cache, "thing", cfg)

    scores, denom, seen, stats = stream_scores(model, scorer, views, cfg, ["thing"], dev)
    # Cached path stores weights as fp16 (a_w.half()); streaming keeps full precision,
    # so they match to ~1e-3 (streaming is the more precise of the two).
    assert torch.allclose(scores[0], cached, atol=3e-3)
    assert stats["views_used"] == len(views) and not stats["early_stopped"]
    # every covered gaussian gets the constant score; unseen ones are 0.
    assert torch.allclose(scores[0][denom > 0], torch.full_like(scores[0][denom > 0], 0.8), atol=3e-3)


def test_streaming_chunking_is_invariant():
    """Chunked batched encode must not change the result vs. per-view streaming."""
    model = StreamFakeModel(nviews=6)
    scorer = StreamFakeScorer(score=0.8)
    views = _views(nviews=6)
    dev = torch.device("cpu")

    cfg1 = _cfg()
    cfg1.stream_chunk = 1
    one, *_ = stream_scores(model, scorer, views, cfg1, ["thing"], dev)

    cfg4 = _cfg()
    cfg4.stream_chunk = 4
    four, *_ = stream_scores(model, scorer, views, cfg4, ["thing"], dev)

    assert torch.allclose(one[0], four[0], atol=1e-6)


def test_streaming_competition_suppresses_losing_query():
    """Low-VRAM path scores the query + distractors in one stream, then competes:
    a Gaussian a distractor wins is dropped from the query's selection -- exactly the
    pool-leaks-into-building case, fixed without the warm cache."""
    from geolangsplat.select import select_query

    cfg = _cfg()
    cfg.min_weight = 0.0  # isolate the competition gate from floater culling
    model = StreamFakeModel(nviews=6)
    scorer = PromptStreamScorer({"car": 0.9, "building": 0.5})
    views = _views(nviews=6)
    dev = torch.device("cpu")

    scores, denom, seen, _stats = stream_scores(model, scorer, views, cfg, ["car", "building"], dev)
    covered = denom > 0
    assert int(covered.sum()) > 0

    # 'car' (0.9) beats distractor 'building' (0.5) everywhere -> kept.
    sel_car = select_query(
        scores[0],
        seen,
        cfg,
        query="car",
        compete=True,
        denom=denom,
        dist_scores=scores[1:].t().contiguous(),
        dist_names=["building"],
    )
    assert bool(sel_car[covered].all())

    # 'building' (0.5) clears the 0.30 threshold on its own ...
    sel_bld_solo = select_query(
        scores[1],
        seen,
        cfg,
        query="building",
        compete=False,
        denom=denom,
    )
    assert int(sel_bld_solo.sum()) > 0
    # ... but loses to 'car' (0.9) under competition -> fully suppressed.
    sel_bld = select_query(
        scores[1],
        seen,
        cfg,
        query="building",
        compete=True,
        denom=denom,
        dist_scores=scores[0:1].t().contiguous(),
        dist_names=["car"],
    )
    assert int(sel_bld.sum()) == 0


def test_fast_views_match_full_over_all_views():
    """Subset aggregation over ALL views reproduces the full cached lift, and a
    half-view subset still gives the constant score on covered Gaussians with a
    smaller (subset) denominator -- the fast query path's contract."""
    cfg = _cfg()
    model = StreamFakeModel()
    scorer = StreamFakeScorer(score=0.8)
    views = _views()
    dev = torch.device("cpu")
    cache = build_alpha_cache(model, views, cfg, dev)
    states = scorer.encode([v.pil for v in views], views[0].height, views[0].width)

    full = aggregate_alpha(scorer, states, cache, "thing", cfg)
    allv, denom_all, seen_all = aggregate_alpha_views(scorer, states, cache, "thing", cfg, list(range(len(views))))
    assert torch.allclose(allv, full, atol=3e-3)

    half = list(range(0, len(views), 2))
    sub, denom_sub, seen_sub = aggregate_alpha_views(scorer, states, cache, "thing", cfg, half)
    cov = denom_sub > 0
    assert torch.allclose(sub[cov], torch.full_like(sub[cov], 0.8), atol=3e-3)
    assert float(denom_sub.sum()) < float(denom_all.sum())  # fewer views -> less weight


def test_subset_indices_are_spread():
    from geolangsplat.engine import GeoLangSplatEngine

    eng = GeoLangSplatEngine.__new__(GeoLangSplatEngine)
    eng.views = list(range(100))
    idx = eng._subset_indices(10)
    assert len(idx) == 10 and idx == sorted(idx) and idx[0] == 0 and idx[-1] == 99
    assert eng._subset_indices(0) == list(range(100))  # 0 -> all
    assert eng._subset_indices(500) == list(range(100))  # cap at n


def test_streaming_early_stops_when_views_agree():
    cfg = _cfg()
    cfg.agree_k = 3
    cfg.stream_chunk = 1  # let early-stop fire view-by-view, not per big chunk
    cfg.min_azimuth_spread = 0.0  # only require the count + convergence here
    cfg.converge_frac = 1.0  # constant scores converge immediately
    model = StreamFakeModel(nviews=6)
    scorer = StreamFakeScorer(score=0.9)
    views = _views(nviews=6)

    _scores, _denom, _seen, stats = stream_scores(
        model, scorer, views, cfg, ["thing"], torch.device("cpu"), early_stop=True
    )
    assert stats["early_stopped"]
    assert stats["views_used"] < len(views)
