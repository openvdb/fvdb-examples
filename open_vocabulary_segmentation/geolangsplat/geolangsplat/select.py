# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Turn per-Gaussian scores into selections / labels.

* :func:`select_query` - threshold + concept competition for a single prompt.
* :func:`spatial_cleanup` - voxel cull of isolated selected Gaussians (floaters).
* :func:`assign_labels` - multi-class argmax with confidence/ambiguity/weight gates.
"""
from __future__ import annotations

import torch

_EPS = 1e-8

# Coarse "is a kind of" map so we don't compete a query against its own
# hypernym/synonym (e.g. don't let the "vehicle" distractor suppress "car").
HYPERNYMS: dict[str, set[str]] = {
    "car": {"vehicle"},
    "bus": {"vehicle"},
    "truck": {"vehicle"},
    "van": {"vehicle"},
    "house": {"building"},
    "garage": {"building"},
    "road": {"pavement", "ground"},
    "sidewalk": {"pavement", "ground"},
    "pavement": {"ground"},
    "grass": {"vegetation", "ground"},
    "tree": {"vegetation"},
}


def _sing(w: str) -> str:
    """Crude singularizer so 'cars' competes like 'car'."""
    w = w.lower().strip()
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def competitor_idx(query: str, dist_names) -> list[int]:
    """Indices of distractors the query should compete against (drops the query
    itself, substring synonyms, and the query's hypernyms/co-hyponyms)."""
    ql = _sing(query)
    q_parents = HYPERNYMS.get(ql, set())
    keep: list[int] = []
    for i, n in enumerate(dist_names):
        nl = n.lower().strip()
        if nl in query.lower() or query.lower() in nl:
            continue
        if _sing(nl) == ql or _sing(nl) in q_parents:
            continue
        keep.append(i)
    return keep


def smooth_scores(
    means: torch.Tensor,
    qscore: torch.Tensor,
    denom: torch.Tensor | None,
    cfg,
) -> torch.Tensor:
    """Voxel-grid smoothing of per-Gaussian scores (training-free regularization).

    Blends each Gaussian's score with the (optionally render-weighted) mean score of
    the Gaussians sharing its voxel. This denoises the lifted score and fills objects
    -- the cheap analog of LangSplat's per-scene field optimization -- and is applied
    *before* thresholding so it lifts both recall (filled interiors) and precision
    (isolated false positives get pulled down toward their empty neighborhood).
    """
    if not getattr(cfg, "smooth", False) or float(cfg.smooth_beta) <= 0:
        return qscore
    span = float((means.max(dim=0).values - means.min(dim=0).values).max())
    vox = max(cfg.smooth_vox_frac * span, 1e-9)
    keys = torch.floor(means / vox).long()
    _uniq, inv = torch.unique(keys, dim=0, return_inverse=True)
    M = int(inv.max()) + 1 if inv.numel() else 0
    if M == 0:
        return qscore
    if denom is not None and getattr(cfg, "smooth_weighted", True):
        w = denom.clamp_min(_EPS)
    else:
        w = torch.ones_like(qscore)
    num = torch.zeros(M, device=qscore.device, dtype=qscore.dtype).scatter_add_(0, inv, qscore * w)
    den = torch.zeros(M, device=qscore.device, dtype=qscore.dtype).scatter_add_(0, inv, w)
    local = (num / den.clamp_min(_EPS))[inv]
    beta = float(cfg.smooth_beta)
    return (1.0 - beta) * qscore + beta * local


def select_query(
    qscore: torch.Tensor,
    seen: torch.Tensor,
    cfg,
    *,
    query: str = "",
    select: float | None = None,
    margin: float | None = None,
    compete: bool | None = None,
    denom: torch.Tensor | None = None,
    dist_scores: torch.Tensor | None = None,
    dist_names=None,
    select_mode: str | None = None,
    select_rel: float | None = None,
    min_keep: int | None = None,
    support: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boolean ``[N]`` selection mask for a single prompt.

    The candidate set (observed, well-observed, competition-surviving Gaussians) is
    fixed first; the score threshold is then applied within it. ``select_mode``:

    * ``"fixed"``    -- keep candidates with ``qscore >= select`` (absolute floor).
    * ``"relative"`` -- keep candidates with ``qscore >= select_rel * max(qscore)``
      over the candidates. Robust to prompts whose grounding scores are globally
      weak (the main recall killer); always keeps at least the peak candidate.

    ``min_keep`` is a non-empty guard for ``fixed`` mode: if the threshold selects
    nothing, the top-``min_keep`` candidates by ``qscore`` are kept instead.
    """
    select = cfg.select if select is None else select
    margin = cfg.margin if margin is None else margin
    compete = cfg.compete if compete is None else compete
    select_mode = getattr(cfg, "select_mode", "fixed") if select_mode is None else select_mode
    select_rel = getattr(cfg, "select_rel", 0.5) if select_rel is None else select_rel
    min_keep = getattr(cfg, "min_keep", 0) if min_keep is None else min_keep

    cand = seen > 0
    if denom is not None and cfg.min_weight > 0:
        cand = cand & (denom >= cfg.min_weight)
    if support is not None and getattr(cfg, "consensus", False):
        need = torch.clamp(
            float(getattr(cfg, "consensus_frac", 0.0)) * seen,
            min=float(getattr(cfg, "consensus_min", 1)),
        )
        cand = cand & (support >= need)
    if compete and dist_scores is not None and dist_names:
        keep = competitor_idx(query, dist_names)
        if keep:
            dmax = dist_scores[:, keep].max(dim=1).values
            cand = cand & (qscore >= dmax + margin)

    if select_mode == "relative":
        cand_scores = qscore[cand]
        thr = select_rel * float(cand_scores.max()) if cand_scores.numel() else select
        sel = cand & (qscore >= thr)
    else:
        sel = cand & (qscore >= select)

    if min_keep > 0 and int(sel.sum()) == 0 and int(cand.sum()) > 0:
        cand_idx = cand.nonzero(as_tuple=True)[0]
        k = min(int(min_keep), cand_idx.numel())
        top = qscore[cand_idx].topk(k).indices
        sel = torch.zeros_like(cand)
        sel[cand_idx[top]] = True
    return sel


def spatial_cleanup(means: torch.Tensor, sel: torch.Tensor, cfg) -> torch.Tensor:
    """Drop isolated selected Gaussians: voxelize, keep voxels with >= min_pts."""
    if not cfg.clean3d or int(sel.sum()) == 0:
        return sel
    pts = means[sel]
    span = float((means.max(dim=0).values - means.min(dim=0).values).max())
    vox = max(cfg.voxel_frac * span, 1e-9)
    keys = torch.floor(pts / vox).long()
    _uniq, inv, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
    survive = counts[inv] >= cfg.min_pts
    out = sel.clone()
    out[sel.nonzero(as_tuple=True)[0]] = survive
    return out


def assign_labels(
    scores: torch.Tensor,
    denom: torch.Tensor,
    min_weight: float,
    tau: float,
    delta: float,
) -> tuple[torch.Tensor, dict]:
    """Multi-class label per Gaussian via argmax with three gates.

    ``scores`` is ``[N, C]`` (one column per vocabulary word). A Gaussian is
    labeled with its top class only if: the top score clears ``tau`` (confidence),
    the top1-top2 margin clears ``delta`` (unambiguous), and the accumulated
    render weight clears ``min_weight`` (well-observed). Otherwise label = -1.

    Returns ``(label_ids [N] long, stats)``.
    """
    N, C = scores.shape
    dev = scores.device
    k = min(2, C)
    top = scores.topk(k, dim=1)
    top1v = top.values[:, 0]
    top2v = top.values[:, 1] if C > 1 else torch.zeros(N, device=dev)
    lab = top.indices[:, 0].clone().long()

    confident = top1v >= tau
    unambiguous = (top1v - top2v) >= delta if C > 1 else torch.ones(N, dtype=torch.bool, device=dev)
    enough = denom >= min_weight
    multi = (scores >= tau).sum(dim=1) >= 2  # how many gaussians have >=2 classes above tau

    keep = confident & unambiguous & enough
    lab[~keep] = -1

    stats = {
        "total": int(N),
        "well_observed": int(enough.sum()),
        "confident": int((confident & enough).sum()),
        "ambiguous_dropped": int((confident & enough & ~unambiguous).sum()),
        "multi_class_candidates": int((multi & enough).sum()),
        "labeled": int((lab >= 0).sum()),
    }
    return lab, stats
