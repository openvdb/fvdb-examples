# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Lift 2D SAM3 score maps onto per-Gaussian scores.

The default, recommended lift is alpha-weighted back-projection: for each view we
ask fVDB which Gaussians contribute to each pixel and by how much
(``render_contributing_gaussian_ids``), then distribute that pixel's SAM3 score
to those Gaussians weighted by their contribution. Summing over views and
dividing by the accumulated weight gives a continuous per-Gaussian score (no
binary "is this Gaussian in the mask" decision, which is what makes naive
back-projection look splatty).

A legacy depth-band lift (single visible Gaussian per pixel + multi-view
consensus) is also provided for the rendered-orbit path.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .cameras import _perp_basis, intrinsics, project, up_vector
from .sam3 import encode_progress, encode_progress_done
from .views import View, core_extent

_EPS = 1e-8


# --- alpha-weighted lift ---------------------------------------------------


@dataclass
class AlphaCache:
    """Per-view contribution lists + accumulated per-Gaussian weight."""

    a_ids: list = field(default_factory=list)  # [Mi] long, gaussian id per contribution
    a_pix: list = field(default_factory=list)  # [Mi] long, pixel index per contribution
    a_w: list = field(default_factory=list)  # [Mi] half, contribution weight
    a_hw: list = field(default_factory=list)  # (h, w) render resolution per view
    denom: torch.Tensor | None = None  # [N] total accumulated weight per gaussian
    seen: torch.Tensor | None = None  # [N] number of views each gaussian appears in


def _lift_resolution(height: int, width: int, lift_res: int) -> tuple[int, int]:
    """Scale a view down so its long side is ``lift_res`` (caps render cost)."""
    scale = lift_res / max(height, width) if max(height, width) > lift_res else 1.0
    rh = max(int(round(height * scale)), 1)
    rw = max(int(round(width * scale)), 1)
    return rh, rw


@torch.no_grad()
def view_contrib(model, v: View, cfg, device: torch.device):
    """Per-pixel top-k contributing Gaussians for one view.

    Returns ``(ids[M] long, pix[M] long, w[M] float, (rh, rw))`` -- the flat list
    of (gaussian, pixel, weight) contributions at the lift resolution. Shared by the
    cached build (:func:`build_alpha_cache`) and the streaming lift, so both compute
    the lift identically.
    """
    rh, rw = _lift_resolution(v.height, v.width, cfg.lift_res)
    sx, sy = rw / v.width, rh / v.height
    K = v.K.clone().float()
    K[0, :] *= sx
    K[1, :] *= sy
    ids_j, w_j = model.render_contributing_gaussian_ids(
        world_to_camera_matrices=v.w2c.unsqueeze(0).float(),
        projection_matrices=K.unsqueeze(0),
        image_width=rw,
        image_height=rh,
        near=0.01,
        far=1e12,
        top_k_contributors=cfg.top_k,
    )
    ids = ids_j.jdata.reshape(-1).long()
    w = w_j.jdata.reshape(-1).float()
    off = ids_j.joffsets.reshape(-1).long()
    counts = off[1:] - off[:-1]
    npix = rh * rw
    pix = torch.repeat_interleave(torch.arange(counts.numel(), device=device) % npix, counts)
    valid = ids >= 0
    return ids[valid], pix[valid], w[valid], (rh, rw)


@torch.no_grad()
def build_alpha_cache(model, views: list[View], cfg, device: torch.device) -> AlphaCache:
    """Render per-pixel top-k contributing Gaussians for every view and cache them."""
    N = model.means.shape[0]
    cache = AlphaCache(denom=torch.zeros(N, device=device), seen=torch.zeros(N, device=device))
    t0 = time.time()
    for v in views:
        ids, pix, w, (rh, rw) = view_contrib(model, v, cfg, device)
        cache.denom.scatter_add_(0, ids, w)
        # seen counts distinct VIEWS per Gaussian (not per-pixel contributions),
        # so it is bounded by len(views) and is a meaningful coverage signal.
        cache.seen[torch.unique(ids)] += 1.0
        cache.a_ids.append(ids)
        cache.a_pix.append(pix)
        cache.a_w.append(w.half())
        cache.a_hw.append((rh, rw))
    print(f"[lift] alpha cache for {len(views)} views in {time.time() - t0:.1f}s", flush=True)
    return cache


@torch.inference_mode()
def aggregate_alpha(
    scorer,
    states,
    cache: AlphaCache,
    prompt: str,
    cfg,
    text_outputs=None,
    want_peak: bool = False,
    want_support: bool = False,
):
    """Alpha-weighted per-Gaussian score for ``prompt`` (mean over contributions),
    optionally also the per-Gaussian peak and a multi-view support count.

    Returns ``mean [N]``; with ``want_peak`` and/or ``want_support`` it returns the
    extra tensors after ``mean`` in the order ``(mean[, peak][, support])``. ``support``
    counts, per Gaussian, the number of views in which its best contributed score
    cleared ``cfg.consensus_thr`` -- the signal the consensus gate thresholds on.
    """
    N = cache.denom.shape[0]
    dev = cache.denom.device
    num = torch.zeros(N, device=dev)
    peak = torch.zeros(N, device=dev) if want_peak else None
    support = torch.zeros(N, device=dev) if want_support else None
    sthr = float(getattr(cfg, "consensus_thr", 0.3))
    amp_dtype = getattr(scorer, "amp_dtype", torch.bfloat16)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=(dev.type == "cuda")):
        if text_outputs is None:
            text_outputs = scorer.forward_text(prompt)
        for i, state in enumerate(states):
            ids = cache.a_ids[i]
            if ids.numel() == 0:
                continue
            rh, rw = cache.a_hw[i]
            # Decode masks straight to the lift resolution: avoids upsampling every
            # instance mask to full photo res just to immediately shrink it.
            smap = scorer.scoremap(prompt, state, text_outputs, out_hw=(rh, rw))
            if smap is None:
                continue
            if smap.shape != (rh, rw):  # band/stand-in scorers may ignore out_hw
                smap = F.interpolate(smap[None, None].float(), size=(rh, rw), mode="bilinear", align_corners=False)[
                    0, 0
                ]
            vals = smap.reshape(-1)[cache.a_pix[i].long()]
            w = cache.a_w[i].float()
            num.scatter_add_(0, ids, w * vals)
            if want_peak:
                peak.scatter_reduce_(0, ids, vals, reduce="amax", include_self=True)
            if want_support:
                # per-view best score per Gaussian, then a vote if it clears the floor
                vview = torch.zeros(N, device=dev)
                vview.scatter_reduce_(0, ids, vals, reduce="amax", include_self=True)
                support += (vview >= sthr).float()
    mean = num / cache.denom.clamp_min(_EPS)
    out = [mean]
    if want_peak:
        out.append(peak)
    if want_support:
        out.append(support)
    return tuple(out) if len(out) > 1 else mean


def alpha_qscore(scorer, states, cache: AlphaCache, prompt: str, cfg, text_outputs=None) -> torch.Tensor:
    """Per-Gaussian query score blending mean and (optionally) peak per ``cfg.peak``."""
    if cfg.peak > 0:
        qmean, qpk = aggregate_alpha(scorer, states, cache, prompt, cfg, text_outputs, want_peak=True)
        return (1 - cfg.peak) * qmean + cfg.peak * qpk
    return aggregate_alpha(scorer, states, cache, prompt, cfg, text_outputs)


def alpha_qscore_support(scorer, states, cache: AlphaCache, prompt: str, cfg, text_outputs=None):
    """Like :func:`alpha_qscore` but also returns the per-Gaussian multi-view support
    count for the consensus gate. Returns ``(qscore [N], support [N])``."""
    if cfg.peak > 0:
        qmean, qpk, support = aggregate_alpha(
            scorer, states, cache, prompt, cfg, text_outputs, want_peak=True, want_support=True
        )
        return (1 - cfg.peak) * qmean + cfg.peak * qpk, support
    qmean, support = aggregate_alpha(scorer, states, cache, prompt, cfg, text_outputs, want_support=True)
    return qmean, support


@torch.inference_mode()
def aggregate_alpha_views(
    scorer, states, cache: AlphaCache, prompt: str, cfg, view_idx, text_outputs=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alpha aggregation over a SUBSET of views (the fast query path).

    Identical math to :func:`aggregate_alpha`, but only the views in ``view_idx`` are
    decoded and the denominator/seen counts are accumulated over *just that subset* (so
    the per-Gaussian mean stays correct). The grounding decode is the per-query cost, so
    decoding k of N views is ~N/k faster. Returns ``(qscore [N], denom [N], seen [N])``
    -- denom/seen feed selection so it behaves like the full path on the chosen views.
    """
    N = cache.denom.shape[0]
    dev = cache.denom.device
    num = torch.zeros(N, device=dev)
    denom = torch.zeros(N, device=dev)
    seen = torch.zeros(N, device=dev)
    peak = torch.zeros(N, device=dev) if cfg.peak > 0 else None
    amp_dtype = getattr(scorer, "amp_dtype", torch.bfloat16)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=(dev.type == "cuda")):
        if text_outputs is None:
            text_outputs = scorer.forward_text(prompt)
        for i in view_idx:
            ids = cache.a_ids[i]
            if ids.numel() == 0:
                continue
            rh, rw = cache.a_hw[i]
            w = cache.a_w[i].float()
            denom.scatter_add_(0, ids, w)
            seen[torch.unique(ids)] += 1.0
            smap = scorer.scoremap(prompt, states[i], text_outputs, out_hw=(rh, rw))
            if smap is None:
                continue
            if smap.shape != (rh, rw):
                smap = F.interpolate(smap[None, None].float(), size=(rh, rw), mode="bilinear", align_corners=False)[
                    0, 0
                ]
            vals = smap.reshape(-1)[cache.a_pix[i].long()]
            num.scatter_add_(0, ids, w * vals)
            if peak is not None:
                peak.scatter_reduce_(0, ids, vals, reduce="amax", include_self=True)
    mean = num / denom.clamp_min(_EPS)
    if peak is not None:
        mean = (1 - cfg.peak) * mean + cfg.peak * peak
    return mean, denom, seen


# --- streaming (low-VRAM) alpha lift ---------------------------------------


def _view_azimuth(v: View, center: np.ndarray, up: np.ndarray) -> float:
    """Azimuth (deg) of a view's camera around the up axis, in the perpendicular
    plane. Used to measure whether agreeing views come from *different sides*."""
    c2w = torch.linalg.inv(v.w2c.float()).cpu().numpy()
    d = c2w[:3, 3].astype(np.float64) - center
    d = d - float(d @ up) * up  # project onto the plane perpendicular to up
    e1, e2 = _perp_basis(up)
    return math.degrees(math.atan2(float(d @ e2), float(d @ e1)))


def _azimuth_coverage(azimuths: list[float]) -> float:
    """How much of the 360 deg ring the azimuths span = 360 - largest gap.

    Two views on opposite sides -> ~180; views clustered on one side -> small. This
    is the "across each other enough" guard so we never early-stop from a cluster of
    near-identical viewpoints all looking from the same direction.
    """
    if len(azimuths) < 2:
        return 0.0
    a = sorted(x % 360.0 for x in azimuths)
    gaps = [a[i + 1] - a[i] for i in range(len(a) - 1)] + [a[0] + 360.0 - a[-1]]
    return 360.0 - max(gaps)


@torch.inference_mode()
def stream_scores(
    model,
    scorer,
    views: list[View],
    cfg,
    prompts,
    device: torch.device,
    *,
    want_peak: bool = False,
    early_stop: bool = False,
):
    """Low-VRAM lift: render + encode + score + accumulate + EVICT, one view at a time.

    Equivalent to building the full cache and running :func:`aggregate_alpha` for each
    prompt, but it never holds more than a single view's embedding resident -- so peak
    VRAM is bounded to ``SAM3 weights + one view`` regardless of view count (vs. the
    warm path's all-views embedding cache). The trade is no reuse: every call re-encodes,
    so this backs the one-shot path while ``gls serve`` keeps the fast warm cache.

    With a single prompt and ``early_stop``, it stops once ``cfg.agree_k`` views with a
    hit span ``cfg.min_azimuth_spread`` of azimuth (agreement from different sides) and
    the selection has converged -- bounding work to "as many views as the object needs".
    Early-stop is only sound for a *compact* subject; the engine gates it to
    object/dome captures (see ``GeoLangSplat._stream_early_stop``).

    Views are processed in chunks of ``cfg.stream_chunk``: each chunk is encoded in one
    batched SAM3 call (fast) then evicted before the next, so peak VRAM is bounded to
    ``SAM3 weights + one chunk`` regardless of view count -- vs. the warm path's
    all-views embedding cache. The trade is no reuse: every call re-encodes, so this
    backs the one-shot path while ``gls serve`` keeps the fast warm cache.

    Returns ``(scores [C, N], denom [N], seen [N], stats)``.
    """
    N = model.means.shape[0]
    C = len(prompts)
    num = torch.zeros(C, N, device=device)
    denom = torch.zeros(N, device=device)
    seen = torch.zeros(N, device=device)
    peak = torch.zeros(C, N, device=device) if want_peak else None

    cap = int(getattr(cfg, "view_cap", 0) or 0)
    cap = min(cap, len(views)) if cap > 0 else len(views)
    chunk = max(1, int(getattr(cfg, "stream_chunk", 8)))
    batch_pref = bool(getattr(cfg, "batch_encode", True))
    agree_k = int(getattr(cfg, "agree_k", 12))
    min_spread = float(getattr(cfg, "min_azimuth_spread", 120.0))
    conv = float(getattr(cfg, "converge_frac", 0.02))
    center, _r = core_extent(model.means)
    up = up_vector(getattr(cfg, "up", "+z"))
    hit_az: list[float] = []
    prev_sel = -1
    used = 0
    stopped = False
    amp_dtype = getattr(scorer, "amp_dtype", torch.bfloat16)

    def _encode_chunk(vlist):
        """Batched encode, grouping by resolution so a single odd-sized view in the
        chunk doesn't force the whole chunk onto the slow per-view path (matters for
        aerial real photos, which can vary in size). Order is preserved."""
        if not batch_pref or len(vlist) <= 1:
            return [scorer.encode([v.pil], v.height, v.width, batch_encode=False)[0] for v in vlist]
        groups: dict = {}
        for i, v in enumerate(vlist):
            groups.setdefault((v.height, v.width), []).append(i)
        states = [None] * len(vlist)
        for (h, w), idxs in groups.items():
            pils = [vlist[i].pil for i in idxs]
            enc = scorer.encode(pils, h, w, batch_encode=(len(idxs) > 1), batch_size=len(idxs))
            for k, i in enumerate(idxs):
                states[i] = enc[k]
        return states

    with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
        text_outs = [scorer.forward_text(p) for p in prompts]
        for c0 in range(0, cap, chunk):
            chunk_views = [views[vi] for vi in range(c0, min(c0 + chunk, cap))]
            # 1) render contributions for the chunk (cheap, no embeddings resident)
            contribs = [view_contrib(model, v, cfg, device) for v in chunk_views]
            # 2) one batched SAM3 encode for the whole chunk
            states = _encode_chunk(chunk_views)
            # 3) score + accumulate, then evict the chunk
            for v, (ids, pix, w, (rh, rw)), state in zip(chunk_views, contribs, states):
                used += 1
                if ids.numel() == 0:
                    continue
                denom.scatter_add_(0, ids, w)
                seen[torch.unique(ids)] += 1.0
                hit = False
                for c, p in enumerate(prompts):
                    smap = scorer.scoremap(p, state, text_outs[c], out_hw=(rh, rw))
                    if smap is None:
                        continue
                    if smap.shape != (rh, rw):
                        smap = F.interpolate(
                            smap[None, None].float(), size=(rh, rw), mode="bilinear", align_corners=False
                        )[0, 0]
                    vals = smap.reshape(-1)[pix]
                    num[c].scatter_add_(0, ids, w * vals)
                    if want_peak:
                        peak[c].scatter_reduce_(0, ids, vals, reduce="amax", include_self=True)
                    if c == 0 and float((vals > cfg.sam_conf).sum()) > 0:
                        hit = True
                if early_stop and C == 1 and hit:
                    hit_az.append(_view_azimuth(v, center, up))
            del contribs, states  # evict the chunk's footprint before the next
            encode_progress(min(c0 + chunk, cap), cap)

            if early_stop and C == 1 and len(hit_az) >= agree_k and _azimuth_coverage(hit_az) >= min_spread:
                cur = int(((num[0] / denom.clamp_min(_EPS)) >= cfg.select).sum())
                if prev_sel >= 0 and abs(cur - prev_sel) <= conv * max(prev_sel, 1):
                    stopped = True
                    break
                prev_sel = cur
        encode_progress_done()

    mean = num / denom.clamp_min(_EPS)
    if want_peak:
        mean = (1 - cfg.peak) * mean + cfg.peak * peak
    stats = {"views_used": used, "views_total": len(views), "early_stopped": stopped}
    return mean, denom, seen, stats


# --- legacy depth-band lift (rendered-orbit path) --------------------------


@dataclass
class BandCache:
    """Single-visible-Gaussian-per-pixel cache for the depth-band lift."""

    cidx: list = field(default_factory=list)  # [Ki] gaussian ids visible in view i
    cvv: list = field(default_factory=list)  # [Ki] pixel rows
    cuu: list = field(default_factory=list)  # [Ki] pixel cols
    hw: list = field(default_factory=list)
    seen: torch.Tensor | None = None


@torch.no_grad()
def build_band_cache(model, views: list[View], cfg, device: torch.device) -> BandCache:
    """Project means into each view and keep the depth-visible ones."""
    means = model.means.detach()
    N = means.shape[0]
    cache = BandCache(seen=torch.zeros(N, device=device))
    for v in views:
        H, W = v.height, v.width
        f = float(v.K[0, 0])
        cx, cy = float(v.K[0, 2]), float(v.K[1, 2])
        img, alpha = model.render_images_and_depths(
            world_to_camera_matrices=v.w2c.unsqueeze(0).float(),
            projection_matrices=v.K.unsqueeze(0).float(),
            image_width=W,
            image_height=H,
            near=0.01,
            far=1e12,
        )
        depth = img[0, ..., 3]
        a0 = alpha[0, ..., 0]
        u, vv, zc = project(means, v.w2c.float(), f, cx, cy)
        ui, vj = u.round().long(), vv.round().long()
        infr = (zc > 0.01) & (ui >= 0) & (ui < W) & (vj >= 0) & (vj < H)
        uic, vjc = ui.clamp(0, W - 1), vj.clamp(0, H - 1)
        vis = infr & (a0[vjc, uic] > 0.5) & (zc <= depth[vjc, uic] * (1.0 + cfg.depth_tol))
        idx = vis.nonzero(as_tuple=True)[0]
        cache.cidx.append(idx)
        cache.cvv.append(vjc[idx])
        cache.cuu.append(uic[idx])
        cache.hw.append((H, W))
        cache.seen[idx] += 1
    return cache


@torch.inference_mode()
def aggregate_band(
    scorer, states, cache: BandCache, prompt: str, cfg, text_outputs=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Footprint-sampled, multi-view-consensus per-Gaussian score. Returns
    ``(smax [N], hits [N])`` where ``smax`` is the best per-view score and
    ``hits`` counts views scoring above ``cfg.view_thresh``."""
    N = cache.seen.shape[0]
    dev = cache.seen.device
    smax = torch.zeros(N, device=dev)
    hits = torch.zeros(N, device=dev)
    k = 2 * int(cfg.foot) + 1
    amp_dtype = getattr(scorer, "amp_dtype", torch.bfloat16)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=(dev.type == "cuda")):
        if text_outputs is None:
            text_outputs = scorer.forward_text(prompt)
        for i, state in enumerate(states):
            idx = cache.cidx[i]
            if idx.numel() == 0:
                continue
            smap = scorer.scoremap(prompt, state, text_outputs)
            if smap is None:
                continue
            if k > 1:
                smap = F.max_pool2d(smap[None, None].float(), kernel_size=k, stride=1, padding=int(cfg.foot))[0, 0]
            vals = smap[cache.cvv[i], cache.cuu[i]].float()
            smax[idx] = torch.maximum(smax[idx], vals)
            hits[idx] += (vals >= cfg.view_thresh).float()
    return smax, hits
