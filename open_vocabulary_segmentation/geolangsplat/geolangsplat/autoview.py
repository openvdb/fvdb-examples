# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Automatic view-generation config from scene geometry.

The named recipes (``satellite``, ``aerial`` ...) hard-code absolute orbit radii
that only frame scenes at one particular scale. The ``auto`` recipe instead
*measures* the splat and derives a render config scaled to that scene, so the
same command frames a tiny object scene and a city-block satellite tile alike.

Flow: :func:`measure_geometry` probes the point cloud (robustly, ignoring
floaters), :func:`recommend_view_config` turns those stats into orbit tiers and
selection defaults, and :func:`apply_auto_view_config` fills them into a config
(respecting any value the caller set explicitly, exactly like ``apply_recipe``).
This is deterministic, so ``gls check`` and ``gls segment`` derive the *same*
config from a scene -- check just shows you what segment will do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .cameras import up_vector
from .config import GeoLangSplatConfig, Tier

# --- up-axis (RANSAC ground-plane) estimation -----------------------------
_UP_MIN_INLIERS = 0.3  # min inlier fraction to trust an estimated ground plane
_UP_PLANE_TOL = 0.02  # plane inlier band as a fraction of the bbox diagonal
_UP_ITERS = 300
_UP_SAMPLE = 20000  # subsample cap for the RANSAC fit (speed)

# --- geometry probing ------------------------------------------------------
_LO_Q = 0.02
_HI_Q = 0.98

# --- aerial recommender ----------------------------------------------------
_OVERVIEW_RADIUS_FRAC = 1.25
_OBLIQUE_RADIUS_FRAC = 0.55
_GRID_MIN = 2
_GRID_MAX = 8
_OVERVIEW_SHARE = 0.3
_VRAM_BASE_GB = 6
_VRAM_PER_VIEW_GB = 0.16

# --- object-centric recommender -------------------------------------------
_OBJECT_FLATNESS = 0.5  # min(extent)/max(extent) above this => roughly isotropic => object
_OBJECT_RADIUS_FRAC = 1.1
_OBJECT_FOV = 55
_OBJECT_ELEVATIONS = (-30, 0, 30, 60)
_OBJECT_AZIMUTH_MIN = 8
_OBJECT_AZIMUTH_MAX = 36

# Dome orbit: elevation rings looking DOWN at the scene core, from low-oblique
# (side-on, to catch object *flanks*) up to near-nadir. We stop above the horizon
# (lowest ring well >0 deg): exactly-level and below-horizon angles see through to
# the unbounded background halo and come out blurry, so we never go under or on-
# level. The lower rings ride closer to the subject's sides; the upper rings cap
# coverage from above. The orbit radius is auto-framed from the dense core.
_GLOBE_ELEVATIONS = (18.0, 32.0, 46.0, 60.0, 74.0, 88.0)
_GLOBE_AZ_MIN = 3


def globe_rings(budget: int, elevations: tuple[float, ...] = _GLOBE_ELEVATIONS):
    """Distribute ``budget`` views across the dome's elevation rings (azimuth count
    per ring proportional to ``cos(elevation)``, so lower rings -- which see more of
    the scene -- get more azimuths and the near-nadir cap isn't over-sampled).
    Returns ``(elevations, az_counts)``; ``sum(az_counts)`` is approximately
    ``budget``.
    """
    weights = [max(0.25, math.cos(math.radians(e))) for e in elevations]
    total = sum(weights)
    az_counts = [max(_GLOBE_AZ_MIN, int(round(budget * w / total))) for w in weights]
    return list(elevations), az_counts


@dataclass
class GeometryStats:
    """Robust scene-extent measurements used to derive a view plan."""

    n: int
    dx: float
    dy: float
    dz: float
    footprint: float
    aspect: float
    vertical_ratio: float
    flatness: float
    span: float


def measure_geometry(means) -> GeometryStats:
    """Probe scene extents robustly (quantile-trimmed to ignore floaters)."""
    m = means.detach().float()
    n = int(m.shape[0])
    q = torch.tensor([_LO_Q, _HI_Q], device=m.device, dtype=m.dtype)
    lo_hi = torch.quantile(m, q, dim=0)
    ext = (lo_hi[1] - lo_hi[0]).clamp_min(1e-6)
    dx = float(ext[0])
    dy = float(ext[1])
    dz = float(ext[2])
    footprint = max(dx, dy)
    aspect = footprint / max(min(dx, dy), 1e-6)
    flatness = min(dx, dy, dz) / max(dx, dy, dz, 1e-6)
    return GeometryStats(
        n=n,
        dx=dx,
        dy=dy,
        dz=dz,
        footprint=footprint,
        aspect=aspect,
        vertical_ratio=dz / max(footprint, 1e-6),
        flatness=flatness,
        span=float(ext.norm()),
    )


def estimate_up(means, seed: int = 0) -> tuple[np.ndarray, float]:
    """Estimate the gravity-up axis from geometry via RANSAC ground-plane fit.

    Real captures (rooms, tabletops, terrain, city tiles) contain a dominant flat
    surface; its normal is gravity. Returns ``(unit_up_vec, confidence)`` where
    confidence is the plane's inlier fraction. The sign is chosen so the bulk of the
    scene sits *above* the plane. Deterministic for a given seed.
    """
    m = means.detach().float().cpu().numpy()
    n = m.shape[0]
    if n < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0
    rng = np.random.default_rng(seed)
    # Drop the unbounded background shell first: in 360/interior captures a sparse
    # far halo of Gaussians dominates the bbox and corrupts the plane fit (the
    # ground plane should come from the dense core, not the floaters).
    lo = np.quantile(m, 0.02, axis=0)
    hi = np.quantile(m, 0.98, axis=0)
    core = m[np.all((m >= lo) & (m <= hi), axis=1)]
    if core.shape[0] >= 3:
        m = core
    if m.shape[0] > _UP_SAMPLE:
        m = m[rng.choice(m.shape[0], _UP_SAMPLE, replace=False)]
    diag = float(np.linalg.norm(m.max(0) - m.min(0)))
    if diag <= 0:
        return np.array([0.0, 0.0, 1.0]), 0.0
    tol = _UP_PLANE_TOL * diag
    best_n, best_in = None, 0
    for _ in range(_UP_ITERS):
        idx = rng.choice(m.shape[0], 3, replace=False)
        p0, p1, p2 = m[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = np.abs((m - p0) @ nrm)
        inl = int((d < tol).sum())
        if inl > best_in:
            best_n, best_in = nrm, inl
    if best_n is None:
        return np.array([0.0, 0.0, 1.0]), 0.0
    conf = best_in / float(m.shape[0])
    # Orient the normal "up" by where the STRUCTURE rises relative to the ground:
    # isolate the dominant ground slab (the half-max-width band around the density
    # peak along the normal), then compare the mass *beyond* it on each side. Above
    # the ground sit buildings/trees (a lot of mass); below it is near-empty (only
    # sparse sub-ground floaters) -- so the heavier side is up. Excluding the slab
    # itself is the crucial part: the ground peak is roughly symmetric and dominates
    # any thin band right at the mode (after subsampling its sign is just noise),
    # while skew/mean-position get tipped by heavy below-ground floaters. Mass beyond
    # the slab is the stable signal (verified +z on JAX_264/JAX_175 across seeds).
    # A wrong sign puts the whole auto orbit *below* the scene, so cameras look up
    # through the ground and miss every rooftop (JAX_264: 0 buildings, coverage 0.38).
    ts = m @ best_n
    lo_t, hi_t = (float(x) for x in np.quantile(ts, [0.02, 0.98]))
    if hi_t > lo_t:
        hist, edges = np.histogram(ts, bins=256, range=(lo_t, hi_t))
        k = int(hist.argmax())
        half = hist[k] / 2.0
        lo_i = k
        while lo_i > 0 and hist[lo_i] > half:
            lo_i -= 1
        hi_i = k
        while hi_i < len(hist) - 1 and hist[hi_i] > half:
            hi_i += 1
        slab_lo, slab_hi = edges[lo_i], edges[hi_i + 1]
        above = int((ts > slab_hi).sum())  # structure rising off the ground
        below = int((ts < slab_lo).sum())  # sparse sub-ground floaters
        if below > above:  # heavier side is below the slab -> normal points down
            best_n = -best_n
    return best_n.astype(np.float64), float(conf)


def resolve_up(up_spec, means) -> np.ndarray:
    """Resolve a config ``up`` to a concrete unit vector.

    Explicit axis (``"+z"``) or a 3-vector is honoured as-is; ``auto``/unset triggers
    geometry estimation, falling back to +z when no confident ground plane is found.
    """
    if isinstance(up_spec, (list, tuple, np.ndarray)):
        return up_vector(up_spec)
    s = str(up_spec).lower().replace(" ", "")
    if s in ("", "auto", "none"):
        vec, conf = estimate_up(means)
        if conf >= _UP_MIN_INLIERS:
            return vec
        return np.array([0.0, 0.0, 1.0])
    return up_vector(s)


def detect_capture(stats: GeometryStats) -> str:
    """Classify the capture as ``"aerial"`` (flat, z-up, top-down) or ``"object"``
    (roughly isotropic, arbitrary frame, 360 ring) from extent isotropy."""
    if stats.flatness >= _OBJECT_FLATNESS:
        return "object"
    return "aerial"


def budget_views(max_views, vram_budget_gb) -> int:
    """Resolve the effective view budget.

    If ``vram_budget_gb`` is set, convert it to a view count via the rough VRAM
    model; otherwise use ``max_views`` directly. Clamped to a sane range.
    """
    if vram_budget_gb and vram_budget_gb > 0:
        n = int((vram_budget_gb - _VRAM_BASE_GB) / _VRAM_PER_VIEW_GB)
        return max(40, min(400, n))
    return max(8, int(max_views))


def _grid_for_budget(budget: int, n_elev: int, num_azimuth: int) -> int:
    """Largest square grid whose view count (grid^2 * n_elev * azimuths) fits the
    per-tier budget, clamped to [_GRID_MIN, _GRID_MAX]."""
    per_center = max(1, n_elev * num_azimuth)
    g = int(math.isqrt(max(1, budget // per_center)))
    return max(_GRID_MIN, min(_GRID_MAX, g))


def _oblique_angles(vertical_ratio: float):
    """Pick oblique azimuth count + elevations from how much vertical structure
    the scene has. Taller structure -> more headings (so every facade is seen) and
    lower (more side-on) elevations."""
    if vertical_ratio > 0.1:
        return (4, (52, 34))
    if vertical_ratio > 0.04:
        return (3, (54, 38))
    return (2, (56, 44))


def _tier_views(tier: Tier) -> int:
    return tier.grid * tier.grid * len(tier.elevations) * tier.num_azimuth


def recommend_view_config(stats: GeometryStats, budget: int) -> dict:
    """Derive an orbit config scaled to ``stats``, sized near ``budget`` total views.

    Dispatches on capture type: a flat z-up aerial slab gets a nadir + oblique
    top-down orbit; a roughly isotropic object-centric scene (arbitrary frame, 360
    ring) gets a spherical orbit around the centroid. Returns config fields to fill
    (``view_source``, ``tiers``, ``select``, ``margin``) plus ``capture``,
    ``n_views_planned`` and a human-readable ``rationale``.
    """
    if detect_capture(stats) == "object":
        return _recommend_object(stats, budget)
    return _recommend_aerial(stats, budget)


def _recommend_object(stats: GeometryStats, budget: int) -> dict:
    """Upper-dome orbit around the core for object-centric / indoor captures.

    The eye sweeps full azimuth rings at elevations looking DOWN at the core
    (oblique -> near-nadir). A bare splat only renders sharply near its original
    inward/above camera region; horizontal and below-horizon angles see through to
    the unbounded background and come out blurry, so the dome stays in the reliable
    region. The radius is auto-framed from the splat's dense core at render time
    rather than guessed from a fixed multiple of the extent (which over/under-zooms
    depending on how the scene was scaled).
    """
    elevs, az_counts = globe_rings(budget)
    n_views = sum(az_counts)
    # Placeholder tier (radius is auto-framed at render; kept only as a fallback if
    # globe rendering is unavailable for some reason).
    tier = Tier(
        radius=round(stats.span * _OBJECT_RADIUS_FRAC, 2),
        elevations=tuple(float(e) for e in elevs),
        fov_deg=_OBJECT_FOV,
        grid=1,
        num_azimuth=max(az_counts),
    )
    rationale = [
        f"capture: object-centric (flatness {stats.flatness:.2f} >= {_OBJECT_FLATNESS}) -> upper-dome orbit (auto-framed)",
        f"span {stats.span:.1f} (x={stats.dx:.1f}, y={stats.dy:.1f}, z={stats.dz:.1f})",
        f"view budget {budget} -> {n_views} views across {len(elevs)} elevation rings "
        f"({elevs[0]:.0f}..{elevs[-1]:.0f} deg, looking down)",
        "radius: auto-framed at render (zoom out until the scene shrinks, then one step in)",
    ]
    return {
        "view_source": "globe",
        "tiers": (tier,),
        "select": 0.33,
        "margin": 0.1,
        "capture": "object",
        "n_views_planned": n_views,
        "rationale": rationale,
    }


def _recommend_aerial(stats: GeometryStats, budget: int) -> dict:
    """Smart nadir + oblique-multi-azimuth orbit scaled to ``stats``, sized
    to land near ``budget`` total views (the cost knob; VRAM/time/query ~ view count).

    The capture has two tiers, mirroring photogrammetry practice:

    * **overview** -- high and near-nadir, frames the whole footprint for ground/
      roof coverage. Cheap (one altitude shot already sees the tile), so it gets a
      small share of the budget.
    * **oblique** -- moves in and looks across the scene from several compass
      headings (azimuths) at lower elevations, so vertical faces (building sides,
      tree canopies, car bodies) are actually observed. This is what disambiguates
      3D structure and lifts quality, so it gets the bulk of the budget.

    Returns config fields to fill (``view_source``, ``tiers``, ``select``,
    ``margin``) plus ``n_views_planned`` and a human-readable ``rationale``.
    """
    fp = stats.footprint
    ov_elev = (80, 60)
    ov_budget = max(2 * _GRID_MIN * _GRID_MIN, int(round(_OVERVIEW_SHARE * budget)))
    g_ov = _grid_for_budget(ov_budget, n_elev=len(ov_elev), num_azimuth=1)
    overview = Tier(
        radius=round(fp * _OVERVIEW_RADIUS_FRAC, 2), elevations=ov_elev, fov_deg=52, grid=g_ov, num_azimuth=1
    )
    ov_views = g_ov * g_ov * len(ov_elev)
    obl_az, obl_elev = _oblique_angles(stats.vertical_ratio)
    obl_budget = max(obl_az * len(obl_elev) * _GRID_MIN * _GRID_MIN, budget - ov_views)
    g_obl = _grid_for_budget(obl_budget, n_elev=len(obl_elev), num_azimuth=obl_az)
    oblique = Tier(
        radius=round(fp * _OBLIQUE_RADIUS_FRAC, 2), elevations=obl_elev, fov_deg=42, grid=g_obl, num_azimuth=obl_az
    )
    tiers = (overview, oblique)
    n_views = sum(_tier_views(t) for t in tiers)
    rationale = [
        f"footprint {fp:.1f} (x={stats.dx:.1f}, y={stats.dy:.1f}), height {stats.dz:.1f}, aspect {stats.aspect:.2f}, vertical_ratio {stats.vertical_ratio:.2f}",
        f"view budget {budget} -> {n_views} views",
        f"overview (nadir): radius {overview.radius:.1f} (={_OVERVIEW_RADIUS_FRAC}x fp), grid {g_ov}, fov 52, elevations {ov_elev}, 1 heading -> {ov_views} views",
        f"oblique (structure): radius {oblique.radius:.1f} (={_OBLIQUE_RADIUS_FRAC}x fp), grid {g_obl}, fov 42, elevations {obl_elev}, {obl_az} headings -> {n_views - ov_views} views",
    ]
    return {
        "view_source": "render",
        "tiers": tiers,
        "select": 0.33,
        "margin": 0.1,
        "capture": "aerial",
        "n_views_planned": n_views,
        "rationale": rationale,
    }


def apply_auto_view_config(cfg: GeoLangSplatConfig, means) -> dict:
    """Fill an auto-derived view config into ``cfg`` in place.

    Only fills fields the caller left at their default (so an explicit
    ``--select`` / ``--view-source`` still wins), mirroring ``apply_recipe``.
    Returns a report dict (``stats``, ``tiers``, ``n_views_planned``,
    ``rationale``, ``applied``) for callers/CLIs to display.
    """
    stats = measure_geometry(means)
    budget = budget_views(cfg.max_views, cfg.vram_budget_gb)
    rec = recommend_view_config(stats, budget=budget)
    default = GeoLangSplatConfig()
    applied: list[str] = []
    for k in ("view_source", "tiers", "select", "margin"):
        if getattr(cfg, k) == getattr(default, k):
            setattr(cfg, k, rec[k])
            applied.append(k)
    if str(cfg.up).lower() in ("", "auto", "none"):
        vec, conf = estimate_up(means)
        if conf >= _UP_MIN_INLIERS:
            cfg.up = vec
            up_note = (
                f"up: estimated from ground plane " f"{tuple(round(float(x), 2) for x in vec)} (inlier {conf:.0%})"
            )
        else:
            cfg.up = np.array([0.0, 0.0, 1.0])
            up_note = f"up: no confident ground plane (inlier {conf:.0%}) -> default +z"
        applied.append("up")
        rec["rationale"] = list(rec["rationale"]) + [up_note]
    return {
        "stats": stats,
        "tiers": cfg.tiers,
        "capture": rec["capture"],
        "up": cfg.up,
        "n_views_planned": rec["n_views_planned"],
        "rationale": rec["rationale"],
        "applied": applied,
    }
