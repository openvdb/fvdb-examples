# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Unified configuration and capture-type recipes for GeoLangSplat.

A single :class:`GeoLangSplatConfig` drives both aerial and satellite captures.
:data:`RECIPES` are named presets that fill sensible defaults for a capture
type; :func:`apply_recipe` only fills fields the caller left at their default, so
an explicitly set value always wins over the preset.
"""
from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field, fields
from typing import Literal

from .errors import GeoLangSplatError

# Canonical recipe names. Used as a CLI choice type so `tyro` validates the value
# (and can tab-complete it) instead of failing deep inside apply_recipe.
RecipeName = Literal["auto", "satellite", "satellite_dense", "aerial"]

# Generic scene vocabulary used as the default competition set (when competition
# is enabled without an explicit distractor list).
DEFAULT_DISTRACTORS: tuple[str, ...] = (
    "building",
    "ground",
    "road",
    "grass",
    "tree",
    "water",
    "car",
    "sidewalk",
)


@dataclass(frozen=True)
class Tier:
    """One level of the synthesized camera ladder (used when ``view_source='render'``).

    A tier renders a grid of orbit cameras at a given distance (``radius``), set
    of ``elevations``, and field of view (``fov_deg``; lower = optical zoom-in,
    which enlarges small objects). Tiers stack to form a multi-scale view set.
    """

    radius: float
    elevations: tuple[float, ...]
    fov_deg: float = 50.0
    grid: int = 5
    num_azimuth: int = 1


@dataclass
class GeoLangSplatConfig:
    """All knobs for the unified open-vocabulary segmentation pipeline.

    Grouped by stage. The most commonly tuned knobs are ``view_source``,
    ``select``, ``min_weight`` and ``distractors``; prefer a recipe and override
    only what you need.
    """

    # --- scene / IO --------------------------------------------------------
    ply: str = ""
    sfm: str = ""

    # --- view generation ---------------------------------------------------
    # When True, derive the view set from scene geometry (footprint, height,
    # capture type) instead of a fixed tier ladder. The "auto" recipe sets this.
    auto_views: bool = False
    # "render" = synthesize an orbit/ladder from scene geometry;
    # "globe"  = synthesized object/interior close dome;
    # "images" = use the scene's ground-truth SfM photos (COLMAP).
    view_source: str = "render"
    # Inside-out: anchor the eye at the scene core and look OUTWARD (interior /
    # unbounded room captures, where an outside-in orbit only sees wall backs).
    inside_out: bool = False
    inside_out_fov: float = 80.0  # wide FOV for inside-out sweeps
    # Cap on the synthesized/subsampled view count (build VRAM & query latency ~
    # view count). 0 = no cap beyond the recipe/auto plan.
    max_views: int = 200
    vram_budget_gb: float = 0.0  # if >0, trim the view plan to fit this VRAM budget
    # If >0, single-prompt queries decode only this many angularly-spread views
    # (no competition) for a faster interactive answer; 0 uses the full view set.
    fast_views: int = 0
    # Low-VRAM streaming lift: encode + score + project + EVICT one view at a time
    # instead of caching every view's embedding. Peak VRAM is bounded to ~one view
    # regardless of count, at the cost of no warm reuse (re-encodes per call) -- it
    # backs the one-shot path while `gls serve` keeps the fast all-views cache.
    low_vram: bool = False
    stream_chunk: int = 8  # views encoded (batched) then evicted per streaming step
    # Early-stop only helps for a COMPACT subject (object/dome capture): a few
    # angularly-diverse views see the whole thing. On aerial/satellite the class is
    # spread across the scene, so stopping early guts recall -- there we stream every
    # view (bounded VRAM, full recall). "auto" = on only for object/globe captures.
    stream_early_stop: str = "auto"  # "auto" | "on" | "off"
    view_cap: int = 0  # hard ceiling on streamed views (0 = all candidate views)
    agree_k: int = 12  # early-stop: views with a hit that must agree (single prompt)
    min_azimuth_spread: float = 120.0  # ...spanning at least this much azimuth (diff sides)
    converge_frac: float = 0.02  # ...with the selection changing < this fraction to stop
    n_views: int = 90  # how many ground-truth images to subsample (view_source="images")
    orbit_radius_frac: float = 1.2  # camera distance as a fraction of scene span
    lift_res: int = 640  # resolution at which contributing-Gaussian weights are rendered
    size: int = 640  # render size for synthesized views (view_source="render")
    grid_frac: float = 0.9  # footprint fraction covered by the render grid
    zoom: float = 1.0  # global multiplier on every tier radius
    # Dome (globe) view tuning. dome_low_zoom pulls the LOW/oblique rings closer
    # (radius scale at the lowest ring; 1.0 at near-nadir) so side-on subjects that
    # were "covered/blocked" fill more pixels. reject_blur enables the per-view
    # quality gate on the rendered dome: a view is rejected when too little of its
    # center is covered by the splat (view_min_coverage -> see-through/empty), when a
    # near foreground blob blocks the subject (view_max_occlusion), or when it is far
    # blurrier than the set median (blur_rel). Rejected views are resampled once from
    # a higher, closer angle (to clear floor clutter) and kept only if they pass.
    dome_low_zoom: float = 0.72
    reject_blur: bool = True
    blur_rel: float = 0.40
    view_min_coverage: float = 0.22  # min central alpha coverage to keep a dome view
    view_max_occlusion: float = 0.45  # max fraction of the center blocked by a near floater
    # Scene up axis: "auto" estimates it from the dominant ground plane (RANSAC);
    # otherwise one of +z,-z,+y,-y,+x,-x. Drives view generation and viewer framing.
    up: str = "auto"
    tiers: tuple[Tier, ...] = (Tier(radius=120.0, elevations=(80.0, 60.0), fov_deg=50.0, grid=5),)

    # --- lift / scoring ----------------------------------------------------
    # "alpha" = alpha-weighted back-projection (continuous, recommended);
    # "band"  = depth-band footprint voting (legacy, render view source).
    lift: str = "alpha"
    top_k: int = 8  # top-k contributing Gaussians kept per pixel (alpha lift)
    peak: float = 0.0  # blend weight of per-Gaussian peak score into the mean (0 = mean only)
    min_weight: float = 0.03  # min accumulated render weight for a Gaussian to count (floater cull)
    depth_band: float = 0.0025  # absolute two-sided depth band (band lift)
    depth_tol: float = 0.05  # multiplicative depth tolerance for visibility (render cache)
    foot: int = 0  # max-pool radius for footprint mask sampling (band lift)
    view_thresh: float = 0.35  # per-view score floor when counting consensus hits (band lift)
    min_views: int = 3  # min number of views that must hit a Gaussian (band lift consensus)
    strong_select: float = 0.99  # score above which a single strong view is enough (band lift)
    # Multi-view consensus gate (alpha lift): require a Gaussian to be supported by
    # enough views whose per-view score clears consensus_thr before it can be selected.
    consensus: bool = False
    consensus_thr: float = 0.3  # per-view score floor that counts as support
    consensus_frac: float = 0.0  # required supporting views as a fraction of the views that saw it
    consensus_min: int = 1  # absolute minimum number of supporting views

    # --- selection / competition ------------------------------------------
    select: float = 0.30  # score threshold to select a Gaussian for the query
    margin: float = 0.08  # query must beat the best distractor by this margin
    # "fixed" = keep candidates with qscore >= select (absolute floor);
    # "relative" = keep candidates with qscore >= select_rel * max(qscore) -- robust
    # to prompts whose grounding scores are globally weak (the main recall killer).
    select_mode: str = "fixed"
    select_rel: float = 0.5  # relative-mode threshold as a fraction of the peak candidate score
    min_keep: int = 0  # fixed-mode non-empty guard: keep top-k candidates if the threshold selects none
    # Concept competition is OFF by default for single queries ("show me X" should
    # just work); turn it on to suppress near-synonyms. The fixed-vocab bake does
    # its own implicit competition (argmax across the vocabulary), independent of this.
    compete: bool = False
    # Generic scene vocabulary used as the competition set when competition is on
    # and no distractors are supplied (recipes override with curated sets).
    distractors: tuple[str, ...] = DEFAULT_DISTRACTORS

    # --- spatial cleanup ---------------------------------------------------
    clean3d: bool = True  # drop isolated selected Gaussians via voxel cull
    voxel_frac: float = 0.01  # voxel size as a fraction of scene span
    min_pts: int = 4  # min selected Gaussians per voxel to survive

    # --- score smoothing (training-free voxel regularization) -------------
    # Blend each Gaussian's score with its voxel-neighborhood mean *before*
    # thresholding: fills object interiors (recall) and pulls isolated false
    # positives down toward their empty neighborhood (precision).
    smooth: bool = False
    smooth_beta: float = 0.5  # blend weight toward the neighborhood mean (0 = off)
    smooth_vox_frac: float = 0.02  # smoothing voxel size as a fraction of scene span
    smooth_weighted: bool = True  # weight the neighborhood mean by render contribution

    # --- SAM3 dual-head fusion (training-free instance + semantic) --------
    # Blend SAM3's presence-gated instance head with its dense prompt-conditioned
    # semantic head in one decode. Recovers amorphous/"stuff" classes the instance
    # head hard-filters out (the dominant recall failure) at no extra forward cost.
    dual_head: bool = False
    sem_weight: float = 0.5  # blend weight of the semantic head (0 = instance only)
    sem_mode: str = "mean"  # "mean" = (1-w)*inst + w*sem ; "max" = max((1-w)*inst, w*sem)

    # --- instances / catalog (connected-component object split) -----------
    inst_link_frac: float = 0.02  # voxel/link size as a fraction of scene span
    inst_min_size: int = 25  # drop instances smaller than this many Gaussians
    cat_iou: float = 0.5  # merge objects from different prompts when their 3D IoU >= this

    # --- multi-class labelling (bake) -------------------------------------
    tau: float = 0.15  # absolute confidence floor for a label
    delta: float = 0.02  # top1-top2 margin below which a Gaussian is left unlabeled

    # --- display -----------------------------------------------------------
    highlight_color: tuple[float, float, float] = (1.0, 0.95, 0.1)
    blend: float = 0.75  # how strongly to tint selected Gaussians in overlays

    # --- SAM3 --------------------------------------------------------------
    # Path to the SAM3 checkpoint. Defaults to the GEOLANGSPLAT_SAM_CKPT env var
    # (set it once), or pass --sam-ckpt / config.sam_ckpt explicitly.
    sam_ckpt: str = field(default_factory=lambda: os.environ.get("GEOLANGSPLAT_SAM_CKPT", ""))
    sam_res: int = 1008
    sam_conf: float = 0.20
    amp: str = "bf16"  # autocast precision: "bf16" or "fp16"
    batch_encode: bool = True
    batch_size: int = 8
    # Resident embedding-cache dtype. "auto" keeps SAM3's native output dtype; "amp"
    # casts the cached per-view features to the autocast dtype (safe -- matches what
    # scoring runs in, and halves the cache when the native output is fp32);
    # "fp16"/"bf16" force a specific half precision. This is our training-free
    # "embedding compression": the cache is the dominant resident VRAM consumer.
    cache_dtype: str = "auto"

    # --- runtime / viewer ports (used by `gls explore`) -------------------
    device: str = "cuda"
    viewer_port: int = 8080
    web_port: int = 8090
    vk_device_id: int = 0


# Capture-type presets. Each maps field name -> preset value; apply_recipe fills
# only fields still at their default. Keep these general; fine-tune per scene.
RECIPES: "dict[str, dict]" = {
    # Geometry-driven default. Derives the whole view plan (radii, elevations,
    # azimuths, capture type) from the scene itself -- works across satellite,
    # aerial and object scales without hand-tuned tiers.
    "auto": {
        "view_source": "render",
        "auto_views": True,
    },
    # Near-nadir satellite / high-altitude captures. Synthesizes a top-down orbit
    # at altitude (radius 200, grid 7 -> ~49 views/elev) -- the configuration
    # validated on the JAX scenes. Works without source photos.
    "satellite": {
        "view_source": "render",
        "tiers": (Tier(radius=200.0, elevations=(80.0, 65.0, 50.0), fov_deg=50.0, grid=7),),
        "select": 0.35,
        "margin": 0.10,
        "distractors": ("road", "grass", "tree", "water"),
    },
    # Satellite with extra coverage for small objects: the altitude tier plus a
    # closer, zoomed-in multi-azimuth tier (lower fov enlarges small structures).
    "satellite_dense": {
        "view_source": "render",
        "tiers": (
            Tier(radius=200.0, elevations=(80.0, 65.0, 50.0), fov_deg=50.0, grid=7),
            Tier(radius=140.0, elevations=(70.0, 55.0), fov_deg=42.0, grid=6, num_azimuth=2),
        ),
        "select": 0.33,
        "margin": 0.10,
        "distractors": ("road", "grass", "tree", "water"),
    },
    # Oblique drone / aerial photogrammetry (e.g. SafetyPark): segment the scene's
    # REAL source photos -- SAM3 segments them most cleanly -- and back-project.
    # This is the validated "ground-truth-image" configuration; it requires --sfm.
    "aerial": {
        "view_source": "images",
        "n_views": 90,
        "select": 0.30,
        "margin": 0.08,
        "distractors": (
            "building",
            "ground",
            "road",
            "grass",
            "tree",
            "water",
            "car",
            "bus",
            "truck",
            "person",
        ),
    },
}


def list_recipes() -> list[str]:
    """Return the available recipe names."""
    return sorted(RECIPES)


def apply_recipe(cfg: GeoLangSplatConfig, name: str | None) -> list[str]:
    """Fill recipe presets into ``cfg`` in place, but only for fields the caller
    left at their default. An explicitly set value always wins over the preset.

    Returns the list of field names actually overwritten by the preset.
    """
    if not name:
        return []
    if name not in RECIPES:
        hint = difflib.get_close_matches(name, list_recipes(), n=1)
        suggest = f" (did you mean {hint[0]!r}?)" if hint else ""
        raise GeoLangSplatError(f"unknown recipe {name!r}{suggest}; choices: {list_recipes()}")
    default = GeoLangSplatConfig()
    applied: list[str] = []
    for k, v in RECIPES[name].items():
        if not hasattr(cfg, k):
            raise KeyError(f"recipe {name!r} sets unknown field {k!r}")
        if getattr(cfg, k) == getattr(default, k):
            setattr(cfg, k, v)
            applied.append(k)
    return applied


def config_field_names() -> list[str]:
    """All field names of :class:`GeoLangSplatConfig` (handy for CLIs/tests)."""
    return [f.name for f in fields(GeoLangSplatConfig)]
