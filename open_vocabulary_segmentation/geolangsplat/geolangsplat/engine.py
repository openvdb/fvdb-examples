# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""The unified GeoLangSplat engine.

It loads a splat, builds the SAM3 image-embedding cache + the per-Gaussian lift
cache once, precomputes distractor scores, and then answers text queries cheaply
(text encode + grounding decode + scatter). One engine serves aerial and
satellite captures; only the recipe presets differ.
"""
from __future__ import annotations

import pathlib
import threading
import time

import torch

from . import autoview as _autoview
from . import lift as _lift
from .profile import Stopwatch
from .errors import GeoLangSplatError
from .sam3 import Sam3Scorer
from .select import select_query, smooth_scores, spatial_cleanup
from .views import generate_views


_PERF_FLAGS_SET = False


def enable_perf_flags() -> None:
    """Idempotent inference perf flags (CUDA only).

    Turns on TF32 matmul/cuDNN and cuDNN autotuning. These speed up SAM3's conv
    backbone and the fvdb rasterization with only minor numeric drift -- harmless
    for thresholded masks / argmax labels. Set once per process.
    """
    global _PERF_FLAGS_SET
    if _PERF_FLAGS_SET:
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    _PERF_FLAGS_SET = True


def resolve_device(requested) -> torch.device:
    """Resolve and validate the compute device.

    GeoLangSplat needs CUDA: both fvdb rasterization (used to render views and the
    per-Gaussian alpha weights) and SAM3 inference are GPU paths. If CUDA was
    requested but is unavailable, fail early with a clear message instead of a deep
    ``cudaGetDeviceCount`` traceback. An explicit non-cuda device is honored (so
    CPU-only unit tests and tooling work), but real segmentation will require a GPU.
    """
    want = str(requested)
    if want.startswith("cuda") and not torch.cuda.is_available():
        raise GeoLangSplatError(
            "GeoLangSplat needs a CUDA GPU (fvdb rasterization + SAM3), but no CUDA "
            "device is available.\n"
            "  - On a GPU box: check drivers / `nvidia-smi` and CUDA_VISIBLE_DEVICES.\n"
            "  - Geometry-only `gls check` and the unit tests can run with `--device cpu`, "
            "but text segmentation cannot."
        )
    dev = torch.device(want)
    if dev.type == "cuda":
        enable_perf_flags()
    return dev


def _load_model(model_or_path, device) -> tuple[object, dict]:
    """Load a splat and its metadata. Returns ``(model, metadata)``.

    Prefers ``fvdb_reality_capture``'s loader so ``.ply`` AND training checkpoints
    (``.pt``/``.pth``) work and metadata round-trips; falls back to a plain
    ``GaussianSplat3d.from_ply`` when frc is not importable.
    """
    if not (isinstance(model_or_path, (str, bytes)) or hasattr(model_or_path, "__fspath__")):
        return model_or_path, {}  # already a GaussianSplat3d

    path = pathlib.Path(str(model_or_path))
    if not path.exists():
        raise GeoLangSplatError(f"no such model file: {path}")

    try:  # reuse frc's IO so `frgs segment` behaves identically (ply + checkpoints)
        from fvdb_reality_capture.cli.frgs._common import load_splats_from_file

        return load_splats_from_file(path, device)
    except GeoLangSplatError:
        raise
    except Exception:
        from fvdb import GaussianSplat3d

        gs = GaussianSplat3d.from_ply(str(path), device=device)
        if isinstance(gs, tuple):
            model = gs[0]
            meta = gs[1] if len(gs) > 1 and isinstance(gs[1], dict) else {}
            return model, meta
        return gs, {}


class GeoLangSplatEngine:
    """Build-once, query-many open-vocabulary segmentation over one splat."""

    def __init__(self, model_or_path, cfg, scorer=None, build: bool = True):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.sw = Stopwatch(self.device)
        is_path = isinstance(model_or_path, (str, bytes)) or hasattr(model_or_path, "__fspath__")
        self.source_path = pathlib.Path(str(model_or_path)) if is_path else None
        with self.sw.span("load_splat"):
            self.model, self.metadata = _load_model(model_or_path, self.device)
        self.means = self.model.means.detach()
        self.N = self.means.shape[0]
        self.lock = threading.Lock()
        # Scorer is created lazily (only when SAM3 is actually needed), so the
        # geometry-only readiness check does not require SAM3 weights.
        self.scorer = scorer
        self.views: list = []
        self.cache = None
        self.states = None
        self.seen = None
        self.denom = None
        self.dist_names: list[str] = []
        self.dist_scores: torch.Tensor | None = None
        self.auto_report: dict | None = None
        if build:
            self.build()

    # -- one-time build -----------------------------------------------------

    def build(self) -> None:
        """Full build: scene geometry/lift cache + SAM3 embeddings + distractors.

        This is the one-time cost (load splat, load SAM3, render + encode views).
        Every query afterwards reuses this cache and is fast, so to benefit you must
        keep the engine alive (a session, the viewer, or reusing it via the API).
        """
        if getattr(self.cfg, "low_vram", False):
            return self.build_streaming()
        print("[build] one-time setup (load + views + SAM3 embeddings); queries after this reuse the cache", flush=True)
        t0 = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.build_geometry()
        self.build_scorer()
        print(
            f"[ready] cached {len(self.views)} views, {self.N:,} gaussians in {time.time() - t0:.1f}s",
            flush=True,
        )
        print(self.sw.report("build"), flush=True)
        print(self.vram_report(), flush=True)

    def build_streaming(self) -> None:
        """Low-VRAM build: render views + load SAM3, but DON'T encode/cache embeddings.

        The per-view embeddings (the dominant resident VRAM) are produced and evicted
        inside each query (:func:`lift.stream_scores`), so peak VRAM is bounded to
        ~one view. There is no warm cache: each query re-encodes. This is the one-shot
        backend; use ``gls serve`` for the fast all-views cache.
        """
        print("[build] low-VRAM mode", flush=True)
        t0 = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._resolve_auto_views()
        with self.sw.span("generate_views"):
            self.views = generate_views(self.model, self.cfg, self.device, self.metadata)
        if self.scorer is None:
            with self.sw.span("sam3_load"):
                self.scorer = Sam3Scorer(
                    self.cfg.sam_ckpt,
                    sam_res=self.cfg.sam_res,
                    sam_conf=self.cfg.sam_conf,
                    amp=self.cfg.amp,
                    device=self.cfg.device,
                    dual_head=self.cfg.dual_head,
                    sem_weight=self.cfg.sem_weight,
                    sem_mode=self.cfg.sem_mode,
                )
        self.dist_names = list(self.cfg.distractors)
        self.dist_scores = None
        print(f"[ready] {self.N:,} gaussians, {len(self.views)} views in {time.time() - t0:.1f}s", flush=True)
        print(self.sw.report("build"), flush=True)
        print(self.vram_report(), flush=True)

    def vram_report(self) -> str:
        """One-line peak-VRAM summary (or a note when running on CPU)."""
        if self.device.type != "cuda":
            return "[vram] cpu device - no GPU memory in use"
        peak = torch.cuda.max_memory_allocated(self.device) / 1e9
        resv = torch.cuda.max_memory_reserved(self.device) / 1e9
        return f"[vram] peak {peak:.2f} GB ({resv:.2f} GB reserved)"

    def _resolve_auto_views(self) -> None:
        """Derive scene-scaled synthesized view tiers from geometry (auto recipe), once.

        Idempotent: safe to call from build_streaming and again from build_geometry.
        No-op for the ground-truth-image path or an explicit recipe.
        """
        cfg = self.cfg
        if not getattr(cfg, "auto_views", False) or self.auto_report is not None:
            return
        if cfg.view_source == "images":
            return
        self.auto_report = _autoview.apply_auto_view_config(cfg, self.means)

    def build_geometry(self) -> None:
        """Generate views and the per-Gaussian lift cache. No SAM3 weights needed.

        This is enough for the scene-readiness check (:meth:`assess`).
        """
        cfg = self.cfg
        self._resolve_auto_views()
        with self.sw.span("generate_views"):
            self.views = generate_views(self.model, cfg, self.device, self.metadata)
        if cfg.lift == "alpha":
            with self.sw.span("alpha_lift"):
                self.cache = _lift.build_alpha_cache(self.model, self.views, cfg, self.device)
            self.denom = self.cache.denom
            self.seen = self.cache.seen
        elif cfg.lift == "band":
            with self.sw.span("band_lift"):
                self.cache = _lift.build_band_cache(self.model, self.views, cfg, self.device)
            self.seen = self.cache.seen
        else:
            raise ValueError(f"unknown lift {cfg.lift!r} (expected 'alpha' or 'band')")

    def build_scorer(self) -> None:
        """Create SAM3 (if needed), cache per-view embeddings, precompute distractors."""
        cfg = self.cfg
        if not self.views:
            self.build_geometry()
        if self.scorer is None:
            with self.sw.span("sam3_load"):
                self.scorer = Sam3Scorer(
                    cfg.sam_ckpt,
                    sam_res=cfg.sam_res,
                    sam_conf=cfg.sam_conf,
                    amp=cfg.amp,
                    device=cfg.device,
                    dual_head=cfg.dual_head,
                    sem_weight=cfg.sem_weight,
                    sem_mode=cfg.sem_mode,
                )
        pils = [v.pil for v in self.views]
        sizes = {(v.height, v.width) for v in self.views}
        h, w = self.views[0].height, self.views[0].width
        batch = cfg.batch_encode and len(sizes) == 1
        te = time.time()
        with self.sw.span("sam3_encode"):
            self.states = self.scorer.encode(pils, h, w, batch_encode=batch, batch_size=cfg.batch_size, progress=True)
        self._compress_embeddings()
        print(f"[engine] {len(self.states)} embeddings cached in {time.time() - te:.1f}s", flush=True)
        self._build_distractors()

    def _compress_embeddings(self) -> None:
        """Optionally shrink the resident per-view embedding cache (the dominant VRAM
        consumer) by casting its float tensors to a half dtype. Off by default."""
        want = str(getattr(self.cfg, "cache_dtype", "auto")).lower()
        if want in ("", "auto", "native") or not self.states:
            return
        if want == "amp":
            target = getattr(self.scorer, "amp_dtype", torch.bfloat16)
        elif want == "fp16":
            target = torch.float16
        elif want == "bf16":
            target = torch.bfloat16
        else:
            raise GeoLangSplatError(f"unknown cache_dtype {want!r}; use auto|amp|fp16|bf16")

        def _cast(obj):
            if torch.is_tensor(obj):
                return obj.to(target) if obj.is_floating_point() and obj.dtype != target else obj
            if isinstance(obj, dict):
                return {k: _cast(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_cast(v) for v in obj)
            return obj

        self.states = _cast(self.states)
        print(f"[engine] embedding cache cast to {str(target).replace('torch.', '')}", flush=True)

    # -- scene readiness ----------------------------------------------------

    def assess(self) -> dict:
        """Geometry-only verdict on whether the scene is segmentable.

        Reports, without SAM3:
          * ``coverage`` - fraction of Gaussians any view sees as a top-k
            contributor. This is naturally bounded below 1.0 on dense splats
            (interior/occluded Gaussians never surface), so it is informational,
            not the verdict driver.
          * ``mean_views_per_observed_gaussian`` - how many distinct views see a
            typical observed Gaussian (multi-view redundancy).
          * ``well_observed_frac`` - fraction of Gaussians with enough accumulated
            render weight (``>= min_weight``) to be scored reliably. This is the
            quantity that actually gates segmentation quality, so the verdict is
            driven by it.
        """
        if self.seen is None:
            self.build_geometry()
        seen = self.seen
        observed = seen > 0
        cov = float(observed.float().mean())
        mean_views = float(seen[observed].float().mean()) if bool(observed.any()) else 0.0
        report = {
            "gaussians": int(self.N),
            "views": int(len(self.views)),
            "coverage": cov,
            "mean_views_per_observed_gaussian": mean_views,
        }
        if self.auto_report is not None:
            report["capture"] = self.auto_report.get("capture")
            report["auto_views_planned"] = self.auto_report.get("n_views_planned")
            report["auto_rationale"] = self.auto_report.get("rationale")
        well = cov
        if self.denom is not None:
            well = float((self.denom >= self.cfg.min_weight).float().mean())
            report["well_observed_frac"] = well

        # Verdict is driven by the well-observed fraction. Coverage and multi-view
        # redundancy (mean_views) are reported but NOT gated on: the tiled render
        # recipe sees each Gaussian from only ~1-2 views by design, and dense
        # splats cap coverage well below 1.0 - yet both still segment fine. These
        # thresholds are anchored on known-good scenes; for the strongest signal,
        # follow up with an actual query (a few well_observed_frac of ~0.4 is healthy).
        if well >= 0.35:
            verdict, note = "good", "enough well-observed gaussians; segmentation should work well"
        elif well >= 0.15:
            verdict, note = (
                "fair",
                "usable; thin in places - prefer real photos (--view-source images) or a denser recipe",
            )
        else:
            verdict, note = (
                "poor",
                "few well-observed gaussians; use real photos, a denser recipe, or improve the reconstruction",
            )
        report["verdict"] = verdict
        report["note"] = note
        return report

    def _build_distractors(self) -> None:
        # Competition is opt-in; only precompute the distractor scores if it is
        # enabled, so a plain "show me X" query does not pay for them.
        self.dist_names = list(self.cfg.distractors)
        self.dist_scores = None
        if self.cfg.compete:
            self._ensure_distractors()

    def _ensure_distractors(self) -> None:
        """Compute distractor scores on demand (cached). No-op if already done."""
        if self.dist_scores is not None or not self.dist_names:
            return
        t0 = time.time()
        cols = [self._score(n) for n in self.dist_names]
        self.dist_scores = torch.stack(cols, dim=1)  # [N, D]
        print(f"[engine] distractors {self.dist_names} in {time.time() - t0:.1f}s", flush=True)

    # -- scoring ------------------------------------------------------------

    def _score(self, prompt: str) -> torch.Tensor:
        """Per-Gaussian scalar score for ``prompt`` (lift-agnostic)."""
        if self.cfg.lift == "alpha":
            return _lift.alpha_qscore(self.scorer, self.states, self.cache, prompt, self.cfg)
        smax, _hits = _lift.aggregate_band(self.scorer, self.states, self.cache, prompt, self.cfg)
        return smax

    def score(self, prompt: str) -> torch.Tensor:
        """Public: per-Gaussian score for a single prompt."""
        return self._score(prompt)

    # -- query --------------------------------------------------------------

    def _subset_indices(self, k: int) -> list[int]:
        """``k`` view indices spread evenly across the (structured) view list, so the
        subset spans angles rather than clustering. Used by the fast query path."""
        n = len(self.views)
        k = min(int(k), n)
        if k <= 0 or k >= n:
            return list(range(n))
        return sorted(set(torch.linspace(0, n - 1, k).round().long().tolist()))

    @torch.no_grad()
    def query(self, prompt: str, *, select=None, margin=None, compete=None, fast_views=None):
        """Return ``(scores [N], selected [N] bool, dt_seconds)`` for a prompt."""
        cfg = self.cfg
        if getattr(cfg, "low_vram", False):
            return self._query_streaming(prompt, select=select, margin=margin, compete=compete)
        use_compete = cfg.compete if compete is None else compete
        # Fast path: decode only a spread subset of views (no competition support).
        fv = cfg.fast_views if fast_views is None else fast_views
        if cfg.lift == "alpha" and fv and not use_compete:
            with self.lock:
                t0 = time.time()
                idx = self._subset_indices(fv)
                qscore, denom, seen = _lift.aggregate_alpha_views(
                    self.scorer, self.states, self.cache, prompt, cfg, idx
                )
                if getattr(cfg, "smooth", False):
                    qscore = smooth_scores(self.means, qscore, denom, cfg)
                sel = select_query(
                    qscore,
                    seen,
                    cfg,
                    query=prompt,
                    select=select,
                    margin=margin,
                    compete=False,
                    denom=denom,
                )
                sel = spatial_cleanup(self.means, sel, cfg)
                return qscore, sel, time.time() - t0
        if use_compete:
            self._ensure_distractors()
        with self.lock:
            t0 = time.time()
            if cfg.lift == "alpha":
                if getattr(cfg, "consensus", False):
                    qscore, support = _lift.alpha_qscore_support(self.scorer, self.states, self.cache, prompt, cfg)
                else:
                    qscore = _lift.alpha_qscore(self.scorer, self.states, self.cache, prompt, cfg)
                    support = None
                if getattr(cfg, "smooth", False):
                    qscore = smooth_scores(self.means, qscore, self.denom, cfg)
                sel = select_query(
                    qscore,
                    self.seen,
                    cfg,
                    query=prompt,
                    select=select,
                    margin=margin,
                    compete=compete,
                    denom=self.denom,
                    dist_scores=self.dist_scores,
                    dist_names=self.dist_names,
                    support=support,
                )
            else:
                qscore, hits = _lift.aggregate_band(self.scorer, self.states, self.cache, prompt, cfg)
                sel_v = cfg.select if select is None else select
                sel = (qscore >= sel_v) & ((hits >= cfg.min_views) | (qscore >= cfg.strong_select)) & (self.seen > 0)
                use_compete = cfg.compete if compete is None else compete
                if use_compete and self.dist_scores is not None and self.dist_names:
                    from .select import competitor_idx

                    keep = competitor_idx(prompt, self.dist_names)
                    if keep:
                        mrg = cfg.margin if margin is None else margin
                        dmax = self.dist_scores[:, keep].max(dim=1).values
                        sel = sel & (qscore >= dmax + mrg)
            sel = spatial_cleanup(self.means, sel, cfg)
            return qscore, sel, time.time() - t0

    def _stream_early_stop(self) -> bool:
        """Whether the streaming lift may early-stop for this scene.

        Early-stop assumes a *compact* subject: a few angularly-diverse views see the
        whole thing, so once they agree the rest are redundant. That holds for an
        object/dome capture but NOT for aerial/satellite, where the queried class
        (buildings, roads, trees) is spread across the whole footprint -- stopping
        early there truncates recall (measured: IoU collapses to ~0.05-0.27). So
        "auto" enables it only for object/globe captures; aerial/sat stream every
        view (still VRAM-bounded, full recall).
        """
        mode = str(getattr(self.cfg, "stream_early_stop", "auto")).lower()
        if mode == "on":
            return True
        if mode == "off":
            return False
        if getattr(self.cfg, "view_source", "") == "globe":
            return True
        return bool(self.auto_report) and self.auto_report.get("capture") == "object"

    @torch.no_grad()
    def _query_streaming(self, prompt: str, *, select=None, margin=None, compete=None):
        """Single-prompt query via the low-VRAM streaming lift (see build_streaming).

        Concept competition is supported here too: when enabled, the queried prompt
        and the distractor set are scored together in the SAME streaming pass -- the
        expensive per-view SAM3 encode is shared across all prompts, only the cheap
        text-vs-features scoring multiplies -- then the query must beat the best
        distractor by ``margin``. Selection is then identical to the warm path.
        Competition scores every view (no early-stop), which is also what spread-out
        aerial/satellite classes want. Sets denom/seen from the stream so
        selection/labelling behave identically to the cached path.
        """
        cfg = self.cfg
        use_compete = cfg.compete if compete is None else compete
        dist_names = list(cfg.distractors) if use_compete else []
        prompts = [prompt] + dist_names
        # Competition needs every view; only a lone prompt may early-stop (and
        # stream_scores only early-stops when it is scoring a single prompt).
        early = self._stream_early_stop() and not use_compete
        with self.lock:
            t0 = time.time()
            scores, denom, seen, stats = _lift.stream_scores(
                self.model,
                self.scorer,
                self.views,
                cfg,
                prompts,
                self.device,
                want_peak=(cfg.peak > 0),
                early_stop=early,
            )
            self.denom, self.seen = denom, seen
            qscore = scores[0]
            if getattr(cfg, "smooth", False):
                qscore = smooth_scores(self.means, qscore, denom, cfg)
            dist_scores = scores[1:].t().contiguous() if (use_compete and len(prompts) > 1) else None
            sel = select_query(
                qscore,
                seen,
                cfg,
                query=prompt,
                select=select,
                margin=margin,
                compete=use_compete,
                denom=denom,
                dist_scores=dist_scores,
                dist_names=dist_names,
            )
            sel = spatial_cleanup(self.means, sel, cfg)
            if stats["early_stopped"]:
                note = " (early-stopped)"
            elif use_compete:
                note = f" (+{len(dist_names)} distractors)"
            else:
                note = ""
            print(f"[stream] {stats['views_used']}/{stats['views_total']} views{note}", flush=True)
            return qscore, sel, time.time() - t0

    # -- fixed-vocabulary bake ---------------------------------------------

    @torch.no_grad()
    def bake_vocab(self, vocab) -> torch.Tensor:
        """Per-Gaussian scores for a fixed vocabulary: returns ``[N, C]``.

        Only supported for the alpha lift (continuous scores comparable across
        words). These become instant lookups for multi-class labelling.
        """
        if self.cfg.lift != "alpha":
            raise NotImplementedError("bake_vocab requires the alpha lift")
        t0 = time.time()
        if getattr(self.cfg, "low_vram", False):
            # One bounded streaming pass scores the whole vocab per view (no early-stop
            # for multi-class -- every class needs full coverage), then evicts.
            scores, denom, seen, _stats = _lift.stream_scores(
                self.model,
                self.scorer,
                self.views,
                self.cfg,
                list(vocab),
                self.device,
                want_peak=(self.cfg.peak > 0),
                early_stop=False,
            )
            self.denom, self.seen = denom, seen
            out = scores.transpose(0, 1).contiguous()  # [C, N] -> [N, C]
        else:
            cols = [_lift.alpha_qscore(self.scorer, self.states, self.cache, w, self.cfg) for w in vocab]
            out = torch.stack(cols, dim=1)
        print(f"[engine] baked {len(vocab)} words in {time.time() - t0:.1f}s", flush=True)
        return out

    # -- segment catalog ----------------------------------------------------

    @torch.no_grad()
    def catalog(self, vocab=None, *, select=None, iou=None, link_frac=None, min_size=None):
        """Build a :class:`~geolangsplat.catalog.SegmentCatalog` over ``vocab``.

        Scores the whole vocabulary in one pass (:meth:`bake_vocab`), then clusters
        each prompt's selection into spatial objects and merges duplicates across
        prompts. ``vocab`` defaults to
        :data:`~geolangsplat.catalog.DEFAULT_CATALOG_VOCAB`.
        """
        from .catalog import DEFAULT_CATALOG_VOCAB, catalog_from_scores

        vocab = list(vocab) if vocab else list(DEFAULT_CATALOG_VOCAB)
        scores = self.bake_vocab(vocab)
        cat = catalog_from_scores(
            self.model,
            vocab,
            self.cfg,
            scores,
            seen=self.seen,
            denom=self.denom,
            select=select,
            iou=iou,
            link_frac=link_frac,
            min_size=min_size,
            metadata=getattr(self, "metadata", {}) or {},
        )
        cat._engine = self  # keep warm so SegmentCatalog.browse() can re-query in place
        return cat


def load_or_build_engine(model_or_path, cfg, *, scorer=None) -> GeoLangSplatEngine:
    """Build an engine for ``model_or_path`` under ``cfg`` and return it ready to query.

    Dispatches on ``cfg.low_vram``: the streaming one-shot backend (default) or the
    full all-views build. Keep the returned engine alive (a session, the viewer, or
    ``gls serve``) to amortize the one-time build across queries.
    """
    engine = GeoLangSplatEngine(model_or_path, cfg, scorer=scorer, build=False)
    engine.build()
    return engine
