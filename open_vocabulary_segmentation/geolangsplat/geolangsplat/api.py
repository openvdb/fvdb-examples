# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Public, pure API: a text prompt in, per-Gaussian results out.

``segment`` is the one entry point. With a single prompt it returns a per-Gaussian
selection; with several prompts it returns a multi-class labelling (argmax with
confidence/ambiguity gates). Results carry helpers to write the segmented ``.ply``,
a highlight-overlay ``.ply``, or a per-Gaussian report.
"""
from __future__ import annotations

import copy
import pathlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from .config import GeoLangSplatConfig, apply_recipe
from .engine import GeoLangSplatEngine, load_or_build_engine
from . import outputs as _out
from .select import assign_labels

OutputMode = Literal["mask", "ply_segmented", "ply_overlay", "report"]


@dataclass
class SegmentResult:
    """Per-Gaussian segmentation result, in input ``.ply`` order.

    Attributes
    ----------
    prompts: the query word(s).
    scores: ``[N]`` for a single prompt, ``[N, C]`` for multiple.
    selected: ``[N]`` bool selection mask (for multi-prompt: any labeled Gaussian).
    label_ids: ``[N]`` long, class index per Gaussian (``-1`` = unlabeled);
        ``None`` for a single prompt.
    """

    prompts: list[str]
    scores: torch.Tensor
    selected: torch.Tensor
    label_ids: torch.Tensor | None
    config: GeoLangSplatConfig
    stats: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    _model: object = None

    @property
    def num_selected(self) -> int:
        return int(self.selected.sum())

    @property
    def means_np(self) -> np.ndarray:
        return self._model.means.detach().cpu().numpy()

    # -- writers ------------------------------------------------------------

    def to_ply_segmented(self, path) -> int:
        return _out.write_ply_segmented(self._model, self.selected, path, metadata=self.metadata)

    def to_ply_overlay(self, path, color=None, blend=None) -> int:
        if self.label_ids is not None:
            return _out.write_ply_labels(self._model, self.label_ids, path, metadata=self.metadata)
        color = self.config.highlight_color if color is None else color
        blend = self.config.blend if blend is None else blend
        return _out.write_ply_overlay(
            self._model, self.selected, path, color=color, blend=blend, metadata=self.metadata
        )

    def to_report(self, path) -> None:
        means = self.means_np
        if self.label_ids is not None:
            labels = self.label_ids.detach().cpu().numpy()
            scores = self.scores.detach().cpu().numpy()
        else:
            labels = np.where(self.selected.detach().cpu().numpy(), 0, -1).astype(np.int64)
            scores = self.scores.detach().cpu().numpy()
        path = pathlib.Path(path)
        if path.suffix == ".npz":
            _out.write_report_npz(means, labels, scores, self.prompts, path)
        else:
            _out.write_report_csv(means, labels, scores, self.prompts, path)


def build_catalog(
    model,
    vocab=None,
    *,
    config: GeoLangSplatConfig | None = None,
    recipe: str | None = None,
    engine: GeoLangSplatEngine | None = None,
    scorer=None,
    select: float | None = None,
    iou: float | None = None,
    link_frac: float | None = None,
    min_size: int | None = None,
):
    """Build a browsable, ID'd object catalog for ``model`` over a prompt ``vocab``.

    Runs the vocabulary in one pass and clusters every prompt's segments into
    distinct physical objects, returning a
    :class:`~geolangsplat.catalog.SegmentCatalog` (a table + per-object ``.ply``
    extraction). ``vocab`` defaults to a general scene vocabulary. Reuse a warm
    ``engine`` for repeat catalogs of the same scene.
    """
    cfg = engine.cfg if engine is not None else _resolve_config(config, recipe)
    eng = engine if engine is not None else load_or_build_engine(model, cfg, scorer=scorer)
    return eng.catalog(vocab, select=select, iou=iou, link_frac=link_frac, min_size=min_size)


def assess_scene(
    model,
    *,
    config: GeoLangSplatConfig | None = None,
    recipe: str | None = None,
) -> dict:
    """Geometry-only readiness check: does this reconstruction look segmentable?

    Builds the views + lift cache (no SAM3 weights required) and returns coverage
    statistics plus a ``verdict`` of ``"good" | "fair" | "poor"``. Run this before
    a full segment to catch sparse/low-quality scenes early.
    """
    cfg = _resolve_config(config, recipe)
    eng = GeoLangSplatEngine(model, cfg, build=False)
    eng.build_geometry()
    return eng.assess()


def _resolve_config(config: GeoLangSplatConfig | None, recipe: str | None) -> GeoLangSplatConfig:
    cfg = copy.deepcopy(config) if config is not None else GeoLangSplatConfig()
    if recipe:
        applied = apply_recipe(cfg, recipe)
        if applied:
            print(f"[recipe] {recipe}: filled {applied}", flush=True)
    return cfg


def segment(
    model,
    prompts: "str | list[str]",
    *,
    cameras=None,  # reserved; ground-truth image views are loaded from config.sfm (COLMAP)
    config: GeoLangSplatConfig | None = None,
    recipe: str | None = None,
    output: OutputMode = "mask",
    out_path=None,
    engine: GeoLangSplatEngine | None = None,
    scorer=None,
) -> SegmentResult:
    """Segment ``model`` by ``prompts``.

    Parameters
    ----------
    model: a ``GaussianSplat3d`` or a path to a ``.ply``.
    prompts: a single prompt (selection) or several (multi-class labelling).
    config: a :class:`GeoLangSplatConfig`; defaults are used if omitted.
    recipe: ``"aerial" | "satellite" | "satellite_dense"`` preset (fills unset fields).
    output: ``"mask"`` (no file), ``"ply_segmented"``, ``"ply_overlay"`` or ``"report"``.
    out_path: where to write when ``output != "mask"``.
    engine: reuse a prebuilt engine (skips the one-time build); keep one warm
        (``gls serve`` / ``GeoLangSplatEngine``) for fast repeated queries.
    scorer: inject a SAM3 scorer (e.g. a mock in tests).
    """
    single = isinstance(prompts, str)
    plist = [prompts] if single else list(prompts)
    if not plist:
        raise ValueError("prompts must be a non-empty string or list of strings")

    cfg = engine.cfg if engine is not None else _resolve_config(config, recipe)
    if engine is not None:
        eng = engine
    else:
        eng = load_or_build_engine(model, cfg, scorer=scorer)

    stats: dict = {}
    if single:
        scores, selected, dt = eng.query(plist[0])
        label_ids = None
        stats["query_seconds"] = dt
        stats["num_selected"] = int(selected.sum())
    else:
        scores = eng.bake_vocab(plist)  # [N, C]
        denom = eng.denom if eng.denom is not None else eng.seen
        label_ids, lab_stats = assign_labels(scores, denom, cfg.min_weight, cfg.tau, cfg.delta)
        selected = label_ids >= 0
        stats.update(lab_stats)

    result = SegmentResult(
        prompts=plist,
        scores=scores,
        selected=selected,
        label_ids=label_ids,
        config=cfg,
        stats=stats,
        metadata=getattr(eng, "metadata", {}) or {},
        _model=eng.model,
    )

    if output != "mask":
        if out_path is None:
            raise ValueError(f"output={output!r} requires out_path")
        if output == "ply_segmented":
            result.to_ply_segmented(out_path)
        elif output == "ply_overlay":
            result.to_ply_overlay(out_path)
        elif output == "report":
            result.to_report(out_path)
        else:
            raise ValueError(f"unknown output mode {output!r}")
    return result
