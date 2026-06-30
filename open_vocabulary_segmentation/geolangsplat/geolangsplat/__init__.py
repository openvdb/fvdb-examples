# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""GeoLangSplat: unified, training-free open-vocabulary 3D segmentation for
Gaussian splats.

A text prompt is segmented in 2D (SAM3) over a set of views of the splat, then
lifted to per-Gaussian scores by alpha-weighted back-projection. The same engine
serves both aerial (oblique drone) and satellite (near-nadir) captures; only the
recipe presets differ.

Typical use::

    from fvdb import GaussianSplat3d
    from geolangsplat import segment

    model = GaussianSplat3d.from_ply("scene.ply")
    result = segment(model, "house", recipe="satellite", output="ply_overlay",
                     out_path="house.ply")
    print(result.num_selected)
"""
from __future__ import annotations

import time as _time

# Single source of truth is pyproject's [project].version; read it from the installed
# metadata when available (pip install) and fall back to a literal for a bare checkout
# / zip drop-in (e.g. shipped into a DLI lab image without `pip install`).
try:  # pragma: no cover - trivial
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        __version__ = _pkg_version("fvdb-geolangsplat")
    except PackageNotFoundError:
        __version__ = "0.0.1"
except Exception:  # pragma: no cover
    __version__ = "0.0.1"

# Captured at first import (≈ process start, before torch is imported). Lets the
# CLI report total wall = python+torch import + CUDA init + engine work, so the
# fixed per-invocation startup cost (which a warm engine amortizes) is visible.
_PROCESS_T0 = _time.perf_counter()

from .config import RECIPES, GeoLangSplatConfig, apply_recipe, list_recipes
from .errors import GeoLangSplatError

__all__ = [
    "__version__",
    "GeoLangSplatConfig",
    "RECIPES",
    "apply_recipe",
    "list_recipes",
    "segment",
    "assess_scene",
    "SegmentResult",
    "build_catalog",
    "SegmentCatalog",
    "CatalogObject",
    "GeoLangSplatError",
]

# Lazily expose the API (which imports torch/engine) so that just importing the
# package -- e.g. for the CLI's lightweight paths that only talk to a running
# daemon (segment-attach, status, stop) -- does NOT pay torch import + CUDA init.
_LAZY = {"segment", "assess_scene", "SegmentResult", "build_catalog"}
_LAZY_CATALOG = {"SegmentCatalog", "CatalogObject"}


def __getattr__(name: str):
    if name in _LAZY:
        from . import api

        return getattr(api, name)
    if name in _LAZY_CATALOG:
        from . import catalog

        return getattr(catalog, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
