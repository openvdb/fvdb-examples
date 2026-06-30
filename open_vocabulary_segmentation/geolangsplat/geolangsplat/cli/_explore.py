# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""`gls explore` - interactive web catalog browser + fvdb viewer.

An optional visual companion to `gls catalog`: launches the fvdb.viz viewer plus a small web
UI for the one-class-at-a-time catalog flow. Type a class ("building", "car"); every instance
lights up in its own colour; click one in the list and the viewer swaps to a cutout of
just that object; Back returns to the highlighted view. Tune the confidence / split /
min-size knobs live, and Export all to dump per-object ``.ply``. Handy for eyeballing and
curation -- the stable surface is `gls segment` / `gls catalog` and the `segment` &
`build_catalog` APIs.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Annotated, Optional

import tyro
from tyro.conf import arg

from ..config import RecipeName
from ._common import BaseCommand, build_config, resolve_recipe


@dataclass
class Explore(BaseCommand):
    """
    Launch the interactive viewer + web query UI for live catalog browsing.

    Example:

        gls explore scene.ply --recipe satellite
        # then open the web UI (default :8090) and the 3D viewer (default :8080)
    """

    # Path to the input Gaussian splat .ply.
    model_path: tyro.conf.Positional[pathlib.Path]

    # UI backend: catalog (GLS instance catalog) | refimg (REF_IMG open-vocab heatmap
    # with live gate sliders). refimg builds a teacher bake + reference-photo prototypes
    # at startup and imports the dev-tree field builder (see --backend-root).
    backend: Annotated[str, arg(aliases=["-b"])] = "catalog"

    # Root dir holding query_field.py / render_queries.py / distill_field.py (for
    # --backend refimg). Defaults to $GLS_QUERY_ROOT or the current working directory.
    backend_root: Optional[str] = None

    # Capture-type preset: auto (default) | satellite | satellite_dense | aerial.
    recipe: Annotated[Optional[RecipeName], arg(aliases=["-r"])] = None

    # Path to the SfM/COLMAP scene (required for --view-source images).
    sfm: Annotated[str, arg(aliases=["-s"])] = ""

    # View source: render (synthetic orbit/ladder) | globe (dome) | images (real photos via --sfm).
    view_source: Optional[str] = None

    # Cap the synthesized/subsampled view count.
    max_views: Optional[int] = None

    # If >0, trim the view plan to fit this VRAM budget (GB).
    vram_budget_gb: Optional[float] = None

    # Scene up axis: auto (estimate from ground plane) or +z,-z,+y,-y,+x,-x.
    up: Annotated[Optional[str], arg(aliases=["-u"])] = None

    # Inside-out: place the eye at the scene core looking outward (interior scenes).
    inside_out: bool = False

    # Blend per-Gaussian peak score into the mean (0..1) for sparse-class recall.
    peak: Optional[float] = None

    # Device.
    device: Annotated[str, arg(aliases=["-d"])] = "cuda"

    # Viewer (fvdb.viz) port.
    viewer_port: int = 8080

    # Web query-UI port.
    web_port: int = 8090

    # Vulkan device id for the viewer.
    vk_device_id: int = 0

    def execute(self) -> None:
        import os

        from ..viewer import run_viewer

        if not self.model_path.exists():
            print(f"[explore] no such file: {self.model_path}", flush=True)
            return

        if self.backend_root:
            os.environ["GLS_QUERY_ROOT"] = self.backend_root

        cfg = build_config(
            recipe=resolve_recipe(self.recipe, self.view_source),
            sfm=self.sfm or None,
            view_source=self.view_source,
            max_views=self.max_views,
            vram_budget_gb=self.vram_budget_gb,
            up=self.up,
            inside_out=(True if self.inside_out else None),
            peak=self.peak,
            device=self.device,
            viewer_port=self.viewer_port,
            web_port=self.web_port,
            vk_device_id=self.vk_device_id,
        )
        # recipe already folded into cfg (recipe-first); pass recipe=None so run_viewer
        # does not re-apply it and clobber the explicit --view-source override.
        run_viewer(str(self.model_path), config=cfg, recipe=None, backend=self.backend)
