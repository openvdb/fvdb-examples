# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""`gls check` - scene-readiness smoke test (no SAM3 weights required)."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Annotated, Optional

import tyro
from tyro.conf import arg

from ..config import RecipeName
from ._common import BaseCommand, build_config, resolve_recipe


@dataclass
class Check(BaseCommand):
    """
    Report whether a reconstruction is dense enough for segmentation to work.

    Builds the views + per-Gaussian lift cache and prints coverage statistics and
    a verdict (good / fair / poor). This needs a GPU to render, but does NOT need
    SAM3 weights -- run it before a full segment to catch weak scenes early.

    Example usage:

        gls check model.ply --sfm /path/scene --recipe satellite
    """

    # Path to the input Gaussian splat .ply.
    model_path: tyro.conf.Positional[pathlib.Path]

    # Path to the SfM/COLMAP scene (required for --view-source images).
    sfm: Annotated[str, arg(aliases=["-s"])] = ""

    # Capture-type preset: auto (default) | satellite | satellite_dense | aerial.
    recipe: Annotated[Optional[RecipeName], arg(aliases=["-r"])] = None

    # View source override: cameras | render | images.
    view_source: Optional[str] = None

    # Number of views to sample / render (view_source=images).
    n_views: Optional[int] = None

    # Cap the synthesized/subsampled view count.
    max_views: Optional[int] = None

    # If >0, trim the view plan to fit this VRAM budget (GB).
    vram_budget_gb: Optional[float] = None

    # Scene up axis: auto (estimate from ground plane) or +z,-z,+y,-y,+x,-x.
    up: Annotated[Optional[str], arg(aliases=["-u"])] = None

    # Inside-out: place the eye at the scene core looking outward (interior scenes).
    inside_out: bool = False

    # Device.
    device: Annotated[str, arg(aliases=["-d"])] = "cuda"

    def execute(self) -> None:
        from ..api import assess_scene  # lazy: keeps `gls` startup torch-free

        cfg = build_config(
            sfm=self.sfm or None,
            view_source=self.view_source,
            n_views=self.n_views,
            max_views=self.max_views,
            vram_budget_gb=self.vram_budget_gb,
            up=self.up,
            inside_out=(True if self.inside_out else None),
            device=self.device,
        )
        report = assess_scene(self.model_path, config=cfg, recipe=resolve_recipe(self.recipe, self.view_source))
        print("\n=== scene readiness ===")
        for k in (
            "gaussians",
            "views",
            "capture",
            "coverage",
            "mean_views_per_observed_gaussian",
            "well_observed_frac",
        ):
            if k in report and report[k] is not None:
                v = report[k]
                print(f"  {k:<34} {v:.3f}" if isinstance(v, float) else f"  {k:<34} {v}")
        rationale = report.get("auto_rationale")
        if rationale:
            print("\n  recommended views (auto):")
            for line in rationale:
                print(f"    - {line}")
        print(f"\n  verdict: {report['verdict'].upper()}  -  {report['note']}\n")
