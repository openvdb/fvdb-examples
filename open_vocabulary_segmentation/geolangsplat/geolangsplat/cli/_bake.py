# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""`gls bake` - fixed-vocabulary multi-class labelling of every Gaussian."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Annotated, Optional

import tyro
from tyro.conf import arg

from ..config import RecipeName
from ._common import BaseCommand, build_config, read_vocab_file, resolve_recipe


@dataclass
class Bake(BaseCommand):
    """
    Assign every Gaussian one label from a fixed vocabulary (argmax with
    confidence/ambiguity gates), and write a per-Gaussian report (+ recoloured ply).

    The vocabulary can be given inline (--vocab) or from a text file
    (--vocab-file, one word per line, '#' for comments).

    Example usage:

        gls bake model.ply --sfm /path/scene --recipe satellite \\
            --vocab house tree grass sand water road -o labels/

        gls bake model.ply --sfm /path/scene --vocab-file classes.txt -o labels/
    """

    # Path to the input Gaussian splat .ply.
    model_path: tyro.conf.Positional[pathlib.Path]

    # The fixed label vocabulary (ignored if --vocab-file is given).
    vocab: list[str] = field(default_factory=lambda: ["house", "tree", "grass", "sand", "water", "road"])

    # Read the vocabulary from a .txt file instead (one word per line, '#' comments).
    vocab_file: Optional[pathlib.Path] = None

    # Path to the SfM/COLMAP scene (required for --view-source images).
    sfm: Annotated[str, arg(aliases=["-s"])] = ""

    # Capture-type preset: auto (default) | satellite | satellite_dense | aerial.
    recipe: Annotated[Optional[RecipeName], arg(aliases=["-r"])] = None

    # View source override: cameras | render | images.
    view_source: Optional[str] = None

    # Cap the synthesized/subsampled view count.
    max_views: Optional[int] = None

    # If >0, trim the view plan to fit this VRAM budget (GB).
    vram_budget_gb: Optional[float] = None

    # Scene up axis: auto (estimate from ground plane) or +z,-z,+y,-y,+x,-x.
    up: Annotated[Optional[str], arg(aliases=["-u"])] = None

    # Output directory (labels.csv, labels.npz, legend.json, labels.ply).
    out: Annotated[pathlib.Path, arg(aliases=["-o"])] = pathlib.Path("labels")

    # Confidence floor for a label.
    tau: Optional[float] = None

    # Ambiguity margin (top1 - top2) below which a Gaussian is left unlabeled.
    delta: Optional[float] = None

    # Also write a label-recoloured .ply.
    viz_ply: bool = True

    # Device.
    device: Annotated[str, arg(aliases=["-d"])] = "cuda"

    # Low-VRAM streaming: one bounded pass scores the whole vocab per view then evicts,
    # so peak VRAM is bounded to ~one view. ON by default; --no-low-vram for the full
    # all-views cache (higher VRAM, only worth it when re-baking many vocabularies).
    low_vram: bool = True

    def execute(self) -> None:
        from ..api import segment  # lazy: keeps `gls` startup torch-free

        cfg = build_config(
            sfm=self.sfm or None,
            view_source=self.view_source,
            max_views=self.max_views,
            vram_budget_gb=self.vram_budget_gb,
            up=self.up,
            tau=self.tau,
            delta=self.delta,
            device=self.device,
            low_vram=self.low_vram,
        )
        vocab = read_vocab_file(self.vocab_file) if self.vocab_file else list(self.vocab)
        if not vocab:
            raise ValueError("empty vocabulary")
        print(f"[gls bake] vocab ({len(vocab)}): {vocab}", flush=True)
        result = segment(
            self.model_path,
            vocab,
            config=cfg,
            recipe=resolve_recipe(self.recipe, self.view_source),
        )
        out = pathlib.Path(self.out)
        out.mkdir(parents=True, exist_ok=True)
        result.to_report(out / "labels.csv")
        result.to_report(out / "labels.npz")
        if self.viz_ply:
            result.to_ply_overlay(out / "labels.ply")
        print(f"[gls bake] {result.stats}", flush=True)
        print(f"[gls bake] wrote report + ply -> {out}/", flush=True)
