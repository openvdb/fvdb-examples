# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""`gls catalog` - run a vocabulary and build a browsable object catalog.

One pass over a prompt list, clustered into distinct physical objects and written
to a folder you can browse (``catalog.csv``) and pull single objects from
(``objects/<id>_<label>.ply``) -- e.g. for import into a downstream tool. The
notebook surface is :func:`geolangsplat.build_catalog` / ``engine.catalog``.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Annotated, Optional

import tyro
from tyro.conf import arg

from ..config import RecipeName
from ._common import BaseCommand, build_config, read_vocab_file, resolve_recipe


@dataclass
class Catalog(BaseCommand):
    """
    Build an ID'd object catalog for a splat over a prompt vocabulary.

    Each prompt is segmented and split into spatial objects; objects hit by several
    prompts are merged. The result is a table plus one ``.ply`` per object.

    Examples:

        gls catalog model.ply --vocab building car tree road -o scene_catalog/
        gls catalog model.ply --vocab-file classes.txt -o scene_catalog/
    """

    # Path to the input Gaussian splat .ply.
    model_path: tyro.conf.Positional[pathlib.Path]

    # The prompt vocabulary to catalog (ignored if --vocab-file is given).
    vocab: list[str] = field(
        default_factory=lambda: ["building", "house", "car", "road", "tree", "grass", "water", "sidewalk"]
    )

    # Read the vocabulary from a .txt file instead (one word per line, '#' comments).
    vocab_file: Optional[pathlib.Path] = None

    # Output directory (catalog.csv, objects/<id>_<label>.ply, catalog_labeled.ply).
    out: Annotated[pathlib.Path, arg(aliases=["-o"])] = pathlib.Path("catalog")

    # Capture-type preset: auto (default) | satellite | satellite_dense | aerial.
    recipe: Annotated[Optional[RecipeName], arg(aliases=["-r"])] = None

    # View source: render | globe | images (real photos via --sfm).
    view_source: Optional[str] = None

    # Path to the SfM/COLMAP scene (required for --view-source images).
    sfm: Annotated[str, arg(aliases=["-s"])] = ""

    # Cap the synthesized/subsampled view count.
    max_views: Optional[int] = None

    # If >0, trim the view plan to fit this VRAM budget (GB).
    vram_budget_gb: Optional[float] = None

    # Scene up axis: auto (estimate from ground plane) or +z,-z,+y,-y,+x,-x.
    up: Annotated[Optional[str], arg(aliases=["-u"])] = None

    # Per-prompt selection score threshold (defaults to the config's `select`).
    select: Annotated[Optional[float], arg(aliases=["-t"])] = None

    # Merge objects from different prompts when their 3D IoU is >= this.
    iou: Optional[float] = None

    # Also write the full splat recoloured by object id.
    labeled_ply: bool = True

    # Device.
    device: Annotated[str, arg(aliases=["-d"])] = "cuda"

    # Low-VRAM streaming build (see `gls segment`). --no-low-vram for the full cache.
    low_vram: bool = True

    def execute(self) -> None:
        from ..api import build_catalog  # lazy: keeps `gls` startup torch-free

        cfg = build_config(
            sfm=self.sfm or None,
            view_source=self.view_source,
            max_views=self.max_views,
            vram_budget_gb=self.vram_budget_gb,
            up=self.up,
            device=self.device,
            low_vram=self.low_vram,
        )
        vocab = read_vocab_file(self.vocab_file) if self.vocab_file else list(self.vocab)
        if not vocab:
            raise ValueError("empty vocabulary")
        print(f"[gls catalog] vocab ({len(vocab)}): {vocab}", flush=True)
        cat = build_catalog(
            self.model_path,
            vocab,
            config=cfg,
            recipe=resolve_recipe(self.recipe, self.view_source),
            select=self.select,
            iou=self.iou,
        )
        cat.export_all(self.out, labeled_ply=self.labeled_ply)
        print(f"[gls catalog] {len(cat)} objects:", flush=True)
        print(cat, flush=True)
