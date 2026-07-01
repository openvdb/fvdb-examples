# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""`gls segment` - segment a splat by a single text prompt.

The one command you always use to query. It does the right thing automatically:

* if a warm engine is already serving this model (you ran ``gls serve``), it
  attaches and answers instantly;
* otherwise it runs one-shot -- build, query, write, exit -- leaving nothing
  behind.

So there are two ways to use GeoLangSplat: one-shot ``gls segment`` (above), or
``gls serve`` to build an engine once, ``gls segment`` to query it many times,
``gls stop`` to free it. Fine-grained tuning beyond ``--select`` lives in the
Python API.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Annotated, Literal, Optional

import tyro
from tyro.conf import arg

from ..config import RecipeName
from ._common import BaseCommand, build_config, resolve_recipe


@dataclass
class Segment(BaseCommand):
    """
    Open-vocabulary 3D segmentation of a Gaussian splat from a text prompt.

    SAM3 segments the prompt in 2D over views of the splat; the masks are lifted
    to per-Gaussian scores by alpha-weighted back-projection. Aerial and satellite
    captures use the same path -- just pick a recipe.

    Examples:

        # one-shot highlight overlay (synthesized views, no photos needed)
        gls segment model.ply "house" --recipe satellite -O ply_overlay -o house.ply

        # engine mode: build once with `gls serve`, then query instantly
        gls serve   model.ply --recipe satellite -b
        gls segment model.ply "house" -O ply_overlay -o house.ply   # attaches, instant
        gls segment model.ply "road"  -O ply_overlay -o road.ply    # attaches, instant
        gls stop    model.ply

        # use the scene's real photos when it ships a COLMAP reconstruction
        gls segment model.ply "tree" --view-source images --sfm /path/scene -O report -o tree.csv
    """

    # Path to the input Gaussian splat .ply.
    model_path: tyro.conf.Positional[pathlib.Path]

    # Text prompt to segment (e.g. "house", "road", "tree").
    prompt: tyro.conf.Positional[str]

    # Capture-type preset: auto (default) | satellite | satellite_dense | aerial.
    recipe: Annotated[Optional[RecipeName], arg(aliases=["-r"])] = None

    # Output mode: mask (no file) | ply_segmented | ply_overlay | report.
    output: Annotated[Literal["mask", "ply_segmented", "ply_overlay", "report"], arg(aliases=["-O"])] = "ply_overlay"

    # Where to write the output (.ply for ply_*, .csv/.npz for report).
    out_path: Annotated[pathlib.Path, arg(aliases=["-o"])] = pathlib.Path("segmented.ply")

    # Selection score threshold -- the main knob. Higher = stricter/fewer gaussians.
    select: Annotated[Optional[float], arg(aliases=["-t"])] = None

    # Blend per-Gaussian peak score into the mean (0..1) to recover diluted, sparsely
    # observed classes (trees, grass) without broadly lowering the threshold.
    peak: Optional[float] = None

    # View source: render (synthetic orbit/ladder) |
    # globe (auto-framed dome for objects/interiors) | images (real photos via --sfm).
    view_source: Optional[Literal["render", "globe", "images"]] = None

    # Cap the synthesized/subsampled view count (build VRAM & query latency ~ views).
    max_views: Optional[int] = None

    # If >0, trim the view plan to fit this VRAM budget (GB).
    vram_budget_gb: Optional[float] = None

    # Scene up axis: auto (estimate from ground plane) or +z,-z,+y,-y,+x,-x.
    up: Annotated[Optional[str], arg(aliases=["-u"])] = None

    # Inside-out: place the eye at the scene core looking outward (interior scenes).
    inside_out: bool = False

    # Path to the SfM/COLMAP scene (required for --view-source images).
    sfm: Annotated[str, arg(aliases=["-s"])] = ""

    # Suppress near-synonyms by competing against the recipe's distractor set.
    compete: bool = False

    # Print the total wall time (incl. python/torch import + CUDA init) at the end.
    profile: bool = False

    # Device.
    device: Annotated[str, arg(aliases=["-d"])] = "cuda"

    # Resident embedding-cache dtype (VRAM lever): auto|amp|fp16|bf16. "amp" matches
    # the autocast dtype and halves the cache when SAM3's features are fp32.
    cache_dtype: str = "auto"

    # Low-VRAM streaming: encode + score + project + evict one view at a time, so peak
    # VRAM is bounded to ~one view (vs. caching every view). ON by default for one-shot;
    # it re-encodes per query, so for fast repeated queries use `gls serve` (warm cache).
    # Pass --no-low-vram for the full-quality, higher-VRAM one-shot build.
    low_vram: bool = True

    # Streaming early-stop: auto|on|off. "auto" stops once enough diverse views agree
    # for a COMPACT subject (object/dome), but streams every view for aerial/satellite
    # (the class is spread across the scene, so stopping early loses recall). Force
    # "off" to always stream all views (max recall), "on" to always early-stop.
    stream_early_stop: Literal["auto", "on", "off"] = "auto"

    # Views encoded (batched) then evicted per streaming step. Higher = faster encode,
    # more transient VRAM; lower = tighter VRAM. Only affects --low-vram.
    stream_chunk: Optional[int] = None

    def execute(self) -> None:
        from ..ipc import daemon_alive, default_socket

        if not self.model_path.exists():
            print(f"[gls] no such file: {self.model_path}", flush=True)
            return

        sock = default_socket(self.model_path)

        # If `gls serve` already built an engine for this model, attach and answer
        # instantly. The engine's build settings win, so this path ignores the
        # build-affecting flags (-r/-s); only query flags (-t/--compete/-O/-o)
        # apply. Otherwise run one-shot in-process and exit, leaving nothing behind.
        if daemon_alive(sock):
            self._via_daemon(sock)
        else:
            self._one_shot()

    def _via_daemon(self, sock: str) -> None:
        from ..ipc import request

        req = {
            "prompt": self.prompt,
            "output": self.output,
            "out_path": (str(self.out_path) if self.output != "mask" else None),
            "select": self.select,
            "compete": (True if self.compete else None),
        }
        resp = request(sock, req)
        if not resp.get("ok"):
            print(f"[gls] error: {resp.get('error') or 'unknown'}", flush=True)
            return
        t = resp.get("t")
        ts = f"{t:.2f}s" if isinstance(t, (int, float)) else "?"
        print(f'[gls] "{resp["prompt"]}" -> {resp["n"]:,} / {resp["N"]:,} gaussians ({ts}, warm)', flush=True)
        if resp.get("path"):
            print(f"[gls] wrote {resp['output']} -> {resp['path']}", flush=True)

    def _one_shot(self) -> None:
        from ..api import segment  # lazy: keeps the daemon-attach path torch-free

        cfg = build_config(
            sfm=self.sfm or None,
            view_source=self.view_source,
            select=self.select,
            peak=self.peak,
            max_views=self.max_views,
            vram_budget_gb=self.vram_budget_gb,
            up=self.up,
            inside_out=(True if self.inside_out else None),
            compete=(True if self.compete else None),
            device=self.device,
            cache_dtype=(self.cache_dtype if self.cache_dtype != "auto" else None),
            low_vram=self.low_vram,
            stream_early_stop=(self.stream_early_stop if self.stream_early_stop != "auto" else None),
            stream_chunk=self.stream_chunk,
        )
        result = segment(
            self.model_path,
            self.prompt,
            config=cfg,
            recipe=resolve_recipe(self.recipe, self.view_source),
            output=self.output,
            out_path=self.out_path if self.output != "mask" else None,
        )
        print(
            f'[gls] "{self.prompt}" -> {result.num_selected:,} / {result.scores.shape[0]:,} gaussians',
            flush=True,
        )
        if self.output != "mask":
            print(f"[gls] wrote {self.output} -> {self.out_path}", flush=True)
