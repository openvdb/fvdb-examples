# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""`gls render` - render a splat .ply to a montage PNG (viewer-free preview).

Renders any `GaussianSplat3d` .ply from a ring of orbit views and tiles them into
one image. Handy for confirming a `segment`/`bake` overlay result over SSH or while
travelling, where the interactive `fvdb.viz` viewer isn't reachable. The up axis is
estimated from the scene's ground plane (same as the rest of the pipeline), so the
montage comes out upright on object/COLMAP scenes too.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Annotated, Optional

import tyro
from tyro.conf import arg

from ._common import BaseCommand


@dataclass
class Render(BaseCommand):
    """
    Render a splat .ply to a single montage PNG (no interactive viewer needed).

    Example:

        gls segment scene.ply "table" -O ply_overlay -o table.ply
        gls render table.ply -o table.png      # eyeball the result as an image
    """

    # Path to the splat .ply (or .pt/.pth checkpoint) to render.
    model_path: tyro.conf.Positional[pathlib.Path]

    # Output PNG path (default: <ply>_views.png).
    out: Annotated[Optional[pathlib.Path], arg(aliases=["-o"])] = None

    # Number of views around the orbit.
    n_views: Annotated[int, arg(aliases=["-n"])] = 8

    # Elevation(s) above the horizon plane, in degrees (comma-separated for rings).
    elevation: str = "15,45"

    # Up axis: "auto" (estimate from ground plane) or +z,-z,+y,-y,+x,-x.
    up: Annotated[str, arg(aliases=["-u"])] = "auto"

    # Orbit radius as a fraction of the scene's bounding-box diagonal (globe off).
    radius_frac: float = 1.1

    # Globe zoom: <1 pulls the camera in, >1 backs it out (default frames the core).
    zoom: Annotated[float, arg(aliases=["-z"])] = 1.0

    # Globe (default): auto-frame the distance from the scene's dense core and sweep
    # an upper dome of elevation rings looking DOWN (where bare-splat renders stay
    # sharp). Turn off (--no-globe) for a fixed --radius-frac / --elevation orbit.
    globe: bool = True

    # Inside-out: place the eye at the scene core and look outward (interior scenes).
    inside_out: bool = False

    # COLMAP scene dir (with sparse/ + images/). When given, render from a sample of
    # the scene's ACTUAL camera poses instead of a synthesized orbit -- crisp and
    # upright by construction (this is the production / FRC path). Overrides --globe.
    sfm: Annotated[Optional[pathlib.Path], arg(aliases=["-s"])] = None

    # Per-tile render size (pixels).
    size: int = 512

    # Camera field of view (degrees).
    fov: float = 55.0

    # Device.
    device: Annotated[str, arg(aliases=["-d"])] = "cuda"

    def execute(self) -> None:
        import math
        import os

        import numpy as np
        import torch
        from PIL import Image

        from ..autoview import measure_geometry, resolve_up
        from ..cameras import inside_out_cameras, intrinsics, orbit_cameras
        from ..engine import _load_model, resolve_device
        from ..views import auto_frame_radius, core_extent

        if not self.model_path.exists():
            print(f"[render] no such file: {self.model_path}", flush=True)
            return

        # Default outputs to scratch, never the (small) home dir: honour
        # GEOLANGSPLAT_OUT_DIR if set, else write next to the source .ply.
        if self.out is not None:
            out = self.out
        else:
            base = os.environ.get("GEOLANGSPLAT_OUT_DIR")
            name = self.model_path.stem + "_views.png"
            out = (pathlib.Path(base) / name) if base else self.model_path.with_name(name)

        dev = resolve_device(self.device)
        model, _meta = _load_model(self.model_path, dev)
        means = model.means.detach()

        stats = measure_geometry(means)
        up = resolve_up(self.up, means)
        radius = float(stats.span) * self.radius_frac
        try:
            elevs = [float(e) for e in str(self.elevation).split(",") if e.strip()]
        except ValueError:
            elevs = [15.0, 45.0]

        # Build the camera list. Three modes, in priority order:
        #   inside-out : eye at core looking out (interior scenes)
        #   globe      : auto-framed radius + upper dome rings looking down
        #   orbit      : fixed --radius-frac at the user's --elevation rings
        grid_cols = None  # when set, montage lays out rows=elevation, cols=azimuth
        if self.sfm is not None:
            # Render from the scene's real COLMAP poses (sampled evenly across the
            # trajectory). No view synthesis, no up-axis guessing -- this is what the
            # splat actually looks like from where it was captured.
            from ..views import _colmap_c2w, _find_colmap, _read_colmap_images

            _cam, img_path, _img_dir = _find_colmap(self.sfm)
            recs = sorted(_read_colmap_images(img_path), key=lambda d: d["name"])
            n = max(1, self.n_views * len(elevs))
            idx = np.linspace(0, len(recs) - 1, num=min(n, len(recs))).round().astype(int)
            fov = self.fov
            cams = [np.linalg.inv(_colmap_c2w(recs[i])) for i in idx]
            elevs = []  # not a ring layout
        elif self.inside_out:
            center = means.float().median(0).values.cpu().numpy().astype(np.float64)
            fov = max(self.fov, 80.0)
            cams = [w2c for w2c, _ in inside_out_cameras(center, elevs, self.n_views, up=up)]
        elif self.globe:
            from ..views import dome_radius_scale

            center, _r = core_extent(model.means)
            fov = self.fov
            radius = auto_frame_radius(model, center, up, fov=fov, zoom=self.zoom)
            elevs = [18.0, 38.0, 58.0, 78.0]  # dome: low-oblique (sides) -> near-nadir, looking down
            grid_cols = self.n_views
            e_lo, e_hi = min(elevs), max(elevs)
            cams = []
            for elev in elevs:
                # Pull low/oblique rings closer (matches the segment dome path) so
                # blocked side-on subjects fill more of the frame.
                ring_radius = radius * dome_radius_scale(elev, e_lo, e_hi, 0.72)
                for ai in range(self.n_views):
                    az = 360.0 * ai / self.n_views
                    cams.append(
                        orbit_cameras(center, elev, ring_radius, num_azimuth=1, azimuth_offset_deg=az, up=up)[0][0]
                    )
        else:
            center = (
                ((means.float().min(0).values + means.float().max(0).values) / 2.0).cpu().numpy().astype(np.float64)
            )
            fov = self.fov
            cams = []
            for elev in elevs:
                for ai in range(self.n_views):
                    az = 360.0 * ai / self.n_views
                    cams.append(orbit_cameras(center, elev, radius, num_azimuth=1, azimuth_offset_deg=az, up=up)[0][0])

        K_np = intrinsics(fov, self.size, self.size)
        K = torch.from_numpy(K_np).float().unsqueeze(0).to(dev)
        tiles: list[Image.Image] = []
        with torch.no_grad():
            for w2c_np in cams:
                w2c = torch.from_numpy(w2c_np).float().to(dev)
                img, _a = model.render_images_and_depths(
                    world_to_camera_matrices=w2c.unsqueeze(0),
                    projection_matrices=K,
                    image_width=self.size,
                    image_height=self.size,
                    near=0.01,
                    far=1e12,
                )
                rgb = (img[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                tiles.append(Image.fromarray(rgb))

        # Lay out rows=elevation, cols=azimuth for the ring modes (clean pattern);
        # otherwise a square grid.
        cols = grid_cols or math.ceil(math.sqrt(len(tiles)))
        rows = math.ceil(len(tiles) / cols)
        montage = Image.new("RGB", (cols * self.size, rows * self.size), (10, 12, 16))
        for i, t in enumerate(tiles):
            montage.paste(t, ((i % cols) * self.size, (i // cols) * self.size))
        out.parent.mkdir(parents=True, exist_ok=True)
        montage.save(out)
        if self.sfm is not None:
            print(f"[render] {len(tiles)} real-camera views (from {self.sfm}) -> {out}", flush=True)
        else:
            mode = "inside-out" if self.inside_out else ("globe" if self.globe else "orbit")
            print(
                f"[render] {len(tiles)} {mode} views ({len(elevs)} elev x {self.n_views} az) "
                f"-> {out}  (up={tuple(round(float(x), 2) for x in up)})",
                flush=True,
            )
