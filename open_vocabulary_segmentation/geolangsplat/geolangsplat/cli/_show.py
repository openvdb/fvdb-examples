# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""`gls show` - open a splat .ply in the fvdb viewer to eyeball a result.

Renders any `GaussianSplat3d` .ply (or checkpoint) in `fvdb.viz`, the same stack
as `frgs show`. Our `segment`/`bake` outputs are standard splats, so the overlay
tint / recolouring shows up directly -- handy for confirming a query worked.
For LIVE prompt-by-prompt exploration use `geolangsplat.viewer.run_viewer` instead.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Annotated

import tyro
from tyro.conf import arg

from ._common import BaseCommand


@dataclass
class Show(BaseCommand):
    """
    View a splat .ply (e.g. a `segment` overlay output) in the fvdb viewer.

    Example:

        gls segment scene.ply "house" -r satellite -O ply_overlay -o house.ply
        gls show house.ply           # confirm the tinted result in 3D
    """

    # Path to the splat .ply (or .pt/.pth checkpoint) to display.
    model_path: tyro.conf.Positional[pathlib.Path]

    # Viewer port.
    viewer_port: Annotated[int, arg(aliases=["-p"])] = 8080

    # IP to bind the viewer on.
    ip: str = "0.0.0.0"

    # Vulkan device id for the viewer.
    vk_device_id: int = 0

    # Scene up axis: auto (estimate from ground plane) or +z,-z,+y,-y,+x,-x.
    up: Annotated[str, arg(aliases=["-u"])] = "auto"

    # Device.
    device: Annotated[str, arg(aliases=["-d"])] = "cuda"

    def execute(self) -> None:
        import time

        import numpy as np

        from ..autoview import resolve_up
        from ..cameras import orbit_cameras
        from ..engine import _load_model, resolve_device

        if not self.model_path.exists():
            print(f"[show] no such file: {self.model_path}", flush=True)
            return

        import fvdb.viz as viz

        dev = resolve_device(self.device)
        model, _meta = _load_model(self.model_path, dev)
        viz.init(ip_address=self.ip, port=self.viewer_port, vk_device_id=self.vk_device_id)
        scene = viz.get_scene("GeoLangSplat Viewer")
        scene.add_gaussian_splat_3d("splat", model)
        # Orient + frame the camera (object/COLMAP plys aren't z-up, and the default
        # near clip is too far to zoom into small scenes).
        try:
            up = resolve_up(self.up, model.means)
            m = model.means.detach().float().cpu().numpy()
            lo = np.quantile(m, 0.02, axis=0)
            hi = np.quantile(m, 0.98, axis=0)
            center = (lo + hi) / 2
            radius = max(float(np.linalg.norm(hi - lo)) / 2, 0.001)
            scene.camera_up_direction = up
            scene.camera_near = max(radius * 0.002, 0.0001)
            scene.camera_far = radius * 100
            c2w = orbit_cameras(center, 18, radius * 2.2, num_azimuth=1, azimuth_offset_deg=35, up=up)[0][1]
            scene.set_camera_lookat(c2w[:3, 3], center, up)
        except Exception as e:
            print(f"[show] could not frame scene (up={self.up!r}): {e}", flush=True)
        if hasattr(viz, "show"):
            viz.show()
        print(f"[show] viewer at http://<host>:{self.viewer_port}  (Ctrl-C to stop)", flush=True)
        try:
            time.sleep(10**9)
        except KeyboardInterrupt:
            print("\n[show] bye", flush=True)
