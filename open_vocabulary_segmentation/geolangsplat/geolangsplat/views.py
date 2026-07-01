# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Unified view generation.

Two ways to get the 2D views that SAM3 segments and that we back-project from:

* ``render`` - synthesize a multi-tier grid of orbit cameras over the splat and
  render RGB for each (good when there are no photos, or for uniform coverage).
* ``images`` - use the scene's ground-truth SfM photos and their COLMAP camera
  poses (SAM3 segments real photos most cleanly).

Both produce a list of :class:`View` (image + camera), which the lift consumes
identically, so everything downstream is view-source agnostic.
"""
from __future__ import annotations

import math
import pathlib
import struct
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .cameras import inside_out_cameras, intrinsics, orbit_cameras, up_vector
from .errors import GeoLangSplatError

# COLMAP camera model id -> number of intrinsic params.
_COLMAP_MODEL_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


@dataclass
class View:
    """One 2D view of the scene: an image plus its camera (world->camera)."""

    pil: Image.Image
    w2c: torch.Tensor  # [4, 4] float
    K: torch.Tensor  # [3, 3] float
    height: int
    width: int


def scene_span(means: torch.Tensor) -> float:
    """Diagonal extent of the scene's axis-aligned bounding box (world units)."""
    return float((means.float().max(dim=0).values - means.float().min(dim=0).values).norm())


def _grid_centers(means: torch.Tensor, grid: int, grid_frac: float) -> list[np.ndarray]:
    """Footprint-grid of look-at centers at the scene's median height."""
    lo = 0.5 - grid_frac / 2.0
    hi = 0.5 + grid_frac / 2.0
    qx = torch.quantile(means[:, 0], torch.tensor([lo, hi], device=means.device))
    qy = torch.quantile(means[:, 1], torch.tensor([lo, hi], device=means.device))
    zc = float(torch.median(means[:, 2]))
    if grid <= 1:
        return [np.array([float(qx.mean()), float(qy.mean()), zc])]
    xs = torch.linspace(float(qx[0]), float(qx[1]), grid)
    ys = torch.linspace(float(qy[0]), float(qy[1]), grid)
    return [np.array([float(x), float(y), zc]) for x in xs for y in ys]


@torch.no_grad()
def render_orbit_views(model, cfg, device: torch.device) -> list[View]:
    """Synthesize and render the multi-tier orbit grid (``view_source='render'``)."""
    means = model.means.detach()
    up = up_vector(getattr(cfg, "up", "+z"))
    views: list[View] = []
    t0 = time.time()
    for tier in cfg.tiers:
        K_np = intrinsics(tier.fov_deg, cfg.size, cfg.size)
        K = torch.from_numpy(K_np).float().unsqueeze(0).to(device)
        rad = tier.radius * cfg.zoom
        for center in _grid_centers(means, tier.grid, cfg.grid_frac):
            for elev in tier.elevations:
                for ai in range(tier.num_azimuth):
                    az = 360.0 * ai / tier.num_azimuth
                    w2c_np = orbit_cameras(center, elev, rad, num_azimuth=1, azimuth_offset_deg=az, up=up)[0][0]
                    w2c = torch.from_numpy(w2c_np).float().to(device)
                    img, _alpha = model.render_images_and_depths(
                        world_to_camera_matrices=w2c.unsqueeze(0),
                        projection_matrices=K,
                        image_width=cfg.size,
                        image_height=cfg.size,
                        near=0.01,
                        far=1e12,
                    )
                    rgb = img[0, ..., :3]
                    pil = Image.fromarray((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
                    views.append(View(pil=pil, w2c=w2c, K=K[0], height=cfg.size, width=cfg.size))
    print(f"[views] rendered {len(views)} orbit views in {time.time() - t0:.1f}s", flush=True)
    return views


@torch.no_grad()
def render_inside_out_views(model, cfg, device: torch.device) -> list[View]:
    """Render from the scene core looking OUTWARD (interior / unbounded scenes).

    For rooms and unbounded captures (Mip-NeRF360 indoor), an outside-in orbit only
    sees the backs of walls. We anchor the eye at the robust scene center (per-axis
    median, ignoring the background shell) and sweep a wide-FOV azimuth ring at a few
    pitches, so the surrounding content is framed.
    """
    means = model.means.detach()
    up = up_vector(getattr(cfg, "up", "+z"))
    center = means.float().median(dim=0).values.cpu().numpy().astype(np.float64)
    elevations = (-25.0, 0.0, 25.0)
    budget = int(getattr(cfg, "max_views", 0) or 120)
    num_az = max(6, budget // len(elevations))
    fov = float(getattr(cfg, "inside_out_fov", 80.0))
    K_np = intrinsics(fov, cfg.size, cfg.size)
    K = torch.from_numpy(K_np).float().unsqueeze(0).to(device)
    views: list[View] = []
    t0 = time.time()
    for w2c_np, _c2w in inside_out_cameras(center, elevations, num_az, up=up):
        w2c = torch.from_numpy(w2c_np).float().to(device)
        img, _alpha = model.render_images_and_depths(
            world_to_camera_matrices=w2c.unsqueeze(0),
            projection_matrices=K,
            image_width=cfg.size,
            image_height=cfg.size,
            near=0.01,
            far=1e12,
        )
        rgb = img[0, ..., :3]
        pil = Image.fromarray((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        views.append(View(pil=pil, w2c=w2c, K=K[0], height=cfg.size, width=cfg.size))
    print(
        f"[views] rendered {len(views)} inside-out views ({len(elevations)} pitch x {num_az} az, "
        f"fov {fov:.0f}) in {time.time() - t0:.1f}s",
        flush=True,
    )
    return views


@torch.no_grad()
# --- globe orbit: auto-framed radius + full-sphere rings -------------------


@torch.no_grad()
def core_extent(means: torch.Tensor, q_lo: float = 0.1, q_hi: float = 0.9):
    """Center + radius of the scene's DENSE core (quantile-trimmed).

    Unbounded 360/interior captures carry a sparse far halo of Gaussians that
    blows up the bbox; the q10-q90 box ignores it and tracks the actual subject.
    Returns ``(center[3] np.float64, radius float)`` where radius is half the core
    box diagonal.
    """
    m = means.detach().float()
    q = torch.tensor([q_lo, q_hi], device=m.device, dtype=m.dtype)
    lo_hi = torch.quantile(m, q, dim=0)
    lo, hi = lo_hi[0], lo_hi[1]
    center = ((lo + hi) / 2.0).cpu().numpy().astype(np.float64)
    radius = 0.5 * float((hi - lo).norm())
    return center, max(radius, 1e-4)


# Default frame fill: <1 lets the dense core overfill the frame so the subject reads
# large -- this is what SAM3 wants (small objects like a mitten get enough pixels to
# fire). Shared by the `globe` segment views and the `render` preview so what you
# eyeball is exactly what SAM sees. Tune per-call with --zoom.
_FRAME_FILL = 0.55


def auto_frame_radius(model, center, up, *, fov: float, zoom: float = 1.0, **_ignored) -> float:
    """Look-at distance that frames the scene's dense core to ~fill the view.

    Frames the q10-q90 core (not the floater-inflated bbox, which is what parked the
    camera absurdly far on 360 scenes). Analytic and floater-robust: distance so the
    core's angular size matches the field of view, scaled by ``_FRAME_FILL`` (a fixed
    "sit a little closer" default) and the caller's ``zoom`` (``<1`` pulls in, ``>1``
    backs out). ``center``/``up``/extra kwargs are accepted for call-site
    compatibility but the framing distance only depends on the core size.
    """
    _center, core_radius = core_extent(model.means)
    half = math.radians(fov) / 2.0
    dist = core_radius / max(math.tan(half), 1e-3)
    return max(dist * _FRAME_FILL * zoom, core_radius * 0.4)


# Dome view-quality tuning (see config.dome_low_zoom / reject_blur / blur_rel).
_DOME_LOW_ZOOM = 0.72  # default radius scale at the lowest ring (1.0 at nadir)
_DOME_RESAMPLE_ZOOM = 0.82  # pull a rejected view this much closer on its one retry
_BLUR_MAX_REJECT = 0.45  # never drop more than this fraction of the dome (safety)


def dome_radius_scale(elev: float, elev_min: float, elev_max: float, low: float) -> float:
    """Per-ring radius multiplier: ``low`` at the lowest (oblique) ring rising to
    1.0 at the top (near-nadir) ring. Low/oblique rings see through more foreground
    clutter, so pulling them closer reframes the subject larger."""
    if elev_max <= elev_min:
        return 1.0
    t = (elev - elev_min) / (elev_max - elev_min)
    return low + (1.0 - low) * max(0.0, min(1.0, t))


def view_sharpness(pil: Image.Image) -> float:
    """Cheap detail/blur metric: variance of the image Laplacian (grayscale).

    Sharp, well-framed renders carry high-frequency edges -> high variance; a view
    that renders mostly smooth blur or empty background -> near-zero. Used relative
    to the median across the view set, so it adapts per scene with no magic absolute
    threshold. Pure numpy (no cv2/scipy)."""
    g = np.asarray(pil.convert("L"), dtype=np.float32) / 255.0
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = -4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


@torch.no_grad()
def _render_w2c(model, w2c_np: np.ndarray, K: torch.Tensor, size: int, device: torch.device):
    """Render one orbit camera; returns ``(pil, w2c_tensor, depth[H,W], alpha[H,W])``.

    Depth and alpha come free from the same rasterization (depth is RGBA channel 3,
    alpha is the separate coverage buffer) and drive the dome view-quality gate.
    """
    w2c = torch.from_numpy(w2c_np).float().to(device)
    img, a = model.render_images_and_depths(
        world_to_camera_matrices=w2c.unsqueeze(0),
        projection_matrices=K,
        image_width=size,
        image_height=size,
        near=0.01,
        far=1e12,
    )
    rgb = img[0, ..., :3]
    depth = img[0, ..., 3]
    alpha = a[0, ..., 0]
    pil = Image.fromarray((rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    return pil, w2c, depth, alpha


def dome_view_metrics(
    pil: Image.Image, depth: torch.Tensor, alpha: torch.Tensor, radius: float, center_frac: float = 0.6
):
    """Quality metrics for a dome view, from the central region where the subject sits.

    Returns ``(coverage, occlusion, sharpness)``:

    * ``coverage`` - fraction of the center actually covered by the splat (low =>
      the camera is staring through to empty/see-through background).
    * ``occlusion`` - fraction of the covered center that sits *much* closer than the
      orbit radius, i.e. a foreground floater/furniture blob blocking the subject
      (this is the dark-blob failure on the low/oblique rings).
    * ``sharpness`` - detail energy (variance of Laplacian), a soft-blur backstop.

    Coverage/occlusion are physically grounded (alpha + depth vs. the known camera
    distance), so the gate generalizes across scenes without appearance tuning.
    """
    H, W = alpha.shape
    m = center_frac
    y0, y1 = int(H * (0.5 - m / 2)), int(H * (0.5 + m / 2))
    x0, x1 = int(W * (0.5 - m / 2)), int(W * (0.5 + m / 2))
    a = alpha[y0:y1, x0:x1]
    d = depth[y0:y1, x0:x1]
    solid = a > 0.5
    nsolid = float(solid.sum())
    coverage = nsolid / float(a.numel())
    occlusion = float((solid & (d < 0.5 * radius)).sum()) / nsolid if nsolid > 0 else 1.0
    return coverage, occlusion, view_sharpness(pil)


@torch.no_grad()
def render_globe_views(model, cfg, device: torch.device) -> list[View]:
    """Render an upper-dome orbit around the scene core (``view_source='globe'``).

    Auto-frames the radius (see :func:`auto_frame_radius`) then sweeps full azimuth
    rings at elevations looking DOWN at the core (oblique -> near-nadir). A bare
    splat only renders sharply near its original (inward/above) camera region, so the
    dome stays there and avoids the blurry see-through-background angles. This is the
    generalizable path for object/interior captures with no embedded cameras.
    """
    from .autoview import globe_rings

    up = up_vector(getattr(cfg, "up", "+z"))
    center, _r = core_extent(model.means)
    fov = 55.0
    base_radius = auto_frame_radius(model, center, up, fov=fov, zoom=float(getattr(cfg, "zoom", 1.0) or 1.0))
    budget = int(getattr(cfg, "max_views", 0) or 200)
    elevs, az_counts = globe_rings(budget)
    e_lo, e_hi = min(elevs), max(elevs)
    low_zoom = float(getattr(cfg, "dome_low_zoom", _DOME_LOW_ZOOM))
    K = torch.from_numpy(intrinsics(fov, cfg.size, cfg.size)).float().unsqueeze(0).to(device)
    gate = bool(getattr(cfg, "reject_blur", True))
    min_cov = float(getattr(cfg, "view_min_coverage", 0.22))
    max_occ = float(getattr(cfg, "view_max_occlusion", 0.45))
    t0 = time.time()

    def _render(elev, ai, naz, radius):
        cams = orbit_cameras(center, elev, radius, num_azimuth=naz, up=up)
        return _render_w2c(model, cams[ai % len(cams)][0], K, cfg.size, device)

    # Render every ring; pull low/oblique rings closer (dome_radius_scale) so blocked
    # side-on subjects fill more pixels. Gate each view on coverage + occlusion, and
    # give a failing view one retry from a HIGHER, closer angle (clears floor clutter
    # and near floaters -- exactly the unusable low-ring failure). This per-view
    # quality control is what makes the dome generalize to a new scene unattended.
    total = sum(az_counts)
    pils: list[Image.Image] = []
    w2cs: list[torch.Tensor] = []
    sharps: list[float] = []
    resampled = dropped = 0
    for elev, naz in zip(elevs, az_counts):
        ring_radius = base_radius * dome_radius_scale(elev, e_lo, e_hi, low_zoom)
        for ai in range(naz):
            pil, w2c, depth, alpha = _render(elev, ai, naz, ring_radius)
            cov, occ, sharp = dome_view_metrics(pil, depth, alpha, ring_radius)
            ok = (not gate) or (cov >= min_cov and occ <= max_occ)
            if not ok:
                # rise above the clutter (+ closer) and try once more
                pil2, w2c2, depth2, alpha2 = _render(elev + 8.0, ai, naz, ring_radius * _DOME_RESAMPLE_ZOOM)
                cov2, occ2, sharp2 = dome_view_metrics(pil2, depth2, alpha2, ring_radius * _DOME_RESAMPLE_ZOOM)
                if cov2 >= min_cov and occ2 <= max_occ:
                    pil, w2c, sharp, ok = pil2, w2c2, sharp2, True
                    resampled += 1
            if ok:
                pils.append(pil)
                w2cs.append(w2c)
                sharps.append(sharp)
            else:
                dropped += 1

    # Soft-blur backstop on survivors (catch blur the coverage gate let through),
    # but never let total drops exceed the safety cap.
    keep = list(range(len(pils)))
    if gate and len(pils) >= 4:
        sh = np.array(sharps, dtype=np.float64)
        thr = float(getattr(cfg, "blur_rel", 0.40)) * float(np.median(sh))
        budget = max(0, int(_BLUR_MAX_REJECT * total) - dropped)
        worst = [i for i in sorted(range(len(pils)), key=lambda i: sh[i]) if sh[i] < thr]
        drop2 = set(worst[:budget])
        dropped += len(drop2)
        keep = [i for i in keep if i not in drop2]

    views = [View(pil=pils[i], w2c=w2cs[i], K=K[0], height=cfg.size, width=cfg.size) for i in keep]
    extra = f", {resampled} resampled higher, {dropped} dropped (blocked/empty/blur)" if (resampled or dropped) else ""
    print(
        f"[views] rendered {len(views)} dome views (auto-framed radius {base_radius:.2f}, "
        f"low rings x{low_zoom:.2f} closer, {len(elevs)} rings {e_lo:.0f}..{e_hi:.0f} deg "
        f"looking down{extra}) in {time.time() - t0:.1f}s",
        flush=True,
    )
    return views


# --- COLMAP sparse-model reader (ground-truth image views) -----------------


def _qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def _read_colmap_cameras(path: pathlib.Path) -> dict:
    cams = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            num_params = _COLMAP_MODEL_NUM_PARAMS.get(model_id, 4)
            params = struct.unpack("<" + "d" * num_params, f.read(8 * num_params))
            cams[cam_id] = {"model_id": model_id, "width": width, "height": height, "params": params}
    return cams


def _colmap_K(cam: dict) -> np.ndarray:
    p = cam["params"]
    mid = cam["model_id"]
    if mid == 0:  # SIMPLE_PINHOLE: f, cx, cy
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    else:  # PINHOLE / radial / opencv all start fx, fy(or f), cx, cy
        if mid in (1, 4, 6, 10):  # fx, fy, cx, cy, ...
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else:  # SIMPLE_RADIAL etc: f, cx, cy, ...
            fx = fy = p[0]
            cx, cy = p[1], p[2]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _read_colmap_images(path: pathlib.Path) -> list[dict]:
    imgs = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (num_pts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * num_pts)  # skip 2D points (x, y double + point3D_id int64)
            imgs.append(
                {
                    "name": name.decode(),
                    "qvec": np.array([qw, qx, qy, qz]),
                    "tvec": np.array([tx, ty, tz]),
                    "cam_id": cam_id,
                }
            )
    return imgs


def _find_colmap(sfm: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Locate (cameras.bin, images.bin, images_dir) under a scene directory."""
    for sub in ("sparse/0", "sparse", "colmap/sparse/0", "."):
        d = sfm / sub
        if (d / "cameras.bin").is_file() and (d / "images.bin").is_file():
            for img_sub in ("images", "../images", "../../images"):
                cand = (d / img_sub).resolve()
                if cand.is_dir():
                    return d / "cameras.bin", d / "images.bin", cand
            return d / "cameras.bin", d / "images.bin", sfm / "images"
    raise FileNotFoundError(f"no COLMAP cameras.bin/images.bin found under {sfm}")


def _to_np(x) -> np.ndarray:
    """Tensor/array-like -> float64 numpy on CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _colmap_c2w(rec: dict) -> np.ndarray:
    """COLMAP record (world->camera) -> 4x4 camera->world in the ORIGINAL frame."""
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = _qvec2rotmat(rec["qvec"])
    w2c[:3, 3] = rec["tvec"]
    return np.linalg.inv(w2c)


def match_metadata_to_colmap(meta_c2w: np.ndarray, col_c2w: np.ndarray, nt: np.ndarray) -> np.ndarray:
    """Pair each training (metadata) camera with its source photo.

    The splat's metadata cameras live in the *normalized* splat frame; the COLMAP
    poses live in the scene's *original* (e.g. UTM) frame. We bring the COLMAP
    camera centers into the normalized frame via ``normalization_transform`` and,
    for each metadata camera, pick the nearest COLMAP image by position. Returns an
    index array ``match[M]`` into the COLMAP set. (Faithful to the validated
    LangSplat-Aerial ``_match_real_images``.)
    """
    norm_c2w = nt[None] @ col_c2w  # [P, 4, 4] colmap cameras -> normalized frame
    mt = meta_c2w[:, :3, 3]  # [M, 3]
    nc = norm_c2w[:, :3, 3]  # [P, 3]
    d = np.linalg.norm(mt[:, None, :] - nc[None, :, :], axis=2)  # [M, P]
    return d.argmin(axis=1)


def _load_metadata_image_views(cfg, device: torch.device, metadata: dict) -> list[View]:
    """Build views from the splat's *own* training cameras (correct frame).

    A splat saved during training carries its camera poses (already in the
    normalized splat frame), intrinsics, and image sizes in metadata. We render
    the per-Gaussian lift from those poses and load pixels from the matching
    source photos -- so the splat and the cameras are always in the same frame
    (unlike raw COLMAP poses, which sit in the original geospatial frame and would
    project nothing onto a recentered splat).
    """
    if not cfg.sfm:
        raise GeoLangSplatError(
            "view-source 'images' (the 'aerial' recipe) needs the scene's source photos: "
            "pass --sfm <scene-dir> (the COLMAP folder with sparse/ and images/). "
            "If you have no photos, use --view-source render instead."
        )
    sfm = pathlib.Path(cfg.sfm)
    if not sfm.is_dir():
        raise GeoLangSplatError(f"--sfm path does not exist or is not a directory: {sfm}")
    _cam_path, img_path, img_dir = _find_colmap(sfm)

    meta_c2w = _to_np(metadata["camera_to_world_matrices"]).astype(np.float64)  # [M, 4, 4]
    Kall = _to_np(metadata["projection_matrices"]).astype(np.float64)  # [M, 3, 3]
    isz = _to_np(metadata["image_sizes"]).astype(np.int64)  # [M, 2] = (H, W)
    nt = metadata.get("normalization_transform")
    nt = _to_np(nt).astype(np.float64) if nt is not None else np.eye(4)

    recs = sorted(_read_colmap_images(img_path), key=lambda d: d["name"])
    if not recs:
        raise GeoLangSplatError(f"no images found in {img_path}")
    names = [r["name"] for r in recs]
    col_c2w = np.stack([_colmap_c2w(r) for r in recs], axis=0)
    match = match_metadata_to_colmap(meta_c2w, col_c2w, nt)

    M = meta_c2w.shape[0]
    n = min(cfg.n_views, M) if cfg.n_views > 0 else M
    pick = np.unique(np.linspace(0, M - 1, n).round().astype(int))

    views: list[View] = []
    for i in pick:
        H, W = int(isz[i][0]), int(isz[i][1])
        w2c = np.linalg.inv(meta_c2w[i])
        pil = Image.open(img_dir / names[int(match[i])]).convert("RGB")
        if pil.size != (W, H):
            pil = pil.resize((W, H), Image.BILINEAR)
        views.append(
            View(
                pil=pil,
                w2c=torch.from_numpy(w2c).float().to(device),
                K=torch.from_numpy(Kall[i]).float().to(device),
                height=H,
                width=W,
            )
        )
    print(
        f"[views] loaded {len(views)} ground-truth views (metadata cameras, photos from {img_dir})",
        flush=True,
    )
    return views


def _load_colmap_image_views(cfg, device: torch.device) -> list[View]:
    """Legacy path: use raw COLMAP poses directly.

    Only correct when the splat is in the *same* frame as the COLMAP
    reconstruction (no training-time normalization). Used as a fallback when the
    splat has no camera metadata.
    """
    sfm = pathlib.Path(cfg.sfm)
    cam_path, img_path, img_dir = _find_colmap(sfm)
    cameras = _read_colmap_cameras(cam_path)
    images = sorted(_read_colmap_images(img_path), key=lambda d: d["name"])
    if not images:
        raise GeoLangSplatError(f"no images found in {img_path}")

    n = min(cfg.n_views, len(images)) if cfg.n_views > 0 else len(images)
    pick = np.unique(np.linspace(0, len(images) - 1, n).round().astype(int))

    views: list[View] = []
    for i in pick:
        rec = images[i]
        cam = cameras[rec["cam_id"]]
        R = _qvec2rotmat(rec["qvec"])
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = rec["tvec"]
        K = _colmap_K(cam)
        pil = Image.open(img_dir / rec["name"]).convert("RGB")
        views.append(
            View(
                pil=pil,
                w2c=torch.from_numpy(w2c).float().to(device),
                K=torch.from_numpy(K).float().to(device),
                height=cam["height"],
                width=cam["width"],
            )
        )
    print(f"[views] loaded {len(views)} ground-truth image views from {img_dir}", flush=True)
    return views


def load_image_views(cfg, device: torch.device, metadata: dict | None = None) -> list[View]:
    """Load ground-truth photo views (``view_source='images'``).

    Prefers the splat's own training cameras from ``metadata`` (which are in the
    splat's frame); falls back to raw COLMAP poses when no camera metadata exists.
    """
    if not cfg.sfm:
        raise GeoLangSplatError(
            "view-source 'images' (the 'aerial' recipe) needs the scene's source photos: "
            "pass --sfm <scene-dir> (the COLMAP folder with sparse/ and images/). "
            "If you have no photos, use --view-source render instead."
        )
    sfm = pathlib.Path(cfg.sfm)
    if not sfm.is_dir():
        raise GeoLangSplatError(f"--sfm path does not exist or is not a directory: {sfm}")
    if metadata and "camera_to_world_matrices" in metadata:
        return _load_metadata_image_views(cfg, device, metadata)
    return _load_colmap_image_views(cfg, device)


def generate_views(model, cfg, device: torch.device, metadata: dict | None = None) -> list[View]:
    """Dispatch on ``cfg.view_source`` to produce the view list.

    ``render`` synthesizes an orbit/ladder from scene geometry; ``globe`` synthesizes
    an object/interior close dome; ``images`` loads the ground-truth SfM photos.
    """
    # Explicit inside-out wins (the user asked for interior coverage), regardless of source.
    if getattr(cfg, "inside_out", False):
        return render_inside_out_views(model, cfg, device)
    if cfg.view_source == "globe":
        return render_globe_views(model, cfg, device)
    if cfg.view_source == "render":
        return render_orbit_views(model, cfg, device)
    if cfg.view_source == "images":
        return load_image_views(cfg, device, metadata)
    raise ValueError(f"unknown view_source {cfg.view_source!r} (expected 'render', 'globe', or 'images')")
