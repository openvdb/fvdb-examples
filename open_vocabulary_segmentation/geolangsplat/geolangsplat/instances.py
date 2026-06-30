# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Split a semantic selection into individual instances and pick one.

A text query ("house") selects every Gaussian of that class -- often many
separate buildings. This module turns that one mask into a list of distinct
instances via **3D connected components** (voxelize the selection, union
spatially-adjacent voxels), ranks them largest-first, and exposes a notebook-
friendly result:

    res = engine.query_instances("house")
    res.show()              # inline top-down PNG with 0,1,2... drawn on each house
    res.extract(0, "h.ply") # write just the largest house

The connected-components core (:func:`connected_components`) is pure tensor/numpy
and runs on CPU; only the rendering helpers touch the GPU/fvdb.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


# --- connected components (pure, CPU-testable) -----------------------------

# 13 "forward" neighbor offsets for 26-connectivity. Union is symmetric, so
# visiting half the neighborhood links every adjacent voxel pair exactly once.
_OFFSETS_26 = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (1, -1, -1),
)


def _union_find(n: int, edges) -> list[int]:
    """Connected-component roots for ``n`` nodes given an iterable of (a, b) edges."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def connected_components(
    means: torch.Tensor,
    sel: torch.Tensor,
    cfg,
    *,
    link_frac: float | None = None,
    min_size: int | None = None,
    span: float | None = None,
) -> tuple[torch.Tensor, list[dict]]:
    """Split selected Gaussians into spatial instances via voxel connected components.

    Voxelize the selected points at ``link_frac * scene_span`` and union voxels
    that are 26-adjacent, so a contiguous blob of Gaussians (one building) becomes
    one component while spatially separated blobs split apart. Components smaller
    than ``min_size`` Gaussians are dropped (speckle).

    ``span`` (the scene extent the voxel size is a fraction of) defaults to the
    full min/max range; pass a robust value (e.g. a 1-99% quantile span) when the
    scene has far-away floaters that would otherwise inflate the voxel and merge
    separate objects.

    Returns ``(comp_ids, infos)``:

    * ``comp_ids`` -- ``[N]`` long, in input order. ``-1`` for unselected or
      dropped Gaussians, else the instance index (``0`` = largest).
    * ``infos`` -- per-instance dicts (largest first) with ``idx``, ``size``,
      ``centroid`` (3,), ``bbox_min`` (3,), ``bbox_max`` (3,).
    """
    link_frac = cfg.inst_link_frac if link_frac is None else link_frac
    min_size = cfg.inst_min_size if min_size is None else min_size

    N = sel.shape[0]
    comp_ids = torch.full((N,), -1, dtype=torch.long)
    sel_idx = sel.nonzero(as_tuple=True)[0]
    if sel_idx.numel() == 0:
        return comp_ids, []

    pts_t = means[sel_idx].detach().float().cpu()
    pts = pts_t.numpy()
    if span is None:
        mn = means.detach().float().min(dim=0).values
        mx = means.detach().float().max(dim=0).values
        span = float((mx - mn).max().cpu())
    vox = max(link_frac * span, 1e-9)

    keys = np.floor(pts / vox).astype(np.int64)  # [M, 3]
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)  # uniq [V, 3], inv [M]
    inv = inv.reshape(-1)
    V = uniq.shape[0]

    lut = {(int(r[0]), int(r[1]), int(r[2])): i for i, r in enumerate(uniq)}
    edges = []
    for i in range(V):
        x, y, z = int(uniq[i, 0]), int(uniq[i, 1]), int(uniq[i, 2])
        for dx, dy, dz in _OFFSETS_26:
            j = lut.get((x + dx, y + dy, z + dz))
            if j is not None:
                edges.append((i, j))
    roots = np.asarray(_union_find(V, edges), dtype=np.int64)

    point_root = roots[inv]  # [M] component root per selected point
    uniq_roots, sizes = np.unique(point_root, return_counts=True)
    order = np.argsort(-sizes)  # largest first

    infos: list[dict] = []
    new_id = 0
    for r in uniq_roots[order]:
        size = int((point_root == r).sum())
        if size < min_size:
            continue
        member = sel_idx[torch.from_numpy(point_root == r)]
        cpts = means[member].detach().float()
        infos.append(
            {
                "idx": new_id,
                "size": size,
                "centroid": cpts.mean(dim=0).cpu().numpy(),
                "bbox_min": cpts.min(dim=0).values.cpu().numpy(),
                "bbox_max": cpts.max(dim=0).values.cpu().numpy(),
            }
        )
        comp_ids[member] = new_id
        new_id += 1
    return comp_ids, infos


# --- notebook-facing result ------------------------------------------------


@dataclass
class Instance:
    """One spatial instance from a semantic selection (largest first = idx 0)."""

    idx: int
    size: int
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray


@dataclass
class InstanceResult:
    """Instances of one prompt, ready to inspect/extract from a notebook.

    Hold a reference to the splat ``model`` so :meth:`show` can render a numbered
    top-down and :meth:`extract` can write a single instance's ``.ply``.
    """

    prompt: str
    instances: list[Instance]
    comp_ids: torch.Tensor  # [N] long, -1 = none, else instance idx
    config: object
    _model: object = None
    metadata: dict = field(default_factory=dict)

    # -- container sugar ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, k: int) -> Instance:
        return self.instances[k]

    def __repr__(self) -> str:
        head = f"InstanceResult({self.prompt!r}: {len(self.instances)} instances)"
        rows = [
            f"  #{ins.idx}: {ins.size:,} gaussians  centroid=("
            f"{ins.centroid[0]:.2f}, {ins.centroid[1]:.2f}, {ins.centroid[2]:.2f})"
            for ins in self.instances[:20]
        ]
        if len(self.instances) > 20:
            rows.append(f"  ... (+{len(self.instances) - 20} more)")
        return "\n".join([head, *rows])

    # -- masks / extraction -------------------------------------------------

    def mask(self, k: int) -> torch.Tensor:
        """Boolean ``[N]`` mask selecting only instance ``k``."""
        self._check(k)
        return self.comp_ids == k

    @property
    def all_mask(self) -> torch.Tensor:
        """Boolean ``[N]`` mask of every kept instance (the whole class)."""
        return self.comp_ids >= 0

    def extract(self, k: int, path) -> int:
        """Write only instance ``k`` to ``path`` as a ``.ply``. Returns its count."""
        from . import outputs as _out

        self._check(k)
        return _out.write_ply_segmented(self._model, self.mask(k), path, metadata=self.metadata)

    def _check(self, k: int) -> None:
        if not (0 <= k < len(self.instances)):
            raise IndexError(
                f"instance {k} out of range; this query found {len(self.instances)} "
                f"(valid 0..{len(self.instances) - 1})"
            )

    # -- visualization ------------------------------------------------------

    def show(self, *, size: int = 900, device=None, path=None):
        """Inline top-down render with each instance's index drawn on it.

        Returns a ``PIL.Image`` (Jupyter displays it inline). Pass ``path`` to also
        save the PNG. The number drawn on each instance is exactly the ``k`` you
        pass to :meth:`extract`.
        """
        pil, uv = _render_topdown(self._model, self.config, self.instances, size=size, device=device)
        labels = [(int(ins.idx), uv[i]) for i, ins in enumerate(self.instances) if uv[i] is not None]
        out = _annotate(pil, labels)
        if path is not None:
            out.save(path)
            print(f"[instances] wrote {path}", flush=True)
        return out

    def show_one(self, k: int, *, size: int = 900, device=None, path=None):
        """Top-down render with only instance ``k`` highlighted (rest dimmed)."""
        self._check(k)
        pil, uv = _render_topdown(
            self._model, self.config, self.instances, size=size, device=device, highlight=self.mask(k)
        )
        out = _annotate(pil, [(k, uv[k])] if uv[k] is not None else [])
        if path is not None:
            out.save(path)
            print(f"[instances] wrote {path}", flush=True)
        return out


# --- rendering helpers (GPU / fvdb) ----------------------------------------


def _topdown_camera(means: torch.Tensor, cfg, size: int, fov: float = 60.0):
    """A nadir camera framing the scene footprint. Returns ``(w2c_np, K_np)``."""
    from .autoview import resolve_up
    from .cameras import _perp_basis, intrinsics, orbit_cameras, up_vector

    m = means.detach().float().cpu().numpy()
    center = 0.5 * (m.min(axis=0) + m.max(axis=0))
    up = resolve_up(getattr(cfg, "up", "auto"), means)
    up_hat = up_vector(up)
    e1, e2 = _perp_basis(up_hat)
    rel = m - center
    a, b, c = rel @ e1, rel @ e2, rel @ up_hat
    half = 0.5 * max(float(a.max() - a.min()), float(b.max() - b.min())) * 1.1
    half = max(half, 1e-6)
    height = half / np.tan(np.radians(fov) / 2.0) + max(0.0, float(c.max()))
    w2c_np = orbit_cameras(center, 90.0, height, num_azimuth=1, up=up_hat)[0][0]
    K_np = intrinsics(fov, size, size)
    return w2c_np, K_np


@torch.no_grad()
def _render_topdown(model, cfg, instances, *, size: int, device=None, highlight=None, fov: float = 60.0):
    """Render the splat top-down; project instance centroids to pixels.

    Returns ``(pil, uv)`` where ``uv[i]`` is the ``(u, v)`` pixel of instance ``i``'s
    centroid (or ``None`` if it falls outside / behind the camera). When
    ``highlight`` (a bool mask) is given, those Gaussians are tinted and the rest
    dimmed before rendering.
    """
    from PIL import Image

    from .cameras import project

    means = model.means.detach()
    dev = means.device if device is None else torch.device(device)
    w2c_np, K_np = _topdown_camera(means, cfg, size, fov=fov)
    w2c = torch.from_numpy(w2c_np).float().to(dev)
    K = torch.from_numpy(K_np).float().to(dev)

    disp = _highlight_model(model, highlight) if highlight is not None else model
    img, _a = disp.render_images_and_depths(
        world_to_camera_matrices=w2c.unsqueeze(0),
        projection_matrices=K.unsqueeze(0),
        image_width=size,
        image_height=size,
        near=0.01,
        far=1e12,
    )
    rgb = (img[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(rgb)

    uv: list = []
    if instances:
        cents = torch.from_numpy(np.stack([ins.centroid for ins in instances])).float().to(dev)
        u, v, z = project(cents, w2c, float(K[0, 0]), float(K[0, 2]), float(K[1, 2]))
        for i in range(len(instances)):
            zi = float(z[i])
            ui, vi = float(u[i]), float(v[i])
            if zi > 0 and 0 <= ui < size and 0 <= vi < size:
                uv.append((ui, vi))
            else:
                uv.append(None)
    return pil, uv


def _highlight_model(model, mask: torch.Tensor):
    """A copy of the splat with ``mask`` Gaussians tinted bright and the rest dimmed."""
    from fvdb import GaussianSplat3d

    from .outputs import _recolor_sh0

    sh0 = model.sh0.detach().clone()
    shN = model.shN.detach().clone()
    other = ~mask
    if bool(other.any()):  # dim the rest toward grey so the pick pops
        sh0 = _recolor_sh0(sh0, other, (0.18, 0.18, 0.20), 0.7)
        shN[other] = shN[other] * 0.3
    if bool(mask.any()):
        sh0 = _recolor_sh0(sh0, mask, (1.0, 0.95, 0.1), 0.85)
        shN[mask] = shN[mask] * 0.15
    return GaussianSplat3d.from_tensors(
        means=model.means.detach(),
        quats=model.quats.detach(),
        log_scales=model.log_scales.detach(),
        logit_opacities=model.logit_opacities.detach(),
        sh0=sh0,
        shN=shN,
    )


def _annotate(pil, labels):
    """Draw each ``(index, (u, v))`` label as a numbered marker on a copy of ``pil``."""
    from PIL import ImageDraw, ImageFont

    img = pil.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    r = max(10, img.size[0] // 60)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(r * 1.6))
    except Exception:
        font = ImageFont.load_default()
    for idx, uv in labels:
        if uv is None:
            continue
        u, v = uv
        draw.ellipse([u - r, v - r, u + r, v + r], fill=(255, 235, 40), outline=(20, 20, 20), width=2)
        txt = str(idx)
        try:  # center the glyph in the marker (textbbox carries the baseline offset)
            x0, y0, x1, y1 = draw.textbbox((0, 0), txt, font=font)
            tx, ty = u - (x1 - x0) / 2 - x0, v - (y1 - y0) / 2 - y0
        except Exception:
            tw, th = draw.textsize(txt, font=font)
            tx, ty = u - tw / 2, v - th / 2
        draw.text((tx, ty), txt, fill=(10, 10, 10), font=font)
    return img


def build_instances(
    model,
    selected: torch.Tensor,
    cfg,
    prompt: str,
    *,
    link_frac: float | None = None,
    min_size: int | None = None,
    metadata: dict | None = None,
) -> InstanceResult:
    """Connected-components a selection into an :class:`InstanceResult`."""
    comp_ids, infos = connected_components(model.means.detach(), selected, cfg, link_frac=link_frac, min_size=min_size)
    instances = [
        Instance(
            idx=d["idx"],
            size=d["size"],
            centroid=d["centroid"],
            bbox_min=d["bbox_min"],
            bbox_max=d["bbox_max"],
        )
        for d in infos
    ]
    return InstanceResult(
        prompt=prompt,
        instances=instances,
        comp_ids=comp_ids,
        config=cfg,
        _model=model,
        metadata=metadata or {},
    )
