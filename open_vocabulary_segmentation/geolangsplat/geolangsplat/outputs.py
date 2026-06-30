# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Output writers: segmented ``.ply``, highlight-overlay ``.ply``, and
per-Gaussian reports (``.csv`` / ``.npz``).

All per-Gaussian arrays stay in the input ``.ply`` order so a consumer (e.g. a
co-worker holding the same ``.ply``) can index directly by row.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

SH_C0 = 0.28209479177387814

# A reusable categorical palette (RGB 0..1) for label recolouring.
_PALETTE = np.array(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.70, 0.20],
        [0.15, 0.45, 0.95],
        [0.95, 0.75, 0.10],
        [0.65, 0.25, 0.85],
        [0.10, 0.80, 0.80],
        [0.95, 0.45, 0.10],
        [0.55, 0.55, 0.55],
        [0.85, 0.30, 0.55],
        [0.40, 0.30, 0.20],
    ],
    dtype=np.float32,
)


def _gs():
    from fvdb import GaussianSplat3d

    return GaussianSplat3d


def _save_ply(disp, path: str, metadata: dict | None) -> None:
    """Save a splat, round-tripping metadata when the backend accepts it.

    fvdb's ``save_ply`` only takes scalar/tensor metadata values; if the dict it
    came with has anything richer (e.g. nested camera info from a checkpoint), we
    retry without metadata rather than failing the whole write.
    """
    if metadata:
        try:
            disp.save_ply(path, metadata=metadata)
            return
        except Exception:
            pass
    disp.save_ply(path)


def _subset(model, mask: torch.Tensor):
    return _gs().from_tensors(
        means=model.means.detach()[mask],
        quats=model.quats.detach()[mask],
        log_scales=model.log_scales.detach()[mask],
        logit_opacities=model.logit_opacities.detach()[mask],
        sh0=model.sh0.detach()[mask],
        shN=model.shN.detach()[mask],
    )


def write_ply_segmented(model, selected: torch.Tensor, path, metadata: dict | None = None) -> int:
    """Write a ``.ply`` containing only the selected Gaussians."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = _subset(model, selected)
    _save_ply(sub, str(path), metadata)
    return int(selected.sum())


def _recolor_sh0(sh0: torch.Tensor, mask: torch.Tensor, color, blend: float) -> torch.Tensor:
    tgt = torch.tensor(color, device=sh0.device, dtype=sh0.dtype).view(1, 1, 3)
    cur = 0.5 + SH_C0 * sh0[mask]
    sh0 = sh0.clone()
    sh0[mask] = ((1 - blend) * cur + blend * tgt - 0.5) / SH_C0
    return sh0


def write_ply_overlay(
    model, selected: torch.Tensor, path, color=(1.0, 0.95, 0.1), blend: float = 0.75, metadata: dict | None = None
) -> int:
    """Write the full ``.ply`` with the selected Gaussians tinted ``color``."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sh0 = model.sh0.detach()
    shN = model.shN.detach().clone()
    if bool(selected.any()):
        sh0 = _recolor_sh0(sh0, selected, color, blend)
        shN[selected] = shN[selected] * (1 - blend)
    else:
        sh0 = sh0.clone()
    disp = _gs().from_tensors(
        means=model.means.detach(),
        quats=model.quats.detach(),
        log_scales=model.log_scales.detach(),
        logit_opacities=model.logit_opacities.detach(),
        sh0=sh0,
        shN=shN,
    )
    _save_ply(disp, str(path), metadata)
    return int(selected.sum())


def write_ply_labels(model, label_ids: torch.Tensor, path, blend: float = 0.85, metadata: dict | None = None) -> int:
    """Recolour the full ``.ply`` by per-Gaussian label (unlabeled stay original)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sh0 = model.sh0.detach().clone()
    shN = model.shN.detach().clone()
    labels = label_ids.detach().cpu().numpy()
    for lid in np.unique(labels):
        if lid < 0:
            continue
        mask = torch.from_numpy(labels == lid).to(sh0.device)
        color = tuple(_PALETTE[int(lid) % len(_PALETTE)].tolist())
        sh0 = _recolor_sh0(sh0, mask, color, blend)
        shN[mask] = shN[mask] * (1 - blend)
    disp = _gs().from_tensors(
        means=model.means.detach(),
        quats=model.quats.detach(),
        log_scales=model.log_scales.detach(),
        logit_opacities=model.logit_opacities.detach(),
        sh0=sh0,
        shN=shN,
    )
    _save_ply(disp, str(path), metadata)
    return int((label_ids >= 0).sum())


def write_report_csv(means: np.ndarray, label_ids: np.ndarray, scores: np.ndarray, vocab, path) -> None:
    """Write ``index,x,y,z,label_id,label,score`` plus a sibling ``legend.json``."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vocab = list(vocab)
    lines = ["index,x,y,z,label_id,label,score"]
    for i in range(means.shape[0]):
        lid = int(label_ids[i])
        name = vocab[lid] if 0 <= lid < len(vocab) else ""
        sc = float(scores[i]) if scores.ndim == 1 else float(scores[i, lid]) if lid >= 0 else 0.0
        x, y, z = means[i]
        lines.append(f"{i},{x:.6f},{y:.6f},{z:.6f},{lid},{name},{sc:.4f}")
    path.write_text("\n".join(lines) + "\n")
    legend = {str(i): v for i, v in enumerate(vocab)}
    legend["-1"] = "unlabeled"
    (path.parent / "legend.json").write_text(json.dumps(legend, indent=2))


def write_report_npz(means: np.ndarray, label_ids: np.ndarray, scores: np.ndarray, vocab, path) -> None:
    """Write a compact ``.npz`` (means, label_ids, scores, vocab)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), means=means, label_ids=label_ids, scores=scores, vocab=np.array(list(vocab)))
