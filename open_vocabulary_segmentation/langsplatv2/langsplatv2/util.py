# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
from typing import Literal

import torch
import torch.nn.functional as F
from fvdb import GaussianSplat3d
from fvdb_reality_capture.radiance_fields import GaussianSplatReconstruction
from fvdb_reality_capture.sfm_scene import SfmScene

DatasetType = Literal["colmap", "simple_directory", "e57"]


def load_splats_from_file(path: pathlib.Path, device: str | torch.device) -> tuple[GaussianSplat3d, dict]:
    """
    Load a PLY or a checkpoint file and metadata.

    Args:
        path: Path to the PLY or checkpoint file.
        device: Device to load the model onto.

    Returns:
        model: The loaded Gaussian Splat model.
        metadata: The metadata associated with the model.
    """
    if path.suffix.lower() == ".ply":
        model, metadata = GaussianSplat3d.from_ply(path, device)
    elif path.suffix.lower() in (".pt", ".pth"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        runner = GaussianSplatReconstruction.from_state_dict(checkpoint, device=device)
        model = runner.model
        metadata = runner.reconstruction_metadata
    else:
        raise ValueError("Input path must end in .ply, .pt, or .pth")

    return model, metadata


def load_sfm_scene(path: pathlib.Path, dataset_type: DatasetType) -> SfmScene:
    """
    Load an SfM scene from the specified dataset path and type.

    Args:
        path: Path to the dataset folder.
        dataset_type: Type of the dataset.

    Returns:
        SfmScene: The loaded SfM scene.
    """
    if dataset_type == "colmap":
        sfm_scene = SfmScene.from_colmap(path)
    elif dataset_type == "simple_directory":
        sfm_scene = SfmScene.from_simple_directory(path)
    elif dataset_type == "e57":
        sfm_scene = SfmScene.from_e57(path)
    else:
        raise ValueError(f"Unsupported dataset_type {dataset_type}")

    return sfm_scene


def cosine_error_map(
    predicted: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-pixel cosine error and visualize as a turbo-colormapped heatmap.

    Computes ``1 - cosine_similarity`` at each pixel and maps the scalar
    error through the *turbo* colormap for RGB visualization.

    Args:
        predicted: Predicted feature tensor of shape ``[B, H, W, C]``.
        gt: Ground truth feature tensor of shape ``[B, H, W, C]``.
        mask: Optional boolean mask of shape ``[B, H, W]`` indicating valid
            pixels.  Invalid pixels are shown as black.

    Returns:
        RGB heatmap of shape ``[B, H, W, 3]`` with values in ``[0, 1]``.
    """
    cos_sim = F.cosine_similarity(predicted, gt, dim=-1)  # [B, H, W]
    error = (1.0 - cos_sim).clamp(0.0, 1.0)

    heatmap = _apply_turbo_colormap(error)  # [B, H, W, 3]

    if mask is not None:
        heatmap = heatmap * mask.unsqueeze(-1).float()

    return heatmap


def _apply_turbo_colormap(values: torch.Tensor) -> torch.Tensor:
    """Map scalar values in ``[0, 1]`` through the *turbo* colormap.

    Uses ``matplotlib.colormaps["turbo"]`` when available, otherwise falls
    back to a simple blue-to-red gradient.

    Args:
        values: Scalar tensor of any shape with values in ``[0, 1]``.

    Returns:
        RGB tensor with an extra trailing dimension of size 3.
    """
    device = values.device
    values_np = values.detach().cpu().numpy()

    try:
        import matplotlib

        cmap = matplotlib.colormaps["turbo"]
        rgb_np = cmap(values_np)[..., :3]  # drop alpha channel
    except (ImportError, AttributeError):
        # Fallback: simple blue -> red linear gradient
        import numpy as np

        r = values_np
        g = np.zeros_like(values_np)
        b = 1.0 - values_np
        rgb_np = np.stack([r, g, b], axis=-1)

    return torch.from_numpy(rgb_np.astype("float32")).to(device)


_SH_SCALE = 1.0 / 0.28209479177387814  # ≈ 3.5449077018110318
_SH_OFFSET = -0.5 * _SH_SCALE  # ≈ -1.7724538509055159


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB values to spherical harmonics coefficients.

    Args:
        rgb: [N, 3] Tensor of RGB values

    Returns:
        [N, 3] Tensor of spherical harmonics coefficients
    """
    return rgb * _SH_SCALE + _SH_OFFSET
