# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
from typing import Literal

import torch
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
