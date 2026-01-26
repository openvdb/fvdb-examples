# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from dataclasses import dataclass, field

import numpy as np
import torch
from fvdb import GaussianSplat3d
from fvdb_reality_capture.transforms import (
    Compose,
    CropScene,
    CropSceneToPoints,
    DownsampleImages,
    FilterImagesWithLowPoints,
    Identity,
    PercentileFilterPoints,
    TransformScene,
)
from garfvdb.scene_transforms import ComputeImageSegmentationMasksWithScales


@dataclass
class GARfVDBModelConfig:
    """Configuration parameters for the GARfVDB segmentation model."""

    depth_samples: int = 24
    """Number of depth samples per ray for feature computation."""

    use_grid: bool = True
    """If True, use 3D feature grids (GARField-style). If False, use per-Gaussian features."""

    use_grid_conv: bool = False
    """If True, apply sparse convolutions to grid features."""

    enc_feats_one_idx_per_ray: bool = False
    """If True, stochastically sample one feature per ray instead of weighted averaging."""

    num_grids: int = 24
    """Number of feature grids at different resolutions."""

    grid_feature_dim: int = 8
    """Feature dimension per grid."""

    mlp_hidden_dim: int = 256
    """Hidden layer dimension in the MLP."""

    mlp_num_layers: int = 4
    """Number of hidden layers in the MLP."""

    mlp_output_dim: int = 256
    """Output dimension of the MLP (feature embedding size)."""


@dataclass
class GaussianSplatSegmentationTrainingConfig:
    """Configuration parameters for the segmentation training process."""

    seed: int = 42
    """Random seed for reproducibility."""

    max_steps: int | None = None
    """Maximum number of training steps. If None, uses max_epochs."""

    max_epochs: int = 100
    """Maximum number of training epochs."""

    sample_pixels_per_image: int = 256
    """Number of pixels to sample per image for training."""

    batch_size: int = 8
    """Number of images per training batch."""

    accumulate_grad_steps: int = 1
    """Number of gradient accumulation steps."""

    model: GARfVDBModelConfig = field(default_factory=GARfVDBModelConfig)
    """Model architecture configuration."""

    log_test_images: bool = False
    """Whether to log test images during training."""

    eval_at_percent: list[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 75, 100])
    """Percentages of total epochs at which to run evaluation."""

    save_at_percent: list[int] = field(default_factory=lambda: [10, 20, 100])
    """Percentages of total epochs at which to save checkpoints."""


@dataclass
class SfmSceneSegmentationTransformConfig:
    """Configuration for SfmScene transforms applied before segmentation training."""

    image_downsample_factor: int = 1
    """Factor by which to downsample images."""

    rescale_jpeg_quality: int = 95
    """JPEG quality (0-100) when resaving downsampled images."""

    points_percentile_filter: float = 0.0
    """Percentile of outlier points to filter based on distance from median (0.0 = no filtering)."""

    crop_bbox: tuple[float, float, float, float, float, float] | None = None
    """Optional bounding box to crop the scene to (xmin, xmax, ymin, ymax, zmin, zmax)."""

    crop_to_points: bool = False
    """If True, crop scene bounds to the point cloud extent."""

    min_points_per_image: int = 5
    """Minimum visible 3D points required for an image to be included in training."""

    compute_segmentation_masks: bool = True
    """Whether to compute SAM2 segmentation masks."""

    sam2_points_per_side: int = 40
    """SAM2 grid density for automatic mask generation."""

    sam2_pred_iou_thresh: float = 0.80
    """SAM2 predicted IoU threshold for mask filtering."""

    sam2_stability_score_thresh: float = 0.80
    """SAM2 stability score threshold for mask filtering."""

    device: torch.device | str = "cuda"
    """Device for SAM2 model inference."""

    def build_scene_transforms(self, gs3d: GaussianSplat3d, normalization_transform: torch.Tensor | None):
        # SfmScene transform
        transforms = [
            (
                TransformScene(normalization_transform.cpu().numpy())
                if normalization_transform is not None
                else Identity()
            ),
            PercentileFilterPoints(
                percentile_min=np.full((3,), self.points_percentile_filter),
                percentile_max=np.full((3,), 100.0 - self.points_percentile_filter),
            ),
            DownsampleImages(
                image_downsample_factor=self.image_downsample_factor,
                rescaled_jpeg_quality=self.rescale_jpeg_quality,
            ),
            FilterImagesWithLowPoints(min_num_points=self.min_points_per_image),
            (
                ComputeImageSegmentationMasksWithScales(
                    gs3d=gs3d,
                    checkpoint="large",
                    points_per_side=self.sam2_points_per_side,
                    pred_iou_thresh=self.sam2_pred_iou_thresh,
                    stability_score_thresh=self.sam2_stability_score_thresh,
                    device=self.device,
                )
                if self.compute_segmentation_masks
                else Identity()
            ),
        ]
        if self.crop_bbox is not None:
            transforms.append(CropScene(self.crop_bbox))
        if self.crop_to_points:
            transforms.append(CropSceneToPoints(margin=0.0))
        return Compose(*transforms)
