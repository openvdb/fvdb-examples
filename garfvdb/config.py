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
    """Configuration parameters for the GARfVDB segmentation model.

    Attributes:
        depth_samples: Number of depth samples per ray for feature computation.
        use_grid: If True, use 3D feature grids (GARField-style). If False, use
            per-Gaussian features.
        use_grid_conv: If True, apply sparse convolutions to grid features.
        enc_feats_one_idx_per_ray: If True, stochastically sample one feature
            per ray instead of weighted averaging.
        num_grids: Number of feature grids at different resolutions.
        grid_feature_dim: Feature dimension per grid.
        mlp_hidden_dim: Hidden layer dimension in the MLP.
        mlp_num_layers: Number of hidden layers in the MLP.
        mlp_output_dim: Output dimension of the MLP (feature embedding size).
    """

    depth_samples: int = 24
    use_grid: bool = True
    use_grid_conv: bool = False
    enc_feats_one_idx_per_ray: bool = False
    num_grids: int = 24
    grid_feature_dim: int = 8
    mlp_hidden_dim: int = 256
    mlp_num_layers: int = 4
    mlp_output_dim: int = 256


@dataclass
class GaussianSplatSegmentationTrainingConfig:
    """Configuration parameters for the segmentation training process.

    Attributes:
        seed: Random seed for reproducibility.
        max_steps: Maximum number of training steps. If None, uses max_epochs.
        max_epochs: Maximum number of training epochs.
        sample_pixels_per_image: Number of pixels to sample per image for training.
        batch_size: Number of images per training batch.
        accumulate_grad_steps: Number of gradient accumulation steps.
        model: Model architecture configuration.
        log_test_images: Whether to log test images during training.
        eval_at_percent: Percentages of total epochs at which to run evaluation
            (e.g., [10, 50, 100] runs eval at 10%, 50%, and 100% of training).
        save_at_percent: Percentages of total epochs at which to save checkpoints.
    """

    seed: int = 42
    max_steps: int | None = None
    max_epochs: int = 100
    sample_pixels_per_image: int = 256
    batch_size: int = 8
    accumulate_grad_steps: int = 1
    model: GARfVDBModelConfig = field(default_factory=GARfVDBModelConfig)
    log_test_images: bool = False
    eval_at_percent: list[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 75, 100])
    save_at_percent: list[int] = field(default_factory=lambda: [10, 20, 100])


@dataclass
class SfmSceneSegmentationTransformConfig:
    """Configuration for SfmScene transforms applied before segmentation training.

    Attributes:
        image_downsample_factor: Factor by which to downsample images.
        rescale_jpeg_quality: JPEG quality (0-100) when resaving downsampled images.
        points_percentile_filter: Percentile of outlier points to filter based on
            distance from median (0.0 = no filtering).
        crop_bbox: Optional bounding box to crop the scene to, specified as
            (xmin, xmax, ymin, ymax, zmin, zmax) in normalized coordinates.
        crop_to_points: If True, crop scene bounds to the point cloud extent.
        min_points_per_image: Minimum visible 3D points required for an image
            to be included in training.
        compute_segmentation_masks: Whether to compute SAM2 segmentation masks.
        sam2_points_per_side: SAM2 grid density for automatic mask generation.
        sam2_pred_iou_thresh: SAM2 predicted IoU threshold for mask filtering.
        sam2_stability_score_thresh: SAM2 stability score threshold for mask filtering.
        device: Device for SAM2 model inference.
    """

    image_downsample_factor: int = 1
    rescale_jpeg_quality: int = 95
    points_percentile_filter: float = 0.0
    crop_bbox: tuple[float, float, float, float, float, float] | None = None
    crop_to_points: bool = False
    min_points_per_image: int = 5
    compute_segmentation_masks: bool = True
    sam2_points_per_side: int = 40
    sam2_pred_iou_thresh: float = 0.80
    sam2_stability_score_thresh: float = 0.80
    device: torch.device | str = "cuda"

    def build_scene_transforms(self, gs3d: GaussianSplat3d, normalization_transform: torch.Tensor | None):
        # SfmScene transform
        transforms = [
            TransformScene(normalization_transform.numpy()) if normalization_transform is not None else Identity(),
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
