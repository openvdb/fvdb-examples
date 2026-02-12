# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
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

from .scene_transforms import ComputeCLIPFeatures, ComputeMultiScaleSAM2Masks


@dataclass
class SAM2Config:
    """Configuration for SAM2 multi-scale mask generation."""

    checkpoint: Literal["large", "small", "tiny", "base_plus"] = "large"
    """SAM2 checkpoint size to use."""

    points_per_side: int = 32
    """Grid density for point prompts."""

    pred_iou_thresh: float = 0.7
    """Predicted IoU threshold for mask filtering."""

    stability_score_thresh: float = 0.85
    """Stability score threshold for mask filtering."""

    nms_iou_thr: float = 0.8
    """IoU threshold for mask NMS post-processing."""

    nms_score_thr: float = 0.7
    """Score threshold for mask NMS."""

    nms_inner_thr: float = 0.5
    """Inner overlap threshold for mask NMS."""


@dataclass
class OpenCLIPConfig:
    """Configuration for OpenCLIP feature encoding."""

    clip_model_type: str = "ViT-B-16"
    """CLIP model architecture type."""

    clip_model_pretrained: str = "laion2b_s34b_b88k"
    """Pretrained weights identifier."""

    clip_n_dims: int = 512
    """Dimensionality of CLIP embeddings."""


@dataclass
class LangSplatV2PreprocessConfig:
    """Configuration for the full LangSplatV2 preprocessing pipeline.


    Example usage:

    .. code-block:: python

        from langsplatv2 import LangSplatV2PreprocessConfig
        from fvdb_reality_capture.sfm_scene import SfmScene

        # Create configuration
        config = LangSplatV2PreprocessConfig(
            image_downsample_factor=2,
            sam2=SAM2Config(checkpoint="large"),
        )

        # Load scene
        scene = SfmScene.from_colmap("path/to/colmap")

        # Build and apply transforms
        transforms = config.build_scene_transforms()
        preprocessed_scene = transforms(scene)
    """

    # Image preprocessing
    image_downsample_factor: int = 1
    """Factor by which to downsample images before processing."""

    rescale_jpeg_quality: int = 95
    """JPEG quality (0-100) when resaving downsampled images."""

    # Point cloud filtering
    points_percentile_filter: float = 0.0
    """Percentile of outlier points to filter based on distance from median."""

    min_points_per_image: int = 5
    """Minimum visible 3D points required for an image to be included."""

    # Scene cropping
    crop_bbox: tuple[float, float, float, float, float, float] | None = None
    """Optional bounding box to crop the scene to (xmin, xmax, ymin, ymax, zmin, zmax)."""

    crop_to_points: bool = False
    """If True, crop scene bounds to the point cloud extent."""

    # SAM2 configuration
    sam2: SAM2Config = field(default_factory=SAM2Config)
    """Configuration for SAM2 mask generation."""

    compute_sam_masks: bool = True
    """Whether to compute SAM2 segmentation masks."""

    # CLIP configuration
    clip: OpenCLIPConfig = field(default_factory=OpenCLIPConfig)
    """Configuration for CLIP feature encoding."""

    compute_clip_features: bool = True
    """Whether to compute CLIP features for masked regions."""

    # Device
    device: torch.device | str = "cuda"
    """Device for model inference."""

    def build_scene_transforms(
        self,
        normalization_transform: torch.Tensor | None = None,
    ):
        """
        Build the scene transform pipeline for LangSplatV2 preprocessing.

        This creates a Compose transform that applies all configured
        preprocessing steps in order:
        1. Scene normalization (optional)
        2. Point cloud percentile filtering
        3. Image downsampling
        4. Image filtering by visible points
        5. Multi-scale SAM2 mask generation
        6. CLIP feature encoding
        7. Scene cropping (optional)

        Args:
            normalization_transform: Optional 4x4 transformation matrix
                to apply to the scene for normalization.

        Returns:
            Compose transform that applies all preprocessing steps.
        """
        transforms = [
            # Scene normalization
            (
                TransformScene(normalization_transform.cpu().numpy())
                if normalization_transform is not None
                else Identity()
            ),
            # Point cloud filtering
            PercentileFilterPoints(
                percentile_min=np.full((3,), self.points_percentile_filter),
                percentile_max=np.full((3,), 100.0 - self.points_percentile_filter),
            ),
            # Image preprocessing
            DownsampleImages(
                image_downsample_factor=self.image_downsample_factor,
                rescaled_jpeg_quality=self.rescale_jpeg_quality,
            ),
            FilterImagesWithLowPoints(min_num_points=self.min_points_per_image),
            # SAM2 mask generation
            (
                ComputeMultiScaleSAM2Masks(
                    checkpoint=self.sam2.checkpoint,
                    points_per_side=self.sam2.points_per_side,
                    pred_iou_thresh=self.sam2.pred_iou_thresh,
                    stability_score_thresh=self.sam2.stability_score_thresh,
                    nms_iou_thr=self.sam2.nms_iou_thr,
                    nms_score_thr=self.sam2.nms_score_thr,
                    nms_inner_thr=self.sam2.nms_inner_thr,
                    device=self.device,
                )
                if self.compute_sam_masks
                else Identity()
            ),
            # CLIP feature encoding
            (
                ComputeCLIPFeatures(
                    clip_model_type=self.clip.clip_model_type,
                    clip_model_pretrained=self.clip.clip_model_pretrained,
                    clip_n_dims=self.clip.clip_n_dims,
                    device=self.device,
                )
                if self.compute_clip_features
                else Identity()
            ),
        ]

        # Optional scene cropping
        if self.crop_bbox is not None:
            transforms.append(CropScene(self.crop_bbox))
        if self.crop_to_points:
            transforms.append(CropSceneToPoints(margin=0.0))

        return Compose(*transforms)

    def build_sam2_transform(self):
        """
        Build only the SAM2 mask generation transform.

        Returns:
            ComputeMultiScaleSAM2Masks transform.
        """
        return ComputeMultiScaleSAM2Masks(
            checkpoint=self.sam2.checkpoint,
            points_per_side=self.sam2.points_per_side,
            pred_iou_thresh=self.sam2.pred_iou_thresh,
            stability_score_thresh=self.sam2.stability_score_thresh,
            nms_iou_thr=self.sam2.nms_iou_thr,
            nms_score_thr=self.sam2.nms_score_thr,
            nms_inner_thr=self.sam2.nms_inner_thr,
            device=self.device,
        )

    def build_clip_transform(self):
        """
        Build only the CLIP feature encoding transform.

        Returns:
            ComputeCLIPFeatures transform.
        """
        return ComputeCLIPFeatures(
            clip_model_type=self.clip.clip_model_type,
            clip_model_pretrained=self.clip.clip_model_pretrained,
            clip_n_dims=self.clip.clip_n_dims,
            device=self.device,
        )


@dataclass
class LangSplatV2ModelConfig:
    """Configuration for the LangSplatV2 language feature model."""

    vq_layer_num: int = 1
    """Number of residual vector quantization layers."""

    codebook_size: int = 64
    """Number of entries in each codebook."""

    clip_n_dims: int = 512
    """Dimensionality of CLIP embeddings."""

    topk: int = 4
    """Number of non-zero sparse coefficients per VQ layer."""


@dataclass
class LangSplatV2TrainingConfig:
    """Configuration for LangSplatV2 language feature training."""

    seed: int = 42
    """Random seed for reproducibility."""

    feature_level: int = 1
    """Which SAM scale level to train on (1=small, 2=medium, 3=large).

    Following the original LangSplatV2 paper, separate models are trained
    for each scale level and combined at evaluation time.
    """

    max_steps: int | None = None
    """Maximum number of training steps. If None, uses max_epochs."""

    max_epochs: int = 100
    """Maximum number of training epochs."""

    learning_rate: float = 0.0025
    """Learning rate for language feature parameters (logits + codebooks)."""

    batch_size: int = 1
    """Number of images per training batch."""

    accumulate_grad_steps: int = 1
    """Number of gradient accumulation steps before optimizer update."""

    use_cosine_loss: bool = True
    """Whether to use cosine similarity loss."""

    use_l1_loss: bool = False
    """Whether to use L1 loss."""

    normalize_features: bool = False
    """Whether to L2-normalize predicted features before computing loss."""

    model: LangSplatV2ModelConfig = field(default_factory=LangSplatV2ModelConfig)
    """Model architecture configuration."""

    log_test_images: bool = False
    """Whether to log visualization images (PCA features, error heatmaps)
    during training steps.  Eval images are always logged when the writer
    supports image output, regardless of this flag."""

    eval_at_percent: list[int] = field(default_factory=lambda: [25, 50, 75, 100])
    """Percentages of total epochs at which to run evaluation."""

    save_at_percent: list[int] = field(default_factory=lambda: [50, 100])
    """Percentages of total epochs at which to save checkpoints."""
