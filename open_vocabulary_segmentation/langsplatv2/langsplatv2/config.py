# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from dataclasses import dataclass, field
from pathlib import Path
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

from .scene_transforms import (
    ComputeCLIPFeatures,
    ComputeMultiScaleSAM1Masks,
    ComputeMultiScaleSAM2Masks,
    ImportOriginalLangSplatV2Features,
)


@dataclass
class SAM2Config:
    """Configuration for SAM2 multi-scale mask generation."""

    checkpoint: Literal["large", "small", "tiny", "base_plus"] = "large"
    """SAM2 checkpoint size to use.  Larger models produce higher-quality masks
    but run a heavier image encoder.  "base_plus" or "small" can noticeably
    reduce per-image encoding time at some cost to mask quality."""

    points_per_side: int = 32
    """Grid density for point prompts.  Total points = points_per_side**2
    (e.g. 32 -> 1024, 16 -> 256).  Reducing this value is the single
    largest lever for speeding up mask generation because each point
    requires a decoder forward pass."""

    points_per_batch: int = 256
    """Points processed simultaneously by SAM2.  Larger values reduce the
    number of decoder forward passes (fewer kernel launches) at the cost
    of higher peak GPU memory.  256 is safe on 24 GB+ GPUs."""

    pred_iou_thresh: float = 0.5
    """Predicted IoU threshold for mask filtering.  SAM2 predicts
    substantially lower IoU scores than SAM1 (e.g. small-mask means
    of 0.3-0.6 vs 0.8-0.9), so the threshold is set lower to achieve
    comparable mask survival rates."""

    stability_score_thresh: float = 0.85
    """Stability score threshold for mask filtering."""

    crop_n_layers: int = 1
    """Number of crop layers.  With ``crop_n_layers=1`` (the LangSplatV2
    default), 5 crops are generated per image (1 full + 4 overlapping
    sub-crops), each running a full encoder + decoder pass.  Setting
    this to 0 reduces to a single full-image crop (~5x fewer passes)
    at the cost of losing detail from sub-crop masks."""

    crop_n_points_downscale_factor: int = 2
    """Point grid downscale factor per crop layer.  Sub-crop layers use
    (points_per_side / 2)**2 points instead of points_per_side**2,
    reducing decoder cost on sub-crops by ~4x while keeping the
    full-image crop at full density.  The original LangSplatV2 uses 1
    with SAM ViT-H; 2 is a reasonable default with SAM2."""

    min_mask_region_area: int = 100
    """Minimum mask region area for post-processing (matching the original
    LangSplatV2 which uses ``min_mask_region_area=100``)."""

    box_nms_thresh: float = 0.7
    """Box NMS IoU threshold within each crop."""

    nms_iou_thr: float = 0.8
    """IoU threshold for mask NMS post-processing."""

    nms_score_thr: float = 0.7
    """Score threshold for mask NMS."""

    nms_inner_thr: float = 0.5
    """Inner overlap threshold for mask NMS."""


@dataclass
class SAM1Config:
    """Configuration for SAM1 multi-scale mask generation.

    Default values match the original LangSplatV2 ``preprocess.py`` exactly
    (SAM ViT-H with ``crop_n_layers=1``, ``crop_n_points_downscale_factor=1``).
    """

    checkpoint: Literal["vit_h", "vit_l", "vit_b"] = "vit_h"
    """SAM1 model variant.  The original LangSplatV2 uses ViT-H."""

    points_per_side: int = 32
    """Grid density for point prompts."""

    points_per_batch: int = 256
    """Points processed simultaneously by SAM1."""

    pred_iou_thresh: float = 0.7
    """Predicted IoU threshold for mask filtering."""

    stability_score_thresh: float = 0.85
    """Stability score threshold for mask filtering."""

    crop_n_layers: int = 1
    """Number of crop layers (1 = also run on crops, matching original)."""

    crop_n_points_downscale_factor: int = 1
    """Point grid downscale factor per crop layer.  The original LangSplatV2
    uses 1 (no downscaling on sub-crops)."""

    min_mask_region_area: int = 100
    """Minimum mask region area for post-processing."""

    box_nms_thresh: float = 0.7
    """Box NMS IoU threshold within each crop."""

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

    # SAM configuration
    sam_model: Literal["sam1", "sam2"] = "sam2"
    """Which SAM model to use for mask generation.  ``"sam1"`` uses the
    original Segment Anything Model (ViT-H by default) matching the original
    LangSplatV2 pipeline.  ``"sam2"`` uses SAM2 (Hiera-Large by default)."""

    sam1: SAM1Config = field(default_factory=SAM1Config)
    """Configuration for SAM1 mask generation (used when ``sam_model="sam1"``)."""

    sam2: SAM2Config = field(default_factory=SAM2Config)
    """Configuration for SAM2 mask generation (used when ``sam_model="sam2"``)."""

    compute_sam_masks: bool = True
    """Whether to compute SAM segmentation masks."""

    # CLIP configuration
    clip: OpenCLIPConfig = field(default_factory=OpenCLIPConfig)
    """Configuration for CLIP feature encoding."""

    compute_clip_features: bool = True
    """Whether to compute CLIP features for masked regions."""

    # Import original features (bypasses SAM2 + CLIP)
    original_features_dir: Path | None = None
    """Path to the original LangSplatV2 ``language_features/`` directory.

    When set, imports pre-computed ``_f.npy`` / ``_s.npy`` files from the
    original LangSplatV2 ``preprocess.py`` instead of running SAM2 mask
    generation and CLIP feature encoding.  Useful for A/B testing against
    the original pipeline."""

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
        5a. Multi-scale SAM1/SAM2 mask generation + CLIP feature encoding, OR
        5b. Import original LangSplatV2 features (when ``original_features_dir``
            is set)
        6. Scene cropping (optional)

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
        ]

        if self.original_features_dir is not None:
            # Import pre-computed features from original LangSplatV2
            transforms.append(
                ImportOriginalLangSplatV2Features(
                    original_features_dir=self.original_features_dir,
                    clip_n_dims=self.clip.clip_n_dims,
                )
            )
        else:
            # Standard pipeline: SAM masks then CLIP features
            if self.compute_sam_masks:
                if self.sam_model == "sam1":
                    transforms.append(self.build_sam1_transform())
                else:
                    transforms.append(self.build_sam2_transform())
            else:
                transforms.append(Identity())
            transforms.append(
                ComputeCLIPFeatures(
                    clip_model_type=self.clip.clip_model_type,
                    clip_model_pretrained=self.clip.clip_model_pretrained,
                    clip_n_dims=self.clip.clip_n_dims,
                    device=self.device,
                )
                if self.compute_clip_features
                else Identity()
            )

        # Optional scene cropping
        if self.crop_bbox is not None:
            transforms.append(CropScene(self.crop_bbox))
        if self.crop_to_points:
            transforms.append(CropSceneToPoints(margin=0.0))

        return Compose(*transforms)

    def build_sam1_transform(self):
        """
        Build only the SAM1 mask generation transform.

        Returns:
            ComputeMultiScaleSAM1Masks transform.
        """
        return ComputeMultiScaleSAM1Masks(
            checkpoint=self.sam1.checkpoint,
            points_per_side=self.sam1.points_per_side,
            points_per_batch=self.sam1.points_per_batch,
            pred_iou_thresh=self.sam1.pred_iou_thresh,
            stability_score_thresh=self.sam1.stability_score_thresh,
            crop_n_layers=self.sam1.crop_n_layers,
            crop_n_points_downscale_factor=self.sam1.crop_n_points_downscale_factor,
            min_mask_region_area=self.sam1.min_mask_region_area,
            box_nms_thresh=self.sam1.box_nms_thresh,
            nms_iou_thr=self.sam1.nms_iou_thr,
            nms_score_thr=self.sam1.nms_score_thr,
            nms_inner_thr=self.sam1.nms_inner_thr,
            device=self.device,
        )

    def build_sam2_transform(self):
        """
        Build only the SAM2 mask generation transform.

        Returns:
            ComputeMultiScaleSAM2Masks transform.
        """
        return ComputeMultiScaleSAM2Masks(
            checkpoint=self.sam2.checkpoint,
            points_per_side=self.sam2.points_per_side,
            points_per_batch=self.sam2.points_per_batch,
            pred_iou_thresh=self.sam2.pred_iou_thresh,
            stability_score_thresh=self.sam2.stability_score_thresh,
            crop_n_layers=self.sam2.crop_n_layers,
            crop_n_points_downscale_factor=self.sam2.crop_n_points_downscale_factor,
            min_mask_region_area=self.sam2.min_mask_region_area,
            box_nms_thresh=self.sam2.box_nms_thresh,
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
    """Number of non-zero sparse coefficients per VQ layer.

    The original LangSplatV2 uses topk=4 for both training and evaluation."""


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

    init_codebooks_all_levels: bool = True
    """If True, initialize codebooks using features from ALL scale levels
    (matching original LangSplatV2).  If False, use only the target
    ``feature_level``."""

    model: LangSplatV2ModelConfig = field(default_factory=LangSplatV2ModelConfig)
    """Model architecture configuration."""

    log_test_images: bool = False
    """Whether to log visualization images (PCA features, error heatmaps)
    during training steps.  Eval images are always logged when the writer
    supports image output, regardless of this flag."""

    eval_at_percent: list[int] = field(default_factory=lambda: [5, 10, 20, 30, 40, 50, 75, 100])
    """Percentages of total epochs at which to run evaluation."""

    save_at_percent: list[int] = field(default_factory=lambda: [50, 100])
    """Percentages of total epochs at which to save checkpoints."""
