# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Evaluation script for NVOS dataset segmentation performance.

This script evaluates trained GARfVDB segmentation models on the NVOS dataset by:
1. Loading trained segmentation checkpoints
2. Finding the reference image (scribble) and test image (mask) pairs
3. Mining masks at different scales (similar to GARField completeness evaluation)
4. Computing mIoU between predicted and ground truth masks

The NVOS dataset structure:
    /nvos_root/
    ├── labels/llff/
    │   ├── masks/<scene_name>/<img_name>.JPG, <img_name>_mask.png
    │   ├── reference_image/<scene_name>/<ref_img_name>.JPG
    │   └── scribbles/<scene_name>/pos_0_<scene>.png, neg_0_<scene>.png
    └── scenes/<scene>_undistort/  (colmap data)

Input Modes:
    The script supports two input modes for specifying query points:

    1. "scribbles" (default): Uses NVOS scribble annotations on the reference image
       to get a single click point (centroid of positive scribble region).

    2. "points": Uses hardcoded input points from a JSON file (SAGA v2 style).
       Multiple points can be specified per scene, and the evaluation computes
       the UNION of all matching regions (max affinity across all points).
       This is useful for scenes where the SAGA ground truth mask represents multiple
       objects because SAGA measures affinity of the query point to all potential
       objects in the scene, GARfVDB produces instance masks for each object at a
       given scale, so when evaluating GARfVDB against these masks, we need to compute
       the UNION of all matching regions that would be indicated by each point.

Negative Scribbles:
    Optionally, negative scribbles from the NVOS dataset can be used to suppress
    false positives. When enabled with --use-negative-scribbles, three combination
    modes are available (--negative-mode):

    - "dominate" (default, recommended): Keep pixels where positive affinity
      exceeds threshold AND positive > negative * weight. This requires positive
      affinity to "dominate" negative by a configurable factor.

    - "subtract": Original aggressive approach where
      mask = (pos - weight * neg) >= threshold. Can over-suppress foreground.

    - "veto": Keep pixels where positive >= threshold AND negative < weight.
      Uses negative_weight as a veto threshold for negative affinity.

    Works with both "scribbles" and "points" input modes.

SAGA-Style Scribble Processing:
    When --saga-style-scribbles is enabled (with --input-mode scribbles), the
    scribbles are processed using the SAGA paper approach:
    1. Skeletonization: thin scribbles to 1-pixel wide strokes
    2. Random sampling: multiply by random values and threshold
       - Positive: keep ~2% of skeleton (threshold 0.98)
       - Negative: keep ~0.5% of skeleton (threshold 0.995)

    This produces multiple well-distributed points along the scribble strokes
    instead of a single centroid point.

Usage:
    # Default: use scribbles
    python eval_nvos.py --nvos-root /ai/segmentation_datasets/nvos --results-root ./nvos_results

    # Use SAGA v2 style hardcoded input points (supports multiple points per scene)
    python eval_nvos.py --nvos-root /path --results-root ./results --input-mode points

    # Use custom input points file
    python eval_nvos.py --input-mode points --input-points-file /path/to/points.json

    # Use 4x downsampling for evaluation (SAGA-style, reduces memory usage)
    python eval_nvos.py --eval-downsample 4 --input-mode points

    # Use negative scribbles to suppress false positives
    python eval_nvos.py --input-mode points --use-negative-scribbles --negative-weight 1.0

    # Use SAGA-style multi-point scribbles with negative scribbles
    python eval_nvos.py --input-mode scribbles --saga-style-scribbles --use-negative-scribbles

    # Evaluate specific scenes
    python eval_nvos.py --nvos-root /ai/segmentation_datasets/nvos --results-root ./nvos_results --scenes fern flower
"""
import argparse
import json
import logging
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Literal, Tuple

import cv2
import fvdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from fvdb_reality_capture.sfm_scene import SfmScene

# Add parent directory to path to import garfvdb modules
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from garfvdb.training.dataset import GARfVDBInput
from garfvdb.training.segmentation import GaussianSplatScaleConditionedSegmentation
from garfvdb.util import load_splats_from_file

matplotlib.use("Agg")  # Use non-interactive backend

# NVOS scene mappings from the README
# Maps: scene_name -> (scene_dir_suffix, mask_dir_name, gt_image_id, reference_image_id)
NVOS_SCENE_INFO = {
    "fern": ("fern_undistort", "fern", "IMG_4027", "IMG_4038"),
    "flower": ("flower_undistort", "flower", "IMG_2962", "IMG_2983"),
    "fortress": ("fortress_undistort", "fortress", "IMG_1800", "IMG_1821"),
    "horns_center": ("horns_undistort", "horns_center", "DJI_20200223_163024_597", "DJI_20200223_163055_437"),
    "horns_left": ("horns_undistort", "horns_left", "DJI_20200223_163024_597", "DJI_20200223_163055_437"),
    "leaves": ("leaves_undistort", "leaves", "IMG_2997", "IMG_3011"),
    "orchids": ("orchids_undistort", "orchids", "IMG_4480", "IMG_4479"),
    "trex": ("trex_undistort", "trex", "DJI_20200223_163619_411", "DJI_20200223_163607_906"),
}


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate IoU (intersection over union) between two binary masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 0.0
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def calculate_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculate pixel accuracy between predicted and ground truth masks.

    Accuracy = (correctly classified pixels) / (total pixels)
    """
    correct = np.sum(pred_mask == gt_mask)
    total = pred_mask.size
    return correct / total if total > 0 else 0.0


def _get_base_scene_name(scene_name: str) -> str:
    """
    Get the base scene name for scribble files.

    e.g., horns_center -> horns, horns_left -> horns, fern -> fern
    """
    base_scene = (
        scene_name.split("_")[0]
        if "_" in scene_name and scene_name not in ["horns_center", "horns_left"]
        else scene_name
    )
    if base_scene in ["horns_center", "horns_left"]:
        base_scene = "horns"
    return base_scene


def get_scribble_click_point(scribbles_dir: pathlib.Path, scene_name: str) -> Tuple[int, int]:
    """
    Get a single click point from the positive scribble image.

    The scribble images are 4x downsampled, so we need to scale the coordinates.
    We pick a point from the positive scribble region (median index for stability).

    Args:
        scribbles_dir: Path to the scribbles directory (e.g., .../scribbles/fern/)
        scene_name: Name of the scene directory (e.g., 'fern', 'horns_center')

    Returns:
        Tuple of (x, y) coordinates at original image resolution
    """
    base_scene = _get_base_scene_name(scene_name)

    pos_scribble_path = scribbles_dir / f"pos_0_{base_scene}.png"
    if not pos_scribble_path.exists():
        # Try without the _0 suffix
        pos_scribble_path = scribbles_dir / f"pos_{base_scene}.png"

    if not pos_scribble_path.exists():
        raise FileNotFoundError(f"Positive scribble not found: {pos_scribble_path}")

    # Read the scribble image
    scribble = cv2.imread(str(pos_scribble_path), cv2.IMREAD_GRAYSCALE)
    if scribble is None:
        raise ValueError(f"Failed to read scribble image: {pos_scribble_path}")

    # Find the positive regions (non-zero pixels)
    positive_pixels = np.where(scribble > 0)

    if len(positive_pixels[0]) == 0:
        raise ValueError(f"No positive pixels found in scribble: {pos_scribble_path}")

    # Pick an actual positive pixel (median index to get a central-ish point)
    # Using median index ensures we pick an actual pixel, not a centroid that
    # could fall outside the scribble region for non-convex shapes
    median_idx = len(positive_pixels[0]) // 2
    y_pixel = int(positive_pixels[0][median_idx])
    x_pixel = int(positive_pixels[1][median_idx])

    # Scale up by 4x (scribbles are at 4x downsampled resolution)
    x_click = x_pixel * 4
    y_click = y_pixel * 4

    return (x_click, y_click)


def get_positive_scribble_points(
    scribbles_dir: pathlib.Path,
    scene_name: str,
    max_points: int = 20,
    logger: logging.Logger | None = None,
) -> list[Tuple[int, int]]:
    """
    Get multiple click points from the positive scribble image using SAGA-style processing.

    Uses skeletonization + random sampling for better point distribution.
    The scribble images are 4x downsampled, so coordinates are scaled up.

    Args:
        scribbles_dir: Path to the scribbles directory (e.g., .../scribbles/fern/)
        scene_name: Name of the scene directory (e.g., 'fern', 'horns_center')
        max_points: Maximum number of positive points to return
        logger: Optional logger instance

    Returns:
        List of (x, y) coordinate tuples at original image resolution.
        Raises FileNotFoundError if no positive scribble is found.
    """
    base_scene = _get_base_scene_name(scene_name)

    pos_scribble_path = scribbles_dir / f"pos_0_{base_scene}.png"
    if not pos_scribble_path.exists():
        # Try without the _0 suffix
        pos_scribble_path = scribbles_dir / f"pos_{base_scene}.png"

    if not pos_scribble_path.exists():
        raise FileNotFoundError(f"Positive scribble not found: {pos_scribble_path}")

    # Process scribble with SAGA-style skeletonization and sampling
    # Use threshold 0.98 for positive (keeps ~2% of skeleton pixels)
    scribble = _load_and_process_scribble(pos_scribble_path, sample_threshold=0.98, logger=logger)
    if scribble is None:
        raise ValueError(f"Failed to process positive scribble: {pos_scribble_path}")

    # Extract points (y, x) -> (x, y)
    positive_pixels = np.where(scribble > 0)
    if len(positive_pixels[0]) == 0:
        raise ValueError(f"No positive pixels after processing: {pos_scribble_path}")

    # Convert to (x, y) format and scale up by 4x
    points = []
    for y_pixel, x_pixel in zip(positive_pixels[0], positive_pixels[1]):
        x_click = int(x_pixel) * 4
        y_click = int(y_pixel) * 4
        points.append((x_click, y_click))

    # Limit to max_points if we got too many
    if len(points) > max_points:
        # Randomly sample to max_points
        indices = np.random.choice(len(points), max_points, replace=False)
        points = [points[i] for i in indices]

    if logger:
        logger.info(f"Loaded {len(points)} positive scribble points (SAGA-style) from {pos_scribble_path}")

    return points


def _load_and_process_scribble(
    scribble_path: pathlib.Path,
    sample_threshold: float,
    logger: logging.Logger | None = None,
) -> np.ndarray | None:
    """
    Load a scribble image and process it using SAGA-style skeletonization and sampling.

    This follows the SAGA paper approach:
    1. Load the scribble image and binarize
    2. Skeletonize to get 1-pixel wide strokes
    3. Multiply by random values and threshold to sample points

    Args:
        scribble_path: Path to the scribble PNG file
        sample_threshold: Threshold for random sampling (higher = fewer points)
            SAGA uses 0.98 for positive, 0.995 for negative
        logger: Optional logger instance

    Returns:
        Binary mask with sampled points, or None if loading failed
    """
    # Read the scribble image (may be grayscale or RGB)
    scribble = cv2.imread(str(scribble_path), cv2.IMREAD_UNCHANGED)
    if scribble is None:
        if logger:
            logger.warning(f"Failed to read scribble image: {scribble_path}")
        return None

    # Convert to binary mask
    if len(scribble.shape) == 3:
        # RGB image - sum channels and binarize (SAGA approach)
        scribble = scribble.astype(np.float32).sum(axis=-1)
        scribble[scribble != 0] = 1
    else:
        # Grayscale - just binarize
        scribble = (scribble > 0).astype(np.float32)

    if scribble.sum() == 0:
        if logger:
            logger.warning(f"No pixels found in scribble: {scribble_path}")
        return None

    # Skeletonize to get 1-pixel wide strokes (SAGA approach)
    try:
        from skimage import morphology

        scribble = morphology.skeletonize(scribble).astype(np.float32)
    except ImportError:
        if logger:
            logger.warning("skimage not available, using raw scribble without skeletonization")

    # Random sampling with threshold (SAGA approach)
    # Multiply by random values and keep only high values
    scribble *= np.random.rand(scribble.shape[0], scribble.shape[1])
    scribble[scribble < sample_threshold] = 0

    return scribble


def get_negative_scribble_points(
    scribbles_dir: pathlib.Path,
    scene_name: str,
    max_points: int = 10,
    logger: logging.Logger | None = None,
) -> list[Tuple[int, int]]:
    """
    Get multiple click points from the negative scribble image.

    Uses SAGA-style processing: skeletonization + random sampling.
    The scribble images are 4x downsampled, so coordinates are scaled up.

    Args:
        scribbles_dir: Path to the scribbles directory (e.g., .../scribbles/fern/)
        scene_name: Name of the scene directory (e.g., 'fern', 'horns_center')
        max_points: Maximum number of negative points to return
        logger: Optional logger instance

    Returns:
        List of (x, y) coordinate tuples at original image resolution.
        Returns empty list if no negative scribble is found.
    """
    base_scene = _get_base_scene_name(scene_name)

    neg_scribble_path = scribbles_dir / f"neg_0_{base_scene}.png"
    if not neg_scribble_path.exists():
        # Try without the _0 suffix
        neg_scribble_path = scribbles_dir / f"neg_{base_scene}.png"

    if not neg_scribble_path.exists():
        if logger:
            logger.warning(f"Negative scribble not found: {neg_scribble_path}")
        return []

    # Process scribble with SAGA-style skeletonization and sampling
    # Use threshold 0.995 for negative (keeps ~0.5% of skeleton pixels)
    scribble = _load_and_process_scribble(neg_scribble_path, sample_threshold=0.995, logger=logger)
    if scribble is None:
        return []

    # Extract points (y, x) -> (x, y)
    negative_pixels = np.where(scribble > 0)
    if len(negative_pixels[0]) == 0:
        if logger:
            logger.warning(f"No negative pixels after processing: {neg_scribble_path}")
        return []

    # Convert to (x, y) format and scale up by 4x
    points = []
    for y_pixel, x_pixel in zip(negative_pixels[0], negative_pixels[1]):
        x_click = int(x_pixel) * 4
        y_click = int(y_pixel) * 4
        points.append((x_click, y_click))

    # Limit to max_points if we got too many
    if len(points) > max_points:
        # Randomly sample to max_points
        indices = np.random.choice(len(points), max_points, replace=False)
        points = [points[i] for i in indices]

    if logger:
        logger.info(f"Loaded {len(points)} negative scribble points (SAGA-style) from {neg_scribble_path}")

    return points


def get_input_points_from_file(
    input_points_file: pathlib.Path,
    scene_name: str,
    logger: logging.Logger,
) -> list[Tuple[int, int]]:
    """
    Get input points from a JSON file (SAGA v2 style).

    Args:
        input_points_file: Path to the JSON file containing input points.
        scene_name: Name of the scene (e.g., 'fern', 'horns_center').
        logger: Logger instance.

    Returns:
        List of (x, y) coordinate tuples at original image resolution.
    """
    if not input_points_file.exists():
        raise FileNotFoundError(f"Input points file not found: {input_points_file}")

    with open(input_points_file, "r") as f:
        all_points = json.load(f)

    # Try exact match first, then try without _undistort suffix
    if scene_name in all_points:
        points = all_points[scene_name]
    elif scene_name.replace("_undistort", "") in all_points:
        points = all_points[scene_name.replace("_undistort", "")]
    else:
        # Try base scene name (e.g., horns_center -> horns)
        base_scene = scene_name.split("_")[0] if "_" in scene_name else scene_name
        if base_scene in all_points:
            points = all_points[base_scene]
        else:
            raise ValueError(
                f"Scene '{scene_name}' not found in input points file. "
                f"Available scenes: {[k for k in all_points.keys() if not k.startswith('_')]}"
            )

    # Convert to list of tuples
    scale = all_points.get("_scale", 1)
    result = [(int(p[0] * scale), int(p[1] * scale)) for p in points]
    logger.info(f"Loaded {len(result)} input points for scene '{scene_name}': {result}")
    return result


def load_segmentation_runner(
    segmentation_path: pathlib.Path,
    reconstruction_path: pathlib.Path,
    device: torch.device,
    logger: logging.Logger,
) -> GaussianSplatScaleConditionedSegmentation:
    """
    Load a trained segmentation runner from checkpoint.

    The runner contains the model, GS model, and the SfmScene with correct transforms.

    Args:
        segmentation_path: Path to the segmentation checkpoint
        reconstruction_path: Path to the Gaussian splat PLY file
        device: Device to load the model on
        logger: Logger instance

    Returns:
        GaussianSplatScaleConditionedSegmentation runner
    """
    # Load the GS model
    gs_model, metadata = load_splats_from_file(reconstruction_path, device)
    logger.info(f"Loaded Gaussian splat with {gs_model.num_gaussians} gaussians")

    # Load the segmentation checkpoint
    checkpoint = torch.load(segmentation_path, map_location=device, weights_only=False)

    runner = GaussianSplatScaleConditionedSegmentation.from_state_dict(
        state_dict=checkpoint,
        gs_model=gs_model,
        gs_model_path=reconstruction_path,
        device=device,
        eval_only=True,
    )

    logger.info(f"Loaded segmentation model with max scale {runner.model.max_grouping_scale:.4f}")
    logger.info(f"Restored SfmScene with {runner.sfm_scene.num_images} images")

    return runner


def get_image_id_from_original_scene(
    scene_path: pathlib.Path,
    image_name: str,
    logger: logging.Logger,
) -> int | None:
    """
    Load the original COLMAP scene to find the image_id for a given image filename.

    Since transformed scenes rename images (e.g., to 'image_0000'), we need to
    look at the original scene to find the mapping from filename to image_id.

    Args:
        scene_path: Path to the COLMAP scene directory
        image_name: Name of the image file (without extension)
        logger: Logger instance

    Returns:
        The image_id if found, None otherwise
    """
    original_scene = SfmScene.from_colmap(scene_path)

    for img_meta in original_scene.images:
        img_filename = pathlib.Path(img_meta.image_path).stem
        if img_filename.lower() == image_name.lower():
            logger.debug(f"Found image_id {img_meta.image_id} for {image_name}")
            return img_meta.image_id

    logger.warning(f"Image {image_name} not found in original scene")
    return None


def find_camera_for_image_by_id(
    sfm_scene: SfmScene,
    image_id: int,
    logger: logging.Logger,
) -> Tuple[torch.Tensor, torch.Tensor, int, int] | None:
    """
    Find camera intrinsics and extrinsics for a specific image by its image_id.

    Args:
        sfm_scene: The SfmScene (already transformed/normalized)
        image_id: The unique image ID (preserved across transforms)
        logger: Logger instance

    Returns:
        Tuple of (camera_to_world, projection_matrix, image_height, image_width) or None if not found
    """
    for img_meta in sfm_scene.images:
        if img_meta.image_id == image_id:
            c2w = torch.from_numpy(img_meta.camera_to_world_matrix).float()
            K = torch.from_numpy(img_meta.camera_metadata.projection_matrix).float()
            h = img_meta.camera_metadata.height
            w = img_meta.camera_metadata.width
            logger.debug(f"Found camera for image_id {image_id}: {w}x{h}")
            return c2w, K, h, w

    logger.warning(f"Image with id {image_id} not found in transformed scene")
    return None


def load_scene_image_by_id(
    sfm_scene: SfmScene,
    image_id: int,
    scene_path: pathlib.Path,
    logger: logging.Logger,
) -> tuple[np.ndarray | None, pathlib.Path | None]:
    """
    Load an image from the transformed scene by image_id.

    This uses the image_path stored in the SfmScene, which remains valid even if
    images were renamed (e.g., image_0000). Falls back to None if not found.
    """
    for img_meta in sfm_scene.images:
        if img_meta.image_id == image_id:
            image_path = pathlib.Path(img_meta.image_path)
            if not image_path.is_absolute():
                image_path = scene_path / image_path
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                logger.warning(f"Failed to read image: {image_path}")
                return None, image_path
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), image_path

    logger.warning(f"Image with id {image_id} not found in scene for visualization")
    return None, None


@dataclass
class EvaluationConfig:
    """Configuration for NVOS evaluation."""

    nvos_root: pathlib.Path = pathlib.Path("/ai/segmentation_datasets/nvos")
    results_root: pathlib.Path = pathlib.Path("nvos_results")
    output_dir: pathlib.Path = pathlib.Path("nvos_eval_results")
    device: str = "cuda"
    num_scale_samples: int = 100  # Number of scales to try during mining
    affinity_threshold: float = 0.90  # Threshold for mask selection
    verbose: bool = False

    # Input mode configuration
    input_mode: Literal["scribbles", "points"] = "scribbles"
    input_points_file: pathlib.Path = field(
        default_factory=lambda: pathlib.Path(__file__).parent / "saga_v2_input_points.json"
    )

    # Rendering resolution
    eval_downsample: int = 1  # Downsample factor for evaluation rendering (e.g., 4 for SAGA-style)

    # Negative scribble configuration
    use_negative_scribbles: bool = False  # Whether to use negative scribbles to suppress false positives
    negative_weight: float = 1.0  # Weight/margin for negative affinity comparison
    max_negative_points: int = 10  # Maximum number of negative points to sample from scribble
    negative_mode: Literal["dominate", "subtract", "veto"] = "dominate"
    # Negative modes:
    # - "dominate": mask where pos >= threshold AND pos > neg * weight (default, most robust)
    # - "subtract": mask where (pos - weight * neg) >= threshold (original, aggressive)
    # - "veto": mask where pos >= threshold AND neg < veto_threshold (neg_weight as veto threshold)

    # SAGA-style scribble processing
    saga_style_scribbles: bool = False  # Use SAGA-style multi-point sampling from scribbles
    max_positive_points: int = 20  # Maximum positive points when using SAGA-style scribbles


def run_nvos_evaluation(
    scene_name: str,
    config: EvaluationConfig,
    logger: logging.Logger,
) -> dict | None:
    """
    Run evaluation on a single NVOS scene.

    The NVOS benchmark evaluates novel view segmentation:
    1. Click point is on the reference image (from scribbles)
    2. Ground truth mask is on a different novel view (test image)
    3. Features are computed from the reference view to get the query feature
    4. Features are computed from the novel view and compared to the query

    Args:
        scene_name: Name of the scene (e.g., 'fern', 'flower')
        config: Evaluation configuration
        logger: Logger instance

    Returns:
        Dictionary with evaluation results or None if evaluation failed
    """
    if scene_name not in NVOS_SCENE_INFO:
        logger.error(f"Unknown scene: {scene_name}. Available: {list(NVOS_SCENE_INFO.keys())}")
        return None

    scene_dir_name, mask_dir_name, gt_image_id, ref_image_id = NVOS_SCENE_INFO[scene_name]

    # Paths
    scenes_dir = config.nvos_root / "scenes"
    labels_dir = config.nvos_root / "labels" / "llff"
    masks_dir = labels_dir / "masks" / mask_dir_name
    scribbles_dir = labels_dir / "scribbles" / mask_dir_name
    ref_images_dir = labels_dir / "reference_image" / mask_dir_name
    scene_path = scenes_dir / scene_dir_name

    reconstruction_path = config.results_root / "reconstructions" / f"{scene_dir_name}.ply"
    segmentation_path = config.results_root / "segmentations" / f"{scene_dir_name}_segmentation.pt"

    # Validate paths
    if not scene_path.exists():
        logger.error(f"Scene path not found: {scene_path}")
        return None
    if not reconstruction_path.exists():
        logger.error(f"Reconstruction not found: {reconstruction_path}")
        return None
    if not segmentation_path.exists():
        logger.error(f"Segmentation checkpoint not found: {segmentation_path}")
        return None

    device = torch.device(config.device)

    # Load segmentation runner (includes model and SfmScene with correct transforms)
    logger.info(f"Loading segmentation model from {segmentation_path}")
    runner = load_segmentation_runner(segmentation_path, reconstruction_path, device, logger)
    model = runner.model
    sfm_scene = runner.sfm_scene
    max_scale = model.max_grouping_scale
    model.eval()

    # Get click points (this is on the reference image)
    # Can come from scribbles (single/multi point) or input points file (multiple points)
    click_points: list[Tuple[int, int]] = []
    try:
        if config.input_mode == "scribbles":
            if config.saga_style_scribbles:
                # SAGA-style: multiple points from skeletonized + sampled scribble
                click_points = get_positive_scribble_points(
                    scribbles_dir, mask_dir_name, config.max_positive_points, logger
                )
                logger.info(f"SAGA-style: {len(click_points)} points from positive scribble")
            else:
                # Default: single point from positive scribble
                click_x, click_y = get_scribble_click_point(scribbles_dir, mask_dir_name)
                click_points = [(click_x, click_y)]
                logger.info(f"Click point from scribble (on reference image): ({click_x}, {click_y})")
        else:
            # Load from input points file (SAGA v2 style)
            click_points = get_input_points_from_file(config.input_points_file, scene_name, logger)
            logger.info(f"Loaded {len(click_points)} input points from file: {click_points}")
    except Exception as e:
        logger.error(f"Failed to get click point(s): {e}")
        return None

    if not click_points:
        logger.error("No click points available for evaluation")
        return None

    # Optionally load negative scribble points (to suppress false positives)
    negative_points: list[Tuple[int, int]] = []
    if config.use_negative_scribbles:
        negative_points = get_negative_scribble_points(scribbles_dir, mask_dir_name, config.max_negative_points, logger)
        if negative_points:
            logger.info(f"Using {len(negative_points)} negative scribble points with weight {config.negative_weight}")
        else:
            logger.warning("Negative scribbles enabled but no negative points found")

    # Load ground truth mask (on test/novel view image)
    gt_mask_path = None
    for ext in [".png", ".PNG"]:
        candidate = masks_dir / f"{gt_image_id}_mask{ext}"
        if candidate.exists():
            gt_mask_path = candidate
            break

    if gt_mask_path is None:
        logger.error(f"Ground truth mask not found in {masks_dir}")
        return None

    gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        logger.error(f"Failed to read ground truth mask: {gt_mask_path}")
        return None

    # Normalize GT mask to binary
    gt_mask = (gt_mask > 127).astype(np.uint8)
    gt_height, gt_width = gt_mask.shape
    logger.info(f"Ground truth mask shape: {gt_width}x{gt_height}")

    # Get image IDs from the original COLMAP scene (before transforms renamed files)
    ref_image_id_num = get_image_id_from_original_scene(scene_path, ref_image_id, logger)
    if ref_image_id_num is None:
        logger.error(f"Could not find image_id for reference image {ref_image_id}")
        return None

    test_image_id_num = get_image_id_from_original_scene(scene_path, gt_image_id, logger)
    if test_image_id_num is None:
        logger.error(f"Could not find image_id for test image {gt_image_id}")
        return None

    # Find camera for the REFERENCE image (where the click/scribble is)
    ref_camera_info = find_camera_for_image_by_id(sfm_scene, ref_image_id_num, logger)
    if ref_camera_info is None:
        logger.error(f"Could not find camera for reference image {ref_image_id} (id={ref_image_id_num})")
        return None

    ref_c2w, ref_K, ref_img_h, ref_img_w = ref_camera_info
    logger.info(f"Reference image camera: {ref_img_w}x{ref_img_h}")
    logger.debug(f"Reference camera c2w position: {ref_c2w[:3, 3].tolist()}")
    logger.debug(f"Reference camera K:\n{ref_K}")

    # Find camera for the TEST image (where the GT mask is)
    test_camera_info = find_camera_for_image_by_id(sfm_scene, test_image_id_num, logger)
    if test_camera_info is None:
        logger.error(f"Could not find camera for test image {gt_image_id} (id={test_image_id_num})")
        return None

    test_c2w, test_K, test_img_h, test_img_w = test_camera_info
    logger.info(f"Test image camera: {test_img_w}x{test_img_h}")
    logger.debug(f"Test camera c2w position: {test_c2w[:3, 3].tolist()}")
    logger.debug(f"Test camera K:\n{test_K}")

    # Apply evaluation downsampling if requested
    if config.eval_downsample > 1:
        ds = config.eval_downsample
        logger.info(f"Applying {ds}x downsampling for evaluation (SAGA-style)")

        # Downsample image dimensions
        ref_img_w = ref_img_w // ds
        ref_img_h = ref_img_h // ds
        test_img_w = test_img_w // ds
        test_img_h = test_img_h // ds

        # Scale intrinsics (fx, fy, cx, cy all scale with image size)
        ref_K = ref_K.clone()
        ref_K[0, 0] /= ds  # fx
        ref_K[1, 1] /= ds  # fy
        ref_K[0, 2] /= ds  # cx
        ref_K[1, 2] /= ds  # cy

        test_K = test_K.clone()
        test_K[0, 0] /= ds  # fx
        test_K[1, 1] /= ds  # fy
        test_K[0, 2] /= ds  # cx
        test_K[1, 2] /= ds  # cy

        logger.info(f"Downsampled reference: {ref_img_w}x{ref_img_h}, test: {test_img_w}x{test_img_h}")

    # Adjust click point for reference image dimensions
    # Scribbles are at 4x downsampled resolution relative to original images
    # But the SfmScene might have different dimensions due to training downsampling
    # First, find the original reference image dimensions
    ref_orig_width, ref_orig_height = gt_width, gt_height  # Assume same original size as GT

    # Try to read actual reference image to get dimensions
    ref_img_path = None
    for ext in [".JPG", ".jpg", ".png", ".PNG"]:
        candidate = ref_images_dir / f"{ref_image_id}{ext}"
        if candidate.exists():
            ref_img_path = candidate
            break

    if ref_img_path is not None:
        ref_img_bgr = cv2.imread(str(ref_img_path))
        if ref_img_bgr is not None:
            ref_orig_height, ref_orig_width = ref_img_bgr.shape[:2]

    # Scale all click points from original resolution to model input resolution
    scale_x = ref_img_w / ref_orig_width
    scale_y = ref_img_h / ref_orig_height

    click_points_scaled: list[Tuple[int, int]] = []
    for i, (click_x, click_y) in enumerate(click_points):
        click_x_scaled = int(click_x * scale_x)
        click_y_scaled = int(click_y * scale_y)

        # Clamp to valid range
        click_x_scaled = max(0, min(click_x_scaled, ref_img_w - 1))
        click_y_scaled = max(0, min(click_y_scaled, ref_img_h - 1))
        click_points_scaled.append((click_x_scaled, click_y_scaled))

        logger.info(f"Click point {i}: original=({click_x}, {click_y}) -> scaled=({click_x_scaled}, {click_y_scaled})")

    logger.info(f"All click points scaled ({scale_x:.4f}x{scale_y:.4f}): {click_points_scaled}")

    # Scale negative points if using negative scribbles
    negative_points_scaled: list[Tuple[int, int]] = []
    if negative_points:
        for neg_x, neg_y in negative_points:
            neg_x_scaled = int(neg_x * scale_x)
            neg_y_scaled = int(neg_y * scale_y)
            # Clamp to valid range
            neg_x_scaled = max(0, min(neg_x_scaled, ref_img_w - 1))
            neg_y_scaled = max(0, min(neg_y_scaled, ref_img_h - 1))
            negative_points_scaled.append((neg_x_scaled, neg_y_scaled))
        logger.info(f"Negative points scaled: {len(negative_points_scaled)} points")

    # Prepare model inputs for BOTH reference and test views
    ref_c2w = ref_c2w.unsqueeze(0).to(device)
    ref_K = ref_K.unsqueeze(0).to(device)

    test_c2w = test_c2w.unsqueeze(0).to(device)
    test_K = test_K.unsqueeze(0).to(device)

    ref_model_input = GARfVDBInput(
        intrinsics=ref_K,
        projection=ref_K,
        cam_to_world=ref_c2w,
        camera_to_world=ref_c2w,
        image_w=[ref_img_w],
        image_h=[ref_img_h],
    )

    test_model_input = GARfVDBInput(
        intrinsics=test_K,
        projection=test_K,
        cam_to_world=test_c2w,
        camera_to_world=test_c2w,
        image_w=[test_img_w],
        image_h=[test_img_h],
    )

    # Mine masks at different scales to find best IoU
    # Per GARField paper: mine masks at 0.05 increments, threshold at 0.9
    scale_increment = max_scale.item() / config.num_scale_samples

    best_iou = 0.0
    best_acc = 0.0
    best_scale = 0.0
    best_mask = None
    best_per_point_scales: list[float] = []

    scale_range = np.arange(scale_increment, max_scale.item() + scale_increment, scale_increment)
    num_scales = len(scale_range)
    num_points = len(click_points_scaled)

    logger.info(f"Mining {num_scales} scales from {scale_increment:.4f} to {max_scale.item():.4f}")

    # Memory-efficient approach: store masks on CPU as packed bits
    # Only move to GPU in small batches for IoU computation
    # Structure: all_masks[point_idx][scale_idx] = numpy bool array
    all_masks: list[list[np.ndarray]] = [[] for _ in range(num_points)]
    best_affinity_scales: list[int] = [0] * num_points

    for scale_idx, curr_scale in enumerate(
        tqdm.tqdm(scale_range, desc=f"Computing masks for {scene_name}", leave=False)
    ):
        with torch.no_grad():
            # Get features from REFERENCE view (to extract query features at click points)
            ref_feats, _ = model.get_mask_output(ref_model_input, scale=float(curr_scale))
            ref_feats = ref_feats[0]  # Remove batch dimension [H, W, F]

            # Get features from TEST view (novel view where GT mask is)
            test_feats, _ = model.get_mask_output(test_model_input, scale=float(curr_scale))
            test_feats = test_feats[0]  # Remove batch dimension [H, W, F]

            # Pre-compute max negative affinity if using negative scribbles
            max_neg_affinity = None
            if negative_points_scaled:
                neg_affinities = []
                for neg_x, neg_y in negative_points_scaled:
                    neg_feat = ref_feats[neg_y, neg_x, :]
                    neg_aff = torch.einsum("hwf,f->hw", test_feats, neg_feat)
                    neg_affinities.append(neg_aff)
                # Stack and take max across all negative points
                max_neg_affinity = torch.stack(neg_affinities, dim=0).max(dim=0)[0]  # [H, W]

            # Compute mask for each click point independently at this scale
            for point_idx, (cx, cy) in enumerate(click_points_scaled):
                click_feat = ref_feats[cy, cx, :]
                affinity = torch.einsum("hwf,f->hw", test_feats, click_feat)

                # Apply negative affinity suppression if available
                if max_neg_affinity is not None:
                    pos_mask = affinity >= config.affinity_threshold

                    if config.negative_mode == "dominate":
                        # Keep pixels where positive affinity exceeds negative * weight
                        # This requires pos to "dominate" neg by a factor of weight
                        neg_mask = affinity > (max_neg_affinity * config.negative_weight)
                        mask = (pos_mask & neg_mask).cpu().numpy()
                    elif config.negative_mode == "subtract":
                        # Original subtraction approach (aggressive)
                        adjusted_affinity = affinity - config.negative_weight * max_neg_affinity
                        mask = (adjusted_affinity >= config.affinity_threshold).cpu().numpy()
                    elif config.negative_mode == "veto":
                        # Veto mode: reject pixels where negative affinity exceeds threshold
                        neg_mask = max_neg_affinity < config.negative_weight
                        mask = (pos_mask & neg_mask).cpu().numpy()
                    else:
                        mask = pos_mask.cpu().numpy()
                else:
                    mask = (affinity >= config.affinity_threshold).cpu().numpy()

                # Resize mask to match GT dimensions if needed
                if mask.shape != gt_mask.shape:
                    mask = cv2.resize(mask.astype(np.uint8), (gt_width, gt_height), interpolation=cv2.INTER_NEAREST)

                all_masks[point_idx].append(mask.astype(bool))

        # Clear GPU memory after each scale
        del ref_feats, test_feats
        if max_neg_affinity is not None:
            del max_neg_affinity
        torch.cuda.empty_cache()

    logger.info(f"Pre-computed {num_points} x {num_scales} masks on CPU")

    # Helper functions for IoU calculation
    def calc_iou_np(pred: np.ndarray, gt: np.ndarray) -> float:
        intersection = np.sum(pred & gt)
        union = np.sum(pred | gt)
        return float(intersection / union) if union > 0 else 0.0

    def calc_accuracy_np(pred: np.ndarray, gt: np.ndarray) -> float:
        return float(np.sum(pred == gt) / pred.size)

    gt_mask_bool = gt_mask.astype(bool)

    # Strategy 1: Single scale for all points
    logger.info("Strategy 1: Finding best single scale for all points...")
    for scale_idx in range(num_scales):
        # Union all point masks at this scale
        combined_mask = np.zeros_like(gt_mask, dtype=bool)
        for point_idx in range(num_points):
            combined_mask |= all_masks[point_idx][scale_idx]

        iou = calc_iou_np(combined_mask, gt_mask_bool)
        if iou > best_iou:
            best_iou = iou
            best_acc = calc_accuracy_np(combined_mask, gt_mask_bool)
            best_scale = float(scale_range[scale_idx])
            best_mask = combined_mask.astype(np.uint8)
            best_per_point_scales = [best_scale] * num_points
            best_affinity_scales = [scale_idx] * num_points

    logger.info(f"  Single-scale best: IoU = {best_iou:.4f} at scale {best_scale:.4f}")

    # Strategy 2: Per-point optimal scales (greedy)
    if num_points > 1:
        logger.info("Strategy 2: Finding per-point optimal scales...")

        # Find best individual scale for each point
        point_best_scales: list[int] = []
        for point_idx in range(num_points):
            best_point_iou = 0.0
            best_scale_idx = 0
            for scale_idx in range(num_scales):
                iou = calc_iou_np(all_masks[point_idx][scale_idx], gt_mask_bool)
                if iou > best_point_iou:
                    best_point_iou = iou
                    best_scale_idx = scale_idx
            point_best_scales.append(best_scale_idx)
            logger.info(
                f"  Point {point_idx}: best individual scale = {scale_range[best_scale_idx]:.4f} "
                f"(IoU={best_point_iou:.4f})"
            )

        # Compute combined mask with per-point best scales
        def get_combined_mask_np(scale_indices: list[int]) -> np.ndarray:
            combined = np.zeros_like(gt_mask, dtype=bool)
            for p_idx, s_idx in enumerate(scale_indices):
                combined |= all_masks[p_idx][s_idx]
            return combined

        combined_mask = get_combined_mask_np(point_best_scales)
        multi_scale_iou = calc_iou_np(combined_mask, gt_mask_bool)
        multi_scale_acc = calc_accuracy_np(combined_mask, gt_mask_bool)
        logger.info(f"  Per-point scales combined: IoU = {multi_scale_iou:.4f}")

        # Greedy refinement - optimized by pre-computing "other masks" union
        improved = True
        iteration = 0
        while improved and iteration < 10:
            improved = False
            iteration += 1

            for point_idx in range(num_points):
                current_scale_idx = point_best_scales[point_idx]
                best_new_iou = multi_scale_iou
                best_new_scale_idx = current_scale_idx

                # Pre-compute the "other points" combined mask (excluding current point)
                other_combined = np.zeros_like(gt_mask, dtype=bool)
                for p_idx in range(num_points):
                    if p_idx != point_idx:
                        other_combined |= all_masks[p_idx][point_best_scales[p_idx]]

                # Try each scale for this point
                for test_scale_idx in range(num_scales):
                    test_combined = other_combined | all_masks[point_idx][test_scale_idx]
                    test_iou = calc_iou_np(test_combined, gt_mask_bool)
                    if test_iou > best_new_iou + 1e-6:
                        best_new_iou = test_iou
                        best_new_scale_idx = test_scale_idx

                if best_new_scale_idx != current_scale_idx:
                    point_best_scales[point_idx] = best_new_scale_idx
                    multi_scale_iou = best_new_iou
                    improved = True

            if improved:
                combined_mask = get_combined_mask_np(point_best_scales)
                multi_scale_acc = calc_accuracy_np(combined_mask, gt_mask_bool)

        logger.info(f"  After refinement (iter={iteration}): IoU = {multi_scale_iou:.4f}")

        # Use multi-scale result if better than single scale
        if multi_scale_iou > best_iou:
            best_iou = multi_scale_iou
            best_acc = multi_scale_acc
            best_mask = combined_mask.astype(np.uint8)
            best_per_point_scales = [float(scale_range[idx]) for idx in point_best_scales]
            best_affinity_scales = point_best_scales
            best_scale = -1.0  # Indicate multi-scale was used
            logger.info(f"  Using per-point scales: {[f'{s:.4f}' for s in best_per_point_scales]}")

    logger.info(f"Scene {scene_name}: Best IoU = {best_iou:.4f}, Acc = {best_acc:.4f}")
    if best_scale > 0:
        logger.info(f"  Using single scale: {best_scale:.4f}")
    else:
        logger.info(f"  Using per-point scales: {[f'{s:.4f}' for s in best_per_point_scales]}")

    # Recompute best affinities for visualization (only the needed scales)
    logger.info("Computing affinity maps for visualization...")
    best_affinities: list[tuple[int, np.ndarray]] = []
    unique_scales = sorted(set(best_affinity_scales))

    with torch.no_grad():
        for scale_idx in unique_scales:
            curr_scale = scale_range[scale_idx]
            ref_feats, _ = model.get_mask_output(ref_model_input, scale=float(curr_scale))
            ref_feats = ref_feats[0]
            test_feats, _ = model.get_mask_output(test_model_input, scale=float(curr_scale))
            test_feats = test_feats[0]

            for point_idx, (cx, cy) in enumerate(click_points_scaled):
                if best_affinity_scales[point_idx] == scale_idx:
                    click_feat = ref_feats[cy, cx, :]
                    affinity = torch.einsum("hwf,f->hw", test_feats, click_feat)
                    # Resize if needed
                    if affinity.shape[0] != gt_height or affinity.shape[1] != gt_width:
                        affinity = torch.nn.functional.interpolate(
                            affinity.unsqueeze(0).unsqueeze(0),
                            size=(gt_height, gt_width),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze()
                    best_affinities.append((point_idx, affinity.cpu().numpy()))

            del ref_feats, test_feats
            torch.cuda.empty_cache()

    # Sort by point index and extract affinities
    best_affinities.sort(key=lambda x: x[0])
    affinities_list = [aff for _, aff in best_affinities]

    if len(affinities_list) == 1:
        best_affinity = affinities_list[0]
    else:
        best_affinity = np.maximum.reduce(affinities_list)

    logger.info(
        f"Best affinity stats: min={best_affinity.min():.4f}, max={best_affinity.max():.4f}, mean={best_affinity.mean():.4f}"
    )

    # Save visualization
    output_path = config.output_dir / f"{scene_name}_eval.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the reference image for visualization (where click point is)
    # Prefer the full-resolution labels reference image, then fall back to scene images.
    ref_img = None
    ref_img_loaded_from = None
    if ref_img_path is not None:
        logger.info(f"Loading reference image from labels directory: {ref_img_path}")
        ref_img_bgr = cv2.imread(str(ref_img_path))
        if ref_img_bgr is not None:
            ref_img = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)
            ref_img_loaded_from = ref_img_path
            logger.info(f"Loaded reference image from labels: {ref_img_path} shape={ref_img.shape}")
        else:
            logger.warning(f"Failed to read reference image from labels: {ref_img_path}")

    if ref_img is None:
        ref_img, ref_img_loaded_from = load_scene_image_by_id(sfm_scene, ref_image_id_num, scene_path, logger)
        if ref_img is not None and ref_img_loaded_from is not None:
            logger.info(f"Loaded reference image from scene: {ref_img_loaded_from} shape={ref_img.shape}")

    # Final fallback to zeros
    if ref_img is None:
        logger.warning(
            f"Reference image not found for {ref_image_id}, using placeholder size=({ref_orig_width}x{ref_orig_height})"
        )
        ref_img = np.zeros((ref_orig_height, ref_orig_width, 3), dtype=np.uint8)
    else:
        # Update dimensions from actual loaded image
        ref_orig_height, ref_orig_width = ref_img.shape[:2]
        logger.info(f"Reference image final size: {ref_orig_width}x{ref_orig_height}")

    # Load the test image for visualization (where GT mask is)
    test_img, test_img_path = load_scene_image_by_id(sfm_scene, test_image_id_num, scene_path, logger)
    if test_img is not None and test_img_path is not None:
        logger.debug(f"Loaded test image from {test_img_path}: {test_img.shape}")
    else:
        logger.warning("Test image not found in scene, using placeholder")
        test_img = np.zeros((gt_height, gt_width, 3), dtype=np.uint8)

    # Also render reference view at model resolution to show where click is sampled
    with torch.no_grad():
        ref_render, ref_alpha = model.gs_model.render_images(
            image_width=ref_img_w,
            image_height=ref_img_h,
            world_to_camera_matrices=torch.linalg.inv(ref_c2w).contiguous(),
            projection_matrices=ref_K,
            near=0.01,
            far=1e10,
            sh_degree_to_use=0,
        )
        ref_render_np = ref_render[0].cpu().numpy()
        ref_render_np = np.clip(ref_render_np, 0, 1)

    # Compute accurate 3D reprojected click points for test view visualization
    # 1. Find which Gaussian is visible at each click point in reference view
    # 2. Look up that Gaussian's 3D world position
    # 3. Project to test view using test camera parameters
    click_points_test_3d: list[Tuple[int, int] | None] = []
    with torch.no_grad():
        # Create pixel coordinates for sparse rendering (only at click points)
        # JaggedTensor expects one list per camera, so we create a single tensor with all pixels
        # IMPORTANT: sparse_render expects (row, col) = (y, x) format, but click_points_scaled is (x, y)
        all_pixels_yx = torch.tensor(
            [(cy, cx) for (cx, cy) in click_points_scaled], device=device, dtype=torch.int64
        )  # [N, 2] in (y, x) format
        pixel_coords = fvdb.JaggedTensor([all_pixels_yx])  # Single camera, N pixels

        # Get the top contributing Gaussian ID at each click point
        ids, _ = model.gs_model.sparse_render_contributing_gaussian_ids(
            pixels_to_render=pixel_coords,
            top_k_contributors=1,
            world_to_camera_matrices=torch.linalg.inv(ref_c2w).contiguous(),
            projection_matrices=ref_K,
            image_width=ref_img_w,
            image_height=ref_img_h,
            near=0.01,
            far=1e10,
        )

        # Get world coordinates for test camera projection
        test_w2c = torch.linalg.inv(test_c2w).squeeze(0)  # World to camera for test view [4, 4]
        test_K_mat = test_K.squeeze(0)  # Remove batch dim [3, 3]

        ids_data = ids.jdata

        for i, (cx, cy) in enumerate(click_points_scaled):
            # Get the Gaussian ID for this click point
            # Handle both 1D [N*top_k] and 2D [N, top_k] shapes
            if ids_data.ndim == 2:
                g_id = int(ids_data[i, 0].item()) if ids_data.shape[1] > 0 else -1
            else:
                # 1D tensor: index directly (top_k=1, so each element is one pixel's result)
                g_id = int(ids_data[i].item()) if i < len(ids_data) else -1

            if g_id >= 0 and g_id < model.gs_model.means.shape[0]:
                # Get the 3D world position of this Gaussian
                world_pt = model.gs_model.means[g_id]  # [3]

                # Transform to test camera space: p_cam = w2c @ [p_world, 1]
                world_pt_h = torch.cat([world_pt, torch.ones(1, device=device)])  # [4]
                cam_pt = test_w2c @ world_pt_h  # [4]

                # Project to 2D: p_2d = K @ p_cam[:3] / p_cam[2]
                if cam_pt[2] > 0:  # Only if point is in front of camera
                    p_2d = test_K_mat @ cam_pt[:3]
                    u = (p_2d[0] / p_2d[2]).item()
                    v = (p_2d[1] / p_2d[2]).item()

                    # Scale from model resolution to test image display resolution
                    test_orig_height_tmp, test_orig_width_tmp = test_img.shape[:2]
                    u_display = int(u * test_orig_width_tmp / test_img_w)
                    v_display = int(v * test_orig_height_tmp / test_img_h)

                    click_points_test_3d.append((u_display, v_display))
                    logger.debug(f"Click point {i}: 3D pos {world_pt.tolist()} -> test view ({u_display}, {v_display})")
                else:
                    click_points_test_3d.append(None)
                    logger.debug(f"Click point {i}: behind test camera")
            else:
                click_points_test_3d.append(None)
                logger.debug(f"Click point {i}: no valid Gaussian ID (g_id={g_id})")

    # Create figure with landscape subplots to match image aspect ratios (~4:3)
    # With 5 columns at 6" wide and 2 rows at 4" tall, each subplot is ~6:4 = 3:2 (landscape)
    fig = plt.figure(figsize=(30, 9))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1], wspace=0.02, hspace=0.12)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(2)])

    # Define a common display size for consistent visualization (use GT mask dimensions)
    display_width, display_height = gt_width, gt_height

    # Row 1: Reference and Test images
    cmap = plt.colormaps.get_cmap("Set1")
    num_pts = len(click_points)
    colors = cmap(np.linspace(0, 1, max(num_pts, 1)))

    # Reference image with click points - resize to common display size
    ref_scale_x = display_width / ref_orig_width
    ref_scale_y = display_height / ref_orig_height
    logger.info(
        f"DEBUG: ref_img shape={ref_img.shape}, ref_orig=({ref_orig_width}x{ref_orig_height}), display=({display_width}x{display_height})"
    )
    ref_img_display = cv2.resize(ref_img, (display_width, display_height))
    logger.info(f"DEBUG: ref_img_display shape={ref_img_display.shape}")
    axes[0, 0].imshow(
        ref_img_display,
        extent=[0, display_width, display_height, 0],
        aspect="auto",
    )
    axes[0, 0].set_xlim(0, display_width)
    axes[0, 0].set_ylim(display_height, 0)
    for i, (cx, cy) in enumerate(click_points):
        cx_disp = int(cx * ref_scale_x)
        cy_disp = int(cy * ref_scale_y)
        axes[0, 0].scatter([cx_disp], [cy_disp], c=[colors[i]], s=25, marker="*", edgecolors="white", linewidths=1)
        axes[0, 0].annotate(
            f"{i}", (cx_disp, cy_disp), xytext=(5, 5), textcoords="offset points", color="white", fontsize=8
        )
    # Draw negative points as red X markers
    if negative_points:
        for j, (nx, ny) in enumerate(negative_points):
            nx_disp = int(nx * ref_scale_x)
            ny_disp = int(ny * ref_scale_y)
            axes[0, 0].scatter([nx_disp], [ny_disp], c="red", s=20, marker="x", linewidths=1.5)
    pts_str = f"{num_pts} pts" if num_pts > 1 else f"({click_points[0][0]}, {click_points[0][1]})"
    neg_str = f", {len(negative_points)} neg" if negative_points else ""
    axes[0, 0].set_title(f"Reference Image ({ref_image_id})\n{pts_str}{neg_str} @ {ref_orig_width}x{ref_orig_height}")
    axes[0, 0].axis("off")

    # Rendered reference at model resolution - resize to common display size
    render_scale_x = display_width / ref_img_w
    render_scale_y = display_height / ref_img_h
    ref_render_display = cv2.resize(ref_render_np, (display_width, display_height))
    axes[0, 1].imshow(ref_render_display)
    for i, (cx, cy) in enumerate(click_points_scaled):
        cx_disp = int(cx * render_scale_x)
        cy_disp = int(cy * render_scale_y)
        axes[0, 1].scatter([cx_disp], [cy_disp], c=[colors[i]], s=25, marker="*", edgecolors="white", linewidths=1)
        axes[0, 1].annotate(
            f"{i}", (cx_disp, cy_disp), xytext=(5, 5), textcoords="offset points", color="white", fontsize=8
        )
    # Draw negative points as red X markers
    if negative_points_scaled:
        for j, (nx, ny) in enumerate(negative_points_scaled):
            nx_disp = int(nx * render_scale_x)
            ny_disp = int(ny * render_scale_y)
            axes[0, 1].scatter([nx_disp], [ny_disp], c="red", s=20, marker="x", linewidths=1.5)
    neg_str = f", {len(negative_points_scaled)} neg" if negative_points_scaled else ""
    axes[0, 1].set_title(f"Rendered Ref (model res)\n{num_pts} pts{neg_str} @ {ref_img_w}x{ref_img_h}")
    axes[0, 1].axis("off")

    # Test image (novel view) - resize to common display size
    test_orig_h, test_orig_w = test_img.shape[:2]
    test_scale_x = display_width / test_orig_w
    test_scale_y = display_height / test_orig_h
    test_img_display = cv2.resize(test_img, (display_width, display_height))
    axes[0, 2].imshow(
        test_img_display,
        extent=[0, display_width, display_height, 0],
        aspect="auto",
    )
    axes[0, 2].set_xlim(0, display_width)
    axes[0, 2].set_ylim(display_height, 0)
    valid_pts_count = 0
    for i, pt in enumerate(click_points_test_3d):
        if pt is not None:
            cx_test, cy_test = pt
            # Scale reprojected points to display resolution
            cx_disp = int(cx_test * test_scale_x)
            cy_disp = int(cy_test * test_scale_y)
            axes[0, 2].scatter([cx_disp], [cy_disp], c=[colors[i]], s=25, marker="*", edgecolors="white", linewidths=1)
            axes[0, 2].annotate(
                f"{i}", (cx_disp, cy_disp), xytext=(5, 5), textcoords="offset points", color="white", fontsize=8
            )
            valid_pts_count += 1
    axes[0, 2].set_title(f"Test Image ({gt_image_id})\n{valid_pts_count}/{num_pts} pts (3D reprojected)")
    axes[0, 2].axis("off")

    # Ground truth mask
    axes[0, 3].imshow(gt_mask, cmap="gray")
    axes[0, 3].set_title(f"Ground Truth Mask\n({gt_width}x{gt_height})")
    axes[0, 3].axis("off")

    # Predicted mask
    if best_mask is not None:
        axes[0, 4].imshow(best_mask, cmap="gray")
        if best_scale > 0:
            axes[0, 4].set_title(f"Predicted Mask\n(single scale={best_scale:.4f})")
        else:
            # Show per-point scales (abbreviated if too many)
            if len(best_per_point_scales) <= 3:
                scales_str = ", ".join([f"{s:.3f}" for s in best_per_point_scales])
            else:
                scales_str = f"{len(best_per_point_scales)} scales"
            axes[0, 4].set_title(f"Predicted Mask\n(multi-scale: {scales_str})")
    else:
        axes[0, 4].set_title("No mask found")
    axes[0, 4].axis("off")

    # Row 2: Debug visualizations
    # Affinity map (before thresholding) at model resolution
    axes[1, 0].imshow(best_affinity, cmap="jet")
    axes[1, 0].set_title(
        f"Affinity (model res {best_affinity.shape[1]}x{best_affinity.shape[0]})\n(min={best_affinity.min():.3f}, max={best_affinity.max():.3f})"
    )
    axes[1, 0].axis("off")

    # Affinity map resized to GT resolution
    affinity_resized = cv2.resize(best_affinity, (gt_width, gt_height), interpolation=cv2.INTER_LINEAR)
    im = axes[1, 1].imshow(affinity_resized, cmap="jet")
    axes[1, 1].set_title(f"Affinity (resized to {gt_width}x{gt_height})")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Affinity map thresholded
    axes[1, 2].imshow(affinity_resized >= config.affinity_threshold, cmap="gray")
    axes[1, 2].set_title(f"Affinity >= {config.affinity_threshold}")
    axes[1, 2].axis("off")

    # GT mask overlaid on test image
    if test_img.shape[:2] != gt_mask.shape:
        test_img_resized = cv2.resize(test_img, (gt_width, gt_height))
    else:
        test_img_resized = test_img
    gt_overlay = test_img_resized.copy().astype(np.float32) / 255.0
    gt_overlay[:, :, 2] = np.clip(gt_overlay[:, :, 2] + gt_mask * 0.5, 0, 1)
    axes[1, 3].imshow(gt_overlay)
    axes[1, 3].set_title("GT Mask on Test Image")
    axes[1, 3].axis("off")

    # Overlay comparison
    if best_mask is not None:
        overlay = test_img_resized.copy().astype(np.float32) / 255.0
        mask_overlay = np.zeros_like(overlay)
        mask_overlay[:, :, 0] = best_mask  # Red channel for prediction
        mask_overlay[:, :, 2] = gt_mask  # Blue channel for GT
        overlay = overlay * 0.5 + mask_overlay * 0.5
        axes[1, 4].imshow(overlay)
        axes[1, 4].contour(gt_mask, colors="blue", linewidths=1, linestyles="solid")
        axes[1, 4].contour(best_mask, colors="red", linewidths=1, linestyles="dashed")
    axes[1, 4].set_title(f"Overlay (Red=Pred, Blue=GT)\n(IoU={best_iou:.4f}, Acc={best_acc:.2%})")
    axes[1, 4].axis("off")

    fig.suptitle(f"NVOS Evaluation: {scene_name}")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved visualization to {output_path}")

    return {
        "scene": scene_name,
        "best_iou": best_iou,
        "best_acc": best_acc,
        "best_scale": best_scale if best_scale > 0 else None,
        "per_point_scales": best_per_point_scales if best_scale <= 0 else None,
        "multi_scale_used": best_scale <= 0,
        "gt_mask_path": str(gt_mask_path),
        "click_points": click_points,
        "input_mode": config.input_mode,
        "num_click_points": len(click_points),
        "num_negative_points": len(negative_points) if negative_points else 0,
        "used_negative_scribbles": config.use_negative_scribbles and len(negative_points) > 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GARfVDB segmentation models on NVOS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nvos-root",
        type=pathlib.Path,
        default=pathlib.Path("/ai/segmentation_datasets/nvos"),
        help="Root directory of the NVOS dataset",
    )
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=pathlib.Path("nvos_results"),
        help="Directory containing trained models (reconstructions/ and segmentations/)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("nvos_eval_results"),
        help="Directory to save evaluation results and visualizations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation (cuda or cuda:N)",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        type=str,
        default=None,
        help="Specific scenes to evaluate. If not specified, evaluates all available scenes.",
    )
    parser.add_argument(
        "--num-scale-samples",
        type=int,
        default=100,
        help="Number of scale values to sample during mask mining",
    )
    parser.add_argument(
        "--affinity-threshold",
        type=float,
        default=0.90,
        help="Affinity threshold for mask selection (GARField uses 0.90)",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["scribbles", "points"],
        default="scribbles",
        help="Input mode for click points: 'scribbles' uses NVOS scribble annotations, "
        "'points' uses hardcoded points from JSON file (SAGA v2 style)",
    )
    parser.add_argument(
        "--input-points-file",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "saga_v2_input_points.json",
        help="JSON file containing input points for each scene (only used with --input-mode points)",
    )
    parser.add_argument(
        "--eval-downsample",
        type=int,
        default=1,
        help="Downsample factor for evaluation rendering. Use 4 for SAGA-style evaluation "
        "(reduces memory usage significantly). Default 1 = no downsampling.",
    )
    parser.add_argument(
        "--use-negative-scribbles",
        action="store_true",
        help="Use negative scribbles from NVOS dataset to suppress false positives. "
        "Works with both 'scribbles' and 'points' input modes.",
    )
    parser.add_argument(
        "--negative-weight",
        type=float,
        default=1.0,
        help="Weight for negative affinity subtraction: final = pos - weight * max_neg. "
        "Higher values more aggressively suppress regions similar to negative scribbles.",
    )
    parser.add_argument(
        "--max-negative-points",
        type=int,
        default=10,
        help="Maximum number of negative points to sample from the negative scribble.",
    )
    parser.add_argument(
        "--negative-mode",
        type=str,
        choices=["dominate", "subtract", "veto"],
        default="dominate",
        help="How to combine positive and negative affinities: "
        "'dominate' (default) keeps pixels where pos > neg * weight; "
        "'subtract' uses pos - weight * neg >= threshold (aggressive); "
        "'veto' rejects pixels where neg >= weight threshold.",
    )
    parser.add_argument(
        "--saga-style-scribbles",
        action="store_true",
        help="Use SAGA-style multi-point sampling from positive scribbles (skeletonization + random sampling). "
        "Only applies when --input-mode is 'scribbles'. Produces multiple positive points instead of one.",
    )
    parser.add_argument(
        "--max-positive-points",
        type=int,
        default=20,
        help="Maximum number of positive points when using --saga-style-scribbles.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("nvos_eval")

    # Build config
    config = EvaluationConfig(
        nvos_root=args.nvos_root,
        results_root=args.results_root,
        output_dir=args.output_dir,
        device=args.device,
        num_scale_samples=args.num_scale_samples,
        affinity_threshold=args.affinity_threshold,
        verbose=args.verbose,
        input_mode=args.input_mode,
        input_points_file=args.input_points_file,
        eval_downsample=args.eval_downsample,
        use_negative_scribbles=args.use_negative_scribbles,
        negative_weight=args.negative_weight,
        max_negative_points=args.max_negative_points,
        negative_mode=args.negative_mode,
        saga_style_scribbles=args.saga_style_scribbles,
        max_positive_points=args.max_positive_points,
    )

    logger.info(f"Input mode: {config.input_mode}")
    if config.input_mode == "scribbles" and config.saga_style_scribbles:
        logger.info(f"SAGA-style scribbles: max {config.max_positive_points} positive points")
    if config.input_mode == "points":
        logger.info(f"Input points file: {config.input_points_file}")
    if config.eval_downsample > 1:
        logger.info(f"Evaluation downsample: {config.eval_downsample}x (SAGA-style)")
    if config.use_negative_scribbles:
        logger.info(
            f"Using negative scribbles: mode={config.negative_mode}, weight={config.negative_weight}, "
            f"max_points={config.max_negative_points}"
        )

    # Determine which scenes to evaluate
    if args.scenes:
        scenes_to_eval = args.scenes
    else:
        # Find all scenes with available checkpoints
        segmentations_dir = config.results_root / "segmentations"
        if segmentations_dir.exists():
            available_checkpoints = list(segmentations_dir.glob("*_segmentation.pt"))
            scenes_to_eval = []
            for ckpt in available_checkpoints:
                # Extract scene dir name from checkpoint filename
                scene_dir_name = ckpt.stem.replace("_segmentation", "")
                # Find matching NVOS scene(s)
                # Note: horns_undistort maps to both horns_center and horns_left
                for scene_name, (dir_name, _, _, _) in NVOS_SCENE_INFO.items():
                    if dir_name == scene_dir_name and scene_name not in scenes_to_eval:
                        scenes_to_eval.append(scene_name)
        else:
            logger.error(f"Segmentations directory not found: {segmentations_dir}")
            return

    logger.info(f"Evaluating scenes: {scenes_to_eval}")

    # Run evaluation on each scene
    results = []
    for scene_name in scenes_to_eval:
        logger.info("=" * 60)
        logger.info(f"Evaluating scene: {scene_name}")
        logger.info("=" * 60)

        result = run_nvos_evaluation(scene_name, config, logger)
        if result is not None:
            results.append(result)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE - Summary:")
    logger.info("=" * 60)

    if results:
        ious = [r["best_iou"] for r in results]
        accs = [r["best_acc"] for r in results]
        mean_iou = np.mean(ious)
        mean_acc = np.mean(accs)

        logger.info(f"{'Scene':<20} {'IoU':>10} {'Acc %':>10} {'Scale(s)':>15}")
        logger.info("-" * 57)
        for r in results:
            if r.get("multi_scale_used"):
                scale_str = f"multi({r['num_click_points']})"
            else:
                scale_str = f"{r['best_scale']:.4f}" if r["best_scale"] else "N/A"
            logger.info(f"{r['scene']:<20} {r['best_iou']:>10.4f} {r['best_acc']*100:>10.2f} {scale_str:>15}")
        logger.info("-" * 57)
        logger.info(f"{'Mean':<20} {mean_iou:>10.4f} {mean_acc*100:>10.2f}")

        # Save results to JSON
        import json

        results_file = config.output_dir / "nvos_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(
                {
                    "results": results,
                    "mean_iou": mean_iou,
                    "mean_acc": mean_acc,
                    "config": {
                        "num_scale_samples": config.num_scale_samples,
                        "affinity_threshold": config.affinity_threshold,
                        "input_mode": config.input_mode,
                        "input_points_file": str(config.input_points_file) if config.input_mode == "points" else None,
                        "eval_downsample": config.eval_downsample,
                        "use_negative_scribbles": config.use_negative_scribbles,
                        "negative_mode": config.negative_mode if config.use_negative_scribbles else None,
                        "negative_weight": config.negative_weight if config.use_negative_scribbles else None,
                        "max_negative_points": config.max_negative_points if config.use_negative_scribbles else None,
                    },
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved to {results_file}")
    else:
        logger.warning("No successful evaluations")


if __name__ == "__main__":
    main()
