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
       This is useful for scenes with multiple parts belonging to the same object.

Usage:
    # Default: use scribbles
    python eval_nvos.py --nvos-root /ai/segmentation_datasets/nvos --results-root ./nvos_results

    # Use SAGA v2 style hardcoded input points (supports multiple points per scene)
    python eval_nvos.py --nvos-root /path --results-root ./results --input-mode points

    # Use custom input points file
    python eval_nvos.py --input-mode points --input-points-file /path/to/points.json

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


def get_scribble_click_point(scribbles_dir: pathlib.Path, scene_name: str) -> Tuple[int, int]:
    """
    Get a click point from the positive scribble image.

    The scribble images are 4x downsampled, so we need to scale the coordinates.
    We find the centroid of the positive scribble region.

    Args:
        scribbles_dir: Path to the scribbles directory (e.g., .../scribbles/fern/)
        scene_name: Name of the scene directory (e.g., 'fern', 'horns_center')

    Returns:
        Tuple of (x, y) coordinates at original image resolution
    """
    # Scene name for scribble files is the base scene name (without _center, _left, etc.)
    # e.g., horns_center -> horns, fern -> fern
    base_scene = (
        scene_name.split("_")[0]
        if "_" in scene_name and scene_name not in ["horns_center", "horns_left"]
        else scene_name
    )
    if base_scene in ["horns_center", "horns_left"]:
        base_scene = "horns"

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
    result = [(int(p[0]), int(p[1])) for p in points]
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
    # Can come from scribbles (single point) or input points file (multiple points)
    click_points: list[Tuple[int, int]] = []
    try:
        if config.input_mode == "scribbles":
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

    scale_range = np.arange(scale_increment, max_scale.item() + scale_increment, scale_increment)

    logger.info(f"Mining {len(scale_range)} scales from {scale_increment:.4f} to {max_scale.item():.4f}")

    first_iteration = True
    for curr_scale in tqdm.tqdm(scale_range, desc=f"Mining scales for {scene_name}", leave=False):
        with torch.no_grad():
            # Get features from REFERENCE view (to extract query features at click points)
            ref_feats, _ = model.get_mask_output(ref_model_input, scale=float(curr_scale))
            ref_feats = ref_feats[0]  # Remove batch dimension [H, W, F]

            if first_iteration:
                logger.info(f"Reference features shape: {ref_feats.shape} (expected: {ref_img_h}x{ref_img_w})")
                logger.info(f"Reference features stats: min={ref_feats.min():.4f}, max={ref_feats.max():.4f}")
                # Check feature norms to see if already normalized
                ref_norms = ref_feats.norm(dim=-1)
                logger.info(
                    f"Reference feature norms: min={ref_norms.min():.4f}, max={ref_norms.max():.4f}, mean={ref_norms.mean():.4f}"
                )
                # Check if all features are the same (would explain affinity=1 everywhere)
                ref_feats_flat = ref_feats.reshape(-1, ref_feats.shape[-1])
                feat_var = ref_feats_flat.var(dim=0).mean().item()
                logger.info(f"Reference feature variance across pixels: {feat_var:.6f}")

            # Features from get_mask_output are already normalized in the model
            # Get query features at ALL click points on reference image
            click_feats = []
            for i, (cx, cy) in enumerate(click_points_scaled):
                click_feat = ref_feats[cy, cx, :]
                click_feats.append(click_feat)
                if first_iteration:
                    logger.info(f"Click feature {i} at ({cx}, {cy}): norm={click_feat.norm().item():.4f}")

            if first_iteration and len(click_feats) > 0:
                logger.info(f"Total click features: {len(click_feats)}")

            # Get features from TEST view (novel view where GT mask is)
            test_feats, _ = model.get_mask_output(test_model_input, scale=float(curr_scale))
            test_feats = test_feats[0]  # Remove batch dimension [H, W, F]

            if first_iteration:
                logger.info(f"Test features shape: {test_feats.shape} (expected: {test_img_h}x{test_img_w})")
                logger.info(f"Test features stats: min={test_feats.min():.4f}, max={test_feats.max():.4f}")
                test_norms = test_feats.norm(dim=-1)
                logger.info(
                    f"Test feature norms: min={test_norms.min():.4f}, max={test_norms.max():.4f}, mean={test_norms.mean():.4f}"
                )
                # Check if all features are the same
                test_feats_flat = test_feats.reshape(-1, test_feats.shape[-1])
                feat_var = test_feats_flat.var(dim=0).mean().item()
                logger.info(f"Test feature variance across pixels: {feat_var:.6f}")
                # Sample a few feature values from different pixels
                logger.info(f"Test feat at (0,0) first 5: {test_feats[0, 0, :5].tolist()}")
                mid_h, mid_w = test_feats.shape[0] // 2, test_feats.shape[1] // 2
                logger.info(f"Test feat at ({mid_h},{mid_w}) first 5: {test_feats[mid_h, mid_w, :5].tolist()}")
                first_iteration = False

        # Calculate affinity (dot product) between each click feature and all test view pixels
        # Features are already normalized in the model, so this is cosine similarity
        # For multiple click points, take the MAX affinity (union of all matching regions)
        affinities = []
        for click_feat in click_feats:
            aff = torch.einsum("hwf,f->hw", test_feats, click_feat)
            affinities.append(aff)

        # Union: take max across all click features
        if len(affinities) == 1:
            affinity = affinities[0]
        else:
            affinity = torch.stack(affinities, dim=0).max(dim=0)[0]

        # Threshold to get mask
        mask = (affinity >= config.affinity_threshold).cpu().numpy().astype(np.uint8)

        # Resize mask to match GT dimensions if needed
        if mask.shape != gt_mask.shape:
            mask = cv2.resize(mask, (gt_width, gt_height), interpolation=cv2.INTER_NEAREST)

        # Calculate IoU
        iou = calculate_iou(mask, gt_mask)

        if iou > best_iou:
            best_iou = iou
            best_acc = calculate_accuracy(mask, gt_mask)
            best_scale = curr_scale
            best_mask = mask

    logger.info(f"Scene {scene_name}: Best IoU = {best_iou:.4f}, Acc = {best_acc:.4f} at scale {best_scale:.4f}")

    # Save best affinity map for debugging
    with torch.no_grad():
        ref_feats, _ = model.get_mask_output(ref_model_input, scale=float(best_scale))
        ref_feats = ref_feats[0]
        test_feats, _ = model.get_mask_output(test_model_input, scale=float(best_scale))
        test_feats = test_feats[0]

        # Compute union affinity from all click points
        affinities = []
        for (cx, cy) in click_points_scaled:
            click_feat = ref_feats[cy, cx, :]
            aff = torch.einsum("hwf,f->hw", test_feats, click_feat)
            affinities.append(aff)

        if len(affinities) == 1:
            best_affinity = affinities[0].cpu().numpy()
        else:
            best_affinity = torch.stack(affinities, dim=0).max(dim=0)[0].cpu().numpy()

    logger.info(
        f"Best affinity stats: min={best_affinity.min():.4f}, max={best_affinity.max():.4f}, mean={best_affinity.mean():.4f}"
    )

    # Save visualization
    output_path = config.output_dir / f"{scene_name}_eval.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the reference image for visualization (where click point is)
    ref_img = np.zeros((ref_orig_height, ref_orig_width, 3), dtype=np.uint8)
    if ref_img_path is not None:
        ref_img_bgr = cv2.imread(str(ref_img_path))
        if ref_img_bgr is not None:
            ref_img = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)

    # Load the test image for visualization (where GT mask is)
    # The test image is in the scene's images directory, not the masks directory
    scene_images_dir = scene_path / "images"
    test_img_path = None
    for ext in [".JPG", ".jpg", ".png", ".PNG"]:
        candidate = scene_images_dir / f"{gt_image_id}{ext}"
        if candidate.exists():
            test_img_path = candidate
            break

    if test_img_path is not None:
        test_img_bgr = cv2.imread(str(test_img_path))
        if test_img_bgr is not None:
            test_img = cv2.cvtColor(test_img_bgr, cv2.COLOR_BGR2RGB)
            logger.debug(f"Loaded test image from {test_img_path}: {test_img.shape}")
        else:
            logger.warning(f"Failed to read test image: {test_img_path}")
            test_img = np.zeros((gt_height, gt_width, 3), dtype=np.uint8)
    else:
        logger.warning(f"Test image not found in {scene_images_dir}")
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

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    # Row 1: Reference and Test images
    # Reference image with click points (original resolution)
    axes[0, 0].imshow(ref_img)
    # Plot all click points with different colors
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(click_points), 1)))
    for i, (cx, cy) in enumerate(click_points):
        axes[0, 0].scatter([cx], [cy], c=[colors[i]], s=150, marker="*", edgecolors="white", linewidths=1)
        axes[0, 0].annotate(f"{i}", (cx, cy), xytext=(5, 5), textcoords="offset points", color="white", fontsize=8)
    num_pts = len(click_points)
    pts_str = f"{num_pts} pts" if num_pts > 1 else f"({click_points[0][0]}, {click_points[0][1]})"
    axes[0, 0].set_title(
        f"Reference Image ({ref_image_id})\n{pts_str} @ {ref_orig_width}x{ref_orig_height}"
    )
    axes[0, 0].axis("off")

    # Rendered reference at model resolution with scaled click points
    axes[0, 1].imshow(ref_render_np)
    for i, (cx, cy) in enumerate(click_points_scaled):
        axes[0, 1].scatter([cx], [cy], c=[colors[i]], s=150, marker="*", edgecolors="white", linewidths=1)
        axes[0, 1].annotate(f"{i}", (cx, cy), xytext=(5, 5), textcoords="offset points", color="white", fontsize=8)
    axes[0, 1].set_title(
        f"Rendered Ref (model res)\n{num_pts} pts @ {ref_img_w}x{ref_img_h}"
    )
    axes[0, 1].axis("off")

    # Test image (novel view)
    axes[0, 2].imshow(test_img)
    axes[0, 2].set_title(f"Test Image ({gt_image_id})")
    axes[0, 2].axis("off")

    # Ground truth mask
    axes[0, 3].imshow(gt_mask, cmap="gray")
    axes[0, 3].set_title(f"Ground Truth Mask\n({gt_width}x{gt_height})")
    axes[0, 3].axis("off")

    # Predicted mask
    if best_mask is not None:
        axes[0, 4].imshow(best_mask, cmap="gray")
        axes[0, 4].set_title(f"Predicted Mask\n(scale={best_scale:.4f})")
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
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved visualization to {output_path}")

    return {
        "scene": scene_name,
        "best_iou": best_iou,
        "best_acc": best_acc,
        "best_scale": best_scale,
        "gt_mask_path": str(gt_mask_path),
        "click_points": click_points,
        "input_mode": config.input_mode,
        "num_click_points": len(click_points),
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
    )

    logger.info(f"Input mode: {config.input_mode}")
    if config.input_mode == "points":
        logger.info(f"Input points file: {config.input_points_file}")

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

        logger.info(f"{'Scene':<20} {'IoU':>10} {'Acc %':>10} {'Scale':>10}")
        logger.info("-" * 52)
        for r in results:
            logger.info(f"{r['scene']:<20} {r['best_iou']:>10.4f} {r['best_acc']*100:>10.2f} {r['best_scale']:>10.4f}")
        logger.info("-" * 52)
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
