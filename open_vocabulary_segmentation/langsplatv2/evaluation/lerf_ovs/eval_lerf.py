# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Evaluation script for LERF dataset open-vocabulary segmentation.

This script evaluates trained LangSplatV2 models on the LERF-OVS dataset by:
1. Loading 3 trained checkpoints (one per SAM scale level)
2. Loading LERF ground-truth annotations (labelme-format JSONs)
3. Rendering CLIP features from each level and computing relevancy maps
4. Computing segmentation mIoU and localization accuracy

The evaluation matches the original LangSplatV2 eval_lerf.py methodology:
- Relevancy is computed using OpenCLIP (ViT-B-16) with pos/neg softmax scoring
- Segmentation uses AvgPool smoothing + normalization + thresholding
- The best level is chosen per-prompt based on max relevancy score
- Localization checks if the peak of the relevancy map is inside any GT bbox

LERF-OVS dataset structure:
    lerf_ovs/
        label/<scene_name>/frame_XXXXX.json, frame_XXXXX.jpg
        <scene_name>/
            images/
            sparse/
            output/<scene_name>/point_cloud/iteration_30000/point_cloud.ply

Usage:
    # Evaluate a single scene
    python eval_lerf.py \\
        --lerf-root /path/to/lerf_ovs \\
        --results-root ./langsplatv2_results \\
        --gs-model-path /path/to/point_cloud.ply \\
        --scenes teatime

    # Evaluate all scenes
    python eval_lerf.py \\
        --lerf-root /path/to/lerf_ovs \\
        --results-root ./langsplatv2_results
"""
import argparse
import glob
import json
import logging
import os
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from fvdb_reality_capture.sfm_scene import SfmScene
from langsplatv2.evaluation.openclip_relevancy import OpenCLIPRelevancy
from langsplatv2.model import LangSplatV2Model
from langsplatv2.training.trainer import LangSplatV2Trainer
from langsplatv2.util import load_splats_from_file

matplotlib.use("Agg")  # Use non-interactive backend


# ---------------------------------------------------------------------------
# Ground truth parsing (matching original eval_gt_lerfdata)
# ---------------------------------------------------------------------------


def polygon_to_mask(img_shape: Tuple[int, int], points_list: list) -> np.ndarray:
    """Convert a polygon to a binary mask.

    Args:
        img_shape: (height, width) of the target image.
        points_list: List of [x, y] polygon vertices.

    Returns:
        Binary mask of shape ``(height, width)`` with dtype uint8.
    """
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def stack_mask(mask_base: np.ndarray, mask_add: np.ndarray) -> np.ndarray:
    """Merge two binary masks (logical OR)."""
    mask = mask_base.copy()
    mask[mask_add != 0] = 1
    return mask


def load_lerf_ground_truth(
    json_folder: pathlib.Path,
    logger: logging.Logger,
) -> Tuple[Dict, Tuple[int, int], list]:
    """Parse LERF-OVS ground truth annotations from labelme-format JSONs.

    Matches the original ``eval_gt_lerfdata`` function exactly.

    Args:
        json_folder: Path to the label folder for a specific scene
            (e.g. ``lerf_ovs/label/teatime``).
        logger: Logger instance.

    Returns:
        Tuple of:
            - gt_ann: ``{str(frame_idx): {label: {"bboxes": ndarray, "mask": ndarray}}}``
            - image_shape: ``(height, width)`` of the ground truth images
            - img_paths: sorted list of GT JPEG image paths
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), "frame_*.json")))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), "frame_*.jpg")))

    if not gt_json_paths:
        raise FileNotFoundError(f"No frame_*.json files found in {json_folder}")

    logger.info(f"Found {len(gt_json_paths)} GT annotations, {len(img_paths)} GT images in {json_folder}")

    gt_ann = {}
    h, w = 0, 0
    for js_path in gt_json_paths:
        img_ann: Dict[str, dict] = defaultdict(dict)
        with open(js_path, "r") as f:
            gt_data = json.load(f)

        h, w = gt_data["info"]["height"], gt_data["info"]["width"]
        # Frame index: frame_00001 -> idx=0  (1-indexed filename to 0-indexed)
        idx = int(gt_data["info"]["name"].split("_")[-1].split(".jpg")[0]) - 1

        for prompt_data in gt_data["objects"]:
            label = prompt_data["category"]
            box = np.asarray(prompt_data["bbox"]).reshape(-1)  # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data["segmentation"])

            if img_ann[label].get("mask", None) is not None:
                # Merge multiple objects with the same label
                mask = stack_mask(img_ann[label]["mask"], mask)
                img_ann[label]["bboxes"] = np.concatenate(
                    [img_ann[label]["bboxes"].reshape(-1, 4), box.reshape(-1, 4)], axis=0
                )
            else:
                img_ann[label]["bboxes"] = box
            img_ann[label]["mask"] = mask

        gt_ann[str(idx)] = dict(img_ann)

    logger.info(f"Parsed GT: {len(gt_ann)} frames, image size {w}x{h}")
    for idx_str, ann in gt_ann.items():
        labels = list(ann.keys())
        logger.debug(f"  Frame {idx_str}: {len(labels)} labels: {labels}")

    return gt_ann, (h, w), img_paths


# ---------------------------------------------------------------------------
# Segmentation and localization (matching original eval_lerf.py)
# ---------------------------------------------------------------------------


def smooth_mask(mask_pred: torch.Tensor) -> torch.Tensor:
    """Smooth a binary mask using average pooling (matching original smooth_cuda).

    Args:
        mask_pred: Binary mask tensor ``[H, W]`` of type uint8.

    Returns:
        Smoothed binary mask ``[H, W]`` of type uint8.
    """
    scale = 7
    avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=3, count_include_pad=False).to(mask_pred.device)
    avg_filtered = avg_pool(mask_pred.float().unsqueeze(0).unsqueeze(0))
    mask = (avg_filtered > 0.5).to(torch.uint8).squeeze(0).squeeze(0)
    return mask


def segmentation_process(
    relevancy_map: torch.Tensor,
    thresh: float,
    img_ann: dict,
    prompts: list,
    device: torch.device,
) -> Tuple[list, list, dict]:
    """Compute segmentation IoU for each prompt across all levels.

    Replicates the original ``segmentation_process_cuda`` exactly:
    1. AvgPool2d(29) smoothing blended 50/50 with raw relevancy
    2. Min-max normalize to [-1, 1] then clip to [0, 1]
    3. Threshold and smooth with AvgPool2d(7)
    4. IoU against GT mask
    5. Choose level with highest max relevancy score

    Args:
        relevancy_map: ``[n_levels, n_prompts, H, W]`` relevancy values.
        thresh: Mask threshold (default 0.4).
        img_ann: GT annotations for this frame ``{label: {"mask": ndarray, "bboxes": ndarray}}``.
        prompts: List of prompt labels (same order as relevancy_map's prompt dim).
        device: Torch device.

    Returns:
        Tuple of:
            - chosen_iou_list: per-prompt IoU at the chosen level
            - chosen_lvl_list: chosen level index per prompt
            - iou_all: ``{prompt: [iou_level_0, iou_level_1, iou_level_2]}``
    """
    n_head, n_prompt, h, w = relevancy_map.shape
    valid_map = relevancy_map.clone()

    chosen_iou_list = []
    chosen_lvl_list = []
    iou_all = {}

    for k in range(n_prompt):
        iou_lvl = torch.zeros(n_head, device=device)
        for i in range(n_head):
            # AvgPool smoothing (kernel=29)
            avg_pool = torch.nn.AvgPool2d(kernel_size=29, stride=1, padding=14, count_include_pad=False).to(device)
            avg_filtered = avg_pool(valid_map[i][k].unsqueeze(0).unsqueeze(0))
            valid_map[i][k] = 0.5 * (avg_filtered.squeeze(0).squeeze(0) + valid_map[i][k])

            # Normalize to [-1, 1] then clip to [0, 1]
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * 2.0 - 1.0  # scale to [-1, 1]
            output = torch.clip(output, 0, 1)

            # Threshold and smooth
            mask_pred = (output > thresh).to(torch.uint8)
            mask_pred = smooth_mask(mask_pred)

            # GT mask
            mask_gt = torch.from_numpy(img_ann[prompts[k]]["mask"].astype(np.uint8)).to(device)

            # IoU
            intersection = torch.sum(torch.logical_and(mask_gt, mask_pred))
            union = torch.sum(torch.logical_or(mask_gt, mask_pred))
            iou = intersection.float() / union.float()
            iou_lvl[i] = iou

        iou_all[prompts[k]] = iou_lvl.tolist()

        # Choose level with highest max relevancy score
        score_lvl = torch.zeros(n_head, device=valid_map.device)
        for i in range(n_head):
            score_lvl[i] = valid_map[i, k].max()
        chosen_lvl = torch.argmax(score_lvl)

        chosen_iou_list.append(iou_lvl[chosen_lvl].cpu().item())
        chosen_lvl_list.append(chosen_lvl.cpu().item())

    return chosen_iou_list, chosen_lvl_list, iou_all


def localization_process(
    relevancy_map: torch.Tensor,
    img_ann: dict,
    device: torch.device,
) -> int:
    """Compute localization accuracy (peak-in-bbox check).

    Replicates the original ``localization_process_cuda`` exactly:
    1. AvgPool2d(29) smoothing of relevancy
    2. For each prompt, find the peak location at the best level
    3. Check if peak falls inside any GT bbox

    Args:
        relevancy_map: ``[n_levels, n_prompts, H, W]`` relevancy values.
        img_ann: GT annotations ``{label: {"mask": ndarray, "bboxes": ndarray}}``.
        device: Torch device.

    Returns:
        Number of correctly localized bboxes.
    """
    n_head, n_prompt, h, w = relevancy_map.shape

    positives = list(img_ann.keys())
    acc_num = 0

    for k in range(n_prompt):
        select_output = relevancy_map[:, k]  # [n_head, H, W]
        avg_pool = torch.nn.AvgPool2d(kernel_size=29, stride=1, padding=14, count_include_pad=False).to(device)
        avg_filtered = avg_pool(select_output.unsqueeze(1)).squeeze(1)  # [n_head, H, W]

        score_lvl = torch.zeros(n_head)
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[i].max()
            coord = torch.nonzero((avg_filtered[i] == score).to(torch.uint8))
            score_lvl[i] = score
            coord_lvl.append(coord)

        selec_head = torch.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]

        for box in img_ann[positives[k]]["bboxes"].reshape(-1, 4):
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            flag = 0
            for cord_list in coord_final:
                # coord is (row, col) = (y, x)
                if cord_list[1] >= x_min and cord_list[1] <= x_max and cord_list[0] >= y_min and cord_list[0] <= y_max:
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break

    return acc_num


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def load_langsplatv2_model(
    checkpoint_path: pathlib.Path,
    gs_model_path: pathlib.Path,
    device: torch.device,
    logger: logging.Logger,
    eval_topk: int | None = None,
) -> Tuple[LangSplatV2Model, SfmScene]:
    """Load a trained LangSplatV2 model from a checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        gs_model_path: Path to the Gaussian splat PLY file.
        device: Device to load the model on.
        logger: Logger instance.
        eval_topk: If set, override the checkpoint's topk value. The
            original LangSplatV2 trains with topk=4.

    Returns:
        Tuple of (LangSplatV2Model, SfmScene) from the checkpoint.
    """
    # Load the base Gaussian splat
    gs_model, _ = load_splats_from_file(gs_model_path, device)
    logger.info(f"Loaded Gaussian splat with {gs_model.num_gaussians} gaussians from {gs_model_path}")

    # Load checkpoint and create trainer (eval-only mode)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    trainer = LangSplatV2Trainer.from_state_dict(
        state_dict=state_dict,
        gs_model=gs_model,
        gs_model_path=gs_model_path,
        device=device,
        eval_only=True,
    )

    model = trainer._model
    sfm_scene = trainer._sfm_scene
    feature_level = trainer._cfg.feature_level

    if eval_topk is not None and model.topk != eval_topk:
        logger.info(f"Overriding topk: {model.topk} (checkpoint) -> {eval_topk} (eval)")
        model.topk = eval_topk

    logger.info(
        f"Loaded LangSplatV2 model (feature_level={feature_level}, topk={model.topk}) " f"from {checkpoint_path}"
    )

    return model, sfm_scene


def render_clip_features(
    model: LangSplatV2Model,
    world_to_camera: torch.Tensor,
    projection: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Render CLIP feature maps from a LangSplatV2 model.

    Args:
        model: Trained LangSplatV2Model.
        world_to_camera: ``[1, 4, 4]`` world-to-camera matrix.
        projection: ``[1, 3, 3]`` camera intrinsics.
        image_width: Render width.
        image_height: Render height.

    Returns:
        Normalized CLIP features ``[H, W, 512]``.
    """
    with torch.no_grad():
        feature_maps, _ = model(
            world_to_camera=world_to_camera,
            projection=projection,
            image_width=image_width,
            image_height=image_height,
        )
    # feature_maps: [1, H, W, 512], normalize
    feat = feature_maps[0]  # [H, W, 512]
    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-10)
    return feat


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_frame_visualization(
    output_path: pathlib.Path,
    frame_idx: int,
    gt_img: np.ndarray | None,
    relevancy_map: torch.Tensor,
    prompts: list,
    img_ann: dict,
    chosen_iou_list: list,
    chosen_lvl_list: list,
    thresh: float,
    device: torch.device,
):
    """Save a per-frame visualization showing relevancy and segmentation.

    Creates a grid with one row per prompt showing:
    - GT image with GT mask overlay
    - Relevancy heatmap at chosen level
    - Predicted mask at chosen level
    - Overlay (predicted=red, GT=blue)

    Args:
        output_path: Directory for output images.
        frame_idx: Frame index.
        gt_img: Optional GT image ``[H, W, 3]`` (RGB uint8).
        relevancy_map: ``[n_levels, n_prompts, H, W]``.
        prompts: List of prompt labels.
        img_ann: GT annotations for this frame.
        chosen_iou_list: IoU values per prompt.
        chosen_lvl_list: Chosen level per prompt.
        thresh: Mask threshold.
        device: Torch device.
    """
    n_prompts = len(prompts)
    n_levels, _, h, w = relevancy_map.shape

    fig, axes = plt.subplots(n_prompts, 4, figsize=(24, 6 * n_prompts))
    if n_prompts == 1:
        axes = axes.reshape(1, -1)

    for k, prompt in enumerate(prompts):
        lvl = chosen_lvl_list[k]
        iou = chosen_iou_list[k]

        # Get relevancy at chosen level
        relev = relevancy_map[lvl, k].cpu().numpy()

        # Recompute predicted mask at chosen level (same as segmentation_process)
        relev_t = relevancy_map[lvl, k].clone()
        avg_pool = torch.nn.AvgPool2d(kernel_size=29, stride=1, padding=14, count_include_pad=False).to(device)
        avg_filtered = avg_pool(relev_t.unsqueeze(0).unsqueeze(0))
        blended = 0.5 * (avg_filtered.squeeze(0).squeeze(0) + relev_t)
        output = blended - torch.min(blended)
        output = output / (torch.max(output) + 1e-9)
        output = output * 2.0 - 1.0
        output = torch.clip(output, 0, 1)
        mask_pred = (output > thresh).to(torch.uint8)
        mask_pred = smooth_mask(mask_pred)
        mask_pred_np = mask_pred.cpu().numpy()

        mask_gt = img_ann[prompt]["mask"]

        # Col 0: GT image with GT mask overlay
        if gt_img is not None:
            overlay_gt = gt_img.copy().astype(np.float32) / 255.0
        else:
            overlay_gt = np.zeros((h, w, 3), dtype=np.float32)
        overlay_gt[:, :, 2] = np.clip(overlay_gt[:, :, 2] + mask_gt * 0.4, 0, 1)
        axes[k, 0].imshow(overlay_gt)
        axes[k, 0].set_title(f'GT: "{prompt}"')
        axes[k, 0].axis("off")

        # Col 1: Relevancy heatmap
        im = axes[k, 1].imshow(relev, cmap="jet", vmin=0, vmax=1)
        axes[k, 1].set_title(f"Relevancy (level {lvl + 1})")
        axes[k, 1].axis("off")
        plt.colorbar(im, ax=axes[k, 1], fraction=0.046, pad=0.04)

        # Col 2: Predicted mask
        axes[k, 2].imshow(mask_pred_np, cmap="gray")
        axes[k, 2].set_title(f"Pred mask (thresh={thresh})")
        axes[k, 2].axis("off")

        # Col 3: Overlay (red=pred, blue=GT)
        if gt_img is not None:
            overlay = gt_img.copy().astype(np.float32) / 255.0
        else:
            overlay = np.zeros((h, w, 3), dtype=np.float32)
        mask_overlay = np.zeros_like(overlay)
        mask_overlay[:, :, 0] = mask_pred_np  # Red for prediction
        mask_overlay[:, :, 2] = mask_gt  # Blue for GT
        overlay = overlay * 0.5 + mask_overlay * 0.5
        axes[k, 3].imshow(np.clip(overlay, 0, 1))
        axes[k, 3].contour(mask_gt, colors="blue", linewidths=1, linestyles="solid")
        axes[k, 3].contour(mask_pred_np, colors="red", linewidths=1, linestyles="dashed")
        axes[k, 3].set_title(f"Overlay (IoU={iou * 100:.1f}%)")
        axes[k, 3].axis("off")

    fig.suptitle(f"Frame {frame_idx}")
    fig.tight_layout()
    fig.savefig(output_path / f"frame_{frame_idx:05d}.jpg", dpi=150, pil_kwargs={"quality": 90})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvaluationConfig:
    """Configuration for LERF evaluation."""

    lerf_root: pathlib.Path = pathlib.Path("data/lerf_ovs")
    """Root directory of the LERF-OVS dataset."""

    results_root: pathlib.Path = pathlib.Path("langsplatv2_results")
    """Directory containing trained model checkpoints.

    Expected layout::

        results_root/
            <scene_name>_level_1.pt
            <scene_name>_level_2.pt
            <scene_name>_level_3.pt
    """

    reconstructions_root: pathlib.Path | None = pathlib.Path("reconstructions")
    """Directory containing per-scene Gaussian splat reconstructions.

    Expected layout::

        reconstructions_root/
            <scene_name>.ply

    If None, falls back to the LERF dataset structure at
    ``lerf_root/<scene>/output/<scene>/point_cloud/iteration_30000/point_cloud.ply``.
    """

    output_dir: pathlib.Path = pathlib.Path("lerf_eval_results")
    """Directory to save evaluation results and visualizations."""

    device: str = "cuda"
    """Device for computation."""

    mask_thresh: float = 0.4
    """Threshold for converting relevancy map to binary mask (matching original)."""

    save_visualizations: bool = True
    """Whether to save per-frame visualization images."""

    eval_topk: int = 4
    """Number of codebook entries to combine at evaluation time.

    The original LangSplatV2 trains with topk=4"""


def get_camera_for_frame(
    sfm_scene,
    frame_idx: int,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[torch.Tensor, torch.Tensor, int, int] | None:
    """Get camera parameters for a specific frame index.

    Images in the SfmScene are sorted by name to match the COLMAP ordering
    used by the original LangSplatV2 dataset.

    Args:
        sfm_scene: The SfmScene from the checkpoint.
        frame_idx: 0-based frame index.
        device: Torch device.
        logger: Logger instance.

    Returns:
        Tuple of (world_to_camera [1,4,4], projection [1,3,3], height, width)
        or None if index is out of range.
    """
    # Sort images by name to match COLMAP ordering
    sorted_images = sorted(sfm_scene.images, key=lambda img: img.image_path)

    if frame_idx >= len(sorted_images):
        logger.error(f"Frame index {frame_idx} out of range ({len(sorted_images)} images)")
        return None

    img_meta = sorted_images[frame_idx]
    c2w = torch.from_numpy(img_meta.camera_to_world_matrix).float()
    K = torch.from_numpy(img_meta.camera_metadata.projection_matrix).float()
    h = img_meta.camera_metadata.height
    w = img_meta.camera_metadata.width

    w2c = torch.linalg.inv(c2w).contiguous()

    return (
        w2c.unsqueeze(0).to(device),
        K.unsqueeze(0).to(device),
        h,
        w,
    )


def run_lerf_evaluation(
    scene_name: str,
    config: EvaluationConfig,
    logger: logging.Logger,
) -> dict | None:
    """Run evaluation on a single LERF scene.

    Args:
        scene_name: Name of the scene (e.g. "teatime", "figurines").
        config: Evaluation configuration.
        logger: Logger instance.

    Returns:
        Dictionary with evaluation results, or None if evaluation failed.
    """
    device = torch.device(config.device)

    # --- Locate checkpoint files ---
    level_checkpoints = []
    for level in [1, 2, 3]:
        ckpt_path = config.results_root / f"{scene_name}_level_{level}.pt"
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {ckpt_path}")
            return None
        level_checkpoints.append(ckpt_path)

    # --- Locate GS model for this scene ---
    gs_path = None
    # Try reconstructions_root/<scene>.ply first
    if config.reconstructions_root is not None:
        candidate = config.reconstructions_root / f"{scene_name}.ply"
        if candidate.exists():
            gs_path = candidate
        else:
            # Also try .pt/.pth extensions
            for ext in (".pt", ".pth"):
                candidate = config.reconstructions_root / f"{scene_name}{ext}"
                if candidate.exists():
                    gs_path = candidate
                    break

    # Fall back to LERF dataset structure
    if gs_path is None:
        gs_path = (
            config.lerf_root
            / scene_name
            / "output"
            / scene_name
            / "point_cloud"
            / "iteration_30000"
            / "point_cloud.ply"
        )

    if not gs_path.exists():
        logger.error(
            f"GS model not found for scene '{scene_name}'. "
            f"Searched: reconstructions_root={config.reconstructions_root}, "
            f"lerf_root={config.lerf_root}/<scene>/output/..."
        )
        return None

    logger.info(f"Using GS model: {gs_path}")

    # --- Load ground truth ---
    label_dir = config.lerf_root / "label" / scene_name
    if not label_dir.exists():
        logger.error(f"Label directory not found: {label_dir}")
        return None

    gt_ann, image_shape, gt_img_paths = load_lerf_ground_truth(label_dir, logger)
    eval_frame_indices = [int(idx) for idx in gt_ann.keys()]
    gt_h, gt_w = image_shape

    # --- Load models (3 levels) ---
    models: list[LangSplatV2Model] = []
    sfm_scene = None
    for i, ckpt_path in enumerate(level_checkpoints):
        model, scene = load_langsplatv2_model(ckpt_path, gs_path, device, logger, eval_topk=config.eval_topk)
        models.append(model)
        if sfm_scene is None:
            sfm_scene = scene

    # --- Load OpenCLIP for relevancy ---
    clip_relevancy = OpenCLIPRelevancy(device=config.device)

    # --- Evaluate each annotated frame ---
    chosen_iou_all = []
    chosen_lvl_list_all = []
    acc_num_total = 0
    total_bboxes = 0
    per_frame_results = []

    scene_output_dir = config.output_dir / scene_name
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    for frame_enum_idx, frame_idx in enumerate(
        tqdm.tqdm(eval_frame_indices, desc=f"Evaluating {scene_name}", leave=False)
    ):
        # Load GT image for visualization (optional)
        gt_img = None
        if frame_enum_idx < len(gt_img_paths) and config.save_visualizations:
            gt_img_bgr = cv2.imread(gt_img_paths[frame_enum_idx])
            if gt_img_bgr is not None:
                gt_img = cv2.cvtColor(gt_img_bgr, cv2.COLOR_BGR2RGB)

        # Get camera parameters for this frame
        cam_info = get_camera_for_frame(sfm_scene, frame_idx, device, logger)
        if cam_info is None:
            logger.warning(f"Skipping frame {frame_idx}: camera not found")
            continue
        w2c, K, img_h, img_w = cam_info

        # Render CLIP features from all 3 levels
        sem_feats = []
        for model in models:
            feat = render_clip_features(model, w2c, K, img_w, img_h)
            sem_feats.append(feat)

        # Stack: [3, H, W, 512]
        sem_map = torch.stack(sem_feats, dim=0)

        # Get GT annotations for this frame
        img_ann = gt_ann[str(frame_idx)]
        prompts = list(img_ann.keys())

        # Set positive prompts in CLIP model
        clip_relevancy.set_positives(prompts)

        # Compute relevancy: [3, n_prompts, H, W]
        relevancy_map = clip_relevancy.get_relevancy_map(sem_map)

        # Resize relevancy to GT resolution if needed
        _, _, relev_h, relev_w = relevancy_map.shape
        if relev_h != gt_h or relev_w != gt_w:
            relevancy_map = torch.nn.functional.interpolate(
                relevancy_map.reshape(-1, 1, relev_h, relev_w),
                size=(gt_h, gt_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(3, len(prompts), gt_h, gt_w)

        # Segmentation IoU
        # Clone relevancy_map because segmentation_process modifies it in-place
        c_iou_list, c_lvl_list, iou_all = segmentation_process(
            relevancy_map.clone(), config.mask_thresh, img_ann, prompts, device
        )
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list_all.extend(c_lvl_list)

        # Localization accuracy
        acc_num_img = localization_process(relevancy_map.clone(), img_ann, device)
        acc_num_total += acc_num_img
        total_bboxes += len(prompts)

        # Per-frame record
        frame_result = {
            "frame_idx": frame_idx,
            "prompts": prompts,
            "ious": c_iou_list,
            "chosen_levels": c_lvl_list,
            "iou_all_levels": iou_all,
            "localization_correct": acc_num_img,
            "num_bboxes": len(prompts),
        }
        per_frame_results.append(frame_result)

        logger.info(
            f"  Frame {frame_idx}: "
            + ", ".join(
                f'"{p}" IoU={iou * 100:.1f}% (lvl {lvl + 1})' for p, iou, lvl in zip(prompts, c_iou_list, c_lvl_list)
            )
            + f" | loc={acc_num_img}/{len(prompts)}"
        )

        # Save visualization
        if config.save_visualizations:
            save_frame_visualization(
                scene_output_dir,
                frame_idx,
                gt_img,
                relevancy_map,
                prompts,
                img_ann,
                c_iou_list,
                c_lvl_list,
                config.mask_thresh,
                device,
            )

    # --- Scene summary ---
    if not chosen_iou_all:
        logger.warning(f"No successful evaluations for scene {scene_name}")
        return None

    mean_iou = float(np.mean(chosen_iou_all))
    loc_accuracy = acc_num_total / total_bboxes if total_bboxes > 0 else 0.0

    logger.info(
        f"Scene {scene_name}: mean IoU = {mean_iou * 100:.1f}%, localization accuracy = {loc_accuracy * 100:.1f}%"
    )
    logger.info(f"  Chosen levels: {chosen_lvl_list_all}")

    return {
        "scene": scene_name,
        "mean_iou": mean_iou,
        "localization_accuracy": loc_accuracy,
        "localization_correct": acc_num_total,
        "localization_total": total_bboxes,
        "per_prompt_ious": chosen_iou_all,
        "chosen_levels": chosen_lvl_list_all,
        "mask_thresh": config.mask_thresh,
        "per_frame_results": per_frame_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LangSplatV2 models on the LERF-OVS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lerf-root",
        type=pathlib.Path,
        default=pathlib.Path("data/lerf_ovs"),
        help="Root directory of the LERF-OVS dataset",
    )
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=pathlib.Path("langsplatv2_results"),
        help="Directory containing trained model checkpoints "
        "(<scene>_level_1.pt, <scene>_level_2.pt, <scene>_level_3.pt)",
    )
    parser.add_argument(
        "--reconstructions-root",
        type=pathlib.Path,
        default=pathlib.Path("reconstructions"),
        help="Directory containing per-scene Gaussian splat reconstructions "
        "(<scene>.ply). If not found, falls back to LERF dataset structure.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("lerf_eval_results"),
        help="Directory to save evaluation results and visualizations",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        type=str,
        default=None,
        help="Specific scenes to evaluate. If not specified, discovers available scenes.",
    )
    parser.add_argument(
        "--mask-thresh",
        type=float,
        default=0.4,
        help="Threshold for converting relevancy map to binary mask",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (cuda or cuda:N)",
    )
    parser.add_argument(
        "--eval-topk",
        type=int,
        default=4,
        help="Number of codebook entries to combine at evaluation time. "
        "The original LangSplatV2 trains with topk=4.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable saving per-frame visualization images",
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
    logger = logging.getLogger("lerf_eval")

    # Build config
    config = EvaluationConfig(
        lerf_root=args.lerf_root,
        results_root=args.results_root,
        reconstructions_root=args.reconstructions_root,
        output_dir=args.output_dir,
        device=args.device,
        mask_thresh=args.mask_thresh,
        save_visualizations=not args.no_visualizations,
        eval_topk=args.eval_topk,
    )

    # Determine scenes to evaluate
    if args.scenes:
        scenes_to_eval = args.scenes
    else:
        # Auto-discover scenes from checkpoint files
        ckpt_files = list(config.results_root.glob("*_level_1.pt"))
        if ckpt_files:
            scenes_to_eval = [f.stem.replace("_level_1", "") for f in ckpt_files]
        else:
            # Fall back to label directories
            label_root = config.lerf_root / "label"
            if label_root.exists():
                scenes_to_eval = [d.name for d in label_root.iterdir() if d.is_dir()]
            else:
                logger.error("No scenes found. Specify --scenes or check paths.")
                return

    logger.info(f"Evaluating scenes: {scenes_to_eval}")
    logger.info(f"Mask threshold: {config.mask_thresh}")

    # Run evaluation
    results = []
    for scene_name in scenes_to_eval:
        logger.info("=" * 60)
        logger.info(f"Evaluating scene: {scene_name}")
        logger.info("=" * 60)

        result = run_lerf_evaluation(scene_name, config, logger)
        if result is not None:
            results.append(result)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE - Summary:")
    logger.info("=" * 60)

    if results:
        ious = [r["mean_iou"] for r in results]
        accs = [r["localization_accuracy"] for r in results]
        overall_mean_iou = float(np.mean(ious))
        overall_loc_acc = float(np.mean(accs))

        logger.info(f"{'Scene':<20} {'mean IoU':>10} {'Loc Acc':>10}")
        logger.info("-" * 42)
        for r in results:
            logger.info(f"{r['scene']:<20} {r['mean_iou'] * 100:>9.1f}% {r['localization_accuracy'] * 100:>9.1f}%")
        logger.info("-" * 42)
        logger.info(f"{'Overall':<20} {overall_mean_iou * 100:>9.1f}% {overall_loc_acc * 100:>9.1f}%")

        # Save results JSON
        results_file = config.output_dir / "lerf_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        # Strip per-frame results for the summary (keep in per-scene files)
        summary_results = []
        for r in results:
            # Save per-scene detailed results
            scene_results_file = config.output_dir / r["scene"] / "results.json"
            scene_results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(scene_results_file, "w") as f:
                json.dump(r, f, indent=2)
            logger.info(f"Per-scene results saved to {scene_results_file}")

            # Summary (without per-frame details)
            summary = {k: v for k, v in r.items() if k != "per_frame_results"}
            summary_results.append(summary)

        with open(results_file, "w") as f:
            json.dump(
                {
                    "results": summary_results,
                    "overall_mean_iou": overall_mean_iou,
                    "overall_localization_accuracy": overall_loc_acc,
                    "config": {
                        "mask_thresh": config.mask_thresh,
                        "device": config.device,
                    },
                },
                f,
                indent=2,
            )
        logger.info(f"Summary results saved to {results_file}")
    else:
        logger.warning("No successful evaluations")


if __name__ == "__main__":
    main()
