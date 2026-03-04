#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Visualise the relevancy mask for a text prompt on a trained LangSplatV2 model.

Loads a single-level checkpoint, renders CLIP features for a chosen camera
view, computes OpenCLIP relevancy against the user's prompt, and writes an
image showing the source photograph, the relevancy heatmap, and the
thresholded binary mask side-by-side.

Usage:
    python scripts/query_prompt.py \\
        --checkpoint langsplatv2_results/scene_level_2.pt \\
        --reconstruction-path reconstructions/scene.ply \\
        --prompt "coffee cup"

    # Specify a different view and output path:
    python scripts/query_prompt.py \\
        --checkpoint langsplatv2_results/scene_level_1.pt \\
        --reconstruction-path reconstructions/scene.ply \\
        --prompt "wooden table" \\
        --image-index 42 \\
        --output table_query.jpg
"""
import argparse
import logging
import pathlib

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from langsplatv2.evaluation.openclip_relevancy import OpenCLIPRelevancy
from langsplatv2.training.trainer import LangSplatV2Trainer
from langsplatv2.util import load_splats_from_file

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _load_model(checkpoint_path, gs_model_path, device, eval_topk=None):
    """Load a trained LangSplatV2 model and its embedded SfmScene."""
    gs_model, _ = load_splats_from_file(gs_model_path, device)
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
        logger.info(f"Overriding topk: {model.topk} -> {eval_topk}")
        model.topk = eval_topk

    logger.info(
        f"Loaded model (feature_level={feature_level}, topk={model.topk}) " f"with {gs_model.num_gaussians:,} Gaussians"
    )
    return model, sfm_scene, feature_level


def _get_camera(sfm_scene, image_index, device):
    """Return (w2c, K, h, w, image_path) for the given image index."""
    sorted_images = sorted(sfm_scene.images, key=lambda img: img.image_path)
    if image_index >= len(sorted_images):
        raise IndexError(f"--image-index {image_index} out of range " f"(scene has {len(sorted_images)} images)")
    img = sorted_images[image_index]
    c2w = torch.from_numpy(img.camera_to_world_matrix).float()
    K = torch.from_numpy(img.camera_metadata.projection_matrix).float()
    w2c = torch.linalg.inv(c2w).contiguous()
    return (
        w2c.unsqueeze(0).to(device),
        K.unsqueeze(0).to(device),
        img.camera_metadata.height,
        img.camera_metadata.width,
        img.image_path,
    )


@torch.no_grad()
def _render_clip(model, w2c, K, width, height):
    """Render normalised CLIP features [H, W, 512]."""
    feat_maps, _ = model(
        world_to_camera=w2c,
        projection=K,
        image_width=width,
        image_height=height,
    )
    feat = feat_maps[0]  # [H, W, 512]
    return feat / (feat.norm(dim=-1, keepdim=True) + 1e-10)


def _smooth_mask(mask: torch.Tensor) -> torch.Tensor:
    pool = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=3, count_include_pad=False)
    pool = pool.to(mask.device)
    smoothed = pool(mask.float().unsqueeze(0).unsqueeze(0))
    return (smoothed > 0.5).to(torch.uint8).squeeze(0).squeeze(0)


def _relevancy_to_mask(relevancy_2d: torch.Tensor, thresh: float) -> torch.Tensor:
    """Convert a raw relevancy map [H, W] to a binary mask.

    Applies the same AvgPool(29) blending, normalisation, and thresholding
    that the original LangSplatV2 evaluation uses.
    """
    pool = torch.nn.AvgPool2d(kernel_size=29, stride=1, padding=14, count_include_pad=False)
    pool = pool.to(relevancy_2d.device)
    blended = 0.5 * (pool(relevancy_2d.unsqueeze(0).unsqueeze(0)).squeeze() + relevancy_2d)
    blended = blended - blended.min()
    blended = blended / (blended.max() + 1e-9)
    blended = blended * 2.0 - 1.0
    blended = torch.clip(blended, 0, 1)
    return _smooth_mask((blended > thresh).to(torch.uint8))


def _save_visualization(
    output_path: pathlib.Path,
    prompt: str,
    feature_level: int,
    relevancy: np.ndarray,
    mask: np.ndarray,
    source_img: np.ndarray | None,
    image_index: int,
):
    """Write a side-by-side panel image to *output_path*."""
    has_source = source_img is not None
    n_cols = 3 if has_source else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 8))

    col = 0

    if has_source:
        overlay = source_img.astype(np.float32) / 255.0
        mask_rgb = np.zeros_like(overlay)
        mask_rgb[:, :, 0] = mask.astype(np.float32)
        blended = np.clip(overlay * 0.6 + mask_rgb * 0.4, 0, 1)
        axes[col].imshow(blended)
        axes[col].set_title(f"Source (image {image_index}) + mask overlay", fontsize=12)
        axes[col].axis("off")
        col += 1

    im = axes[col].imshow(relevancy, cmap="turbo", vmin=0, vmax=1)
    axes[col].set_title(f"Relevancy (level {feature_level})", fontsize=12)
    axes[col].axis("off")
    plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
    col += 1

    axes[col].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[col].set_title("Binary mask", fontsize=12)
    axes[col].axis("off")

    fig.suptitle(f'"{prompt}"', fontsize=16, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved visualisation to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise relevancy for a text prompt on a trained LangSplatV2 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to a trained LangSplatV2 checkpoint (.pt) for one feature level.",
    )
    parser.add_argument(
        "--reconstruction-path",
        type=pathlib.Path,
        required=True,
        help="Path to the Gaussian splat reconstruction (.ply or .pt).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text query to compute relevancy for (e.g. 'coffee cup').",
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="Which dataset image (camera view) to render.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output image path.  Defaults to '<prompt>_level<N>_img<I>.jpg'.",
    )
    parser.add_argument(
        "--mask-thresh",
        type=float,
        default=0.4,
        help="Threshold for converting relevancy to a binary mask.",
    )
    parser.add_argument(
        "--eval-topk",
        type=int,
        default=None,
        help="Override the checkpoint's topk value for codebook decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")

    device = torch.device(args.device)

    model, sfm_scene, feature_level = _load_model(
        args.checkpoint,
        args.reconstruction_path,
        device,
        args.eval_topk,
    )

    w2c, K, img_h, img_w, image_path = _get_camera(sfm_scene, args.image_index, device)
    logger.info(f"Rendering view {args.image_index}: {image_path} ({img_w}x{img_h})")

    source_img = None
    if image_path and pathlib.Path(image_path).is_file():
        bgr = cv2.imread(str(image_path))
        if bgr is not None:
            source_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    clip_feat = _render_clip(model, w2c, K, img_w, img_h)

    clip_relevancy = OpenCLIPRelevancy(device=args.device)
    clip_relevancy.set_positives([args.prompt])

    # get_relevancy_map expects [n_levels, H, W, 512]
    relevancy = clip_relevancy.get_relevancy_map(clip_feat.unsqueeze(0))  # [1, 1, H, W]
    relevancy_2d = relevancy[0, 0]  # [H, W]

    mask = _relevancy_to_mask(relevancy_2d, args.mask_thresh)

    if args.output is None:
        safe_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:40]
        args.output = pathlib.Path(f"{safe_prompt}_level{feature_level}_img{args.image_index}.jpg")

    _save_visualization(
        args.output,
        args.prompt,
        feature_level,
        relevancy_2d.cpu().numpy(),
        mask.cpu().numpy(),
        source_img,
        args.image_index,
    )

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
