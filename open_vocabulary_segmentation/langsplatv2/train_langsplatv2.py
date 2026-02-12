# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Training script for LangSplatV2 language features using fVDB.

This script trains language features on a pre-trained Gaussian Splatting
reconstruction, following the LangSplatV2 approach of using sparse coefficient
fields decoded via shared codebooks into CLIP features.

Usage:
    python train_langsplatv2.py \\
        --sfm-dataset-path /path/to/colmap/scene \\
        --reconstruction-path /path/to/gaussians.ply \\
        --config.feature-level 1

    # Train all 3 levels (as in the original paper):
    for level in 1 2 3; do
        python train_langsplatv2.py \\
            --sfm-dataset-path /path/to/scene \\
            --reconstruction-path /path/to/gaussians.ply \\
            --config.feature-level $level \\
            --log-path langsplatv2_logs
    done
"""
import logging
import pathlib
from typing import Literal

import torch
import tyro

from langsplatv2.config import LangSplatV2PreprocessConfig, LangSplatV2TrainingConfig
from langsplatv2.training import LangSplatV2Trainer
from langsplatv2.util import load_sfm_scene, load_splats_from_file


def main(
    sfm_dataset_path: pathlib.Path,
    reconstruction_path: pathlib.Path,
    config: LangSplatV2TrainingConfig = LangSplatV2TrainingConfig(),
    preprocess: LangSplatV2PreprocessConfig = LangSplatV2PreprocessConfig(),
    dataset_type: Literal["colmap", "simple_directory", "e57"] = "colmap",
    run_name: str | None = None,
    log_path: pathlib.Path | None = pathlib.Path("langsplatv2_logs"),
    use_every_n_as_val: int = -1,
    device: str | torch.device = "cuda",
    log_every: int = 10,
    verbose: bool = False,
    cache_dataset: bool = True,
):
    """Train LangSplatV2 language features on a Gaussian Splatting scene.

    This script performs the following steps:
    1. Loads the SfM scene and Gaussian splat reconstruction
    2. Applies preprocessing transforms (normalization, downsampling, SAM2 masks, CLIP features)
    3. Initializes codebooks via K-means clustering on CLIP features
    4. Trains per-Gaussian sparse coefficient logits and shared codebooks
    5. Saves checkpoints for later evaluation

    Args:
        sfm_dataset_path: Path to the SfM dataset (COLMAP, simple_directory, or E57).
        reconstruction_path: Path to the pre-trained Gaussian splat reconstruction
            (.ply or .pt checkpoint).
        config: Training configuration including model hyperparameters, feature level,
            loss settings, and optimization parameters.
        preprocess: Preprocessing pipeline configuration for scene transforms
            (image downsampling, SAM2 masks, CLIP features).
        dataset_type: Type of input dataset.
        run_name: Name for this training run. If None, auto-generated from timestamp.
        log_path: Directory for saving checkpoints, metrics, and logs.
            Set to None to disable saving.
        use_every_n_as_val: Use every N-th image as validation. -1 = no validation.
        device: Device for training (cuda or cpu).
        log_every: Log training metrics every N steps.
        verbose: Enable debug logging.
        cache_dataset: Cache images and features in memory for faster training.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("LangSplatV2 Training with fVDB")
    logger.info("=" * 60)
    logger.info(f"Dataset: {sfm_dataset_path}")
    logger.info(f"Reconstruction: {reconstruction_path}")
    logger.info(f"Feature level: {config.feature_level}")
    logger.info(f"Max steps: {config.max_steps}")
    logger.info(f"Codebook: {config.model.vq_layer_num} layers x {config.model.codebook_size} entries")
    logger.info(f"Top-k: {config.model.topk}")
    logger.info(f"Loss: cosine={config.use_cosine_loss}, l1={config.use_l1_loss}")

    # Load the SfM scene
    logger.info("Loading SfM scene...")
    sfm_scene = load_sfm_scene(sfm_dataset_path, dataset_type)
    logger.info(f"Loaded scene with {sfm_scene.num_images} images")

    # Load the Gaussian splat model
    logger.info("Loading Gaussian splat reconstruction...")
    gs_model, metadata = load_splats_from_file(reconstruction_path, device)
    normalization_transform = metadata.get("normalization_transform", None)
    logger.info(f"Loaded {gs_model.num_gaussians:,} Gaussians")

    # Apply preprocessing transforms (SAM2 masks + CLIP features)
    logger.info("Applying preprocessing transforms...")
    scene_transforms = preprocess.build_scene_transforms(
        normalization_transform=normalization_transform,
    )
    preprocessed_scene = scene_transforms(sfm_scene)
    logger.info("Preprocessing complete.")

    # Create and run training
    runner = LangSplatV2Trainer.new(
        sfm_scene=preprocessed_scene,
        gs_model=gs_model,
        gs_model_path=reconstruction_path,
        config=config,
        device=device,
        use_every_n_as_val=use_every_n_as_val,
        save_path=log_path,
        run_name=run_name,
        log_interval_steps=log_every,
        cache_dataset=cache_dataset,
    )

    runner.train()

    logger.info("=" * 60)
    logger.info("Training complete!")
    if log_path is not None:
        logger.info(f"Results saved to {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    tyro.cli(main)
