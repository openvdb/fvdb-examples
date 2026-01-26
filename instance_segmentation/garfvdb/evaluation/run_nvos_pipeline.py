# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Batch processing script for NVOS dataset evaluation.

This script runs the full GARfVDB pipeline on all NVOS scenes:
1. Reconstructs Gaussian splats from colmap data
2. Trains segmentation models on the reconstructions

Usage:
    python run_nvos_pipeline.py --dataset-root /media/datasets/segmentation_evaluation/nvos/scenes
    python run_nvos_pipeline.py --dataset-root /media/datasets/segmentation_evaluation/nvos/scenes --output-root ./nvos_results
    python run_nvos_pipeline.py --dataset-root /media/datasets/segmentation_evaluation/nvos/scenes --scenes fern_undistort flower_undistort

    # Using MCMC optimizer with Gaussian limits:
    python run_nvos_pipeline.py --dataset-root /path/to/scenes --use-mcmc-optimizer --max-gaussians 500000 --insertion-rate 1.1
"""
import argparse
import logging
import pathlib
import socket
from dataclasses import dataclass, field
from typing import Literal

import fvdb.viz as fviz
import numpy as np
import torch
from fvdb_reality_capture.radiance_fields import (
    GaussianSplatOptimizerConfig,
    GaussianSplatOptimizerMCMCConfig,
    GaussianSplatReconstruction,
    GaussianSplatReconstructionConfig,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)
from fvdb_reality_capture.sfm_scene import SfmScene
from fvdb_reality_capture.transforms import (
    Compose,
    CropSceneToPoints,
    DownsampleImages,
    FilterImagesWithLowPoints,
    Identity,
    NormalizeScene,
    PercentileFilterPoints,
    TransformScene,
)
from garfvdb.config import GaussianSplatSegmentationTrainingConfig
from garfvdb.scene_transforms import ComputeImageSegmentationMasksWithScales
from garfvdb.training.segmentation import GaussianSplatScaleConditionedSegmentation
from garfvdb.training.segmentation_writer import (
    GaussianSplatSegmentationWriter,
    GaussianSplatSegmentationWriterConfig,
)
from garfvdb.util import load_splats_from_file

# NVOS ground truth mask images that should be excluded from training
# Maps: scene_dir_name -> gt_mask_image_filename (without extension)
NVOS_GT_MASK_IMAGES = {
    "fern_undistort": "IMG_4027",
    "flower_undistort": "IMG_2962",
    "fortress_undistort": "IMG_1800",
    "horns_undistort": "DJI_20200223_163024_597",
    "leaves_undistort": "IMG_2997",
    "orchids_undistort": "IMG_4480",
    "trex_undistort": "DJI_20200223_163619_411",
}


def get_lan_ip() -> str:
    """
    Get the LAN IP address of this machine.

    Returns the IP address that would be used to connect to external networks,
    falling back to 127.0.0.1 if detection fails.
    """
    try:
        # Create a socket and connect to an external address (doesn't send data)
        # This reveals which local interface would be used for external connections
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def fix_camera_metadata_from_images(sfm_scene: SfmScene, logger: logging.Logger | None = None) -> SfmScene:
    """
    Fix camera metadata to match actual image dimensions on disk.

    This is needed when images have been undistorted and their dimensions changed,
    but the colmap camera metadata still has the original (pre-undistortion) dimensions.

    The function intelligently determines whether cx, cy need adjustment:
    - If cx, cy appear to be centered on the metadata dimensions, they are adjusted
      for the crop offset (assuming center crop during undistortion).
    - If cx, cy already appear to be centered on the actual image dimensions,
      they are left unchanged (COLMAP already adjusted them).

    Args:
        sfm_scene: The SfM scene with potentially incorrect camera metadata.
        logger: Optional logger for debug output.

    Returns:
        A new SfmScene with camera metadata updated to match actual image dimensions.
    """
    import cv2
    from fvdb_reality_capture.sfm_scene import SfmCameraMetadata, SfmPosedImageMetadata

    if logger:
        logger.info("Checking and fixing camera metadata to match actual image dimensions...")

    # Read actual dimensions from image files
    new_cameras = {}
    new_images = []

    for img_meta in sfm_scene.images:
        # Read actual image dimensions
        img = cv2.imread(img_meta.image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_meta.image_path}")

        actual_h, actual_w = img.shape[:2]
        expected_w = img_meta.camera_metadata.width
        expected_h = img_meta.camera_metadata.height

        if actual_w != expected_w or actual_h != expected_h:
            old_cam = img_meta.camera_metadata

            # Determine if cx, cy need adjustment by checking which dimensions they're centered on
            # If cx is close to expected_w/2, it's centered on metadata dims and needs adjustment
            # If cx is close to actual_w/2, it's already adjusted for actual dims
            cx_centered_on_metadata = abs(old_cam.cx - expected_w / 2) < abs(old_cam.cx - actual_w / 2)
            cy_centered_on_metadata = abs(old_cam.cy - expected_h / 2) < abs(old_cam.cy - actual_h / 2)

            # Calculate crop offsets (assuming center crop)
            cx_offset = (expected_w - actual_w) / 2 if cx_centered_on_metadata else 0
            cy_offset = (expected_h - actual_h) / 2 if cy_centered_on_metadata else 0

            new_cx = old_cam.cx - cx_offset
            new_cy = old_cam.cy - cy_offset

            if logger:
                logger.debug(
                    f"Image {img_meta.image_id}: metadata {expected_w}x{expected_h} -> actual {actual_w}x{actual_h}, "
                    f"cx: {old_cam.cx:.1f} -> {new_cx:.1f} ({'adjusted' if cx_offset else 'kept'}), "
                    f"cy: {old_cam.cy:.1f} -> {new_cy:.1f} ({'adjusted' if cy_offset else 'kept'})"
                )

            new_cam_meta = SfmCameraMetadata(
                img_width=actual_w,
                img_height=actual_h,
                fx=old_cam.fx,
                fy=old_cam.fy,
                cx=new_cx,
                cy=new_cy,
                camera_type=old_cam.camera_type,
                distortion_parameters=old_cam.distortion_parameters,
            )

            # Use image_id as camera_id since each image may have different adjustments
            new_cameras[img_meta.image_id] = new_cam_meta

            new_images.append(
                SfmPosedImageMetadata(
                    world_to_camera_matrix=img_meta.world_to_camera_matrix,
                    camera_to_world_matrix=img_meta.camera_to_world_matrix,
                    camera_metadata=new_cam_meta,
                    camera_id=img_meta.image_id,
                    image_path=img_meta.image_path,
                    mask_path=img_meta.mask_path,
                    point_indices=img_meta.point_indices,
                    image_id=img_meta.image_id,
                )
            )
        else:
            # Dimensions match, keep original
            if img_meta.camera_id not in new_cameras:
                new_cameras[img_meta.camera_id] = img_meta.camera_metadata
            new_images.append(img_meta)

    # Check if any fixes were needed
    num_fixed = sum(1 for img in new_images if img.camera_id == img.image_id and img.image_id in new_cameras)
    if num_fixed > 0:
        if logger:
            logger.info(f"Fixed camera metadata for {num_fixed} images to match actual dimensions")

        return SfmScene(
            cameras=new_cameras,
            images=new_images,
            points=sfm_scene.points,
            points_err=sfm_scene.points_err,
            points_rgb=sfm_scene.points_rgb,
            scene_bbox=sfm_scene.scene_bbox,
            transformation_matrix=sfm_scene.transformation_matrix,
            cache=sfm_scene.cache,
        )
    else:
        if logger:
            logger.info("All camera metadata already matches actual image dimensions")
        return sfm_scene


@dataclass
class ReconstructionSettings:
    """Configuration for the reconstruction phase."""

    image_downsample_factor: int = 4
    rescale_jpeg_quality: int = 95
    points_percentile_filter: float = 0.0
    normalization_type: Literal["pca", "none", "ecef2enu", "similarity"] = "pca"
    crop_to_points: bool = False
    min_points_per_image: int = 5

    # Reconstruction config overrides
    max_epochs: int = 30
    refine_every_epoch: float = 0.65
    refine_stop_epoch: int = 100  # At which epoch to stop refining Gaussians

    # MCMC optimizer settings
    use_mcmc_optimizer: bool = False
    max_gaussians: int = -1  # -1 means no limit
    insertion_rate: float = 1.05

    def build_scene_transform(self) -> Compose:
        """Build the scene transform pipeline for reconstruction."""
        transforms = [
            NormalizeScene(normalization_type=self.normalization_type),
            PercentileFilterPoints(
                percentile_min=np.full((3,), self.points_percentile_filter),
                percentile_max=np.full((3,), 100.0 - self.points_percentile_filter),
            ),
            DownsampleImages(
                image_downsample_factor=self.image_downsample_factor,
                rescaled_jpeg_quality=self.rescale_jpeg_quality,
            ),
            FilterImagesWithLowPoints(min_num_points=self.min_points_per_image),
        ]
        if self.crop_to_points:
            transforms.append(CropSceneToPoints(margin=0.0))
        return Compose(*transforms)


@dataclass
class SegmentationSettings:
    """Configuration for the segmentation training phase."""

    max_epochs: int = 100
    batch_size: int = 8
    sample_pixels_per_image: int = 256
    use_every_n_as_val: int = -1
    log_every: int = 10
    image_downsample_factor: int = 2
    eval_at_percent: list[int] = field(default_factory=lambda: [100])


@dataclass
class PipelineConfig:
    """Main configuration for the NVOS pipeline."""

    dataset_root: pathlib.Path = pathlib.Path("/ai/segmentation_datasets/nvos/scenes")
    output_root: pathlib.Path = pathlib.Path("nvos_results")
    device: str = "cuda"
    verbose: bool = False
    skip_existing_reconstruction: bool = True
    skip_existing_segmentation: bool = True
    segmentation_only: bool = False  # Skip reconstruction, use existing PLY files

    # Viewer settings
    viewer_port: int = 8080
    viewer_ip: str = "127.0.0.1"
    update_viz_every: float = -1.0  # -1 means no visualization

    reconstruction: ReconstructionSettings = field(default_factory=ReconstructionSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)


def get_nvos_scene_dirs(dataset_root: pathlib.Path) -> list[pathlib.Path]:
    """Find all NVOS scene directories (ending in _undistort)."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    scene_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir() and d.name.endswith("_undistort")])

    if not scene_dirs:
        raise ValueError(f"No scene directories found in {dataset_root}")

    return scene_dirs


def run_reconstruction(
    scene_path: pathlib.Path,
    output_ply_path: pathlib.Path,
    log_path: pathlib.Path,
    config: PipelineConfig,
    logger: logging.Logger,
) -> bool:
    """
    Run Gaussian splat reconstruction on a single scene.

    Args:
        scene_path: Path to the colmap scene directory.
        output_ply_path: Path to save the output PLY file.
        log_path: Path for reconstruction logs.
        config: Pipeline configuration.
        logger: Logger instance.

    Returns:
        True if reconstruction succeeded, False otherwise.
    """
    logger.info(f"Loading scene from {scene_path}")

    try:
        # Load the SfM scene (colmap format)
        sfm_scene = SfmScene.from_colmap(scene_path)
        logger.info(f"Loaded scene with {sfm_scene.num_images} images and {len(sfm_scene.points)} points")

        # Fix camera metadata to match actual image dimensions (important for undistorted images
        # where the undistortion process may have changed the image size)
        sfm_scene = fix_camera_metadata_from_images(sfm_scene, logger)

        # Apply scene transforms
        scene_transform = config.reconstruction.build_scene_transform()
        sfm_scene = scene_transform(sfm_scene)
        logger.info(f"After transforms: {sfm_scene.num_images} images and {len(sfm_scene.points)} points")

        # Set up visualization if enabled
        viz_scene = None
        if config.update_viz_every > 0:
            logger.info(f"Starting viewer server on {config.viewer_ip}:{config.viewer_port}")
            fviz.init(ip_address=config.viewer_ip, port=config.viewer_port, verbose=config.verbose)
            viz_scene = fviz.get_scene(f"Gaussian Splat Reconstruction - {scene_path.name}")

        # Configure reconstruction
        recon_config = GaussianSplatReconstructionConfig(
            max_epochs=config.reconstruction.max_epochs,
            refine_every_epoch=config.reconstruction.refine_every_epoch,
            refine_stop_epoch=config.reconstruction.refine_stop_epoch,
        )

        # Choose optimizer config based on settings
        if config.reconstruction.use_mcmc_optimizer:
            logger.info("Using MCMC optimizer")
            optimizer_config = GaussianSplatOptimizerMCMCConfig(
                max_gaussians=config.reconstruction.max_gaussians,
                insertion_rate=config.reconstruction.insertion_rate,
            )
        else:
            optimizer_config = GaussianSplatOptimizerConfig(
                max_gaussians=config.reconstruction.max_gaussians,
            )

        # Configure writer
        writer_config = GaussianSplatReconstructionWriterConfig(
            save_checkpoints=False,  # We only need the final PLY
            save_images=False,
            save_metrics=True,
        )
        writer = GaussianSplatReconstructionWriter(
            run_name=scene_path.name,
            save_path=log_path,
            config=writer_config,
            exist_ok=True,
        )

        # Create reconstruction runner
        runner = GaussianSplatReconstruction.from_sfm_scene(
            sfm_scene=sfm_scene,
            config=recon_config,
            optimizer_config=optimizer_config,
            writer=writer,
            viz_scene=viz_scene,
            use_every_n_as_val=-1,
            log_interval_steps=10,
            viz_update_interval_epochs=config.update_viz_every,
            device=config.device,
        )

        # Run optimization
        logger.info("Starting reconstruction optimization...")
        if viz_scene is not None:
            logger.info(f"Viewer running at http://{config.viewer_ip}:{config.viewer_port}")
            logger.info(f"Visualization updates every {config.update_viz_every} epoch(s)")
            fviz.show()
        runner.optimize()

        # Save the result
        output_ply_path.parent.mkdir(parents=True, exist_ok=True)
        runner.save_ply(output_ply_path)
        logger.info(f"Saved reconstruction to {output_ply_path}")

        return True

    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        if config.verbose:
            import traceback

            traceback.print_exc()
        return False


def run_segmentation_training(
    scene_path: pathlib.Path,
    reconstruction_path: pathlib.Path,
    output_path: pathlib.Path,
    log_path: pathlib.Path,
    config: PipelineConfig,
    logger: logging.Logger,
    exclude_images: list[str] | None = None,
) -> bool:
    """
    Run segmentation training on a reconstructed scene.

    Args:
        scene_path: Path to the colmap scene directory.
        reconstruction_path: Path to the reconstruction PLY file.
        output_path: Path to save the segmentation checkpoint.
        log_path: Path for segmentation training logs.
        config: Pipeline configuration.
        logger: Logger instance.
        exclude_images: Optional list of image filenames (without extension) to exclude
            from training. For NVOS evaluation, the GT mask images should be excluded.

    Returns:
        True if training succeeded, False otherwise.
    """
    logger.info(f"Loading reconstruction from {reconstruction_path}")

    try:
        # Load the GaussianSplat3D model
        gs_model, metadata = load_splats_from_file(reconstruction_path, config.device)
        normalization_transform = metadata.get("normalization_transform", None)
        logger.info(f"Loaded Gaussian splat with {gs_model.num_gaussians} gaussians")

        # Load the SfM scene
        sfm_scene = SfmScene.from_colmap(scene_path)
        logger.info(f"Loaded scene with {sfm_scene.num_images} images")

        # Fix camera metadata to match actual image dimensions (important for undistorted images)
        sfm_scene = fix_camera_metadata_from_images(sfm_scene, logger)

        # Get image INDICES to exclude BEFORE transforms (we need positional indices for cache consistency)
        # The mask cache uses positional indices, so we track indices not IDs
        exclude_image_indices = []
        if exclude_images:
            # Create a case-insensitive lookup set
            exclude_names_lower = {name.lower() for name in exclude_images}
            for i, img_meta in enumerate(sfm_scene.images):
                img_filename = pathlib.Path(img_meta.image_path).stem
                if img_filename.lower() in exclude_names_lower:
                    exclude_image_indices.append(i)
                    logger.info(f"Will exclude image '{img_filename}' (index={i}) from training")

        # Configure scene transforms for segmentation
        seg_transforms = [
            (
                TransformScene(normalization_transform.cpu().numpy())
                if normalization_transform is not None
                else Identity()
            ),
            PercentileFilterPoints(
                percentile_min=np.full((3,), config.reconstruction.points_percentile_filter),
                percentile_max=np.full((3,), 100.0 - config.reconstruction.points_percentile_filter),
            ),
            DownsampleImages(
                image_downsample_factor=config.segmentation.image_downsample_factor,
                rescaled_jpeg_quality=config.reconstruction.rescale_jpeg_quality,
            ),
            FilterImagesWithLowPoints(min_num_points=config.reconstruction.min_points_per_image),
            ComputeImageSegmentationMasksWithScales(
                gs3d=gs_model,
                checkpoint="large",
                points_per_side=40,
                pred_iou_thresh=0.80,
                stability_score_thresh=0.80,
                device=config.device,
            ),
        ]
        scene_transform = Compose(*seg_transforms)

        # Apply transforms
        sfm_scene = scene_transform(sfm_scene)
        logger.info(f"After transforms: {sfm_scene.num_images} images")

        # Note: We don't filter the scene here. Instead, we pass exclude_image_indices
        # to the runner so mask cache indices remain consistent.

        # Configure segmentation training
        seg_config = GaussianSplatSegmentationTrainingConfig(
            max_epochs=config.segmentation.max_epochs,
            batch_size=config.segmentation.batch_size,
            sample_pixels_per_image=config.segmentation.sample_pixels_per_image,
            eval_at_percent=config.segmentation.eval_at_percent,
            save_at_percent=[100],
        )

        # Configure writer
        writer_config = GaussianSplatSegmentationWriterConfig(
            save_checkpoints=True,
            save_images=False,
            save_metrics=True,
        )
        writer = GaussianSplatSegmentationWriter(
            run_name=scene_path.name,
            save_path=log_path,
            config=writer_config,
            exist_ok=True,
        )

        # Set up visualization if enabled
        viewer_update_interval = -1
        if config.update_viz_every > 0:
            logger.info(f"Starting viewer server on {config.viewer_ip}:{config.viewer_port}")
            fviz.init(ip_address=config.viewer_ip, port=config.viewer_port, verbose=config.verbose)
            viewer_update_interval = int(config.update_viz_every)

        # Create segmentation runner
        runner = GaussianSplatScaleConditionedSegmentation.new(
            sfm_scene=sfm_scene,
            gs_model=gs_model,
            gs_model_path=reconstruction_path,
            writer=writer,
            config=seg_config,
            device=config.device,
            use_every_n_as_val=config.segmentation.use_every_n_as_val,
            exclude_indices=exclude_image_indices if exclude_image_indices else None,
            viewer_update_interval_epochs=viewer_update_interval,
            log_interval_steps=config.segmentation.log_every,
        )

        # Run training
        logger.info("Starting segmentation training...")
        if viewer_update_interval > 0:
            logger.info(f"Viewer running at http://{config.viewer_ip}:{config.viewer_port}")
            logger.info(f"Visualization updates every {viewer_update_interval} epoch(s)")
            fviz.show()
        runner.train()

        # Save final checkpoint
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(runner.state_dict(), output_path)
        logger.info(f"Saved segmentation checkpoint to {output_path}")

        return True

    except Exception as e:
        logger.error(f"Segmentation training failed: {e}")
        if config.verbose:
            import traceback

            traceback.print_exc()
        return False


def run_pipeline(config: PipelineConfig, scene_filter: list[str] | None = None):
    """
    Run the full pipeline on all NVOS scenes.

    Args:
        config: Pipeline configuration.
        scene_filter: Optional list of scene names to process. If None, process all scenes.
    """
    # Set up logging
    log_level = logging.DEBUG if config.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("nvos_pipeline")

    # Get all scene directories
    scene_dirs = get_nvos_scene_dirs(config.dataset_root)
    logger.info(f"Found {len(scene_dirs)} NVOS scenes")

    # Filter scenes if requested
    if scene_filter:
        scene_dirs = [d for d in scene_dirs if d.name in scene_filter]
        logger.info(f"Filtered to {len(scene_dirs)} scenes: {[d.name for d in scene_dirs]}")

    # Create output directories
    config.output_root.mkdir(parents=True, exist_ok=True)
    reconstructions_dir = config.output_root / "reconstructions"
    segmentations_dir = config.output_root / "segmentations"
    recon_logs_dir = config.output_root / "logs" / "reconstruction"
    seg_logs_dir = config.output_root / "logs" / "segmentation"

    # Track results
    results = {"reconstruction": {}, "segmentation": {}}

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        logger.info("=" * 60)
        logger.info(f"Processing scene: {scene_name}")
        logger.info("=" * 60)

        # Define output paths
        ply_path = reconstructions_dir / f"{scene_name}.ply"
        seg_checkpoint_path = segmentations_dir / f"{scene_name}_segmentation.pt"

        # Step 1: Reconstruction
        if config.segmentation_only:
            # Skip reconstruction entirely, require existing PLY
            if not ply_path.exists():
                logger.error(f"Reconstruction file not found: {ply_path} (required for --segmentation-only)")
                results["reconstruction"][scene_name] = "missing"
                results["segmentation"][scene_name] = "skipped"
                continue
            logger.info(f"Using existing reconstruction: {ply_path}")
            results["reconstruction"][scene_name] = "existing"
        elif config.skip_existing_reconstruction and ply_path.exists():
            logger.info(f"Skipping reconstruction (already exists): {ply_path}")
            results["reconstruction"][scene_name] = "skipped"
        else:
            logger.info(f"Running reconstruction for {scene_name}...")
            success = run_reconstruction(
                scene_path=scene_dir,
                output_ply_path=ply_path,
                log_path=recon_logs_dir,
                config=config,
                logger=logger,
            )
            results["reconstruction"][scene_name] = "success" if success else "failed"

            if not success:
                logger.warning(f"Reconstruction failed for {scene_name}, skipping segmentation")
                results["segmentation"][scene_name] = "skipped"
                continue

        # Step 2: Segmentation training
        if config.skip_existing_segmentation and seg_checkpoint_path.exists() and not config.segmentation_only:
            logger.info(f"Skipping segmentation (already exists): {seg_checkpoint_path}")
            results["segmentation"][scene_name] = "skipped"
        else:
            if not ply_path.exists():
                logger.warning(f"Reconstruction file not found: {ply_path}, skipping segmentation")
                results["segmentation"][scene_name] = "skipped"
                continue

            logger.info(f"Running segmentation training for {scene_name}...")

            # Get the GT mask image to exclude from training (per NVOS evaluation protocol)
            exclude_images = []
            if scene_name in NVOS_GT_MASK_IMAGES:
                exclude_images.append(NVOS_GT_MASK_IMAGES[scene_name])
                logger.info(f"Will exclude GT mask image '{NVOS_GT_MASK_IMAGES[scene_name]}' from training")

            success = run_segmentation_training(
                scene_path=scene_dir,
                reconstruction_path=ply_path,
                output_path=seg_checkpoint_path,
                log_path=seg_logs_dir,
                config=config,
                logger=logger,
                exclude_images=exclude_images if exclude_images else None,
            )
            results["segmentation"][scene_name] = "success" if success else "failed"

    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE - Summary:")
    logger.info("=" * 60)
    logger.info("Reconstruction results:")
    for scene, status in results["reconstruction"].items():
        logger.info(f"  {scene}: {status}")
    logger.info("Segmentation results:")
    for scene, status in results["segmentation"].items():
        logger.info(f"  {scene}: {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GARfVDB pipeline on NVOS dataset scenes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=pathlib.Path("/media/datasets/segmentation_evaluation/nvos/scenes"),
        help="Path to the NVOS scenes directory",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path("nvos_results"),
        help="Path to store output files (reconstructions, checkpoints, logs)",
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
        help="Specific scene names to process (e.g., fern_undistort flower_undistort). If not specified, all scenes are processed.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run both reconstruction and segmentation even if output files already exist",
    )
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="Skip reconstruction and use existing PLY files from the reconstructions directory. "
        "Forces segmentation to run even if checkpoint exists.",
    )
    parser.add_argument(
        "--rerun-segmentation",
        action="store_true",
        help="Re-run segmentation even if checkpoint exists (but still skip reconstruction if PLY exists)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )

    # Viewer settings
    parser.add_argument(
        "--viewer-port",
        "-p",
        type=int,
        default=8080,
        help="Port for the interactive viewer server",
    )
    parser.add_argument(
        "--viewer-ip",
        "-ip",
        type=str,
        default=get_lan_ip(),
        help="IP address for the interactive viewer server (default: auto-detected LAN IP)",
    )
    parser.add_argument(
        "--update-viz-every",
        "-uv",
        type=float,
        default=-1.0,
        help="Update the viewer every N epochs during reconstruction (-1 to disable viewer)",
    )

    # Reconstruction settings
    parser.add_argument(
        "--recon-epochs",
        type=int,
        default=300,
        help="Number of epochs for reconstruction",
    )
    parser.add_argument(
        "--refine-stop-epoch",
        type=int,
        default=100,
        help="At which epoch to stop refining Gaussians (inserting/deleting based on optimization)",
    )
    parser.add_argument(
        "--recon-image-downsample",
        type=int,
        default=1,
        help="Image downsample factor for reconstruction",
    )
    parser.add_argument(
        "--use-mcmc-optimizer",
        action="store_true",
        help="Use the MCMC optimizer instead of the standard optimizer for reconstruction",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=-1,
        help="Maximum number of Gaussians for reconstruction (-1 for no limit)",
    )
    parser.add_argument(
        "--insertion-rate",
        type=float,
        default=1.05,
        help="Insertion rate for MCMC optimizer (e.g., 1.05 means 5%% more Gaussians per refinement step)",
    )

    # Segmentation settings
    parser.add_argument(
        "--seg-epochs",
        type=int,
        default=300,
        help="Number of epochs for segmentation training",
    )
    parser.add_argument(
        "--seg-image-downsample",
        type=int,
        default=2,
        help="Image downsample factor for segmentation training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for segmentation training",
    )
    parser.add_argument(
        "--seg-eval-at-percent",
        nargs="+",
        type=int,
        default=[100],
        help="Percentages of total epochs at which to run evaluation during segmentation training",
    )

    args = parser.parse_args()

    # Build configuration
    skip_all = not args.no_skip_existing
    config = PipelineConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        device=args.device,
        verbose=args.verbose,
        skip_existing_reconstruction=skip_all,
        skip_existing_segmentation=skip_all and not args.rerun_segmentation,
        segmentation_only=args.segmentation_only,
        viewer_port=args.viewer_port,
        viewer_ip=args.viewer_ip,
        update_viz_every=args.update_viz_every,
        reconstruction=ReconstructionSettings(
            max_epochs=args.recon_epochs,
            refine_stop_epoch=args.refine_stop_epoch,
            image_downsample_factor=args.recon_image_downsample,
            use_mcmc_optimizer=args.use_mcmc_optimizer,
            max_gaussians=args.max_gaussians,
            insertion_rate=args.insertion_rate,
        ),
        segmentation=SegmentationSettings(
            max_epochs=args.seg_epochs,
            batch_size=args.batch_size,
            image_downsample_factor=args.seg_image_downsample,
            eval_at_percent=args.seg_eval_at_percent,
        ),
    )

    run_pipeline(config, args.scenes)


if __name__ == "__main__":
    main()
