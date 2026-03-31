# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time
from typing import Literal

import cv2
import numpy as np
import torch
import tyro
from garfvdb.config import (
    GaussianSplatSegmentationTrainingConfig,
    SfmSceneSegmentationTransformConfig,
)
from garfvdb.model import GARfVDBModel
from garfvdb.training.segmentation import GaussianSplatScaleConditionedSegmentation
from garfvdb.training.segmentation_writer import (
    GaussianSplatSegmentationWriter,
    GaussianSplatSegmentationWriterConfig,
)
from garfvdb.util import load_sfm_scene, load_splats_from_file, pca_projection_fast


def main(
    sfm_dataset_path: pathlib.Path,
    reconstruction_path: pathlib.Path,
    config: GaussianSplatSegmentationTrainingConfig = GaussianSplatSegmentationTrainingConfig(),
    tx: SfmSceneSegmentationTransformConfig = SfmSceneSegmentationTransformConfig(),
    io: GaussianSplatSegmentationWriterConfig = GaussianSplatSegmentationWriterConfig(),
    dataset_type: Literal["colmap", "simple_directory", "e57"] = "colmap",
    run_name: str | None = None,
    log_path: pathlib.Path | None = pathlib.Path("garfvdb_logs"),
    use_every_n_as_val: int = -1,
    device: str | torch.device = "cuda",
    log_every: int = 10,
    visualize_every: int = -1,
    viewer_port: int = 8080,
    viewer_ip_address: str = "127.0.0.1",
    overlay_width: int = 1440,
    overlay_height: int = 720,
    overlay_downsample: int = 2,
    mask_scale: float = 0.1,
    mask_blend: float = 0.5,
    verbose: bool = False,
    cache_dataset: bool = True,
):
    """
    Train a Gaussian splat segmentation model from a set of images and camera poses.

    Args:
        sfm_dataset_path (pathlib.Path): Path to the dataset. For "colmap" datasets, this should be the
            directory containing the ``images`` and ``sparse`` subdirectories. For "simple_directory"
            datasets, this should be the directory containing the images and a ``cameras.txt`` file.
            For "e57" datasets, this should be the path to the E57 file or directory.
        reconstruction_path (pathlib.Path): Path to the precomputed Gaussian splat reconstruction to load.
        config (GaussianSplatSegmentationTrainingConfig): Configuration for the Gaussian splat
            segmentation training.
        tx (SfmSceneSegmentationTransformConfig): Configuration for the transforms to apply to the
            SfmScene before training.
        io (GaussianSplatSegmentationWriterConfig): Configuration for saving segmentation metrics
            and checkpoints.
        dataset_type (Literal["colmap", "simple_directory", "e57"]): Type of dataset to load.
        run_name (str | None): Name of the run. If None, a name will be generated based on the
            current date and time.
        log_path (pathlib.Path | None): Path to log metrics and checkpoints. If None, no metrics
            or checkpoints will be saved.
        use_every_n_as_val (int): Use every n-th image as a validation image. If -1, do not use
            a separate validation split.
        device (str | torch.device): Device to use for training.
        log_every (int): Log training metrics every n steps.
        visualize_every (int): Update the viewer every n epochs. If -1, do not visualize.
        viewer_port (int): The port to expose the viewer server on if ``visualize_every > 0``.
        viewer_ip_address (str): The IP address to expose the viewer server on if
            ``visualize_every > 0``.
        overlay_width (int): Width of the segmentation overlay in the viewer. Must match
            the nanovdb-editor viewport width for correct alignment (default 1440).
        overlay_height (int): Height of the segmentation overlay in the viewer. Must match
            the nanovdb-editor viewport height for correct alignment (default 720).
        overlay_downsample (int): Downsample factor for rendering. Renders at
            ``overlay_size / overlay_downsample`` and then scales up for better performance.
        mask_scale (float): Fraction of scene max scale to use for rendering segmentation masks
            (0.1 = 10%).
        mask_blend (float): Blend factor for the segmentation overlay (0.0 = transparent,
            1.0 = opaque).
        verbose (bool): Whether to log debug messages.
        cache_dataset (bool): If True, cache images and masks in memory to speed up data loading.
            Set to False to reduce memory usage for large datasets. Default is True.
    """
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s", force=True)
    logger = logging.getLogger(__name__)

    viewer_enabled = visualize_every > 0
    # ---- Initialize viewer BEFORE any CUDA operations ----
    # The Vulkan device created by fviz.init() must be set up before the CUDA
    # context is first created (by load_splats_from_file).  This matches the
    # initialization order used in frgs reconstruction.
    if viewer_enabled:
        import fvdb.viz as fviz
        fviz.init(ip_address=viewer_ip_address, port=viewer_port, verbose=verbose)


    # ---- Load data ----
    sfm_scene = load_sfm_scene(sfm_dataset_path, dataset_type)
    gs_model, metadata = load_splats_from_file(reconstruction_path, device)
    normalization_transform = metadata.get("normalization_transform", None)

    # ---- Create the runner (SAM2 masks + model init) ----
    writer = GaussianSplatSegmentationWriter(run_name=run_name, save_path=log_path, config=io, exist_ok=False)
    runner = GaussianSplatScaleConditionedSegmentation.new(
        sfm_scene=tx.build_scene_transforms(gs_model, normalization_transform)(sfm_scene),
        gs_model=gs_model,
        gs_model_path=reconstruction_path,
        writer=writer,
        config=config,
        device=device,
        use_every_n_as_val=use_every_n_as_val,
        viewer_update_interval_epochs=visualize_every,
        log_interval_steps=log_every,
        viz_callback=None,
        cache_dataset=cache_dataset,
    )

    # ---- Start the viewer ----
    viz_scene = None
    image_view = None
    if viewer_enabled:
        import fvdb.viz as fviz

        viz_scene = fviz.get_scene("Gaussian Splat Segmentation Training")
        viz_scene.add_gaussian_splat_3d("Gaussian Splats", gs_model)

        # Set up initial camera position based on scene geometry
        scene_centroid = gs_model.means.mean(dim=0).detach().cpu().numpy()
        cam_to_world_matrices = metadata.get("camera_to_world_matrices", None)
        if cam_to_world_matrices is not None:
            initial_camera_position = cam_to_world_matrices[0, :3, 3].detach().cpu().numpy()
        else:
            scene_radius = (
                gs_model.means.detach().max(dim=0).values - gs_model.means.detach().min(dim=0).values
            ).max().item() / 2.0
            initial_camera_position = scene_centroid + np.ones(3) * scene_radius

        viz_scene.set_camera_lookat(
            eye=initial_camera_position,
            center=scene_centroid,
            up=[0, 0, 1],
        )

        # Add the segmentation overlay image
        initial_image = np.zeros((overlay_height, overlay_width, 4), dtype=np.uint8)
        initial_image[..., 3] = 128
        try:
            image_view = viz_scene.add_image(
                name="Segmentation Overlay",
                width=overlay_width,
                height=overlay_height,
                rgba_image=initial_image.flatten(),
            )
        except Exception:
            pass

        fviz.show()
        logger.info("=" * 60)
        logger.info(f"Viewer running at http://{viewer_ip_address}:{viewer_port}")
        logger.info(f"Visualization updates every {visualize_every} epoch(s)")
        logger.info("=" * 60)

    # ---- Build the overlay helpers used by the viz_callback ----
    # Compute render dimensions (smaller for performance)
    render_w = overlay_width // overlay_downsample
    render_h = overlay_height // overlay_downsample

    # Build intrinsic matrix from the viewer's FOV to match its perspective
    # camera.  Recomputed whenever the FOV changes.
    cx = render_w / 2.0
    cy = render_h / 2.0
    cached_fov: float | None = None
    reference_projection: torch.Tensor | None = None

    def _update_projection(fov_y_rad: float) -> torch.Tensor:
        nonlocal cached_fov, reference_projection
        fy = render_h / (2.0 * np.tan(fov_y_rad / 2.0))
        reference_projection = torch.tensor(
            [[fy, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).to(device)
        cached_fov = fov_y_rad
        return reference_projection

    # OpenGL to OpenCV conversion matrix
    opengl_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    def _camera_tuple_to_c2w(
        center: np.ndarray,
        eye_direction: np.ndarray,
        radius: float,
        up_world: np.ndarray,
    ) -> torch.Tensor:
        """Convert orbit camera parameters to a 4x4 camera-to-world matrix.

        Constructs a camera-to-world transformation matrix from the viewer's
        orbit parameters (center, direction, radius, up vector) and converts
        it from OpenGL to OpenCV convention for use with the segmentation model.

        Returns:
            A 4x4 camera-to-world matrix in OpenCV convention on ``device``.
        """
        # Camera position: center - eye_direction * radius
        position = center - eye_direction * radius

        # Forward = eye_direction (already the look direction)
        forward = eye_direction / np.linalg.norm(eye_direction)

        # Right vector = forward x up_world
        right = np.cross(forward, up_world)
        right = right / np.linalg.norm(right)

        # Up vector = right x forward
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Build OpenGL-style camera-to-world (X-right, Y-up, Z-backward)
        c2w_gl = np.eye(4, dtype=np.float32)
        c2w_gl[:3, 0] = right
        c2w_gl[:3, 1] = up
        c2w_gl[:3, 2] = -forward  # OpenGL: camera looks along -Z
        c2w_gl[:3, 3] = position

        # Convert to OpenCV convention for the segmentation model
        c2w_cv = c2w_gl @ opengl_to_opencv
        return torch.from_numpy(c2w_cv).float().to(device)

    def render_overlay(model: GARfVDBModel, camera_to_world: torch.Tensor) -> np.ndarray | None:
        """Render the segmentation overlay for a given camera pose.

        Returns an ``(H, W, 4)`` uint8 RGBA image, or ``None`` on failure.
        """
        if reference_projection is None:
            return None
        try:
            from garfvdb.training.dataset import GARfVDBInput

            # Render at lower resolution for performance
            try:
                world_to_camera = torch.linalg.inv(camera_to_world).contiguous()
            except torch.linalg.LinAlgError:
                return None
            model_input = GARfVDBInput(
                {
                    "projection": reference_projection.unsqueeze(0),
                    "camera_to_world": camera_to_world.unsqueeze(0).contiguous(),
                    "world_to_camera": world_to_camera.unsqueeze(0),
                    "image_w": [render_w],
                    "image_h": [render_h],
                }
            )

            # Render at configured fraction of max scale
            max_scale_tensor = model.max_grouping_scale
            desired_scale = float(max_scale_tensor.item()) * mask_scale

            with torch.no_grad():
                mask_features_output, mask_alpha = model.get_mask_output(model_input, desired_scale)

                # Check for invalid values
                if torch.isnan(mask_features_output).any() or torch.isinf(mask_features_output).any():
                    return None

                # Apply PCA projection to get RGB visualization
                mask_pca = pca_projection_fast(mask_features_output, 3, mask=mask_alpha.squeeze(-1) > 0)[0]

                # Blend mask with alpha
                blended_alpha = mask_alpha[0] * mask_blend
                rgba = np.concatenate([mask_pca.detach().cpu().numpy(), blended_alpha.detach().cpu().numpy()], axis=-1)
                rgba_uint8 = (rgba.clip(0.0, 1.0) * 255).astype(np.uint8)

                # Scale up to overlay resolution
                if overlay_downsample > 1:
                    rgba_uint8 = cv2.resize(rgba_uint8, (overlay_width, overlay_height), interpolation=cv2.INTER_LINEAR)
                return rgba_uint8
        except Exception as exc:
            logger.warning(f"Error rendering overlay: {exc}")
            return None

    def get_viewer_camera() -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
        """Read the current orbit camera state directly from the viewer."""
        if viz_scene is None:
            return None
        try:
            return (
                viz_scene.camera_orbit_center.cpu().numpy(),
                viz_scene.camera_orbit_direction.cpu().numpy(),
                float(viz_scene.camera_orbit_radius),
                viz_scene.camera_up_direction.cpu().numpy(),
            )
        except Exception:
            return None

    def update_visualization(runner_arg: GaussianSplatScaleConditionedSegmentation, epoch: int) -> None:
        """Viz callback invoked at epoch boundaries to update the segmentation overlay."""
        if image_view is None or viz_scene is None:
            return
        cam = get_viewer_camera()
        if cam is None:
            return
        fov_y_rad = viz_scene.camera_fov
        if reference_projection is None or fov_y_rad != cached_fov:
            _update_projection(fov_y_rad)
        c2w = _camera_tuple_to_c2w(*cam)
        frame = render_overlay(runner_arg.model, c2w)
        if frame is not None:
            image_view.update(frame.flatten())
            logger.debug(f"Updated segmentation overlay at epoch {epoch}")

    if image_view is not None:
        runner._viz_callback = update_visualization

    # ---- Train ----
    runner.train()

    # ---- Post-training interactive viewing ----
    if viz_scene is not None:
        logger.info("Training complete. Viewer running... Ctrl+C to exit.")
        try:
            while True:
                cam = get_viewer_camera()
                if cam is not None:
                    fov_y_rad = viz_scene.camera_fov
                    if reference_projection is None or fov_y_rad != cached_fov:
                        _update_projection(fov_y_rad)
                    c2w = _camera_tuple_to_c2w(*cam)
                    frame = render_overlay(runner.model, c2w)
                    if frame is not None and image_view is not None:
                        image_view.update(frame.flatten())
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    tyro.cli(main)
