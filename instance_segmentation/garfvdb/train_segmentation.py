# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import threading
import time
from typing import Callable, Literal

import cv2
import fvdb.viz as fviz
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
    camera_check_interval: float = 0.5,
    overlay_width: int = 1920,
    overlay_height: int = 1080,
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
            directory containing the `images` and `sparse` subdirectories. For "simple_directory" datasets,
            this should be the directory containing the images and a `cameras.txt` file. For "e57" datasets,
            this should be the path to the E57 file or directory.
        reconstruction_path (pathlib.Path): Path to the precomputed Gaussian splat reconstruction to load.
        config (GaussianSplatSegmentationTrainingConfig): Configuration for the Gaussian splat segmentation training.
        tx (SfmSceneSegmentationTransformConfig): Configuration for the transforms to apply to the SfmScene before training.
        io (GaussianSplatSegmentationWriterConfig): Configuration for saving segmentation metrics and checkpoints.
        dataset_type (Literal["colmap", "simple_directory", "e57"]): Type of dataset to load.
        run_name (str | None): Name of the run. If None, a name will be generated based on the current date and time.
        log_path (pathlib.Path | None): Path to log metrics and checkpoints. If None, no metrics or checkpoints will be saved.
        use_every_n_as_val (int): Use every n-th image as a validation image. If -1, do not use a separate validation split.
        device (str | torch.device): Device to use for training.
        log_every (int): Log training metrics every n steps.
        visualize_every (int): Update the viewer every n epochs. If -1, do not visualize.
        viewer_port (int): The port to expose the viewer server on if visualize_every > 0.
        viewer_ip_address (str): The IP address to expose the viewer server on if visualize_every > 0.
        camera_check_interval (float): How often to check for camera changes (seconds). The segmentation
            overlay will update when the viewer camera moves.
        overlay_width (int): Width of the segmentation overlay in the viewer.
        overlay_height (int): Height of the segmentation overlay in the viewer.
        overlay_downsample (int): Downsample factor for rendering. Renders at overlay_size / overlay_downsample
            and then scales up for better performance.
        mask_scale (float): Fraction of scene max scale to use for rendering segmentation masks (0.1 = 10%).
        mask_blend (float): Blend factor for the segmentation overlay (0.0 = transparent, 1.0 = opaque).
        verbose (bool): Whether to log debug messages.
        cache_dataset (bool): If True, cache images and masks in memory to speed up data loading.
            Set to False to reduce memory usage for large datasets. Default is True.
    """
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")
    logger = logging.getLogger(__name__)

    # Validate camera_check_interval when visualization is enabled
    if visualize_every > 0 and camera_check_interval <= 0:
        raise ValueError(
            f"camera_check_interval must be positive when visualize_every > 0, got {camera_check_interval}"
        )

    # Load the SfmScene
    sfm_scene = load_sfm_scene(sfm_dataset_path, dataset_type)

    # Load the GaussianSplat3D model
    gs_model, metadata = load_splats_from_file(reconstruction_path, device)
    normalization_transform = metadata.get("normalization_transform", None)

    # Set up visualization if enabled
    viz_scene = None
    viz_callback: Callable[[GaussianSplatScaleConditionedSegmentation, int], None] | None = None

    if visualize_every > 0:
        logger.info(f"Starting viewer server on {viewer_ip_address}:{viewer_port}")
        fviz.init(ip_address=viewer_ip_address, port=viewer_port, verbose=verbose)
        viz_scene = fviz.get_scene("Gaussian Splat Segmentation Training")

        # Add the Gaussian splats to the scene
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

        # Get projection matrix from metadata for camera intrinsics
        projection_matrices = metadata.get("projection_matrices", None)
        image_sizes = metadata.get("image_sizes", None)

        # Compute render dimensions (smaller for performance)
        render_w = overlay_width // overlay_downsample
        render_h = overlay_height // overlay_downsample

        # Try to set up the segmentation overlay image at full resolution
        image_view = None
        try:
            initial_image = np.zeros((overlay_height, overlay_width, 4), dtype=np.uint8)
            initial_image[..., 3] = 128  # Semi-transparent
            image_view = viz_scene.add_image(
                name="Segmentation Overlay",
                width=overlay_width,
                height=overlay_height,
                rgba_image=initial_image.flatten(),
            )
            logger.info(f"Segmentation overlay: {overlay_width}x{overlay_height} (render: {render_w}x{render_h})")
        except Exception as e:
            logger.warning(f"Failed to set up segmentation overlay: {e}")
            logger.info("Running without segmentation overlay")

        # Get reference projection for intrinsics from metadata and scale for render resolution
        reference_projection = None
        if projection_matrices is not None and image_sizes is not None:
            orig_projection = projection_matrices[0].float()
            orig_w = float(image_sizes[0, 0].item())
            orig_h = float(image_sizes[0, 1].item())
            # Scale the projection matrix to the render resolution
            # fx, fy scale with resolution, cx, cy scale with resolution
            scale_x = render_w / orig_w
            scale_y = render_h / orig_h
            scaled_projection = orig_projection.clone()
            scaled_projection[0, 0] *= scale_x  # fx
            scaled_projection[1, 1] *= scale_y  # fy
            scaled_projection[0, 2] *= scale_x  # cx
            scaled_projection[1, 2] *= scale_y  # cy
            reference_projection = scaled_projection.to(device)

        # OpenGL to OpenCV conversion matrix
        opengl_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

        def get_camera_from_viewer() -> torch.Tensor | None:
            """Compute camera-to-world matrix from the viewer's orbit camera state.

            Constructs a 4x4 transformation matrix from the viewer's current
            orbit parameters (center, direction, radius, up vector) and converts
            it from OpenGL to OpenCV convention for use with the segmentation model.

            Returns:
                torch.Tensor | None: A 4x4 camera-to-world matrix in OpenCV
                convention, or ``None`` if the camera state cannot be retrieved.
            """
            try:
                # Get orbit camera state from viewer
                center = viz_scene.camera_orbit_center.cpu().numpy().copy()
                eye_direction = viz_scene.camera_orbit_direction.cpu().numpy().copy()
                radius = viz_scene.camera_orbit_radius
                up_world = viz_scene.camera_up_direction.cpu().numpy().copy()

                # Camera position: center - eye_direction * radius
                position = center - eye_direction * radius

                # Forward = eye_direction (already the look direction)
                forward = eye_direction / np.linalg.norm(eye_direction)

                # Right vector = forward × up_world
                right = np.cross(forward, up_world)
                right = right / np.linalg.norm(right)

                # Up vector = right × forward
                up = np.cross(right, forward)
                up = up / np.linalg.norm(up)

                # Build OpenGL-style camera-to-world (X-right, Y-up, Z-backward)
                c2w_opengl = np.eye(4, dtype=np.float32)
                c2w_opengl[:3, 0] = right
                c2w_opengl[:3, 1] = up
                c2w_opengl[:3, 2] = -forward  # OpenGL: camera looks along -Z
                c2w_opengl[:3, 3] = position

                # Convert to OpenCV convention for the segmentation model
                c2w_opencv = c2w_opengl @ opengl_to_opencv
                return torch.from_numpy(c2w_opencv).float().to(device)
            except Exception as e:
                logger.debug(f"Failed to get camera from viewer: {e}")
                return None

        def render_segmentation_overlay(model: GARfVDBModel, log_prefix: str = "") -> None:
            """Render a segmentation mask overlay in the interactive viewer.

            Captures the current viewer camera pose, renders the model's mask
            features at a reduced resolution, applies PCA to produce an RGB
            visualization, and updates the viewer's overlay image.

            Args:
                model: The segmentation model to render.
                log_prefix: Optional message to log on successful render.
            """
            if image_view is None or reference_projection is None:
                return

            camera_to_world = get_camera_from_viewer()
            if camera_to_world is None:
                return

            try:
                from garfvdb.training.dataset import GARfVDBInput

                # Render at lower resolution for performance
                model_input = GARfVDBInput(
                    {
                        "projection": reference_projection.unsqueeze(0),
                        "camera_to_world": camera_to_world.unsqueeze(0),
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
                        logger.warning("Invalid values in mask features, skipping visualization update")
                        return

                    # Apply PCA projection to get RGB visualization
                    mask_pca = pca_projection_fast(mask_features_output, 3, mask=mask_alpha.squeeze(-1) > 0)[0]

                    # Blend mask with alpha
                    blended_alpha = mask_alpha[0] * mask_blend

                    rgba = np.concatenate(
                        [mask_pca.detach().cpu().numpy(), blended_alpha.detach().cpu().numpy()], axis=-1
                    )
                    rgba_uint8 = (rgba.clip(0.0, 1.0) * 255).astype(np.uint8)

                    # Scale up to overlay resolution
                    if overlay_downsample > 1:
                        rgba_uint8 = cv2.resize(
                            rgba_uint8, (overlay_width, overlay_height), interpolation=cv2.INTER_LINEAR
                        )

                    image_view.update(rgba_uint8.flatten())
                    if log_prefix:
                        logger.debug(f"{log_prefix}")

            except Exception as e:
                logger.warning(f"Error updating visualization: {e}")

        # Create visualization callback that uses the viewer camera
        def update_visualization(runner: GaussianSplatScaleConditionedSegmentation, epoch: int) -> None:
            """Update the visualization overlay during training from current viewer camera."""
            render_segmentation_overlay(runner.model, f"Updated segmentation visualization at epoch {epoch}")

        viz_callback = update_visualization

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
        viz_callback=viz_callback,
        cache_dataset=cache_dataset,
    )

    # Set up camera tracking thread for interactive visualization
    camera_thread = None
    stop_camera_thread = threading.Event()

    if viz_scene is not None and image_view is not None:
        # Camera state for change detection
        prev_center = None
        prev_direction = None
        prev_radius = None
        prev_up = None

        def camera_changed() -> bool:
            """Determine whether the interactive viewer camera state has changed.

            Compares the current camera orbit parameters (center, direction,
            radius, and up vector) from ``viz_scene`` against the previously
            observed values stored in the nonlocal variables ``prev_center``,
            ``prev_direction``, ``prev_radius``, and ``prev_up``.

            Notes:
                This function updates the ``prev_*`` variables when called for
                the first time or when a change is detected.

            Returns:
                bool: ``True`` if this is the first call (to force an initial
                overlay update) or if any of the tracked camera parameters
                differ from the previously stored values; ``False`` otherwise
                or if an error occurs while querying the camera state.
            """
            nonlocal prev_center, prev_direction, prev_radius, prev_up
            try:
                center = viz_scene.camera_orbit_center
                direction = viz_scene.camera_orbit_direction
                radius = viz_scene.camera_orbit_radius
                up = viz_scene.camera_up_direction

                # First time - always update
                if prev_center is None:
                    prev_center = center
                    prev_direction = direction
                    prev_radius = radius
                    prev_up = up
                    return True

                changed = (
                    not torch.allclose(center, prev_center)
                    or not torch.allclose(direction, prev_direction)  # type: ignore
                    or radius != prev_radius
                    or not torch.allclose(up, prev_up)  # type: ignore
                )

                if changed:
                    prev_center = center
                    prev_direction = direction
                    prev_radius = radius
                    prev_up = up

                return changed
            except Exception as exc:
                logger.debug(
                    "Failed to retrieve camera state in camera_changed: %s",
                    exc,
                    exc_info=True,
                )
                return False

        def camera_monitor_loop() -> None:
            """Monitor camera changes in a background thread and trigger overlay updates.

            This function is intended to be run in a dedicated daemon thread started
            from :func:`main`. It periodically checks the interactive viewer camera
            state and, when a change is detected, re-renders the segmentation overlay
            for the current model.

            The loop runs until the ``stop_camera_thread`` :class:`threading.Event`
            is set (e.g. in the main thread's shutdown/KeyboardInterrupt handler),
            at which point the background thread exits cleanly.

            While :meth:`runner.train` executes in the main thread, this monitor
            runs concurrently to keep the visualization in sync with user-driven
            camera movements, and it continues to run during the post-training
            viewing phase until shutdown.
            """
            while not stop_camera_thread.is_set():
                time.sleep(camera_check_interval)
                if camera_changed():
                    render_segmentation_overlay(runner.model, "Updated segmentation overlay from camera")

        # Start the camera monitoring thread
        camera_thread = threading.Thread(target=camera_monitor_loop, daemon=True)
        camera_thread.start()
        logger.info(f"Camera tracking enabled (checking every {camera_check_interval}s)")

    if viz_scene is not None:
        logger.info("=" * 60)
        logger.info(f"Viewer running at http://{viewer_ip_address}:{viewer_port}")
        logger.info(f"Visualization updates every {visualize_every} epoch(s)")
        if camera_thread is not None:
            logger.info(f"Camera tracking updates every {camera_check_interval}s")
        logger.info("=" * 60)
        fviz.show()

    runner.train()

    if viz_scene is not None:
        logger.info("Training complete. Viewer running... Ctrl+C to exit.")
        try:
            # Camera thread continues running during post-training viewing
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            if camera_thread is not None:
                stop_camera_thread.set()
                camera_thread.join(timeout=5.0)
            logger.info("Shutting down...")


if __name__ == "__main__":
    tyro.cli(main)
