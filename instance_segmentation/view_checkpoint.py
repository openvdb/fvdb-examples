# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Annotated

import cv2
import fvdb.viz as fviz
import numpy as np
import torch
import tyro
from fvdb import GaussianSplat3d
from fvdb.types import to_Mat33fBatch, to_Mat44fBatch, to_Vec2iBatch
from fvdb_reality_capture.tools import filter_splats_above_scale
from tyro.conf import arg

from garfvdb.model import GARfVDBModel
from garfvdb.training.dataset import GARfVDBInput
from garfvdb.training.segmentation import GaussianSplatScaleConditionedSegmentation
from garfvdb.util import (
    calculate_pca_projection,
    load_splats_from_file,
    pca_projection_fast,
)


def load_segmentation_runner_from_checkpoint(
    checkpoint_path: pathlib.Path,
    gs_model: GaussianSplat3d,
    gs_model_path: pathlib.Path,
    device: str | torch.device = "cuda",
) -> GaussianSplatScaleConditionedSegmentation:
    """Load a segmentation runner from a checkpoint file.

    Restores the complete training state including the transformed SfmScene
    (with correct scale statistics), the GARfVDB segmentation model, and
    the GaussianSplat3d model.

    Args:
        checkpoint_path: Path to the segmentation checkpoint (.pt or .pth).
        gs_model: GaussianSplat3d model for the scene.
        gs_model_path: Path to the GaussianSplat3d model file.
        device: Device to load the model onto.

    Returns:
        Loaded runner with access to ``model``, ``gs_model``, ``sfm_scene``,
        and ``config`` attributes.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    runner = GaussianSplatScaleConditionedSegmentation.from_state_dict(
        state_dict=checkpoint,
        gs_model=gs_model,
        gs_model_path=gs_model_path,
        device=device,
        eval_only=True,
    )

    torch.cuda.empty_cache()

    return runner


class SegmentationRenderer:
    """Renderer for visualizing segmentation masks from a GARfVDB model.

    Renders the segmentation model's mask features at a specified scale and
    projects them to RGB using PCA for visualization.

    Attributes:
        gs_model: The underlying Gaussian splat model.
        segmentation_model: The GARfVDB segmentation model.
        device: Device for computation.
        scale: Current rendering scale (world-space units).
        mask_blend: Blend factor for mask overlay (0=transparent, 1=opaque).
        freeze_pca: If True, use frozen PCA projection for consistent colors.
        frozen_pca_projection: Cached PCA projection matrix when freeze_pca=True.
    """

    def __init__(
        self,
        gs_model: GaussianSplat3d,
        segmentation_model: GARfVDBModel,
        device: torch.device,
    ) -> None:
        """Initialize the segmentation renderer.

        Args:
            gs_model: The Gaussian splat model for the scene.
            segmentation_model: The trained GARfVDB segmentation model.
            device: Device for computation (e.g., cuda).
        """
        self.gs_model = gs_model
        self.segmentation_model = segmentation_model
        self.device = device

        # Renderer state
        self.scale = float(segmentation_model.max_grouping_scale.item()) * 0.1
        self.mask_blend = 0.5
        self.freeze_pca = False
        self.frozen_pca_projection: torch.Tensor | None = None

        self._logger = logging.getLogger(__name__)

    def _apply_pca_projection(
        self,
        features: torch.Tensor,
        n_components: int = 3,
        valid_feature_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply PCA projection, either using frozen parameters or computing fresh ones."""
        if self.freeze_pca and self.frozen_pca_projection is not None:
            # Use frozen PCA projection matrix
            return pca_projection_fast(features, n_components, V=self.frozen_pca_projection, mask=valid_feature_mask)
        else:
            # Compute fresh PCA
            try:
                result = pca_projection_fast(features, n_components, mask=valid_feature_mask)

                # Store projection matrix if freezing is enabled
                if self.freeze_pca:
                    if valid_feature_mask is not None:
                        features = features[valid_feature_mask]
                    self.frozen_pca_projection = calculate_pca_projection(features, n_components, center=True)

                return result
            except RuntimeError as e:
                if "failed to converge" in str(e):
                    # Fallback: return zeros with correct shape
                    self._logger.warning("PCA failed to converge, returning zero projection")
                    B, H, W, C = features.shape
                    return torch.zeros(B, H, W, n_components, device=features.device, dtype=features.dtype)
                else:
                    raise e

    @torch.no_grad()
    def render_segmentation_image(
        self,
        camera_to_world: torch.Tensor,
        projection: torch.Tensor,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """
        Render an RGBA segmentation image for use with fvdb.viz add_image API.

        Args:
            camera_to_world: Camera to world transformation matrix [4, 4].
            projection: Projection matrix [3, 3].
            img_w: Image width.
            img_h: Image height.

        Returns:
            RGBA segmentation image as uint8 numpy array [H, W, 4].
        """
        # Create input for the model
        model_input = GARfVDBInput(
            {
                "projection": projection.unsqueeze(0),
                "camera_to_world": camera_to_world.unsqueeze(0),
                "image_w": [img_w],
                "image_h": [img_h],
            }
        )

        try:
            # Render the mask features
            mask_features_output, mask_alpha = self.segmentation_model.get_mask_output(model_input, self.scale)

            # Debug: Check for invalid values in render output
            has_nan = torch.isnan(mask_features_output).any().item()
            has_inf = torch.isinf(mask_features_output).any().item()

            # If there are NaN/Inf values, skip PCA and return a fallback
            if has_nan or has_inf:
                self._logger.error("Invalid values detected in mask features! Returning fallback image.")
                return np.zeros((img_h, img_w, 4), dtype=np.uint8)

            # Apply PCA projection
            mask_pca = self._apply_pca_projection(
                mask_features_output, 3, valid_feature_mask=mask_alpha.squeeze(-1) > 0
            )[0]
            # Blend mask output based on blend factor
            mask_alpha *= self.mask_blend

            rgba = np.concatenate([mask_pca.cpu().numpy(), mask_alpha[0].cpu().numpy()], axis=-1)

        except Exception as e:
            self._logger.warning(f"Error in render function: {e}")
            import traceback

            traceback.print_exc()
            # Return a fallback image on error
            rgba = np.zeros((img_h, img_w, 4), dtype=np.float32)

        return (rgba.clip(0.0, 1.0) * 255).astype(np.uint8)


@dataclass
class ViewCheckpoint:
    """Interactive viewer for GARfVDB segmentation models.

    Launches a 3D viewer displaying the Gaussian splat radiance field with a
    live segmentation mask overlay that updates as the camera moves.

    Example:
        View a trained segmentation model::

            python view_checkpoint.py \\
                --segmentation-path ./segmentation_checkpoint.pt \\
                --reconstruction-path ./gsplat_checkpoint.ply
    """

    segmentation_path: Annotated[pathlib.Path, arg(aliases=["-s"])]
    """Path to the GARfVDB segmentation checkpoint (.pt or .pth)."""

    reconstruction_path: Annotated[pathlib.Path, arg(aliases=["-r"])]
    """Path to the Gaussian splat reconstruction checkpoint."""

    viewer_port: Annotated[int, arg(aliases=["-p"])] = 8080
    """Port to expose the viewer server on."""

    viewer_ip_address: Annotated[str, arg(aliases=["-ip"])] = "127.0.0.1"
    """IP address to expose the viewer server on."""

    verbose: Annotated[bool, arg(aliases=["-v"])] = False
    """Enable verbose logging."""

    device: str | torch.device = "cuda"
    """Device for computation (e.g., "cuda" or "cpu")."""

    initial_scale: float = 0.1
    """Initial segmentation scale as a fraction of max scale."""

    initial_blend: float = 0.5
    """Initial mask blend factor (0=beauty only, 1=mask only)."""

    camera_check_interval: float = 0.5
    """Camera change polling interval in seconds."""

    no_overlay: bool = False
    """Disable the segmentation overlay (show Gaussian splats only)."""

    overlay_width: int = 1920
    """Width of the segmentation overlay in pixels."""

    overlay_height: int = 1080
    """Height of the segmentation overlay in pixels."""

    overlay_downsample: int = 2
    """Downsample factor for rendering (renders at overlay_size/downsample)."""

    def execute(self) -> None:
        """Execute the viewer command."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")
        logger = logging.getLogger(__name__)

        device = torch.device(self.device)

        # Validate segmentation checkpoint path
        if not self.segmentation_path.exists():
            raise FileNotFoundError(f"Segmentation checkpoint {self.segmentation_path} does not exist.")

        # Load GS model from explicit path
        if not self.reconstruction_path.exists():
            raise FileNotFoundError(f"Reconstruction checkpoint {self.reconstruction_path} does not exist.")
        logger.info(f"Loading Gaussian splat model from {self.reconstruction_path}")
        gs_model, metadata = load_splats_from_file(self.reconstruction_path, device)
        gs_model = filter_splats_above_scale(gs_model, 0.1)
        logger.info(f"Loaded {gs_model.num_gaussians:,} Gaussians")

        # Load the segmentation runner from checkpoint
        # This restores the trained model, the GS model, and the transformed SfmScene with correct scales
        logger.info(f"Loading segmentation checkpoint from {self.segmentation_path}")
        runner = load_segmentation_runner_from_checkpoint(
            checkpoint_path=self.segmentation_path,
            gs_model=gs_model,
            gs_model_path=self.reconstruction_path,
            device=device,
        )

        # Get models and scene from the runner
        gs_model = runner.gs_model
        segmentation_model = runner.model
        sfm_scene = runner.sfm_scene

        logger.info(f"Loaded {gs_model.num_gaussians:,} Gaussians")
        logger.info(f"Restored SfmScene with {sfm_scene.num_images} images (with correct scale transforms)")
        logger.info(f"Segmentation model max scale: {segmentation_model.max_grouping_scale:.4f}")

        # Create the renderer
        renderer = SegmentationRenderer(
            gs_model=gs_model,
            segmentation_model=segmentation_model,
            device=device,
        )
        renderer.scale = self.initial_scale * float(segmentation_model.max_grouping_scale.item())
        renderer.mask_blend = self.initial_blend

        # Initialize fvdb.viz
        logger.info(f"Starting viewer server on {self.viewer_ip_address}:{self.viewer_port}")
        fviz.init(ip_address=self.viewer_ip_address, port=self.viewer_port, verbose=self.verbose)
        viz_scene = fviz.get_scene("GarfVDB Segmentation Viewer")

        # Set initial camera position
        scene_centroid = gs_model.means.mean(dim=0).cpu().numpy()
        cam_to_world_matrices = metadata.get("camera_to_world_matrices", None)
        if cam_to_world_matrices is not None:
            cam_to_world_matrices = to_Mat44fBatch(cam_to_world_matrices.detach()).cpu()
            initial_camera_position = cam_to_world_matrices[0, :3, 3].numpy()
        else:
            scene_radius = (gs_model.means.max(dim=0).values - gs_model.means.min(dim=0).values).max().item() / 2.0
            initial_camera_position = scene_centroid + np.ones(3) * scene_radius

        logger.info(f"Setting camera to {initial_camera_position} looking at {scene_centroid}")
        viz_scene.set_camera_lookat(
            eye=initial_camera_position,
            center=scene_centroid,
            up=[0, 0, 1],
        )

        # Add cameras to the scene if available
        projection_matrices = metadata.get("projection_matrices", None)
        image_sizes = metadata.get("image_sizes", None)
        if cam_to_world_matrices is not None and projection_matrices is not None:
            projection_matrices = to_Mat33fBatch(projection_matrices.detach()).cpu()
            if image_sizes is not None:
                image_sizes = to_Vec2iBatch(image_sizes.detach()).cpu()
            viz_scene.add_cameras(
                name="Training Cameras",
                camera_to_world_matrices=cam_to_world_matrices,
                projection_matrices=projection_matrices,
                image_sizes=image_sizes,
            )

        # Get original training image size for projection scaling
        assert image_sizes is not None
        orig_img_w = int(image_sizes[0, 0].item())
        orig_img_h = int(image_sizes[0, 1].item())

        # Compute render dimensions (smaller for performance)
        render_w = self.overlay_width // self.overlay_downsample
        render_h = self.overlay_height // self.overlay_downsample
        logger.info(f"Overlay: {self.overlay_width}x{self.overlay_height}, render: {render_w}x{render_h}")

        # Add the Gaussian splat model to the scene
        viz_scene.add_gaussian_splat_3d("Gaussian Splats", gs_model)

        # Set up the segmentation overlay if enabled
        image_view = None
        if not self.no_overlay:
            try:
                # Create an initial blank image at full overlay resolution
                initial_image = np.zeros((self.overlay_height, self.overlay_width, 4), dtype=np.uint8)
                initial_image[..., 3] = 128  # Semi-transparent

                # Add the image overlay using the add_image API
                image_view = viz_scene.add_image(  # type: ignore[call-arg]
                    name="Segmentation Overlay",
                    width=self.overlay_width,
                    height=self.overlay_height,
                    rgba_image=initial_image.flatten(),
                )
                logger.info("Segmentation overlay enabled")

            except AttributeError as e:
                logger.warning(f"add_image API not available: {e}")
                logger.info("Running without segmentation overlay")
            except Exception as e:
                logger.warning(f"Failed to set up segmentation overlay: {e}")
                logger.info("Running without segmentation overlay")

        logger.info("=" * 60)
        logger.info("Viewer running... Ctrl+C to exit.")
        logger.info(f"Open your browser to http://{self.viewer_ip_address}:{self.viewer_port}")
        logger.info("")
        logger.info("Segmentation settings:")
        logger.info(f"  - Scale: {renderer.scale:.4f} (max: {segmentation_model.max_grouping_scale:.4f})")
        logger.info(f"  - Mask blend: {renderer.mask_blend:.2f}")
        if image_view is not None:
            logger.info(f"  - Overlay: {self.overlay_width}x{self.overlay_height} (render: {render_w}x{render_h})")
            logger.info(f"  - Update interval: {self.camera_check_interval}s")
        else:
            logger.info("  - Overlay: DISABLED")
        logger.info("=" * 60)

        fviz.show()

        # Simple loop to check camera changes and update overlay
        prev_center = None
        prev_direction = None
        prev_radius = None
        prev_up = None

        def camera_changed() -> bool:
            """Check if camera state changed using documented fvdb.viz.Scene properties."""
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

                assert prev_center is not None and prev_direction is not None and prev_up is not None
                changed = (
                    not np.allclose(center, prev_center)
                    or not np.allclose(direction, prev_direction)
                    or radius != prev_radius
                    or not np.allclose(up, prev_up)
                )

                if changed:
                    prev_center = center
                    prev_direction = direction
                    prev_radius = radius
                    prev_up = up

                return changed
            except Exception:
                return False

        # Get reference projection from SfmScene and scale for render resolution
        sfm_projection = torch.from_numpy(sfm_scene.projection_matrices).float()
        orig_projection = sfm_projection[0]
        # Scale the projection matrix from original training image size to render resolution
        scale_x = render_w / orig_img_w
        scale_y = render_h / orig_img_h
        scaled_projection = orig_projection.clone()
        scaled_projection[0, 0] *= scale_x  # fx
        scaled_projection[1, 1] *= scale_y  # fy
        scaled_projection[0, 2] *= scale_x  # cx
        scaled_projection[1, 2] *= scale_y  # cy
        reference_projection = scaled_projection.to(device)

        # OpenGL to OpenCV conversion matrix (applied to camera axes)
        # OpenGL: X-right, Y-up, Z-backward
        # OpenCV: X-right, Y-down, Z-forward
        opengl_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

        def update_overlay() -> None:
            """Render and update the segmentation overlay."""
            if image_view is None:
                return
            try:
                # Get orbit camera state from viewer
                # NOTE: Despite the Python API name "camera_orbit_direction", the C++ implementation
                # returns eye_direction which is the direction the camera is LOOKING (toward scene).
                # Camera position = center - eye_direction * distance (see Camera.h line 387-390)
                center = np.array(viz_scene.camera_orbit_center.cpu().numpy())
                eye_direction = np.array(viz_scene.camera_orbit_direction.cpu().numpy())
                radius = viz_scene.camera_orbit_radius
                up_world = np.array(viz_scene.camera_up_direction.cpu().numpy())

                # Camera position: center - eye_direction * radius
                # (eye_direction points FROM camera TOWARD center)
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
                # In OpenGL, camera looks along -Z, so Z column = -forward
                c2w_opengl = np.eye(4, dtype=np.float32)
                c2w_opengl[:3, 0] = right
                c2w_opengl[:3, 1] = up
                c2w_opengl[:3, 2] = -forward  # OpenGL: camera looks along -Z
                c2w_opengl[:3, 3] = position

                # Convert to OpenCV convention for the segmentation model
                c2w_opencv = c2w_opengl @ opengl_to_opencv

                camera_to_world = torch.from_numpy(c2w_opencv).float().to(renderer.device)

                # Render at lower resolution for performance
                rgba_image = renderer.render_segmentation_image(
                    camera_to_world, reference_projection, render_w, render_h
                )
                # Scale up to overlay resolution
                if self.overlay_downsample > 1:
                    rgba_image = cv2.resize(
                        rgba_image, (self.overlay_width, self.overlay_height), interpolation=cv2.INTER_LINEAR
                    )
                image_view.update(rgba_image.flatten())  # type: ignore[attr-defined]
                logger.debug(
                    f"Updated segmentation overlay ({render_w}x{render_h} -> {self.overlay_width}x{self.overlay_height})"
                )
            except Exception as e:
                logger.warning(f"Error updating overlay: {e}")
                import traceback

                traceback.print_exc()

        try:
            while True:
                time.sleep(self.camera_check_interval)

                if image_view is not None and camera_changed():
                    logger.debug("Camera changed, updating overlay...")
                    update_overlay()

        except KeyboardInterrupt:
            logger.info("Shutting down...")


def main():
    """Main entry point."""
    cmd = tyro.cli(ViewCheckpoint)
    cmd.execute()


if __name__ == "__main__":
    main()
