# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Annotated

import fvdb.viz as fviz
import numpy as np
import torch
import tyro
from fvdb import GaussianSplat3d
from fvdb.types import to_Mat33fBatch, to_Mat44fBatch, to_Vec2iBatch
from fvdb_reality_capture.tools import (
    filter_splats_above_scale,
    filter_splats_by_mean_percentile,
    filter_splats_by_opacity_percentile,
)
from garfvdb.evaluation.clustering import (
    compute_cluster_labels,
    split_gaussians_into_clusters,
)
from garfvdb.training.segmentation import GaussianSplatScaleConditionedSegmentation
from garfvdb.util import load_splats_from_file
from tyro.conf import arg


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


@dataclass
class ViewCheckpoint:
    """Interactive viewer for GARfVDB segmentation models.

    Launches a 3D viewer displaying the Gaussian splat radiance field with a
    live segmentation mask overlay that updates as the camera moves.

    Example:
        View a trained segmentation model::

            python visualize_segmentation_clusters.py \\
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

    scale: float = 0.1
    """Segmentation scale as a fraction of max scale."""

    filter_high_variance: bool = True
    """Remove clusters with high spatial variance (multi-center clusters)."""

    variance_threshold: float = 0.1
    """Clusters with normalized variance above this threshold are removed.
    Normalized variance = variance / extent^2. Typical values: ~0.03 (tight), ~0.08 (uniform), >0.1 (scattered)."""

    def execute(self) -> None:
        """Execute the viewer command."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")
        logger = logging.getLogger(__name__)

        device = torch.device(self.device)

        # Validate segmentation checkpoint path
        if not self.segmentation_path.exists():
            raise FileNotFoundError(f"Segmentation checkpoint {self.segmentation_path} does not exist.")

        # Load GS model
        if not self.reconstruction_path.exists():
            raise FileNotFoundError(f"Reconstruction checkpoint {self.reconstruction_path} does not exist.")
        logger.info(f"Loading Gaussian splat model from {self.reconstruction_path}")
        gs_model, metadata = load_splats_from_file(self.reconstruction_path, device)

        # Filter GS model
        gs_model = filter_splats_above_scale(gs_model, 0.1)
        gs_model = filter_splats_by_opacity_percentile(gs_model, percentile=0.85)
        gs_model = filter_splats_by_mean_percentile(gs_model, percentile=[0.96, 0.96, 0.96, 0.96, 0.98, 0.99])
        runner = load_segmentation_runner_from_checkpoint(
            checkpoint_path=self.segmentation_path,
            gs_model=gs_model,
            gs_model_path=self.reconstruction_path,
            device=device,
        )

        gs_model = runner.gs_model
        segmentation_model = runner.model

        logger.info(f"Loaded {gs_model.num_gaussians:,} Gaussians")
        logger.info(f"Segmentation model max scale: {segmentation_model.max_grouping_scale:.4f}")

        ## Segmentation and Clustering
        # Query per-gaussian features at a given scale
        scale = self.scale * float(segmentation_model.max_grouping_scale.item())
        mask_features_output = segmentation_model.get_gaussian_affinity_output(scale)  # [N, 256]

        # Perform clustering
        cluster_labels, cluster_probs = compute_cluster_labels(mask_features_output, device=device)

        # Split gaussian scene into separate GaussianSplat3d instances per cluster
        cluster_splats, cluster_coherence, _ = split_gaussians_into_clusters(cluster_labels, cluster_probs, gs_model)

        ## Filtering
        # Filter clusters by spatial variance (remove multi-center clusters)
        if self.filter_high_variance:
            # Compute normalized spatial variance for filtering multi-center clusters
            cluster_norm_variance: dict[int, float] = {}
            for label, splat in cluster_splats.items():
                means = splat.means  # [N, 3]
                # Spatial extent: max range across any axis
                extent = (means.max(dim=0).values - means.min(dim=0).values).max().item()
                # Normalized variance: mean variance across axes / extent^2
                # High values indicate scattered points relative to the cluster size
                if extent > 1e-6:
                    variance = means.var(dim=0).mean().item()
                    cluster_norm_variance[label] = variance / (extent**2)
                else:
                    cluster_norm_variance[label] = 0.0

            # Log variance statistics to help with threshold tuning
            variance_values = list(cluster_norm_variance.values())
            logger.info(
                f"Normalized variance stats: min={min(variance_values):.4f}, "
                f"max={max(variance_values):.4f}, median={np.median(variance_values):.4f}"
            )

            removed_variance = [
                label for label in cluster_splats.keys() if cluster_norm_variance[label] > self.variance_threshold
            ]
            for label in removed_variance:
                logger.info(
                    f"  Removing cluster {label}: normalized variance {cluster_norm_variance[label]:.4f} "
                    f"above threshold {self.variance_threshold:.4f}"
                )
                del cluster_splats[label]
                del cluster_coherence[label]
            if removed_variance:
                logger.info(f"Removed {len(removed_variance)} spatially incoherent clusters")

            logger.info(f"Remaining clusters: {len(cluster_splats)}")

        ## Visualization
        # Initialize fvdb.viz
        logger.info(f"Starting viewer server on {self.viewer_ip_address}:{self.viewer_port}")
        fviz.init(ip_address=self.viewer_ip_address, port=self.viewer_port, verbose=self.verbose)
        viz_scene = fviz.get_scene("GarfVDB Segmentation Viewer")

        # Add the Gaussian splat models to the scene, sort by coherence
        sorted_clusters = sorted(cluster_splats.items(), key=lambda x: cluster_coherence[x[0]])
        for cluster_id, splat in sorted_clusters:
            viz_scene.add_gaussian_splat_3d(f"Cluster {cluster_id}", splat)

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

        logger.info("=" * 60)
        logger.info("Viewer running... Ctrl+C to exit.")
        logger.info(f"Open your browser to http://{self.viewer_ip_address}:{self.viewer_port}")
        logger.info("")
        logger.info("Segmentation settings:")
        logger.info(f"  - Scale: {scale:.4f} (max: {segmentation_model.max_grouping_scale:.4f})")

        logger.info("=" * 60)

        fviz.show()

        time.sleep(1000000)


def main():
    """Main entry point."""
    cmd = tyro.cli(ViewCheckpoint)
    cmd.execute()


if __name__ == "__main__":
    main()
