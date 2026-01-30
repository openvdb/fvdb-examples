# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Annotated

import cuml
import cuml.cluster.hdbscan
import cupy as cp
import fvdb.viz as fviz
import numpy as np
import torch
import tyro
from fvdb import GaussianSplat3d
from fvdb.types import to_Mat33fBatch, to_Mat44fBatch, to_Vec2iBatch
from fvdb_reality_capture.tools import filter_splats_above_scale
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
        gs_model = filter_splats_above_scale(gs_model, 0.1)
        logger.info(f"Loaded {gs_model.num_gaussians:,} Gaussians")

        # Load the segmentation runner from checkpoint
        logger.info(f"Loading segmentation checkpoint from {self.segmentation_path}")
        runner = load_segmentation_runner_from_checkpoint(
            checkpoint_path=self.segmentation_path,
            gs_model=gs_model,
            gs_model_path=self.reconstruction_path,
            device=device,
        )

        gs_model = runner.gs_model
        segmentation_model = runner.model
        sfm_scene = runner.sfm_scene

        # Get per-gaussian features at a given scale
        scale = self.scale * float(segmentation_model.max_grouping_scale.item())

        mask_features_output = segmentation_model.get_gaussian_affinity_output(scale)  # [N, 256]
        logger.info(f"Got mask features with shape: {mask_features_output.shape}")

        # PCA pre-reduction (256 -> 128)
        logger.info("PCA pre-reduction (256 -> 128 dimensions)...")
        pca = cuml.PCA(n_components=128)
        features_pca = pca.fit_transform(mask_features_output)
        logger.info(f"PCA reduced shape: {features_pca.shape}")

        # UMAP reduction (128 -> 32)
        n_points = features_pca.shape[0]
        reduction_sample_size = min(300_000, n_points)

        logger.info(
            f"UMAP reduction (128 -> 32 dimensions, fitting on {reduction_sample_size:,} / {n_points:,} points)..."
        )
        umap_reducer = cuml.UMAP(
            n_components=32,
            n_neighbors=15,
            min_dist=0.0,
            metric="euclidean",
            random_state=42,
        )

        if n_points > reduction_sample_size:
            # Subsample for fitting, then transform all points
            sample_idx = cp.random.permutation(n_points)[:reduction_sample_size]
            umap_reducer.fit(features_pca[sample_idx])
            features_reduced = umap_reducer.transform(features_pca)
        else:
            features_reduced = umap_reducer.fit_transform(features_pca)

        logger.info(f"UMAP reduced shape: {features_reduced.shape}")

        # Cluster HDBSCAN
        logger.info(f"Clustering with HDBSCAN (fitting on {reduction_sample_size:,} / {n_points:,} points)...")

        clusterer = cuml.HDBSCAN(
            min_samples=100,
            min_cluster_size=200,
            prediction_data=True,  # Required for approximate_predict
        )

        if n_points > reduction_sample_size:
            hdbscan_sample_idx = cp.random.permutation(n_points)[:reduction_sample_size]
            clusterer.fit(features_reduced[hdbscan_sample_idx])
            # Use approximate_predict to assign labels to all points
            cluster_labels_cp, _ = cuml.cluster.hdbscan.approximate_predict(clusterer, features_reduced)
            cluster_labels = torch.as_tensor(cluster_labels_cp, device=gs_model.means.device)
        else:
            clusterer.fit(features_reduced)
            cluster_labels = torch.as_tensor(clusterer.labels_, device=gs_model.means.device)
        unique_labels = torch.unique(cluster_labels)
        num_clusters = (unique_labels >= 0).sum().item()  # Exclude noise label (-1)
        logger.info(f"Found {num_clusters} clusters (+ {(cluster_labels == -1).sum().item()} noise points)")

        # Split gaussians into separate GaussianSplat3d instances per cluster
        cluster_splats: dict[int, GaussianSplat3d] = {}
        for label in unique_labels.tolist():
            if label == -1:
                # Optionally skip noise points, or include them as a separate "noise" cluster
                continue
            cluster_mask = cluster_labels == label
            cluster_splats[label] = gs_model[cluster_mask]
            logger.info(f"  Cluster {label}: {cluster_splats[label].num_gaussians:,} gaussians")

        # Also store noise points if you want them
        noise_mask = cluster_labels == -1
        if noise_mask.any():
            noise_splats = gs_model[noise_mask]
            logger.info(f"  Noise: {noise_splats.num_gaussians:,} gaussians")

        logger.info(f"Loaded {gs_model.num_gaussians:,} Gaussians")
        logger.info(f"Restored SfmScene with {sfm_scene.num_images} images (with correct scale transforms)")
        logger.info(f"Segmentation model max scale: {segmentation_model.max_grouping_scale:.4f}")

        # Initialize fvdb.viz
        logger.info(f"Starting viewer server on {self.viewer_ip_address}:{self.viewer_port}")
        fviz.init(ip_address=self.viewer_ip_address, port=self.viewer_port, verbose=self.verbose)
        viz_scene = fviz.get_scene("GarfVDB Segmentation Viewer")

        # Add the Gaussian splat models to the scene
        logger.info(f"Adding {len(cluster_splats)} clusters to the scene")
        for cluster_id, splat in cluster_splats.items():
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
