import logging

import cuml
import cupy as cp
import numpy as np
import torch
from fvdb import GaussianSplat3d

logger = logging.getLogger(__name__)


def compute_cluster_labels(
    mask_features_output: torch.Tensor,
    pca_n_components: int = 128,
    umap_n_components: int = 32,
    umap_n_neighbors: int = 15,
    hdbscan_min_samples: int = 100,
    hdbscan_min_cluster_size: int = 200,
    fitting_sample_size: int = 300_000,
    random_seed: int = 42,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster per-gaussian features

    To speed up clustering on typically large GaussianSplat3d models (million+ gaussians),
    we perform feature reduction and clustering in a three-stage pipeline:
    1. PCA: Pre-reduction of high-dimensional features to an intermediate representation
    2. UMAP: Non-linear reduction to a low-dimensional manifold
    3. HDBSCAN: Density-based clustering to identify groups of similar gaussians

    Additionally, for scenes (>300k points), subsampling is used during fitting to improve
    performance, and all points are transformed/predicted afterwards.

    Args:
        mask_features_output: Per-gaussian feature vectors from the segmentation
            model. Shape: [N, feature_dim].
        pca_n_components: Number of PCA components for initial reduction.
        umap_n_components: Number of UMAP dimensions for manifold embedding.
        umap_n_neighbors: UMAP neighbor count (higher = more global structure).
        hdbscan_min_samples: Minimum samples for HDBSCAN core points.
        hdbscan_min_cluster_size: Minimum cluster size for HDBSCAN.
        fitting_sample_size: Sample size for fitting UMAP and HDBSCAN.
        random_seed: Random seed for reproducibility.
        device: Device to perform clustering on.
    Returns:
        cluster_labels: Cluster assignment for each gaussian. Shape: [N].
            Label -1 indicates noise points.
        cluster_probs: Membership probability for each gaussian. Shape: [N].
            Higher values indicate stronger cluster membership.
    """
    cp.random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device(device)

    assert umap_n_neighbors < fitting_sample_size, "UMAP n_neighbors must be less than fitting_sample_size"

    # PCA pre-reduction
    n_samples, n_features = mask_features_output.shape[0], mask_features_output.shape[1]
    max_pca_components = min(n_samples, n_features)
    if pca_n_components > max_pca_components:
        logger.warning(
            "Requested pca_n_components=%d is greater than min(n_samples=%d, n_features=%d); " "clamping to %d.",
            pca_n_components,
            n_samples,
            n_features,
            max_pca_components,
        )
        pca_n_components = max_pca_components
    logger.info(f"PCA pre-reduction ({n_features} -> {pca_n_components} dimensions)...")

    pca = cuml.PCA(n_components=pca_n_components)
    features_pca = pca.fit_transform(mask_features_output)
    logger.info(f"PCA reduced shape: {features_pca.shape}")

    # UMAP reduction
    n_points = features_pca.shape[0]
    reduction_sample_size = min(fitting_sample_size, n_points)

    logger.info(
        f"UMAP reduction ({pca_n_components} -> {umap_n_components} dimensions, fitting on {reduction_sample_size:,} / {n_points:,} points)..."
    )
    umap_reducer = cuml.UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=0.0,
        metric="euclidean",
        random_state=random_seed,
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
        min_samples=hdbscan_min_samples,
        min_cluster_size=hdbscan_min_cluster_size,
        prediction_data=True,  # Required for approximate_predict
    )

    if n_points > reduction_sample_size:
        hdbscan_sample_idx = cp.random.permutation(n_points)[:reduction_sample_size]
        clusterer.fit(features_reduced[hdbscan_sample_idx])
        # Use approximate_predict to assign labels to all points
        cluster_labels_cp, cluster_probs_cp = cuml.cluster.hdbscan.approximate_predict(clusterer, features_reduced)
        cluster_labels = torch.as_tensor(cluster_labels_cp, device=device)
        cluster_probs = torch.as_tensor(cluster_probs_cp, device=device)
    else:
        clusterer.fit(features_reduced)
        cluster_labels = torch.as_tensor(clusterer.labels_, device=device)
        cluster_probs = torch.as_tensor(clusterer.probabilities_, device=device)

    return cluster_labels, cluster_probs


def split_gaussians_into_clusters(
    cluster_labels: torch.Tensor, cluster_probs: torch.Tensor, gs_model: GaussianSplat3d
) -> tuple[dict[int, GaussianSplat3d], dict[int, float], GaussianSplat3d]:
    """Split a GaussianSplat3d model into per-cluster subsets.

    Groups gaussians by their cluster labels and computes coherence scores
    (mean membership probability) for each cluster.

    Args:
        cluster_labels: Cluster assignment for each gaussian. Shape: [N].
            Label -1 indicates noise points.
        cluster_probs: Membership probability for each gaussian. Shape: [N].
        gs_model: The GaussianSplat3d model to split.

    Returns:
        cluster_splats: Dictionary mapping cluster ID to GaussianSplat3d subset.
            Excludes noise points (label -1).
        cluster_coherence: Dictionary mapping cluster ID to mean membership
            probability. Higher values indicate tighter, more confident clusters.
        noise_splats: GaussianSplat3d containing all noise points (label -1).
    """
    unique_labels = torch.unique(cluster_labels)
    num_clusters = (unique_labels >= 0).sum().item()  # Exclude noise label (-1)
    logger.info(f"Found {num_clusters} clusters (+ {(cluster_labels == -1).sum().item()} noise points)")

    # Split gaussians into separate GaussianSplat3d instances per cluster
    # Also compute cluster coherence (mean membership probability)
    cluster_splats: dict[int, GaussianSplat3d] = {}
    cluster_coherence: dict[int, float] = {}
    for label in unique_labels.tolist():
        if label == -1:
            # Optionally skip noise points, or include them as a separate "noise" cluster
            continue
        cluster_mask = cluster_labels == label
        cluster_splats[label] = gs_model[cluster_mask]
        cluster_coherence[label] = cluster_probs[cluster_mask].mean().item()
        logger.info(
            f"  Cluster {label}: {cluster_splats[label].num_gaussians:,} gaussians, "
            f"coherence: {cluster_coherence[label]:.3f}"
        )

    # Also store noise points
    noise_mask = cluster_labels == -1
    noise_splats = gs_model[noise_mask]
    if noise_mask.any():
        logger.info(f"  Noise: {noise_splats.num_gaussians:,} gaussians")

    return cluster_splats, cluster_coherence, noise_splats
