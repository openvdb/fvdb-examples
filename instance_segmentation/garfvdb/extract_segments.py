# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Export instance segments from a trained GARfVDB segmentation checkpoint.

Loads a Gaussian splat reconstruction and segmentation checkpoint, clusters
per-Gaussian affinity features at a chosen scale, filters clusters, then writes
``n`` segment ``.ply`` Gaussian splat scenes (one per selected cluster).

Example::

    python extract_segments.py \\
        -s garfvdb_logs/run/checkpoints/00036600/train_ckpt.pt \\
        -r frgs_logs/safety_park_1/checkpoints/00024800/reconstruct_ckpt.pt \\
        -o segments/ \\
        --n 5 \\
        --scale 0.1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fvdb import GaussianSplat3d
from scipy.spatial import cKDTree

# Import directly to avoid fvdb_reality_capture.tools.__init__ pulling optional deps (e.g. DLNR).
from fvdb_reality_capture.tools._filter_splats import (
    filter_splats_above_scale,
    filter_splats_by_mean_percentile,
    filter_splats_by_opacity_percentile,
)
from garfvdb.training.segmentation import GaussianSplatScaleConditionedSegmentation
from garfvdb.util import load_splats_from_file

logger = logging.getLogger(__name__)


def load_segmentation_runner_from_checkpoint(
    checkpoint_path: Path,
    gs_model: GaussianSplat3d,
    gs_model_path: Path,
    device: str | torch.device = "cuda",
) -> GaussianSplatScaleConditionedSegmentation:
    """Restore a segmentation runner from a training checkpoint.
    Args:
        checkpoint_path: Path to the segmentation checkpoint.
        gs_model: The Gaussian splat model to use for the segmentation.
        gs_model_path: Path to the Gaussian splat reconstruction.
        device: The device to use for the segmentation.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    runner = GaussianSplatScaleConditionedSegmentation.from_state_dict(
        state_dict=checkpoint,
        gs_model=gs_model,
        gs_model_path=gs_model_path,
        device=device,
        eval_only=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return runner


def _is_gpu_oom_error(exc: BaseException) -> bool:
    """Return whether an exception indicates GPU out-of-memory.
    Args:
        exc: The exception raised during clustering.
    """
    return "out_of_memory" in str(exc).lower() or isinstance(exc, MemoryError)


def _drop_clusters(
    cluster_splats: dict[int, GaussianSplat3d],
    cluster_coherence: dict[int, float],
    keys: list[int],
) -> None:
    """Remove clusters from the splat and coherence maps in place.
    Args:
        cluster_splats: Cluster label to Gaussian splat mapping.
        cluster_coherence: Cluster label to coherence score mapping.
        keys: Cluster labels to remove.
    """
    for key in keys:
        cluster_splats.pop(key, None)
        cluster_coherence.pop(key, None)


def _subsample_gaussians(
    gs_model: GaussianSplat3d,
    max_gaussians: int,
    seed: int,
    device: torch.device,
) -> tuple[GaussianSplat3d, torch.Tensor]:
    """Randomly subsample Gaussians for memory-bounded clustering.
    Args:
        gs_model: Full-scene Gaussian splat model.
        max_gaussians: Maximum number of Gaussians to keep.
        seed: Random seed for reproducible subsampling.
        device: Torch device for the returned mask.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(gs_model.num_gaussians, size=max_gaussians, replace=False)
    mask = torch.zeros(gs_model.num_gaussians, dtype=torch.bool, device=device)
    mask[torch.from_numpy(indices).to(device)] = True
    return gs_model[mask], mask


def _map_cluster_labels_to_full_scene(
    cluster_labels_sub: torch.Tensor,
    cluster_probs_sub: torch.Tensor,
    gs_model: GaussianSplat3d,
    clustering_gs_model: GaussianSplat3d,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map cluster labels from a subsampled set back to the full scene.
    Args:
        cluster_labels_sub: Cluster labels on the subsampled Gaussians.
        cluster_probs_sub: Cluster probabilities on the subsampled Gaussians.
        gs_model: Full-scene Gaussian splat model.
        clustering_gs_model: Subsampled model used for clustering.
        device: Torch device for returned tensors.
    """
    tree = cKDTree(clustering_gs_model.means.cpu().numpy())
    _, nearest_indices = tree.query(gs_model.means.cpu().numpy(), k=1, workers=-1)
    index_tensor = torch.from_numpy(nearest_indices).to(device)
    return cluster_labels_sub[index_tensor], cluster_probs_sub[index_tensor]


def _filter_high_variance_clusters(
    cluster_splats: dict[int, GaussianSplat3d],
    cluster_coherence: dict[int, float],
    variance_threshold: float,
) -> list[int]:
    """Find spatially incoherent clusters by normalized variance.
    Args:
        cluster_splats: Cluster label to Gaussian splat mapping.
        cluster_coherence: Cluster label to coherence score mapping.
        variance_threshold: Normalized variance cutoff (variance / extent^2).
    """
    removed: list[int] = []
    for label, splat in list(cluster_splats.items()):
        means = splat.means
        extent = (means.max(dim=0).values - means.min(dim=0).values).max().item()
        if extent > 1e-6:
            norm_variance = means.var(dim=0).mean().item() / (extent**2)
        else:
            norm_variance = 0.0
        if norm_variance > variance_threshold:
            removed.append(label)
    return removed


@torch.inference_mode()
def rip_segments(
    *,
    segmentation_path: Path,
    reconstruction_path: Path,
    out_dir: Path,
    n: int,
    scale: float,
    scale_is_fraction_of_max: bool,
    seed: int,
    device: str,
    min_splat_scale: float,
    opacity_percentile: float,
    mean_percentile: tuple[float, ...],
    min_cluster_gaussians: int,
    filter_high_variance: bool,
    variance_threshold: float,
    sample_by: Literal["random", "coherence"],
    max_gaussians_for_clustering: int,
    verbose: bool,
) -> None:
    """Cluster Gaussians and export ``n`` segment PLY files."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")

    # Clustering depends on GPU libs (cuml/cupy); import lazily so --help works without them.
    from garfvdb.evaluation.clustering import (  # noqa: PLC0415
        compute_cluster_labels,
        split_gaussians_into_clusters,
    )

    device_t = torch.device(device)

    if not segmentation_path.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {segmentation_path}")
    if not reconstruction_path.exists():
        raise FileNotFoundError(f"Reconstruction checkpoint not found: {reconstruction_path}")
    if n <= 0:
        raise ValueError(f"--n must be > 0, got {n}")

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Gaussian splat model from %s", reconstruction_path)
    gs_model, original_metadata = load_splats_from_file(reconstruction_path, device_t)
    logger.info("Loaded %s Gaussians (pre-filter)", f"{gs_model.num_gaussians:,}")

    if min_splat_scale > 0:
        gs_model = filter_splats_above_scale(gs_model, min_splat_scale)
    if opacity_percentile > 0:
        gs_model = filter_splats_by_opacity_percentile(gs_model, percentile=opacity_percentile)
    if mean_percentile:
        gs_model = filter_splats_by_mean_percentile(gs_model, percentile=list(mean_percentile))
    logger.info("Remaining %s Gaussians (post-filter)", f"{gs_model.num_gaussians:,}")

    runner = load_segmentation_runner_from_checkpoint(
        checkpoint_path=segmentation_path,
        gs_model=gs_model,
        gs_model_path=reconstruction_path,
        device=device_t,
    )
    gs_model = runner.gs_model
    segmentation_model = runner.model

    max_scale = float(segmentation_model.max_grouping_scale.item())
    scale_abs = float(scale) * max_scale if scale_is_fraction_of_max else float(scale)
    logger.info("Segmentation model max scale: %.6f", max_scale)
    logger.info("Clustering at scale: %.6f", scale_abs)

    clustering_gs_model = gs_model
    subsample_mask: torch.Tensor | None = None
    if max_gaussians_for_clustering > 0 and gs_model.num_gaussians > max_gaussians_for_clustering:
        logger.warning(
            "Scene has %s gaussians (> %s); subsampling for clustering, then mapping labels back.",
            f"{gs_model.num_gaussians:,}",
            f"{max_gaussians_for_clustering:,}",
        )
        clustering_gs_model, subsample_mask = _subsample_gaussians(
            gs_model, max_gaussians_for_clustering, seed, device_t
        )
        logger.info(
            "Clustering on %s subsampled gaussians",
            f"{clustering_gs_model.num_gaussians:,}",
        )

    mask_features = segmentation_model.get_gaussian_affinity_output(scale_abs)
    if subsample_mask is not None:
        mask_features = mask_features[subsample_mask]

    try:
        cluster_labels_sub, cluster_probs_sub = compute_cluster_labels(mask_features, device=device_t)
    except Exception as exc:
        if not _is_gpu_oom_error(exc):
            raise
        logger.error(
            "GPU OOM while clustering %s gaussians. Try tighter pre-filters or a lower "
            "--max-gaussians-for-clustering (current: %s). Target: <500k for clustering.",
            f"{clustering_gs_model.num_gaussians:,}",
            f"{max_gaussians_for_clustering:,}",
        )
        raise RuntimeError(
            f"Clustering failed due to GPU memory ({clustering_gs_model.num_gaussians:,} gaussians)."
        ) from exc

    if subsample_mask is not None:
        logger.info("Mapping cluster labels back to all gaussians via nearest neighbor...")
        cluster_labels, cluster_probs = _map_cluster_labels_to_full_scene(
            cluster_labels_sub,
            cluster_probs_sub,
            gs_model,
            clustering_gs_model,
            device_t,
        )
    else:
        cluster_labels, cluster_probs = cluster_labels_sub, cluster_probs_sub

    cluster_splats, cluster_coherence, _ = split_gaussians_into_clusters(cluster_labels, cluster_probs, gs_model)

    if min_cluster_gaussians > 0:
        removed_small = [
            label for label, splat in cluster_splats.items() if splat.num_gaussians < min_cluster_gaussians
        ]
        _drop_clusters(cluster_splats, cluster_coherence, removed_small)
        if removed_small:
            logger.info(
                "Removed %d clusters with < %d gaussians",
                len(removed_small),
                min_cluster_gaussians,
            )

    if filter_high_variance:
        removed_variance = _filter_high_variance_clusters(cluster_splats, cluster_coherence, variance_threshold)
        _drop_clusters(cluster_splats, cluster_coherence, removed_variance)
        if removed_variance:
            logger.info(
                "Removed %d spatially incoherent clusters (variance_threshold=%.4f)",
                len(removed_variance),
                variance_threshold,
            )

    if not cluster_splats:
        raise RuntimeError("No clusters remaining after filtering; relax filters or try a different --scale.")

    labels = sorted(cluster_splats.keys())
    n_to_export = min(n, len(labels))
    if n_to_export < n:
        logger.warning(
            "Requested n=%d but only %d clusters available; exporting %d.",
            n,
            len(labels),
            n_to_export,
        )

    if sample_by == "coherence":
        chosen_labels = [
            label
            for label, _ in sorted(cluster_coherence.items(), key=lambda item: item[1], reverse=True)[:n_to_export]
        ]
    else:
        rng = np.random.default_rng(seed)
        chosen_labels = rng.choice(np.array(labels, dtype=np.int64), size=n_to_export, replace=False).tolist()

    for i, label in enumerate(chosen_labels):
        splat = cluster_splats[int(label)]
        coherence = float(cluster_coherence[int(label)])
        ply_path = out_dir / (f"segment_{i:04d}_cluster{int(label)}_coh{coherence:.3f}_n{splat.num_gaussians}.ply")
        if ply_path.exists():
            logger.warning("Overwriting existing file: %s", ply_path)

        metadata: dict = {}
        if original_metadata:
            metadata.update(original_metadata)
        metadata.update(
            {
                "cluster_id": int(label),
                "coherence": coherence,
                "num_gaussians": int(splat.num_gaussians),
                "scale_abs": scale_abs,
                "segmentation_ckpt": str(segmentation_path),
                "reconstruction": str(reconstruction_path),
            }
        )

        splat.save_ply(str(ply_path), metadata=metadata)
        logger.info("Wrote %s (%s gaussians)", ply_path, f"{splat.num_gaussians:,}")


def main() -> None:
    """Parse CLI arguments and run segment extraction."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--segmentation-path",
        type=Path,
        required=True,
        help="GARfVDB segmentation checkpoint (.pt / .pth)",
    )
    parser.add_argument(
        "-r",
        "--reconstruction-path",
        type=Path,
        required=True,
        help="Gaussian splat reconstruction (.pt / .ply)",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for segment PLY files",
    )

    parser.add_argument("--n", type=int, default=10, help="Number of segments to export")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.1,
        help="Clustering scale (absolute or fraction of max; see --scale-is-fraction-of-max)",
    )
    parser.add_argument(
        "--scale-is-fraction-of-max",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpret --scale as a fraction of model.max_grouping_scale",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling/subsampling")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--min-splat-scale",
        type=float,
        default=0.1,
        help="Drop Gaussians with scale below this value (0 disables)",
    )
    parser.add_argument(
        "--opacity-percentile",
        type=float,
        default=0.85,
        help="Keep Gaussians above this opacity percentile (0 disables)",
    )
    parser.add_argument(
        "--mean-percentile",
        type=float,
        nargs="*",
        default=[0.96, 0.96, 0.96, 0.96, 0.98, 0.99],
        help="Per-channel mean percentiles for splat pre-filtering",
    )

    parser.add_argument(
        "--min-cluster-gaussians",
        type=int,
        default=200,
        help="Drop clusters smaller than this (0 disables)",
    )
    parser.add_argument(
        "--filter-high-variance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove spatially incoherent clusters",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.1,
        help="Normalized variance cutoff when --filter-high-variance is enabled",
    )
    parser.add_argument(
        "--sample-by",
        choices=["random", "coherence"],
        default="random",
        help="How to pick which clusters to export",
    )
    parser.add_argument(
        "--max-gaussians-for-clustering",
        type=int,
        default=500_000,
        help="Subsample before clustering when scene is larger (0 disables)",
    )

    args = parser.parse_args()

    rip_segments(
        segmentation_path=args.segmentation_path,
        reconstruction_path=args.reconstruction_path,
        out_dir=args.out_dir,
        n=args.n,
        scale=args.scale,
        scale_is_fraction_of_max=args.scale_is_fraction_of_max,
        seed=args.seed,
        device=args.device,
        min_splat_scale=args.min_splat_scale,
        opacity_percentile=args.opacity_percentile,
        mean_percentile=tuple(float(x) for x in args.mean_percentile),
        min_cluster_gaussians=args.min_cluster_gaussians,
        filter_high_variance=args.filter_high_variance,
        variance_threshold=args.variance_threshold,
        sample_by=args.sample_by,
        max_gaussians_for_clustering=args.max_gaussians_for_clustering,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
