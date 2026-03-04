# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans

if TYPE_CHECKING:
    from langsplatv2.training.dataset import LangSplatV2Dataset

logger = logging.getLogger(__name__)

def softmax_to_topk_soft_code(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Generate sparse coefficients from logits via softmax + top-k selection.

    Applies softmax to produce probabilities, selects the top-k entries,
    zeros out the rest, and re-normalizes so the sparse coefficients sum to 1.

    Args:
        logits: Raw logits of shape ``[N, codebook_size]``.
        k: Number of non-zero entries to retain per row.

    Returns:
        Sparse coefficient tensor of shape ``[N, codebook_size]`` with
        exactly ``k`` non-zero entries per row that sum to 1.
    """
    y_soft = logits.softmax(dim=1)  # [N, codebook_size]

    # Work with compact [N, k] tensors instead of full [N, codebook_size]
    topk_vals, topk_idx = torch.topk(y_soft, k, dim=1)  # [N, k], [N, k]

    # Normalize the top-k values directly (much cheaper than full-size ops)
    topk_vals = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-10)  # [N, k]

    # Scatter normalized values into sparse output
    result = torch.zeros_like(y_soft)  # [N, codebook_size]
    result.scatter_(1, topk_idx, topk_vals)

    return result


class ResidualVectorQuantizationWithClustering(nn.Module):
    """Residual vector quantization using K-means clustering.

    Initializes codebooks by fitting K-means on input features, with each
    subsequent level fitting on the residuals from the previous level.
    This follows the LangSplatV2 approach for codebook initialization.

    Attributes:
        num_levels: Number of quantization levels (layers).
        num_clusters: Number of cluster centers per level.
        feature_dim: Dimensionality of the codebook features.
        seed: Random state passed to ``MiniBatchKMeans`` for reproducibility.
        quantizers: List of codebook tensors after fitting.
    """

    def __init__(
        self,
        num_levels: int,
        num_clusters: int,
        feature_dim: int,
        device: torch.device | str = "cuda",
        seed: int = 42,
    ):
        """Initialize the residual VQ module.

        Args:
            num_levels: Number of quantization levels.
            num_clusters: Number of clusters (codebook size) per level.
            feature_dim: Feature dimensionality.
            device: Device for codebook tensors.
            seed: Random state for K-means clustering reproducibility.
        """
        super().__init__()
        self.num_levels = num_levels
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.device = device
        self.seed = seed
        self.quantizers: list[torch.Tensor] = []

    def fit_quantizers(self, features: torch.Tensor) -> None:
        """Fit codebooks on the given features using K-means clustering.

        For each level, clusters the current residuals, stores the cluster
        centers as the codebook, then updates residuals by subtracting
        the quantized values.

        Args:
            features: Feature tensor of shape ``[N, feature_dim]``.
        """

        residuals = features.cpu().detach().numpy()

        for level in range(self.num_levels):
            logger.info(f"Fitting VQ level {level} with {self.num_clusters} clusters on {len(residuals)} samples")
            kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=self.seed)
            kmeans.fit(residuals)

            # Store cluster centers as codebook
            codebook = torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=torch.float32)
            self.quantizers.append(codebook)

            # Compute quantized values and update residuals
            quantized = self._quantize_with_centers(residuals, kmeans.cluster_centers_) #type: ignore
            residuals = residuals - quantized

    def _quantize_with_centers(self, data: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Quantize data by assigning each point to its nearest center.

        Args:
            data: Input data of shape ``[N, D]``.
            centers: Cluster centers of shape ``[K, D]``.

        Returns:
            Quantized data of shape ``[N, D]``.
        """
        # Use torch for efficient distance computation
        data_tensor = torch.tensor(data, device=self.device)
        centers_tensor = torch.tensor(centers, device=self.device)
        distances = torch.cdist(data_tensor, centers_tensor, p=2)
        indices = distances.argmin(dim=1)
        quantized_data = centers_tensor[indices]
        return quantized_data.cpu().numpy()


def load_clip_features_for_level(
    full_dataset: LangSplatV2Dataset,
    feature_level: int
) -> torch.Tensor:
    """Load CLIP features for a specific scale level.

    Only loads features corresponding to the specified scale level
    (0=default, 1=small, 2=medium, 3=large).

    Args:
        full_dataset: The full dataset containing all features.
        feature_level: Which scale level to load (0-3).

    Returns:
        Feature tensor of shape ``[N_level, clip_n_dims]`` containing
        only features from the specified scale level.
    """
    all_features = []

    for image_id in range(len(full_dataset)):
        features, _, lengths = full_dataset.get_feature_data(image_id)

        lengths = lengths.tolist()

        # Compute offset for the requested level
        offset = sum(lengths[:feature_level])
        count = lengths[feature_level]

        if count > 0:
            level_features = features[offset : offset + count]
            all_features.append(level_features)

    if len(all_features) == 0:
        raise RuntimeError(f"No features found for level {feature_level}")

    return torch.cat(all_features, dim=0).float()


def load_all_clip_features(
    full_dataset: LangSplatV2Dataset,
) -> torch.Tensor:
    """Load CLIP features from ALL scale levels across all images.

    This matches the original LangSplatV2 codebook initialization which
    concatenates every ``*_f.npy`` file (each containing features from all
    4 scales) into a single tensor for k-means.

    Args:
        full_dataset: The full dataset containing all features.

    Returns:
        Feature tensor of shape ``[N_all, clip_n_dims]`` containing
        features from all scale levels and all images.
    """
    all_features = []

    for image_id in range(len(full_dataset)):
        features, _, _ = full_dataset.get_feature_data(image_id)
        if features.shape[0] > 0:
            all_features.append(features)

    if len(all_features) == 0:
        raise RuntimeError("No features found across any images")

    return torch.cat(all_features, dim=0).float()
