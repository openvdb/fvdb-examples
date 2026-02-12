# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Sequence, Sized, TypedDict, cast

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from fvdb import JaggedTensor
from fvdb_reality_capture.radiance_fields import SfmDataset
from fvdb_reality_capture.sfm_scene import SfmScene

logger = logging.getLogger(__name__)


class LangSplatV2DataItem(TypedDict):
    """Type definition for a single item in the LangSplatV2 dataset.

    Compact representation: stores the sparse CLIP features and seg_map
    instead of the dense ``[H, W, 512]`` feature map. The dense feature
    map is built on-device after transfer using :func:`build_feature_map`.

    Attributes:
        image: RGB image tensor of shape ``[H, W, 3]``.
        projection: Camera projection matrix of shape ``[3, 3]``.
        camera_to_world: Camera-to-world transformation ``[4, 4]``.
        world_to_camera: World-to-camera transformation ``[4, 4]``.
        features: CLIP feature vectors ``[N_masks, clip_n_dims]``.
        seg_map: Segmentation map ``[H, W]`` mapping pixels to feature indices.
        image_h: Image height.
        image_w: Image width.
    """

    image: torch.Tensor
    projection: torch.Tensor
    camera_to_world: torch.Tensor
    world_to_camera: torch.Tensor
    features: torch.Tensor
    seg_map: torch.Tensor
    image_h: int
    image_w: int


class LangSplatV2Dataset(SfmDataset):
    """Dataset for LangSplatV2 training that loads CLIP features from cache.

    Extends SfmDataset to load pre-computed CLIP features and segmentation
    maps. For each image, constructs the ground truth feature map by looking
    up CLIP features using the segmentation map at the configured scale level.

    Attributes:
        feature_level: Which SAM scale to use (0=default, 1=s, 2=m, 3=l).
        clip_n_dims: Dimensionality of CLIP features.
    """

    def __init__(
        self,
        sfm_scene: SfmScene,
        feature_level: int = 1,
        clip_n_dims: int = 512,
        dataset_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        cache_features: bool = True,
        cache_images: bool = True,
    ):
        """Initialize the LangSplatV2 dataset.

        Args:
            sfm_scene: The SfmScene with CLIP features in its cache.
            feature_level: Which SAM scale level to use (1=s, 2=m, 3=l).
            clip_n_dims: CLIP feature dimensionality.
            dataset_indices: Optional indices to subset the dataset.
            cache_features: If True, cache compact feature data (feature
                vectors + seg_maps) in memory. Feature maps are built
                on-the-fly in ``__getitem__``.
            cache_images: If True, cache decoded images in memory.
        """
        super().__init__(sfm_scene, dataset_indices)
        self.feature_level = feature_level
        self.clip_n_dims = clip_n_dims
        self._cache_features = cache_features
        self._cache_images = cache_images
        self._features_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._images_cache: dict[int, dict] = {}

    @property
    def num_zeropad(self) -> int:
        return len(str(self._sfm_scene.num_images)) + 2

    def warmup_cache(self) -> None:
        """Pre-load feature data and images into cache.

        Caches compact feature vectors and seg_maps. Feature maps
        (dense ``[H,W,512]``) are built on-the-fly in ``__getitem__``.

        Call this BEFORE creating a DataLoader with ``num_workers > 0``.
        Workers forked after warmup inherit the populated cache.
        """
        import tqdm

        if self._cache_features:
            for idx in tqdm.tqdm(range(len(self)), desc="Warming up feature cache"):
                index = self._indices[idx]
                if index not in self._features_cache:
                    self.get_feature_data(index)

        if self._cache_images:
            for idx in tqdm.tqdm(range(len(self)), desc="Warming up image cache"):
                index = self._indices[idx]
                if index not in self._images_cache:
                    sfm_item = super().__getitem__(idx)
                    self._images_cache[index] = sfm_item

    def _read_feature_data(self, index: int) -> dict[str, torch.Tensor]:
        """Read feature data from disk cache."""
        cache_filename = f"features_{index:0{self.num_zeropad}}"
        _, data = self.sfm_scene.cache.read_file(cache_filename)
        return data

    def get_feature_data(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load feature data for an image.

        Returns the feature vectors and seg_map for the configured level.
        Feature maps are built separately via :func:`build_feature_map`.

        Args:
            index: Image index in the SfmScene.

        Returns:
            Tuple of (features, seg_map, lengths) where:
                - features: ``[N_total, clip_n_dims]`` CLIP features
                - seg_map: ``[H, W]`` int32 map of pixel->feature indices (-1 = none)
                - lengths: ``[4]`` number of features at each scale level
        """
        if self._cache_features and index in self._features_cache:
            return self._features_cache[index]

        data = self._read_feature_data(index)

        features = data["features"]  # [N_total, clip_n_dims]
        seg_maps = data["seg_maps"]  # [4, H, W]
        lengths = data["lengths"]  # [4]

        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        if isinstance(seg_maps, np.ndarray):
            seg_maps = torch.from_numpy(seg_maps)
        if isinstance(lengths, np.ndarray):
            lengths = torch.from_numpy(lengths)

        features = features.float()
        seg_map = seg_maps[self.feature_level]  # [H, W]

        result = (features, seg_map, lengths)

        if self._cache_features:
            self._features_cache[index] = result

        return result

    def _get_sfm_item(self, idx: int) -> dict:
        """Get SfmDataset item, using cache if enabled."""
        index = self._indices[idx]
        if self._cache_images and index in self._images_cache:
            return self._images_cache[index]

        sfm_item = super().__getitem__(idx)

        if self._cache_images:
            self._images_cache[index] = sfm_item

        return sfm_item

    def __getitem__(self, idx: int) -> LangSplatV2DataItem:
        sfm_item = self._get_sfm_item(idx)
        index = self._indices[idx]

        features, seg_map, _ = self.get_feature_data(index)

        return LangSplatV2DataItem(
            image=torch.from_numpy(sfm_item["image"]),
            projection=sfm_item["projection"],
            camera_to_world=sfm_item["camera_to_world"],
            world_to_camera=sfm_item["world_to_camera"],
            features=features,
            seg_map=seg_map,
            image_h=sfm_item["image"].shape[0],
            image_w=sfm_item["image"].shape[1],
        )


def build_feature_map(
    features: JaggedTensor | torch.Tensor,
    seg_map: torch.Tensor,
    clip_n_dims: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense GT feature map from compact features and seg_map.

    Runs on whatever device the inputs are on. When called after
    transferring compact data to GPU, this constructs the dense map
    directly on GPU -- avoiding a ~4 GB PCIe transfer for 1080p images.

    Args:
        features: CLIP features as a :class:`JaggedTensor` (batched, each
            element ``[N_masks_i, clip_n_dims]``) or a plain ``torch.Tensor``
            of shape ``[N_masks, clip_n_dims]`` (unbatched).
        seg_map: Segmentation map ``[B, H, W]`` or ``[H, W]`` (unbatched)
            with feature indices (-1 = no feature).
        clip_n_dims: Feature dimensionality (needed for output allocation).

    Returns:
        Tuple of (gt_features, feature_mask) where:
            - gt_features: ``[B, H, W, clip_n_dims]`` or ``[H, W, clip_n_dims]``
            - feature_mask: ``[B, H, W]`` or ``[H, W]`` bool
    """
    if isinstance(features, torch.Tensor):
        # Unbatched path: plain tensor [N_masks, clip_n_dims]
        H, W = seg_map.shape
        feature_mask = seg_map >= 0
        # Use empty -- invalid pixels are masked out in the loss and never read
        gt_features = torch.empty(
            H, W, clip_n_dims, dtype=features.dtype, device=features.device
        )
        if feature_mask.any():
            valid_indices = seg_map[feature_mask].long()
            gt_features[feature_mask] = features[valid_indices]
        return gt_features, feature_mask

    # Batched path: JaggedTensor with B elements
    B = seg_map.shape[0] if seg_map.dim() == 3 else 1
    H, W = seg_map.shape[-2:]
    device = features.jdata.device
    dtype = features.jdata.dtype

    feature_mask = seg_map >= 0  # [B, H, W]
    # Use empty -- invalid pixels are masked out in the loss and never read
    gt_features = torch.empty(
        B, H, W, clip_n_dims, dtype=dtype, device=device
    )
    for b in range(B):
        mask_b = feature_mask[b]
        if mask_b.any():
            feat_b = features[b].jdata
            valid_indices = seg_map[b][mask_b].long()
            gt_features[b][mask_b] = feat_b[valid_indices]

    return gt_features, feature_mask


class LangSplatV2Input(dict):
    """Batched input dictionary for the LangSplatV2 model.

    A typed dictionary wrapper that supports device transfer.
    """

    def to(self, device: torch.device, non_blocking: bool = True) -> "LangSplatV2Input":
        result = {}
        for k, v in self.items():
            if isinstance(v, JaggedTensor):
                result[k] = v.to(device)
            elif isinstance(v, torch.Tensor):
                result[k] = v.to(device, non_blocking=non_blocking)
            else:
                result[k] = v
        return LangSplatV2Input(result)


def LangSplatV2CollateFn(batch: list[LangSplatV2DataItem]) -> LangSplatV2Input:
    """Collate LangSplatV2DataItems into a batched input.

    Args:
        batch: List of data items to collate.

    Returns:
        Batched LangSplatV2Input dictionary.
    """

    return LangSplatV2Input(
        image=torch.stack([cast(torch.Tensor, b["image"]) for b in batch]),
        projection=torch.stack([cast(torch.Tensor, b["projection"]) for b in batch]),
        camera_to_world=torch.stack([cast(torch.Tensor, b["camera_to_world"]) for b in batch]),
        world_to_camera=torch.stack([cast(torch.Tensor, b["world_to_camera"]) for b in batch]),
        features=JaggedTensor([cast(torch.Tensor, b["features"]) for b in batch]),
        seg_map=torch.stack([cast(torch.Tensor, b["seg_map"]) for b in batch]),
        image_h=[cast(int, b["image_h"]) for b in batch],
        image_w=[cast(int, b["image_w"]) for b in batch],
    )


class InfiniteSampler(torch.utils.data.Sampler):
    """Sampler that yields dataset indices infinitely.

    Avoids DataLoader iterator recreation between epochs.
    Epoch boundaries must be tracked by counting samples.
    """

    def __init__(self, dataset: Sized, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        while True:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self._epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))
            yield from indices
            self._epoch += 1

    def __len__(self):
        raise NotImplementedError("InfiniteSampler yields indices indefinitely.")
