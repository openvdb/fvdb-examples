# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, NotRequired, Sequence, Sized, TypedDict, Union, cast

import fvdb
import numpy as np
import torch
from fvdb_reality_capture.radiance_fields import SfmDataset
from fvdb_reality_capture.sfm_scene import SfmScene


class SegmentationDataItem(TypedDict):
    """Type definition for a single item in the SegmentationDataset for linting convenience."""

    image: torch.Tensor  # [H, W, 3] or [num_samples, 3]
    projection: torch.Tensor  # [3, 3]
    camera_to_world: torch.Tensor  # [4, 4]
    world_to_camera: torch.Tensor  # [4, 4]
    scales: torch.Tensor  # [NM] i.e. [0.6053, 0.4358, 0.2108, 0.2107, 0.2090, 0.1880, 0.1320,
    mask_cdf: torch.Tensor  # [H, W, MM] or [num_samples, MM] i.e. [0.5014, 1., 1., 1.]
    mask_ids: torch.Tensor  # [H, W, MM] or [num_samples, MM] i.e. [12, 14, -1, -1],
    image_h: int
    image_w: int
    image_full: NotRequired[torch.Tensor]  # [H, W, 3]
    pixel_coords: NotRequired[torch.Tensor]  # [num_samples, 2]


class SegmentationDataset(SfmDataset):
    """Dataset for loading the SegmentationDataset which loads the images, intrinsics, cam_to_worlds,
    scales, mask_cdfs, mask_ids from disk.  Members of this class can then be modified by further
    data transforms."""

    def __init__(
        self,
        sfm_scene: SfmScene,
        dataset_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        cache_loaded_masks: bool = True,
    ):
        """
        Args:
            sfm_scene: The SfmScene containing images and camera data.
            dataset_indices: Optional indices to subset the dataset.
            cache_loaded_masks: If True, cache loaded mask data in memory to avoid
                repeated decoding. This significantly speeds up data loading at the cost
                of additional RAM usage. Default is True.
        """
        super().__init__(sfm_scene, dataset_indices)
        self._cache_loaded_masks = cache_loaded_masks
        self._loaded_masks_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @property
    def num_zeropad(self) -> int:
        return len(str(self._sfm_scene.num_images)) + 2

    def warmup_cache(self) -> None:
        """Pre-load all mask data into cache.

        Call this BEFORE creating a DataLoader with num_workers > 0.
        When workers are forked, they inherit the populated cache, eliminating
        disk I/O during training.
        """
        if not self._cache_loaded_masks:
            return

        import tqdm

        for idx in tqdm.tqdm(range(len(self)), desc="Warming up mask cache"):
            index = self._indices[idx]
            if index not in self._loaded_masks_cache:
                self._get_mask_data(index)

    def _get_mask_data(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get mask data for the given index, using cache if enabled.

        Returns:
            Tuple of (mask_cdf, mask_ids, scales) tensors.
        """
        if self._cache_loaded_masks and index in self._loaded_masks_cache:
            return self._loaded_masks_cache[index]

        # Read and decode from disk
        _, data = self.sfm_scene.cache.read_file(f"masks_{index:0{self.num_zeropad}}")
        # NOTE: One other strategy we investigated was to RLE-encode the mask_cdf and mask_ids,
        # which are smaller (by about 2x) but are slower to decode on disk and would certainly
        # require caching to be performant.
        mask_cdf = data["mask_cdf"]
        mask_ids = data["pixel_to_mask_id"]
        # Using pixel_to_mask_id as indices, torch requires the indices to be at least int32.
        if mask_ids.dtype in [torch.int8, torch.int16]:
            mask_ids = mask_ids.to(torch.int32)

        scales = data["scales"]

        # Cache if enabled
        if self._cache_loaded_masks:
            self._loaded_masks_cache[index] = (mask_cdf, mask_ids, scales)

        return mask_cdf, mask_ids, scales

    def __getitem__(self, idx: int) -> SegmentationDataItem:  # pyright: ignore[reportIncompatibleMethodOverride]
        sfm_item = super().__getitem__(idx)
        index = self._indices[idx]

        mask_cdf, mask_ids, scales = self._get_mask_data(index)

        return SegmentationDataItem(
            image=torch.from_numpy(sfm_item["image"]),
            projection=sfm_item["projection"],
            camera_to_world=sfm_item["camera_to_world"],
            world_to_camera=sfm_item["world_to_camera"],
            scales=scales,
            mask_cdf=mask_cdf,
            mask_ids=mask_ids,
            image_h=sfm_item["image"].shape[0],
            image_w=sfm_item["image"].shape[1],
        )

    @property
    def scales(self) -> torch.Tensor:
        scales = []
        for index in self._indices:
            # Use cache if available, otherwise read from disk
            if self._cache_loaded_masks and index in self._loaded_masks_cache:
                scales.append(self._loaded_masks_cache[index][2])  # scales is third element
            else:
                _, data = self.sfm_scene.cache.read_file(f"masks_{index:0{self.num_zeropad}}")
                scales.append(data["scales"])
        return torch.cat(scales)

    @property
    def camera_to_world_matrices(self) -> np.ndarray:
        """
        Get the camera to world matrices for all images in the dataset.

        This returns the camera to world matrices as a numpy array of shape (N, 4, 4) where N is the number of images.

        Returns:
            np.ndarray: An Nx4x4 array of camera to world matrices for the cameras in the dataset.
        """
        return self.sfm_scene.camera_to_world_matrices[self._indices]

    @property
    def projection_matrices(self) -> np.ndarray:
        """
        Get the projection matrices mapping camera to pixel coordinates for all images in the dataset.

        This returns the undistorted projection matrices as a numpy array of shape (N, 3, 3) where N is the number of images.

        Returns:
            np.ndarray: An Nx3x3 array of projection matrices for the cameras in the dataset.
        """
        # in fvdb_3dgs/training/sfm_dataset.py
        return np.stack(
            [self._sfm_scene.images[i].camera_metadata.projection_matrix for i in self._indices],
            axis=0,
        )

    @property
    def indices(self) -> np.ndarray:
        """
        Return the indices of the images in the SfmScene used in the dataset.

        Returns:
            np.ndarray: The indices of the images in the SfmScene used in the dataset.
        """
        return self._indices


class InfiniteSampler(torch.utils.data.Sampler):
    """A sampler that yields indices infinitely, avoiding DataLoader iterator recreation pauses.

    This sampler never raises StopIteration, so the DataLoader iterator can run indefinitely.
    Epoch boundaries must be tracked manually by counting batches.
    """

    def __init__(self, dataset: Sized, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        while True:
            if self.shuffle:
                # Use epoch as part of seed for reproducibility across restarts
                g = torch.Generator()
                g.manual_seed(self.seed + self._epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))

            yield from indices
            self._epoch += 1

    def __len__(self):
        """Length is undefined for InfiniteSampler.

        This sampler yields indices indefinitely and does not define epoch
        boundaries. Code must not rely on len(sampler) to determine the number
        of steps per epoch; track epochs/steps explicitly instead.
        """
        raise NotImplementedError(
            "InfiniteSampler does not define __len__; it yields indices infinitely. "
            "Do not rely on len(sampler) for epoch length."
        )


class GARfVDBInput(Dict[str, Union[torch.Tensor, fvdb.JaggedTensor, list[int], None]]):
    """Dictionary with custom behavior for 3D Gaussian splatting inputs."""

    def __repr__(self):
        return f"GARfVDBInput({super().__repr__()})"

    def to(self, device: torch.device, non_blocking: bool = True) -> "GARfVDBInput":
        return GARfVDBInput(
            {k: v.to(device, non_blocking=non_blocking) if (type(v) in (torch.Tensor, fvdb.JaggedTensor)) else v for k, v in self.items()}  # type: ignore
        )


def GARfVDBInputCollateFn(batch: List[SegmentationDataItem]) -> GARfVDBInput:
    """Collate function for a DataLoader to stack the SegmentationDataItems into a GARfVDBInput.
    Args:
        batch: List of SegmentationDataItems.
    Returns:
        GARfVDBInput: A dictionary of tensors that is expected as input to the GARfVDB model.
    """

    kwargs = {
        "image": torch.stack([cast(torch.Tensor, b["image"]) for b in batch]),
        "projection": torch.stack([cast(torch.Tensor, b["projection"]) for b in batch]),
        "camera_to_world": torch.stack([cast(torch.Tensor, b["camera_to_world"]) for b in batch]),
        "world_to_camera": torch.stack([cast(torch.Tensor, b["world_to_camera"]) for b in batch]),
        "image_h": [cast(int, b["image_h"]) for b in batch],
        "image_w": [cast(int, b["image_w"]) for b in batch],
        "scales": torch.stack([cast(torch.Tensor, b["scales"]) for b in batch]),
        # "mask_cdf": torch.nested.nested_tensor([cast(torch.Tensor, b["mask_cdf"]) for b in batch]),
        "mask_ids": torch.stack([cast(torch.Tensor, b["mask_ids"]) for b in batch]),
    }

    if "pixel_coords" in batch[0]:
        kwargs["pixel_coords"] = torch.stack([cast(torch.Tensor, b.get("pixel_coords")) for b in batch])

    return GARfVDBInput(**kwargs)
