# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from garfvdb.training.dataset import SegmentationDataItem, SegmentationDataset


class TransformedSegmentationDataset(Dataset):
    """Wrapper dataset that applies transforms to a SegmentationDataset.

    Applies torchvision-style transforms to each item returned by the base
    dataset, enabling data augmentation and preprocessing pipelines.
    """

    def __init__(self, base_dataset: SegmentationDataset, transform=None) -> None:
        self._base_dataset = base_dataset
        self._transform = transform

    def __getitem__(self, idx):
        item = self._base_dataset[idx]
        if self._transform:
            return self._transform(item)
        return item

    def __len__(self):
        return len(self._base_dataset)

    @property
    def base_dataset(self) -> SegmentationDataset:
        return self._base_dataset

    @property
    def indices(self) -> np.ndarray:
        return self._base_dataset.indices

    def warmup_cache(self) -> None:
        """Pre-load all data into cache before DataLoader workers are spawned."""
        self._base_dataset.warmup_cache()


class RandomSelectMaskIDAndScale:
    """Transform that randomly selects a mask ID and interpolates scale values.

    For each pixel, randomly selects one of the overlapping masks based on the
    mask CDF (which biases towards smaller masks) and interpolates the scale
    value to create smooth transitions between hierarchical groupings.
    """

    def __call__(self, item: SegmentationDataItem) -> SegmentationDataItem:
        """Apply random mask selection and scale interpolation.

        Args:
            item: Input data item with multi-mask information per pixel.

        Returns:
            Modified item with single mask ID and interpolated scale per pixel.
        """

        per_pixel_index = item["mask_ids"]  # [H, W, MM] or [num_samples, MM]
        random_vec_sampling = torch.full(per_pixel_index.shape[:-1], torch.rand((1,)).item()).unsqueeze(-1)  # [H, W, 1]
        random_vec_densify = torch.full(per_pixel_index.shape[:-1], torch.rand((1,)).item())  # [H, W] or [num_samples]

        random_index = torch.sum(random_vec_sampling > item["mask_cdf"], dim=-1)  # [H, W] dtype: torch.int64

        # `per_pixel_index` encodes the list of groups that each pixel belongs to.
        # If there's only one group, then `per_pixel_index` is a 1D tensor
        # -- this will mess up the future `gather` operations.
        if per_pixel_index.shape[-1] == 1:
            per_pixel_mask = per_pixel_index.squeeze()
        else:
            per_pixel_mask = torch.gather(
                per_pixel_index, -1, random_index.unsqueeze(-1)
            ).squeeze()  # [H, W] dtype: torch.int64
            # per_pixel_mask_ is a selection of the *previous* group in the list before the per_pixel_mask selection
            per_pixel_mask_ = torch.gather(
                per_pixel_index,
                -1,
                torch.max(random_index.unsqueeze(-1) - 1, torch.Tensor([0]).int()),
            ).squeeze()

        scales = item["scales"]  # [NM] dtype: torch.float32
        curr_scale = scales[per_pixel_mask]  # [H, W] dtype: torch.float32

        # For pixels in the first group (random_index == 0), randomly scale down their scale value
        # between 0 and the full scale. This creates a smooth transition from zero to the first group's scale,
        # similar to how we interpolate between groups for other indices.
        curr_scale[random_index == 0] = (
            scales[per_pixel_mask][random_index == 0] * random_vec_densify[random_index == 0]
        )
        # For each group, interpolate between the previous group's scale and the current group's scale,
        # based on the random_vec_densify value. This creates a smooth transition between groups.
        for j in range(1, item["mask_cdf"].shape[-1]):
            if (random_index == j).sum() == 0:
                continue
            curr_scale[random_index == j] = (
                scales[per_pixel_mask_][random_index == j]  # type: ignore
                + (scales[per_pixel_mask][random_index == j] - scales[per_pixel_mask_][random_index == j])  # type: ignore
                * random_vec_densify[random_index == j]
            ).squeeze()

        item["scales"] = curr_scale  # [rays_per_image] dtype: torch.float32

        item["mask_ids"] = per_pixel_mask  # [rays_per_image] dtype: torch.int64

        return item


class RandomSamplePixels:
    """Transform that randomly samples pixels from an image.

    Samples a subset of pixels for efficient training. Supports optional
    importance sampling to bias towards smaller-scale regions, which can
    improve learning of fine-grained segmentation boundaries. The original
    full image is preserved in the ``image_full`` field.
    """

    def __init__(self, num_samples_per_image: int, scale_bias_strength: float = 0.0):
        """
        Args:
            num_samples_per_image: Number of pixels to sample per image.
            scale_bias_strength: Strength of bias towards smaller scales.
                                0.0 = uniform random sampling (default behavior)
                                > 0.0 = bias towards smaller scales (higher values = stronger bias)
        """
        self.num_samples_per_image = num_samples_per_image
        self.scale_bias_strength = scale_bias_strength

    def __call__(self, item: SegmentationDataItem) -> SegmentationDataItem:
        """Sample pixels from the image.

        Args:
            item: Input data item with full-resolution data.

        Returns:
            Modified item where ``image``, ``mask_ids``, and ``mask_cdf``
            contain only the sampled pixels. Original coordinates are stored
            in ``pixel_coords`` and the full image in ``image_full``.
        """
        h, w = item["image_h"], item["image_w"]

        if self.scale_bias_strength > 0.0 and "scales" in item:
            # Use importance sampling based on scales (smaller scales = higher probability)
            scales = item["scales"]  # [NM] - scale per mask
            mask_ids = item["mask_ids"]  # [H, W, MM] - mask IDs per pixel

            # Get scale values for each pixel by indexing scales with mask_ids
            # Handle invalid mask IDs (typically -1) by masking them out
            valid_mask = mask_ids >= 0  # [H, W, MM]

            # Vectorized computation of per-pixel scales
            # Clamp mask_ids to valid range to avoid index errors when accessing scales
            clamped_mask_ids = torch.clamp(mask_ids, 0, len(scales) - 1)  # [H, W, MM]

            # Get scale values for all pixels at once using advanced indexing
            pixel_scale_values = scales[clamped_mask_ids]  # [H, W, MM]

            # Mask out invalid entries (set to inf so they don't affect min operation)
            pixel_scale_values = torch.where(valid_mask, pixel_scale_values, float("inf"))

            # Get minimum scale per pixel across the MM dimension
            pixel_scales, _ = torch.min(pixel_scale_values, dim=-1)  # [H, W]

            # Handle pixels with no valid masks (where all scales were inf)
            inf_mask = pixel_scales == float("inf")
            if inf_mask.any():
                median_scale = torch.median(scales)
                pixel_scales[inf_mask] = median_scale

            # Convert scales to sampling probabilities (smaller scales = higher prob)
            inv_scales = 1.0 / (pixel_scales + 1e-8)  # Add small epsilon to avoid division by zero

            # Apply bias strength (higher strength = more bias towards small scales)
            if self.scale_bias_strength != 1.0:
                inv_scales = torch.pow(inv_scales, self.scale_bias_strength)

            # Flatten and normalize to get probabilities
            flat_probs = inv_scales.flatten()
            flat_probs = flat_probs / flat_probs.sum()

            # Determine sampling parameters
            total_pixels = h * w
            num_samples = min(self.num_samples_per_image, total_pixels)

            # Sample according to probabilities - keep as tensor for vectorized ops
            flat_indices = torch.multinomial(flat_probs, num_samples, replacement=False)
            # Compute row/col directly
            pixels = torch.empty((num_samples, 2), dtype=torch.long)
            pixels[:, 0] = flat_indices // w  # row
            pixels[:, 1] = flat_indices % w  # col
        else:
            # Uniform random sampling (with replacement, but duplicates are negligible)
            # For 4096 samples from 2M pixels: ~4 expected duplicates (0.1%)
            total_pixels = h * w
            num_samples = min(self.num_samples_per_image, total_pixels)
            flat_indices = torch.randint(0, total_pixels, (num_samples,))
            pixels = torch.empty((num_samples, 2), dtype=torch.long)
            pixels[:, 0] = flat_indices // w  # row
            pixels[:, 1] = flat_indices % w  # col

        item["image_full"] = item["image"]
        item["image"] = item["image"][pixels[:, 0], pixels[:, 1]]
        item["mask_ids"] = item["mask_ids"][pixels[:, 0], pixels[:, 1]]
        item["mask_cdf"] = item["mask_cdf"][pixels[:, 0], pixels[:, 1]]
        item["pixel_coords"] = pixels

        return item


class Resize:
    """Transform that resizes images and masks by a scale factor."""

    def __init__(self, scale: float) -> None:
        """Initialize the resize transform.

        Args:
            scale: Scale factor to apply (e.g., 0.5 for half resolution).
        """
        self.scale = scale

    def __call__(self, item: SegmentationDataItem) -> SegmentationDataItem:
        """Resize image, masks, and update projection matrix.

        Args:
            item: Input data item to resize.

        Returns:
            Resized item with updated dimensions and scaled projection matrix.
        """
        # Resize image from [H, W, 3] to [H * scale, W * scale, 3]
        item["image"] = (
            F.interpolate(
                item["image"].unsqueeze(0).permute(0, 3, 1, 2),  # [1, 3, H, W]
                scale_factor=self.scale,
                mode="nearest",
            )
            .permute(0, 2, 3, 1)
            .squeeze(0)
        )  # back to [H * scale, W * scale, 3]

        # Update dimensions
        item["image_h"] = int(item["image_h"] * self.scale)
        item["image_w"] = int(item["image_w"] * self.scale)

        # Resize masks similarly
        item["mask_cdf"] = (
            F.interpolate(item["mask_cdf"].unsqueeze(0).permute(0, 3, 1, 2), scale_factor=self.scale, mode="nearest")
            .permute(0, 2, 3, 1)
            .squeeze(0)
        )

        item["mask_ids"] = (
            TF.resize(
                item["mask_ids"].unsqueeze(0).permute(0, 3, 1, 2),
                size=[item["image"].shape[0], item["image"].shape[1]],
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            )
            .permute(0, 2, 3, 1)
            .squeeze(0)
        )

        # scale intrinsics for new image size
        fx = item["projection"][0, 0]
        fy = item["projection"][1, 1]
        cx = item["projection"][0, 2]
        cy = item["projection"][1, 2]
        new_fx = fx / self.scale
        new_fy = fy / self.scale
        new_cx = cx / self.scale
        new_cy = cy / self.scale
        item["projection"] = torch.tensor([[new_fx, 0, new_cx], [0, new_fy, new_cy], [0, 0, 1]], dtype=torch.float32)

        return item
