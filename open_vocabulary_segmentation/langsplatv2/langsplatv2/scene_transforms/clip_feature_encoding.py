# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""CLIP feature encoding transform for segmented image regions.

This transform computes CLIP features for masked image regions generated
by the multi-scale SAM transform, following the LangSplatV2 approach.
"""
import logging
from typing import Any

import cv2
import numpy as np
import torch
import tqdm
from fvdb_reality_capture.sfm_scene import SfmCache, SfmScene
from fvdb_reality_capture.transforms import BaseTransform, transform


def get_seg_img(mask: np.ndarray, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Extract and crop the segmented region from an image.

    Args:
        mask: Binary segmentation mask, shape [H, W].
        image: Source image, shape [H, W, 3].
        bbox: Bounding box in XYWH format.

    Returns:
        Cropped and masked image region.
    """
    x, y, w, h = map(int, bbox)

    # Crop first (view, no copy), then copy only the small region
    cropped_img = image[y : y + h, x : x + w].copy()
    cropped_mask = mask[y : y + h, x : x + w]

    # Apply mask only to the cropped region
    cropped_img[cropped_mask == 0] = 0

    return cropped_img


def pad_img(img: np.ndarray) -> np.ndarray:
    """
    Pad an image to make it square.

    Args:
        img: Input image, shape [H, W, 3].

    Returns:
        Square padded image.
    """
    h, w = img.shape[:2]
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)

    if h > w:
        pad[:, (h - w) // 2 : (h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2 : (w - h) // 2 + h, :, :] = img

    return pad


@transform
class ComputeCLIPFeatures(BaseTransform):
    """
    A transform that computes CLIP features for segmented image regions.

    This implements the LangSplatV2 feature encoding step where each segmented
    region from the multi-scale SAM masks is encoded using OpenCLIP. The output
    is a per-image tensor of CLIP features and segmentation maps that map
    pixels to feature indices.

    This transform must be run after ComputeMultiScaleSAMMasks.
    """

    version = "1.0.0"

    def __init__(
        self,
        clip_model_type: str = "ViT-B-16",
        clip_model_pretrained: str = "laion2b_s34b_b88k",
        clip_n_dims: int = 512,
        device: torch.device | str = "cuda",
    ):
        """
        Create a CLIP feature encoding transform.

        Args:
            clip_model_type: CLIP model architecture.
            clip_model_pretrained: Pretrained weights identifier.
            clip_n_dims: Embedding dimensionality.
            device: Device to run CLIP on.
        """
        self._clip_model_type = clip_model_type
        self._clip_model_pretrained = clip_model_pretrained
        self._clip_n_dims = clip_n_dims
        self._device = device

        # Lazy loading of CLIP model
        self._clip_model = None

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def _get_clip_model(self):
        """Lazily load the CLIP model."""
        if self._clip_model is None:
            from fvdb_reality_capture.foundation_models import OpenCLIPModel

            self._clip_model = OpenCLIPModel(
                model_type=self._clip_model_type,
                pretrained=self._clip_model_pretrained,
                device=self._device,
            )
        return self._clip_model

    def _encode_masked_regions(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        bboxes: np.ndarray,
    ) -> torch.Tensor:
        """
        Encode masked image regions using CLIP.

        Args:
            image: Source image in RGB format, shape [H, W, 3].
            masks: Binary masks, shape [N, H, W].
            bboxes: Bounding boxes in XYWH format, shape [N, 4].

        Returns:
            CLIP embeddings for each masked region, shape [N, clip_n_dims].
        """
        if len(masks) == 0:
            return torch.zeros(0, self._clip_n_dims)

        clip_model = self._get_clip_model()
        image_size = clip_model.image_size

        # Extract each masked region, pad to square, and resize to model's expected size
        seg_imgs = []
        for mask, bbox in zip(masks, bboxes):
            seg_img = get_seg_img(mask, image, bbox)
            pad_seg_img = pad_img(seg_img)
            resized_img = cv2.resize(pad_seg_img, (image_size, image_size))
            seg_imgs.append(resized_img)

        # Stack and convert to tensor with values in [0, 1]
        seg_imgs_np = np.stack(seg_imgs, axis=0)  # [N, image_size, image_size, 3]
        seg_imgs_tensor = (
            torch.from_numpy(seg_imgs_np.astype("float32")).permute(0, 3, 1, 2) / 255.0
        )  # [N, 3, image_size, image_size]

        # Apply model's tensor preprocessing (handles normalization) and encode
        with torch.no_grad():
            seg_imgs_preprocessed = clip_model.preprocess_tensor(seg_imgs_tensor)
            clip_embeds = clip_model.encode_image(seg_imgs_preprocessed)
            clip_embeds = clip_embeds / clip_embeds.norm(dim=-1, keepdim=True)

        return clip_embeds.cpu().half()

    def _create_segmentation_map(
        self,
        masks_dict: dict,
        image_shape: tuple,
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Create a multi-scale segmentation map from masks.

        Following LangSplatV2, this creates a tensor where each pixel maps to
        feature indices at different scales.

        Args:
            masks_dict: Dictionary with masks at each scale.
            image_shape: Shape of the original image (H, W).

        Returns:
            Tuple of (seg_maps, lengths) where:
                - seg_maps: Tensor of shape [4, H, W] mapping pixels to feature indices
                - lengths: Number of masks at each scale level
        """
        h, w = image_shape
        scale_names = ["default", "s", "m", "l"]

        seg_maps = torch.full((4, h, w), -1, dtype=torch.int32)
        lengths = []

        cumsum = 0
        for i, scale_name in enumerate(scale_names):
            masks = masks_dict.get(f"{scale_name}_segmentations", np.zeros((0, h, w)))
            num_masks = len(masks)
            lengths.append(num_masks)

            if num_masks == 0:
                continue

            # Create segmentation map for this scale
            scale_seg_map = np.full((h, w), -1, dtype=np.int32)
            for j, mask in enumerate(masks):
                # Assign global feature index (offset by cumulative sum)
                scale_seg_map[mask > 0] = cumsum + j

            seg_maps[i] = torch.from_numpy(scale_seg_map)
            cumsum += num_masks

        return seg_maps, lengths

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Compute CLIP features for all masked regions in the scene.

        Args:
            input_scene: Scene with multi-scale SAM masks in cache.

        Returns:
            Scene with CLIP features and segmentation maps in cache.
        """
        if len(input_scene.images) == 0:
            self._logger.warning("No images found in the SfmScene. Returning unchanged.")
            return input_scene

        input_cache: SfmCache = input_scene.cache

        # Create cache folder
        model_type_safe = self._clip_model_type.replace("-", "_")
        pretrained_safe = self._clip_model_pretrained.replace("-", "_")
        cache_prefix = f"clip_features_{model_type_safe}_{pretrained_safe}"
        output_cache = input_cache.make_folder(
            cache_prefix,
            description=f"CLIP features using {self._clip_model_type}",
        )

        num_zeropad = len(str(len(input_scene.images))) + 2
        regenerate_cache = False

        # Check if cache is valid
        if output_cache.num_files != input_scene.num_images:
            if output_cache.num_files != 0:
                self._logger.info(
                    f"Cache has {output_cache.num_files} files but expected "
                    f"{input_scene.num_images}. Regenerating cache."
                )
            output_cache.clear_current_folder()
            regenerate_cache = True

        if not regenerate_cache:
            cache_filename = f"features_{0:0{num_zeropad}}"
            if output_cache.has_file(cache_filename):
                cache_meta = output_cache.get_file_metadata(cache_filename)
                value_meta = cache_meta.get("metadata", {})
                if (
                    value_meta.get("clip_model_type") != self._clip_model_type
                    or value_meta.get("clip_model_pretrained") != self._clip_model_pretrained
                ):
                    self._logger.info("Cache parameters mismatch. Regenerating.")
                    output_cache.clear_current_folder()
                    regenerate_cache = True
            else:
                regenerate_cache = True

        if regenerate_cache:
            self._logger.info("Computing CLIP features for all masked regions.")
            pbar = tqdm.tqdm(input_scene.images, unit="imgs", desc="Computing CLIP features")

            for image_meta in pbar:
                image_path = image_meta.image_path
                img = cv2.imread(image_path)

                # Undistort the image if the camera has distortion parameters
                img = image_meta.camera_metadata.undistort_image(img)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]

                # Load SAM2 masks from parent cache
                mask_filename = f"masks_{image_meta.image_id:0{num_zeropad}}"
                if not input_cache.has_file(mask_filename):
                    raise RuntimeError(
                        f"Mask file {mask_filename} not found in cache. " "Run ComputeMultiScaleSAM2Masks first."
                    )

                _, mask_data = input_cache.read_file(mask_filename)

                # Encode all masked regions across scales
                all_features = []
                scale_names = ["default", "s", "m", "l"]

                for scale_name in scale_names:
                    masks = mask_data.get(f"{scale_name}_segmentations", np.zeros((0, h, w)))
                    bboxes = mask_data.get(f"{scale_name}_bboxes", np.zeros((0, 4)))

                    if len(masks) > 0:
                        scale_features = self._encode_masked_regions(img_rgb, masks, bboxes)
                        all_features.append(scale_features)

                # Concatenate features from all scales
                if len(all_features) > 0:
                    features = torch.cat(all_features, dim=0)
                else:
                    features = torch.zeros(0, self._clip_n_dims)

                # Create segmentation maps
                seg_maps, lengths = self._create_segmentation_map(mask_data, (h, w))

                # Verify consistency
                total_masks = sum(lengths)
                assert features.shape[0] == total_masks, f"Feature count mismatch: {features.shape[0]} vs {total_masks}"

                # Save
                cache_filename = f"features_{image_meta.image_id:0{num_zeropad}}"
                output_cache.write_file(
                    name=cache_filename,
                    data={
                        "features": features,  # [N_total, clip_n_dims]
                        "seg_maps": seg_maps,  # [4, H, W]
                        "lengths": torch.tensor(lengths, dtype=torch.int32),  # [4]
                    },
                    data_type="pt",
                    metadata={
                        "clip_model_type": self._clip_model_type,
                        "clip_model_pretrained": self._clip_model_pretrained,
                        "clip_n_dims": self._clip_n_dims,
                    },
                )

            pbar.close()
            self._logger.info(f"Computed CLIP features for {input_scene.num_images} images.")
        else:
            self._logger.info("Loading CLIP features from cache.")

        output_scene = SfmScene(
            cameras=input_scene.cameras,
            images=input_scene.images,
            points=input_scene.points,
            points_err=input_scene.points_err,
            points_rgb=input_scene.points_rgb,
            scene_bbox=input_scene.scene_bbox,
            transformation_matrix=input_scene.transformation_matrix,
            cache=output_cache,
        )

        return output_scene

    @staticmethod
    def name() -> str:
        return "ComputeCLIPFeatures"

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
            "version": self.version,
            "clip_model_type": self._clip_model_type,
            "clip_model_pretrained": self._clip_model_pretrained,
            "clip_n_dims": self._clip_n_dims,
            "device": str(self._device),
        }

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "ComputeCLIPFeatures":
        if state_dict["name"] != "ComputeCLIPFeatures":
            raise ValueError(
                f"Expected state_dict with name 'ComputeCLIPFeatures', " f"got {state_dict['name']} instead."
            )

        return ComputeCLIPFeatures(
            clip_model_type=state_dict["clip_model_type"],
            clip_model_pretrained=state_dict["clip_model_pretrained"],
            clip_n_dims=state_dict["clip_n_dims"],
            device=state_dict["device"],
        )
