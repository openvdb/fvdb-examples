# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""CLIP feature encoding transform for segmented image regions.

This transform computes CLIP features for masked image regions generated
by the multi-scale SAM transform, following the LangSplatV2 approach.

Crop extraction, masking, padding, and resize are performed on the GPU in
float32 to avoid uint8 quantisation artefacts that occur with ``cv2.resize`` on small masks.
"""
import logging
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from fvdb_reality_capture.sfm_scene import SfmCache, SfmScene
from fvdb_reality_capture.transforms import BaseTransform, transform


@transform
class ComputeCLIPFeatures(BaseTransform):
    """
    A transform that computes CLIP features for segmented image regions.

    This implements the LangSplatV2 feature encoding step where each segmented
    region from the multi-scale SAM masks is encoded using OpenCLIP. The output
    is a per-image tensor of CLIP features and segmentation maps that map
    pixels to feature indices.

    This transform must be run after ComputeMultiScaleSAM2Masks.
    """

    version = "1.1.0"

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
        image_gpu: torch.Tensor,
        masks: np.ndarray,
        bboxes: np.ndarray,
    ) -> torch.Tensor:
        """
        Encode masked image regions using CLIP.

        Performs crop-mask-pad-resize entirely on GPU in float32 (matching
        the original LangSplatV2 ``mask2segmap`` + ``_embed_clip_sam_tiles``),
        then normalises and encodes through CLIP in one batch.

        Args:
            image_gpu: Source image on CUDA as float32, shape [H, W, 3],
                values in [0, 255].
            masks: Binary masks, shape [N, H, W] (uint8 0/1).
            bboxes: Bounding boxes in XYWH format, shape [N, 4].

        Returns:
            CLIP embeddings for each masked region, shape [N, clip_n_dims].
        """
        if len(masks) == 0:
            return torch.zeros(0, self._clip_n_dims, device="cpu", dtype=torch.float16)

        clip_model = self._get_clip_model()
        image_size = clip_model.image_size
        n = len(masks)
        device = image_gpu.device

        all_segs = torch.from_numpy(masks).to(device)
        seg_imgs = torch.empty(n, 3, image_size, image_size, device=device)

        for i in range(n):
            x, y, w, h = int(bboxes[i][0]), int(bboxes[i][1]), int(bboxes[i][2]), int(bboxes[i][3])
            w = max(w, 1)
            h = max(h, 1)

            crop = image_gpu[y : y + h, x : x + w].clone()
            crop[~all_segs[i, y : y + h, x : x + w].bool()] = 0
            crop = crop.permute(2, 0, 1)  # HWC -> CHW

            side = max(h, w)
            padded = torch.zeros(3, side, side, device=device)
            if h > w:
                offset = (h - w) // 2
                padded[:, :, offset : offset + w] = crop
            else:
                offset = (w - h) // 2
                padded[:, offset : offset + h, :] = crop

            seg_imgs[i] = F.interpolate(
                padded.unsqueeze(0), size=(image_size, image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        seg_imgs /= 255.0

        # Normalise and encode
        with torch.no_grad():
            seg_imgs_preprocessed = clip_model.preprocess_tensor(seg_imgs)
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
        version_safe = self.version.replace(".", "_")
        cache_prefix = f"clip_features_{model_type_safe}_{pretrained_safe}_{self._clip_n_dims}_v{version_safe}"
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

        for image_id in range(input_scene.num_images):
            if regenerate_cache:
                break
            cache_filename = f"features_{image_id:0{num_zeropad}}"
            if not output_cache.has_file(cache_filename):
                self._logger.info(
                    f"Features {cache_filename} not found in the cache. " f"Clearing cache and regenerating."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break

            cache_meta = output_cache.get_file_metadata(cache_filename)
            value_meta = cache_meta.get("metadata", {})
            if (
                value_meta.get("clip_model_type") != self._clip_model_type
                or value_meta.get("clip_model_pretrained") != self._clip_model_pretrained
                or value_meta.get("clip_n_dims") != self._clip_n_dims
            ):
                self._logger.info(
                    f"Cache metadata does not match expected parameters. " f"Clearing cache and regenerating."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break

        if regenerate_cache:
            self._logger.info("Computing CLIP features for all masked regions.")
            pbar = tqdm.tqdm(input_scene.images, unit="imgs", desc="Computing CLIP features")

            for image_meta in pbar:
                image_path = image_meta.image_path
                img = cv2.imread(image_path)
                assert img is not None, f"Failed to load image {image_path}"

                # Undistort the image if the camera has distortion parameters
                img = image_meta.camera_metadata.undistort_image(img)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]

                img_gpu = torch.from_numpy(img_rgb).to(self._device, dtype=torch.float32)

                mask_filename = f"masks_{image_meta.image_id:0{num_zeropad}}"
                if not input_cache.has_file(mask_filename):
                    raise RuntimeError(
                        f"Mask file {mask_filename} not found in cache. "
                        "Run ComputeMultiScaleSAM2Masks or ComputeMultiScaleSAM1Masks first."
                    )

                _, mask_data = input_cache.read_file(mask_filename)

                all_features = []
                scale_names = ["default", "s", "m", "l"]

                for scale_name in scale_names:
                    masks = mask_data.get(f"{scale_name}_segmentations", np.zeros((0, h, w)))
                    bboxes = mask_data.get(f"{scale_name}_bboxes", np.zeros((0, 4)))

                    if len(masks) > 0:
                        scale_features = self._encode_masked_regions(img_gpu, masks, bboxes)
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

                if features.shape[0] > 0 and torch.isnan(features).any():
                    self._logger.warning(
                        f"Image {image_meta.image_id}: CLIP features contain NaN "
                        f"({torch.isnan(features).sum().item()} / {features.numel()} values)"
                    )

                # Report per-scale mask coverage for the first few images
                if image_meta.image_id < 3:
                    coverage = {sn: int((seg_maps[i] >= 0).sum()) for i, sn in enumerate(scale_names)}
                    self._logger.debug(
                        f"Image {image_meta.image_id}: {total_masks} masks, "
                        f"lengths={lengths}, pixel coverage={coverage}"
                    )

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
