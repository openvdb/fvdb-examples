# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Multi-scale SAM2 segmentation transform.

This transform generates segmentation masks at multiple scales using SAM2,
following the LangSplatV2 preprocessing approach but with SAM2 instead of SAM v1.
"""
import logging
from typing import Any, Literal

import cv2
import numpy as np
import torch
import tqdm
from fvdb_reality_capture.foundation_models import SAM2Model
from fvdb_reality_capture.sfm_scene import SfmCache, SfmScene
from fvdb_reality_capture.transforms import BaseTransform, transform


def mask_nms(
    masks: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float = 0.7,
    score_thr: float = 0.1,
    inner_thr: float = 0.2,
) -> torch.Tensor:
    """
    Perform mask non-maximum suppression (NMS) on a set of masks.

    Removes redundant masks based on IoU overlap and inner containment.
    This implementation is fully vectorized for efficient GPU computation.

    Args:
        masks: Binary masks, shape [num_masks, H, W].
        scores: Mask scores, shape [num_masks].
        iou_thr: IoU threshold for NMS.
        score_thr: Minimum score threshold.
        inner_thr: Inner overlap threshold for removing contained masks.

    Returns:
        Indices of selected masks after NMS.
    """
    if len(masks) == 0:
        return torch.tensor([], dtype=torch.long)

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]

    # Flatten masks for vectorized computation: [N, H*W]
    # Use reshape instead of view since indexing may produce non-contiguous tensor
    masks_flat = masks_ord.reshape(num_masks, -1).float()

    # Compute all pairwise intersections in one matrix multiply
    # For binary masks: intersection[i,j] = sum(mask_i & mask_j) = mask_i @ mask_j.T
    intersection = masks_flat @ masks_flat.T  # [N, N]

    # Compute areas
    masks_area = masks_flat.sum(dim=1)  # [N]

    # Compute unions: union[i,j] = area_i + area_j - intersection[i,j]
    union = masks_area[:, None] + masks_area[None, :] - intersection

    # Compute IoU matrix (avoid division by zero)
    iou_matrix = torch.where(union > 0, intersection / union, torch.zeros_like(union))

    # Compute containment ratios for inner IoU
    # ratio_i[i,j] = intersection[i,j] / area_i
    # ratio_j[i,j] = intersection[i,j] / area_j
    area_i = masks_area[:, None].expand_as(intersection)
    area_j = masks_area[None, :].expand_as(intersection)

    # Avoid division by zero
    ratio_i = torch.where(area_i > 0, intersection / area_i, torch.zeros_like(intersection))
    ratio_j = torch.where(area_j > 0, intersection / area_j, torch.zeros_like(intersection))

    # Inner IoU for containment detection
    # Case 1: j is mostly contained in i (ratio_i < 0.5 and ratio_j >= 0.85)
    # Case 2: i is mostly contained in j (ratio_i >= 0.85 and ratio_j < 0.5)
    inner_iou_values = 1 - ratio_i * ratio_j

    # Build inner_iou_matrix based on containment conditions
    inner_iou_matrix = torch.zeros_like(iou_matrix)

    # Upper triangle: j contained in i
    condition_upper = (ratio_i < 0.5) & (ratio_j >= 0.85)
    inner_iou_matrix = torch.where(condition_upper, inner_iou_values, inner_iou_matrix)

    # Lower triangle: i contained in j (transpose the condition)
    condition_lower = (ratio_i >= 0.85) & (ratio_j < 0.5)
    inner_iou_matrix = torch.where(condition_lower, inner_iou_values.T, inner_iou_matrix)

    # Apply triangular masks and compute max values
    iou_matrix = torch.triu(iou_matrix, diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)

    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=-1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    # If no masks pass thresholds, keep top 3
    if keep_conf.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_conf[index] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_inner_u[index] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_inner_l[index] = True

    keep = keep & keep_conf & keep_inner_u & keep_inner_l
    selected_idx = idx[keep]

    return selected_idx


def masks_update(
    *mask_lists,
    iou_thr: float = 0.8,
    score_thr: float = 0.7,
    inner_thr: float = 0.5,
) -> tuple:
    """
    Apply mask NMS to multiple lists of masks.

    Args:
        *mask_lists: Variable number of mask lists to filter.
        iou_thr: IoU threshold for NMS.
        score_thr: Score threshold.
        inner_thr: Inner overlap threshold.

    Returns:
        Tuple of filtered mask lists.
    """
    masks_new = []

    for masks_lvl in mask_lists:
        if len(masks_lvl) == 0:
            masks_new.append([])
            continue

        seg_pred = torch.from_numpy(np.stack([m["segmentation"] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m["predicted_iou"] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m["stability_score"] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, iou_thr=iou_thr, score_thr=score_thr, inner_thr=inner_thr)

        keep_set = set(keep_mask_nms.int().cpu().numpy().tolist())
        filtered_masks = [m for i, m in enumerate(masks_lvl) if i in keep_set]
        masks_new.append(filtered_masks)

    return tuple(masks_new)


@transform
class ComputeMultiScaleSAM2Masks(BaseTransform):
    """
    A transform that generates multi-scale segmentation masks using SAM2.

    This implements the LangSplatV2 preprocessing approach of generating masks
    at multiple scales (default, small, medium, large) using SAM2 (instead of
    SAM v1 in the original) and applying mask NMS to remove redundant masks.

    The output is cached per-image and includes:
    - masks_default: All masks
    - masks_s: Small-scale masks (area < 1% of image)
    - masks_m: Medium-scale masks (1% <= area < 10% of image)
    - masks_l: Large-scale masks (area >= 10% of image)
    """

    version = "1.0.0"

    def __init__(
        self,
        checkpoint: Literal["large", "small", "tiny", "base_plus"] = "large",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.85,
        nms_iou_thr: float = 0.8,
        nms_score_thr: float = 0.7,
        nms_inner_thr: float = 0.5,
        device: torch.device | str = "cuda",
    ):
        """
        Create a multi-scale SAM2 mask generation transform.

        Args:
            checkpoint: SAM2 checkpoint size to use.
            points_per_side: Grid density for point prompts.
            pred_iou_thresh: Predicted IoU threshold.
            stability_score_thresh: Stability score threshold.
            nms_iou_thr: IoU threshold for mask NMS post-processing.
            nms_score_thr: Score threshold for mask NMS.
            nms_inner_thr: Inner overlap threshold for mask NMS.
            device: Device to run SAM2 on.
        """
        self._checkpoint = checkpoint
        self._points_per_side = points_per_side
        self._pred_iou_thresh = pred_iou_thresh
        self._stability_score_thresh = stability_score_thresh
        self._nms_iou_thr = nms_iou_thr
        self._nms_score_thr = nms_score_thr
        self._nms_inner_thr = nms_inner_thr
        self._device = device

        # Lazy loading of SAM2 model
        self._sam2_model: SAM2Model | None = None

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def _get_sam2_model(self) -> SAM2Model:
        if self._sam2_model is None:
            self._sam2_model = SAM2Model(
                checkpoint=self._checkpoint,
                points_per_side=self._points_per_side,
                pred_iou_thresh=self._pred_iou_thresh,
                stability_score_thresh=self._stability_score_thresh,
                device=self._device,
            )
        return self._sam2_model

    @staticmethod
    def _categorize_masks_by_scale(
        masks: list[dict],
        image_area: int,
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        """
        Categorize masks into scale levels based on area ratio.

        Following LangSplatV2's approach:
        - Large: area >= 10% of image
        - Medium: 1% <= area < 10% of image
        - Small: area < 1% of image
        - Default: all masks

        Args:
            masks: List of mask dictionaries from SAM2.
            image_area: Total image area in pixels.

        Returns:
            Tuple of (masks_default, masks_s, masks_m, masks_l).
        """
        masks_default = []
        masks_s = []
        masks_m = []
        masks_l = []

        for mask in masks:
            area_ratio = mask["area"] / image_area

            # Classify by area ratio
            if area_ratio >= 0.1:
                masks_l.append(mask)
            elif area_ratio >= 0.01:
                masks_m.append(mask)
            else:
                masks_s.append(mask)

            # All masks go to default
            masks_default.append(mask)

        return masks_default, masks_s, masks_m, masks_l

    def _generate_multi_scale_masks(self, image: np.ndarray) -> dict:
        """
        Generate masks at multiple scales for a single image.

        Args:
            image: Input image in BGR format (OpenCV default), shape [H, W, 3].

        Returns:
            Dictionary with masks at each scale level.
        """
        # Convert BGR to RGB for SAM2
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        image_area = h * w

        sam2_model = self._get_sam2_model()

        # Generate masks using SAM2
        all_masks = sam2_model.predict_masks(image_rgb)

        # Categorize by scale
        masks_default, masks_s, masks_m, masks_l = self._categorize_masks_by_scale(all_masks, image_area)

        # Apply mask NMS to remove redundant masks
        masks_default, masks_s, masks_m, masks_l = masks_update(
            masks_default,
            masks_s,
            masks_m,
            masks_l,
            iou_thr=self._nms_iou_thr,
            score_thr=self._nms_score_thr,
            inner_thr=self._nms_inner_thr,
        )

        return {
            "default": masks_default,
            "s": masks_s,
            "m": masks_m,
            "l": masks_l,
        }

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Generate multi-scale SAM2 masks for all images in the scene.

        Args:
            input_scene: Input scene containing images.

        Returns:
            Scene with cache containing multi-scale mask data.
        """
        if len(input_scene.images) == 0:
            self._logger.warning("No images found in the SfmScene. Returning unchanged.")
            return input_scene

        input_cache: SfmCache = input_scene.cache

        # Create cache folder
        cache_prefix = (
            f"sam2_multi_scale_masks_{self._checkpoint}_"
            f"p{self._points_per_side}_"
            f"iou{int(self._pred_iou_thresh * 100)}_"
            f"stab{int(self._stability_score_thresh * 100)}"
        )
        output_cache = input_cache.make_folder(
            cache_prefix,
            description=f"Multi-scale SAM2 masks with {self._checkpoint} checkpoint",
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
            # Verify cache metadata for first image
            cache_filename = f"masks_{0:0{num_zeropad}}"
            if output_cache.has_file(cache_filename):
                cache_meta = output_cache.get_file_metadata(cache_filename)
                value_meta = cache_meta.get("metadata", {})
                if (
                    value_meta.get("checkpoint") != self._checkpoint
                    or value_meta.get("points_per_side") != self._points_per_side
                ):
                    self._logger.info("Cache parameters mismatch. Regenerating.")
                    output_cache.clear_current_folder()
                    regenerate_cache = True
            else:
                regenerate_cache = True

        if regenerate_cache:
            self._logger.info("Generating multi-scale SAM2 masks for all images.")
            pbar = tqdm.tqdm(input_scene.images, unit="imgs", desc="Generating SAM2 masks")

            for image_meta in pbar:
                image_path = image_meta.image_path
                img = cv2.imread(image_path)
                assert img is not None, f"Failed to load image {image_path}"

                # Undistort the image if the camera has distortion parameters
                img = image_meta.camera_metadata.undistort_image(img)

                # Generate multi-scale masks
                masks_dict = self._generate_multi_scale_masks(img)

                # Convert masks to storable format
                mask_data = {}
                for scale_name, masks in masks_dict.items():
                    if len(masks) > 0:
                        # Store segmentation masks and metadata
                        mask_data[f"{scale_name}_segmentations"] = np.stack(
                            [m["segmentation"].astype(np.uint8) for m in masks], axis=0
                        )
                        mask_data[f"{scale_name}_bboxes"] = np.array([m["bbox"] for m in masks], dtype=np.float32)
                        mask_data[f"{scale_name}_areas"] = np.array([m["area"] for m in masks], dtype=np.int32)
                        mask_data[f"{scale_name}_predicted_ious"] = np.array(
                            [m["predicted_iou"] for m in masks], dtype=np.float32
                        )
                        mask_data[f"{scale_name}_stability_scores"] = np.array(
                            [m["stability_score"] for m in masks], dtype=np.float32
                        )
                    else:
                        # Empty arrays for scales with no masks
                        mask_data[f"{scale_name}_segmentations"] = np.zeros(
                            (0, img.shape[0], img.shape[1]), dtype=np.uint8
                        )
                        mask_data[f"{scale_name}_bboxes"] = np.zeros((0, 4), dtype=np.float32)
                        mask_data[f"{scale_name}_areas"] = np.zeros(0, dtype=np.int32)
                        mask_data[f"{scale_name}_predicted_ious"] = np.zeros(0, dtype=np.float32)
                        mask_data[f"{scale_name}_stability_scores"] = np.zeros(0, dtype=np.float32)

                # Save to cache
                cache_filename = f"masks_{image_meta.image_id:0{num_zeropad}}"
                output_cache.write_file(
                    name=cache_filename,
                    data=mask_data,
                    data_type="pt",
                    metadata={
                        "checkpoint": self._checkpoint,
                        "points_per_side": self._points_per_side,
                        "pred_iou_thresh": self._pred_iou_thresh,
                        "stability_score_thresh": self._stability_score_thresh,
                    },
                )

            pbar.close()
            self._logger.info(f"Generated masks for {input_scene.num_images} images.")
        else:
            self._logger.info("Loading masks from cache.")

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
        return "ComputeMultiScaleSAM2Masks"

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
            "version": self.version,
            "checkpoint": self._checkpoint,
            "points_per_side": self._points_per_side,
            "pred_iou_thresh": self._pred_iou_thresh,
            "stability_score_thresh": self._stability_score_thresh,
            "nms_iou_thr": self._nms_iou_thr,
            "nms_score_thr": self._nms_score_thr,
            "nms_inner_thr": self._nms_inner_thr,
            "device": str(self._device),
        }

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "ComputeMultiScaleSAM2Masks":
        if state_dict["name"] != "ComputeMultiScaleSAM2Masks":
            raise ValueError(
                f"Expected state_dict with name 'ComputeMultiScaleSAM2Masks', " f"got {state_dict['name']} instead."
            )

        return ComputeMultiScaleSAM2Masks(
            checkpoint=state_dict["checkpoint"],
            points_per_side=state_dict["points_per_side"],
            pred_iou_thresh=state_dict["pred_iou_thresh"],
            stability_score_thresh=state_dict["stability_score_thresh"],
            nms_iou_thr=state_dict["nms_iou_thr"],
            nms_score_thr=state_dict["nms_score_thr"],
            nms_inner_thr=state_dict["nms_inner_thr"],
            device=state_dict["device"],
        )
