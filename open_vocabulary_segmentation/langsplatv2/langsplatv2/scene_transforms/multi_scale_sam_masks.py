# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Multi-scale SAM2 segmentation transform.

Uses :class:`fvdb_reality_capture.foundation_models.SAM2Model` with
``output_mode="multi_scale"`` for a generic "one image/crop + points -> 4 scale
lists" API. This module implements the LangSplatV2-specific business logic:
multi-crop generation, point grids per layer, cross-crop NMS, and mask NMS.
"""
import logging
from typing import Any, Dict, List, Literal

import cv2
import numpy as np
import torch
import tqdm
import sam2.utils.amg as _sam2_amg

from fvdb_reality_capture.foundation_models import SAM2Model
from fvdb_reality_capture.sfm_scene import SfmCache, SfmScene
from fvdb_reality_capture.transforms import BaseTransform, transform

from .mask_utils import cross_crop_nms, masks_update, postprocess_small_regions


@transform
class ComputeMultiScaleSAM2Masks(BaseTransform):
    """Generate multi-scale segmentation masks using SAM2.

    Uses :class:`fvdb_reality_capture.foundation_models.SAM2Model` with
    ``output_mode="multi_scale"`` to split the 3 multimask outputs per point
    by index (small/medium/large). After generation, mask NMS is applied
    to each scale level independently.
    """

    version = "1.2.0"

    def __init__(
        self,
        checkpoint: Literal["large", "small", "tiny", "base_plus"] = "large",
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.5,
        stability_score_thresh: float = 0.85,
        crop_n_layers: int = 1,
        crop_n_points_downscale_factor: int = 1,
        min_mask_region_area: int = 100,
        box_nms_thresh: float = 0.7,
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
            points_per_batch: Points processed simultaneously.
            pred_iou_thresh: Predicted IoU threshold.
            stability_score_thresh: Stability score threshold.
            crop_n_layers: Number of crop layers (1 = also run on crops,
                matching the original LangSplatV2).
            crop_n_points_downscale_factor: Point grid downscale per crop layer.
            min_mask_region_area: Minimum mask region area for post-processing.
            box_nms_thresh: Box NMS IoU threshold within each crop.
            nms_iou_thr: IoU threshold for mask NMS post-processing.
            nms_score_thr: Score threshold for mask NMS.
            nms_inner_thr: Inner overlap threshold for mask NMS.
            device: Device to run SAM2 on.
        """
        self._checkpoint = checkpoint
        self._points_per_side = points_per_side
        self._points_per_batch = points_per_batch
        self._pred_iou_thresh = pred_iou_thresh
        self._stability_score_thresh = stability_score_thresh
        self._crop_n_layers = crop_n_layers
        self._crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self._min_mask_region_area = min_mask_region_area
        self._box_nms_thresh = box_nms_thresh
        self._nms_iou_thr = nms_iou_thr
        self._nms_score_thr = nms_score_thr
        self._nms_inner_thr = nms_inner_thr
        self._device = device

        self._sam2_model: SAM2Model | None = None
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def _get_sam2_model(self) -> SAM2Model:
        if self._sam2_model is None:
            self._sam2_model = SAM2Model(
                checkpoint=self._checkpoint,
                points_per_side=self._points_per_side,
                points_per_batch=self._points_per_batch,
                pred_iou_thresh=self._pred_iou_thresh,
                stability_score_thresh=self._stability_score_thresh,
                min_mask_region_area=self._min_mask_region_area,
                box_nms_thresh=self._box_nms_thresh,
                output_mode="multi_scale",
                device=self._device,
            )
        return self._sam2_model

    def _generate_multi_scale_masks(self, image: np.ndarray) -> dict:
        """Generate masks at multiple scales.

        Uses multi-crop generation and cross-crop NMS, then mask NMS per scale.

        Args:
            image: Input image in BGR format (OpenCV default), shape ``[H, W, 3]``.

        Returns:
            Dictionary with mask lists keyed by ``"default"``, ``"s"``,
            ``"m"``, ``"l"``.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_size = image_rgb.shape[:2]
        sam2 = self._get_sam2_model()

        # crop boxes and point grids for each layer
        crop_boxes, layer_idxs = _sam2_amg.generate_crop_boxes(
            orig_size, self._crop_n_layers, 512 / 1500
        )
        point_grids = _sam2_amg.build_all_layer_point_grids(
            self._points_per_side,
            self._crop_n_layers,
            self._crop_n_points_downscale_factor,
        )

        all_default: List[Dict[str, Any]] = []
        all_s: List[Dict[str, Any]] = []
        all_m: List[Dict[str, Any]] = []
        all_l: List[Dict[str, Any]] = []

        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            x0, y0, x1, y1 = crop_box
            cropped_im = image_rgb[y0:y1, x0:x1, :]
            cropped_h, cropped_w = cropped_im.shape[:2]
            points_scale = np.array([cropped_w, cropped_h], dtype=np.float64)
            point_coords = point_grids[layer_idx] * points_scale

            md, ms, mm, ml = sam2.predict_masks_multi_scale(
                cropped_im,
                point_coords=point_coords,
                crop_box=crop_box,
                orig_size=orig_size,
            )
            all_default.extend(md)
            all_s.extend(ms)
            all_m.extend(mm)
            all_l.extend(ml)

        log_diag = not hasattr(self, '_diag_count') or self._diag_count < 2
        if log_diag:
            if not hasattr(self, '_diag_count'):
                self._diag_count = 0
            self._diag_count += 1
            self._logger.debug(
                "[diag] after predict: default=%d, s=%d, m=%d, l=%d",
                len(all_default), len(all_s), len(all_m), len(all_l),
            )

        # Cross-crop NMS (prefer smaller crops)
        if len(crop_boxes) > 1:
            all_default = cross_crop_nms(all_default, iou_threshold=self._box_nms_thresh)
            all_s = cross_crop_nms(all_s, iou_threshold=self._box_nms_thresh)
            all_m = cross_crop_nms(all_m, iou_threshold=self._box_nms_thresh)
            all_l = cross_crop_nms(all_l, iou_threshold=self._box_nms_thresh)

        if log_diag:
            self._logger.debug(
                "[diag] after cross-crop NMS: default=%d, s=%d, m=%d, l=%d",
                len(all_default), len(all_s), len(all_m), len(all_l),
            )

        # Remove small disconnected regions and holes (matches original SAM's
        # postprocess_small_regions step in generate_curr_anns)
        if self._min_mask_region_area > 0:
            nms_thresh = self._box_nms_thresh
            all_default = postprocess_small_regions(all_default, self._min_mask_region_area, nms_thresh)
            all_s = postprocess_small_regions(all_s, self._min_mask_region_area, nms_thresh)
            all_m = postprocess_small_regions(all_m, self._min_mask_region_area, nms_thresh)
            all_l = postprocess_small_regions(all_l, self._min_mask_region_area, nms_thresh)

        if log_diag:
            self._logger.debug(
                "[diag] after postprocess_small_regions: default=%d, s=%d, m=%d, l=%d",
                len(all_default), len(all_s), len(all_m), len(all_l),
            )

        # Mask NMS per scale
        masks_default, masks_s, masks_m, masks_l = masks_update(
            all_default,
            all_s,
            all_m,
            all_l,
            iou_thr=self._nms_iou_thr,
            score_thr=self._nms_score_thr,
            inner_thr=self._nms_inner_thr,
        )

        if log_diag:
            self._logger.debug(
                "[diag] after masks_update: default=%d, s=%d, m=%d, l=%d",
                len(masks_default), len(masks_s), len(masks_m), len(masks_l),
            )

        return {
            "default": masks_default,
            "s": masks_s,
            "m": masks_m,
            "l": masks_l,
        }

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """Generate multi-scale SAM2 masks for all images in the scene.

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
        version_safe = self.version.replace(".", "_")
        cache_prefix = (
            f"sam2_multi_scale_masks_{self._checkpoint}_"
            f"p{self._points_per_side}_"
            f"iou{int(self._pred_iou_thresh * 100)}_"
            f"stab{int(self._stability_score_thresh * 100)}_"
            f"crop{self._crop_n_layers}_"
            f"nmsiou{int(self._nms_iou_thr * 100)}_"
            f"nmsscore{int(self._nms_score_thr * 100)}_"
            f"nmsinner{int(self._nms_inner_thr * 100)}_"
            f"v{version_safe}"
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

        for image_id in range(input_scene.num_images):
            if regenerate_cache:
                break
            cache_filename = f"masks_{image_id:0{num_zeropad}}"
            if not output_cache.has_file(cache_filename):
                self._logger.info(
                    f"Masks {cache_filename} not found in the cache. " f"Clearing cache and regenerating."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break

            cache_meta = output_cache.get_file_metadata(cache_filename)
            value_meta = cache_meta.get("metadata", {})
            if (
                value_meta.get("checkpoint") != self._checkpoint
                or value_meta.get("points_per_side") != self._points_per_side
                or value_meta.get("pred_iou_thresh") != self._pred_iou_thresh
                or value_meta.get("stability_score_thresh") != self._stability_score_thresh
                or value_meta.get("crop_n_layers") != self._crop_n_layers
                or value_meta.get("min_mask_region_area") != self._min_mask_region_area
                or value_meta.get("nms_iou_thr") != self._nms_iou_thr
                or value_meta.get("nms_score_thr") != self._nms_score_thr
                or value_meta.get("nms_inner_thr") != self._nms_inner_thr
            ):
                self._logger.info(
                    f"Cache metadata does not match expected parameters. " f"Clearing cache and regenerating."
                )
                output_cache.clear_current_folder()
                regenerate_cache = True
                break

        if regenerate_cache:
            self._logger.info("Generating multi-scale SAM2 masks for all images.")
            # Suppress SAM2's per-image INFO (e.g. "Computing image embeddings...")
            # SAM2 logs through root, so we must suppress root. Setting our own
            # logger level explicitly ensures propagated messages still reach
            # root's handlers (propagation skips the parent's level check).
            _root = logging.getLogger()
            _prev_root = _root.level
            _prev_self = self._logger.level
            _sam2_model_logger = logging.getLogger("fvdb_reality_capture.foundation_models.sam2.SAM2Model")
            _prev_sam2_model = _sam2_model_logger.level
            self._logger.setLevel(logging.INFO)
            _sam2_model_logger.setLevel(logging.INFO)
            _root.setLevel(logging.WARNING)

            try:
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
                            mask_data[f"{scale_name}_bboxes"] = np.array(
                                [m["bbox"] for m in masks], dtype=np.float32
                            )
                            mask_data[f"{scale_name}_areas"] = np.array(
                                [m["area"] for m in masks], dtype=np.int32
                            )
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
                            "crop_n_layers": self._crop_n_layers,
                            "min_mask_region_area": self._min_mask_region_area,
                            "nms_iou_thr": self._nms_iou_thr,
                            "nms_score_thr": self._nms_score_thr,
                            "nms_inner_thr": self._nms_inner_thr,
                        },
                    )

                pbar.close()
                self._logger.info(f"Generated masks for {input_scene.num_images} images.")
            finally:
                _root.setLevel(_prev_root)
                self._logger.setLevel(_prev_self)
                _sam2_model_logger.setLevel(_prev_sam2_model)
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
            "points_per_batch": self._points_per_batch,
            "pred_iou_thresh": self._pred_iou_thresh,
            "stability_score_thresh": self._stability_score_thresh,
            "crop_n_layers": self._crop_n_layers,
            "crop_n_points_downscale_factor": self._crop_n_points_downscale_factor,
            "min_mask_region_area": self._min_mask_region_area,
            "box_nms_thresh": self._box_nms_thresh,
            "nms_iou_thr": self._nms_iou_thr,
            "nms_score_thr": self._nms_score_thr,
            "nms_inner_thr": self._nms_inner_thr,
            "device": str(self._device),
        }

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "ComputeMultiScaleSAM2Masks":
        if state_dict["name"] != "ComputeMultiScaleSAM2Masks":
            raise ValueError(
                f"Expected state_dict with name 'ComputeMultiScaleSAM2Masks', "
                f"got {state_dict['name']} instead."
            )

        return ComputeMultiScaleSAM2Masks(
            checkpoint=state_dict["checkpoint"],
            points_per_side=state_dict["points_per_side"],
            points_per_batch=state_dict.get("points_per_batch", 64),
            pred_iou_thresh=state_dict["pred_iou_thresh"],
            stability_score_thresh=state_dict["stability_score_thresh"],
            crop_n_layers=state_dict.get("crop_n_layers", 1),
            crop_n_points_downscale_factor=state_dict.get("crop_n_points_downscale_factor", 1),
            min_mask_region_area=state_dict.get("min_mask_region_area", 100),
            box_nms_thresh=state_dict.get("box_nms_thresh", 0.7),
            nms_iou_thr=state_dict["nms_iou_thr"],
            nms_score_thr=state_dict["nms_score_thr"],
            nms_inner_thr=state_dict["nms_inner_thr"],
            device=state_dict["device"],
        )
