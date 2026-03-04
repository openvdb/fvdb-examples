# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Shared mask post-processing utilities for multi-scale SAM transforms.

These functions are used by both :class:`ComputeMultiScaleSAM1Masks` and
:class:`ComputeMultiScaleSAM2Masks`.  They have no dependency on either the
``segment_anything`` or ``sam2`` packages.
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms


def remove_small_regions(
    mask: np.ndarray,
    area_thresh: float,
    mode: str,
) -> Tuple[np.ndarray, bool]:
    """Remove small disconnected regions or holes from a binary mask.

    Pure OpenCV implementation matching ``segment_anything.utils.amg.remove_small_regions``.

    Args:
        mask: Boolean mask, shape ``[H, W]``.
        area_thresh: Minimum area; components smaller than this are removed.
        mode: ``"holes"`` to fill small holes, ``"islands"`` to remove small islands.

    Returns:
        ``(cleaned_mask, changed)`` where *changed* is True if the mask was modified.
    """
    assert mode in ("holes", "islands")
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1]
    small_regions = [i for i, s in enumerate(sizes) if s < area_thresh and i != 0]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels or i == 0]
    mask = np.isin(regions, fill_labels)
    return mask, True


def _clean_single_mask(
    seg_raw: np.ndarray,
    min_area: int,
) -> Tuple[np.ndarray, float, List[float]]:
    """Clean one mask and compute its bounding box.  Thread-safe (CPU-only).

    Returns:
        (cleaned_seg, score, box_xyxy) where score is 1.0 if the mask was
        unchanged and 0.0 if it was modified.
    """
    seg = seg_raw.astype(bool)
    seg, changed_holes = remove_small_regions(seg, min_area, mode="holes")
    unchanged = not changed_holes
    seg, changed_islands = remove_small_regions(seg, min_area, mode="islands")
    unchanged = unchanged and not changed_islands

    ys, xs = np.where(seg)
    if len(xs) == 0:
        box = [0.0, 0.0, 0.0, 0.0]
    else:
        box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    return seg, float(unchanged), box


def postprocess_small_regions(
    masks: List[Dict[str, Any]],
    min_area: int,
    nms_thresh: float,
) -> List[Dict[str, Any]]:
    """Remove small disconnected regions and holes from masks, then re-run box NMS.

    Mirrors the ``postprocess_small_regions`` step in the original SAM
    ``SamAutomaticMaskGenerator.generate_curr_anns`` which cleans up each
    mask's binary segmentation before returning annotations.

    The per-mask cleaning (``cv2.connectedComponentsWithStats``) is
    parallelized across CPU cores via :class:`ThreadPoolExecutor`.

    Args:
        masks: List of mask annotation dicts with at least ``segmentation``
            (np.ndarray bool/uint8 HxW) and ``bbox`` ([x, y, w, h]).
        min_area: Minimum area threshold for ``remove_small_regions``.
        nms_thresh: Box NMS IoU threshold for deduplication after cleaning.

    Returns:
        Filtered list of mask annotation dicts with cleaned segmentations.
    """
    if len(masks) == 0:
        return masks

    max_workers = min(len(masks), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_clean_single_mask, m["segmentation"], min_area)
            for m in masks
        ]
        results = [f.result() for f in futures]

    new_segmentations = [r[0] for r in results]
    scores = [r[1] for r in results]
    boxes_list = [r[2] for r in results]

    boxes_xyxy = torch.tensor(boxes_list, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)

    keep = batched_nms(
        boxes_xyxy,
        scores_t,
        torch.zeros(len(masks), dtype=torch.long),
        iou_threshold=nms_thresh,
    )

    result = []
    for idx in keep.tolist():
        m = masks[idx].copy()
        m["segmentation"] = new_segmentations[idx].astype(np.uint8)
        x_min, y_min, x_max, y_max = boxes_list[idx]
        m["bbox"] = [x_min, y_min, x_max - x_min, y_max - y_min]
        m["area"] = int(new_segmentations[idx].sum())
        result.append(m)

    return result


_mask_nms_logger = logging.getLogger(__name__ + ".mask_nms")
_mask_nms_diag_count = 0


def mask_nms(
    masks: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float = 0.7,
    score_thr: float = 0.1,
    inner_thr: float = 0.2,
    **kwargs,
) -> torch.Tensor:
    """
    Perform mask non-maximum suppression (NMS) on a set of masks.

    Faithful reimplementation of the ``mask_nms`` from the original
    LangSplatV2 ``preprocess.py``.

    Args:
        masks: Binary masks, shape ``[num_masks, H, W]``.
        scores: Mask scores, shape ``[num_masks]``.
        iou_thr: IoU threshold for NMS.
        score_thr: Minimum score threshold.
        inner_thr: Inner overlap threshold for removing contained masks.

    Returns:
        Indices of selected masks after NMS.
    """
    global _mask_nms_diag_count
    log_diag = _mask_nms_diag_count < 8
    if log_diag:
        _mask_nms_diag_count += 1

    if len(masks) == 0:
        return torch.tensor([], dtype=torch.long)

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    masks_flat = masks_ord.reshape(num_masks, -1).float()
    intersection = masks_flat @ masks_flat.T
    union = masks_area[:, None] + masks_area[None, :] - intersection
    iou_matrix = intersection / union

    R = intersection / masks_area[:, None]
    inner_val = 1 - R * R.T
    cond = (R < 0.5) & (R.T >= 0.85)
    inner_iou_matrix = torch.where(cond, inner_val, torch.zeros_like(inner_val))

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    if log_diag:
        _mask_nms_logger.info(
            "[mask_nms diag] input=%d masks  scores: min=%.4f max=%.4f  "
            "areas: min=%.0f max=%.0f  iou_max: min=%.4f max=%.4f mean=%.4f  "
            "pass_iou=%d  pass_conf=%d  pass_inner_u=%d  pass_inner_l=%d",
            num_masks,
            scores.min().item(), scores.max().item(),
            masks_area.min().item(), masks_area.max().item(),
            iou_max.min().item(), iou_max.max().item(), iou_max.mean().item(),
            keep.sum().item(), keep_conf.sum().item(),
            keep_inner_u.sum().item(), keep_inner_l.sum().item(),
        )

    if keep_conf.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_conf[index] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_inner_u[index] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_inner_l[index] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    if log_diag:
        _mask_nms_logger.info(
            "[mask_nms diag] final keep=%d / %d", keep.sum().item(), num_masks,
        )

    selected_idx = idx[keep]
    return selected_idx


_masks_update_logger = logging.getLogger(__name__ + ".masks_update")


def masks_update(
    *mask_lists,
    iou_thr: float = 0.8,
    score_thr: float = 0.7,
    inner_thr: float = 0.5,
    max_area_frac: float = 0.95,
) -> tuple:
    """
    Apply mask NMS to multiple lists of masks.

    Args:
        *mask_lists: Variable number of mask lists to filter.
        iou_thr: IoU threshold for NMS.
        score_thr: Score threshold.
        inner_thr: Inner overlap threshold.
        max_area_frac: Discard masks covering more than this fraction of the
            image.  Near-full-image masks poison the inner-containment check
            by appearing to contain every other mask.

    Returns:
        Tuple of filtered mask lists.
    """
    masks_new = []

    for masks_lvl in mask_lists:
        if len(masks_lvl) == 0:
            masks_new.append([])
            continue

        if max_area_frac < 1.0:
            h, w = masks_lvl[0]["segmentation"].shape[:2]
            total_pixels = h * w
            area_limit = total_pixels * max_area_frac
            before = len(masks_lvl)
            masks_lvl = [m for m in masks_lvl if m["segmentation"].sum() <= area_limit]
            n_dropped = before - len(masks_lvl)
            if n_dropped > 0:
                _masks_update_logger.info(
                    "[masks_update] dropped %d masks covering >%.0f%% of image (%d remain)",
                    n_dropped, max_area_frac * 100, len(masks_lvl),
                )
            if len(masks_lvl) == 0:
                masks_new.append([])
                continue

        seg_pred = torch.from_numpy(np.stack([m["segmentation"] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m["predicted_iou"] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m["stability_score"] for m in masks_lvl], axis=0))

        if torch.cuda.is_available():
            seg_pred = seg_pred.cuda()
            iou_pred = iou_pred.cuda()
            stability = stability.cuda()

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, iou_thr=iou_thr, score_thr=score_thr, inner_thr=inner_thr)

        keep_set = set(keep_mask_nms.int().cpu().numpy().tolist())
        filtered_masks = [m for i, m in enumerate(masks_lvl) if i in keep_set]
        masks_new.append(filtered_masks)

    return tuple(masks_new)


def cross_crop_nms(
    mask_list: List[Dict[str, Any]],
    iou_threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """Run NMS across masks from multiple crops; prefer masks from smaller crops.

    Args:
        mask_list: List of mask records with "bbox" (xywh) and "crop_box" (xywh).
        iou_threshold: Box IoU threshold for NMS.

    Returns:
        Filtered list of mask records.
    """
    if len(mask_list) <= 1:
        return mask_list
    boxes_xywh = np.array([m["bbox"] for m in mask_list], dtype=np.float32)
    x1 = boxes_xywh[:, 0]
    y1 = boxes_xywh[:, 1]
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3]
    boxes_xyxy = torch.from_numpy(np.stack([x1, y1, x2, y2], axis=1))
    crop_areas = np.array(
        [m["crop_box"][2] * m["crop_box"][3] for m in mask_list],
        dtype=np.float32,
    )
    scores = torch.from_numpy(1.0 / (crop_areas + 1e-6))
    keep = batched_nms(
        boxes_xyxy.float(),
        scores,
        torch.zeros(len(mask_list), dtype=torch.long),
        iou_threshold=iou_threshold,
    )
    return [mask_list[i] for i in keep.tolist()]
