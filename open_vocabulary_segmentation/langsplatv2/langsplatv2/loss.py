# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch
import torch.nn.functional as F


def cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity loss between prediction and target.

    Returns ``1 - mean(cosine_similarity)`` so that perfect alignment gives 0.

    Args:
        prediction: Predicted features of shape ``[..., C]``.
        target: Target features of shape ``[..., C]``.

    Returns:
        Scalar loss value.
    """
    return 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute L1 (mean absolute error) loss.

    Args:
        prediction: Predicted features of shape ``[..., C]``.
        target: Target features of shape ``[..., C]``.

    Returns:
        Scalar loss value.
    """
    return torch.abs(prediction - target).mean()


def calculate_langsplatv2_loss(
    predicted_features: torch.Tensor,
    gt_features: torch.Tensor,
    mask: torch.Tensor,
    use_cosine_loss: bool = True,
    use_l1_loss: bool = False,
    normalize_features: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute the LangSplatV2 language feature loss.

    Both predicted and ground-truth features are zeroed for unmapped pixels
    (mask == False), matching the original's
    ``language_feature * language_feature_mask`` approach.  Per-pixel loss
    is then zeroed for unmapped pixels and averaged over **all** pixels.
    This means:

    - Unmapped pixels contribute 0 to the reported loss.
    - The mean over all pixels implicitly down-scales gradients from valid
      pixels by ``N_valid / N_total``, matching the original's gradient
      magnitudes exactly.

    Args:
        predicted_features: Predicted feature map, shape ``[B, H, W, C]``
            or ``[H, W, C]``.
        gt_features: Ground truth feature map, same shape.
        mask: Boolean mask, shape ``[B, H, W]`` or ``[H, W]``.
        use_cosine_loss: Whether to include cosine similarity loss.
        use_l1_loss: Whether to include L1 loss.
        normalize_features: Whether to L2-normalize predicted features
            before computing loss.

    Returns:
        Dictionary with loss components:
            - ``"total_loss"``: Combined loss value.
            - ``"cosine_loss"``: Cosine loss component (if enabled).
            - ``"cosine_loss_valid"``: Cosine loss on valid pixels only (for logging).
            - ``"l1_loss"``: L1 loss component (if enabled).
    """
    assert use_cosine_loss or use_l1_loss, "At least one loss type must be enabled"

    if normalize_features:
        predicted_features = predicted_features / (predicted_features.norm(dim=-1, keepdim=True) + 1e-10)

    loss_dict: dict[str, torch.Tensor] = {}
    total_loss = torch.tensor(0.0, device=predicted_features.device)

    mask_expanded = mask.unsqueeze(-1)  # [..., 1]
    masked_pred = predicted_features * mask_expanded
    masked_gt = gt_features * mask_expanded

    if use_cosine_loss:
        per_pixel_cos = F.cosine_similarity(masked_pred, masked_gt, dim=-1)
        per_pixel_loss = (1.0 - per_pixel_cos) * mask.float()
        cos_loss_all = per_pixel_loss.sum() / mask.numel()
        loss_dict["cosine_loss"] = cos_loss_all
        total_loss = total_loss + cos_loss_all

        if mask.any():
            valid_pred = predicted_features[mask]
            valid_gt = gt_features[mask]
            loss_dict["cosine_loss_valid"] = cosine_loss(valid_pred, valid_gt)
        else:
            loss_dict["cosine_loss_valid"] = cos_loss_all

    if use_l1_loss:
        per_pixel_l1 = torch.abs(masked_pred - masked_gt) * mask_expanded
        l1 = per_pixel_l1.sum() / mask.numel()
        loss_dict["l1_loss"] = l1
        total_loss = total_loss + l1

    loss_dict["total_loss"] = total_loss
    return loss_dict
