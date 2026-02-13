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

    Compares predicted CLIP features with ground truth features, masked to only
    include valid pixels (those covered by a SAM mask).

    Args:
        predicted_features: Predicted feature map of shape ``[B, H, W, clip_n_dims]``.
        gt_features: Ground truth feature map of shape ``[B, H, W, clip_n_dims]``.
        mask: Boolean mask of shape ``[B, H, W]`` indicating valid pixels.
        use_cosine_loss: Whether to include cosine similarity loss.
        use_l1_loss: Whether to include L1 loss.
        normalize_features: Whether to L2-normalize predicted features
            before computing loss.

    Returns:
        Dictionary with loss components:
            - ``"total_loss"``: Combined loss value.
            - ``"cosine_loss"``: Cosine loss component (if enabled).
            - ``"l1_loss"``: L1 loss component (if enabled).
    """
    assert use_cosine_loss or use_l1_loss, "At least one loss type must be enabled"

    # Optionally normalize predicted features
    if normalize_features:
        predicted_features = predicted_features / (predicted_features.norm(dim=-1, keepdim=True) + 1e-10)

    loss_dict: dict[str, torch.Tensor] = {}
    total_loss = torch.tensor(0.0, device=predicted_features.device)

    if not mask.any():
        # No valid pixels - return zero loss
        loss_dict["total_loss"] = total_loss
        if use_cosine_loss:
            loss_dict["cosine_loss"] = total_loss
        if use_l1_loss:
            loss_dict["l1_loss"] = total_loss
        return loss_dict

    # Gather only valid pixels (clean signal, no NaN risk from torch.empty).
    valid_pred = predicted_features[mask]  # [N_valid, clip_n_dims]
    valid_gt = gt_features[mask]  # [N_valid, clip_n_dims]

    # The original LangSplatV2 computes .mean() over ALL H*W pixels, where
    # masked-out pixels are zero-vectors that contribute ~0 to the sum but
    # inflate the denominator.  This implicitly scales gradients down by
    # (N_valid / N_total).  We replicate this by computing the loss on valid
    # pixels only (clean, interpretable values) and multiplying by the mask
    # coverage fraction so that gradient magnitudes match the original exactly:
    #
    #   grad_original = (1/N_total)  * sum_valid(dL_i)
    #   grad_ours     = (ratio/N_valid) * sum_valid(dL_i)
    #                 = (1/N_total)  * sum_valid(dL_i)   [identical]
    mask_fraction = mask.sum().float() / mask.numel()

    if use_cosine_loss:
        cos_loss = cosine_loss(valid_pred, valid_gt) * mask_fraction
        loss_dict["cosine_loss"] = cos_loss
        total_loss = total_loss + cos_loss

    if use_l1_loss:
        l1 = l1_loss(valid_pred, valid_gt) * mask_fraction
        loss_dict["l1_loss"] = l1
        total_loss = total_loss + l1

    loss_dict["total_loss"] = total_loss
    return loss_dict
