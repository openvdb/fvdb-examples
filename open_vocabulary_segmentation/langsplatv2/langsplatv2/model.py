# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any

import torch
import torch.nn as nn
from fvdb import GaussianSplat3d
import torch.cuda.nvtx as nvtx

from .util import rgb_to_sh

logger = logging.getLogger(__name__)


class LangSplatV2Model(nn.Module):
    """LangSplatV2 language feature model on a frozen Gaussian Splatting scene.

    This model adds learnable language features to an existing Gaussian splat
    reconstruction. Each Gaussian stores logits over a shared codebook, which
    are converted to sparse coefficients via softmax + top-k selection. These
    sparse coefficients are rendered as per-pixel weight maps using fVDB's
    differentiable Gaussian splatting renderer, then decoded into CLIP features
    via matrix multiplication with the codebook.

    The key components are:
        1. **Codebooks**: Shared embedding matrices ``[vq_layer_num, codebook_size, clip_n_dims]``
           initialized via K-means on CLIP features.
        2. **Logits**: Per-Gaussian logits ``[N, vq_layer_num * codebook_size]``
           that produce sparse coefficients via softmax + top-k.
        3. **Rendering**: Alpha-blends sparse coefficients per pixel using the
           frozen Gaussian geometry.
        4. **Decoding**: Multiplies rendered weight maps by codebooks to produce
           per-pixel CLIP features.

    Attributes:
        gs_model: The GaussianSplat3d model.
        codebooks: Shared codebook parameters.
        logits: Per-Gaussian sparse coefficient logits.
    """

    def __init__(
        self,
        gs_model: GaussianSplat3d,
        vq_layer_num: int = 1,
        codebook_size: int = 64,
        clip_n_dims: int = 512,
        topk: int = 4,
        topk_refresh_every: int = 100,
        device: torch.device | str = "cuda",
    ):
        """Initialize the LangSplatV2 model.

        Args:
            gs_model: Pre-trained GaussianSplat3d model.
            vq_layer_num: Number of residual VQ layers.
            codebook_size: Number of entries per codebook.
            clip_n_dims: CLIP embedding dimensionality.
            topk: Number of non-zero sparse coefficients per layer.
            topk_refresh_every: Recompute top-k indices every N steps.
                Between refreshes only the softmax weights are recomputed,
                avoiding the expensive topk kernel. Set to 1 to disable caching.
            device: Device for model parameters.
        """
        super().__init__()

        self.device = torch.device(device)
        self.vq_layer_num = vq_layer_num
        self.codebook_size = codebook_size
        self.clip_n_dims = clip_n_dims
        self.topk = topk
        self.topk_refresh_every = topk_refresh_every

        self.gs_model = gs_model
        num_gaussians = gs_model.num_gaussians

        # Total dimensionality of sparse weight vector
        self.weight_dim = vq_layer_num * codebook_size

        # Learnable parameters
        # Logits: per-Gaussian logits for sparse coefficient generation
        self.logits = nn.Parameter(
            torch.zeros(num_gaussians, self.weight_dim, device=device)
        )

        # Codebooks: shared embedding matrices
        # Shape: [vq_layer_num, codebook_size, clip_n_dims]
        self.codebooks = nn.Parameter(
            torch.randn(vq_layer_num, codebook_size, clip_n_dims, device=device) * 0.01
        )

        # Cached top-k indices, refreshed every topk_refresh_every steps.
        # Between refreshes, get_render_weights only does gather+softmax+scatter
        # on [N, k] tensors instead of a full topk on [N, codebook_size].
        self._cached_topk_idx: list[torch.Tensor] | None = None
        self._topk_step: int = 0

        # Create a GaussianSplat3d for rendering sparse weights
        # We render the weight vectors as if they were sh0 color features
        self._gs_render = GaussianSplat3d.from_tensors(
            means=gs_model.means.detach(),
            quats=gs_model.quats.detach(),
            log_scales=gs_model.log_scales.detach(),
            logit_opacities=gs_model.logit_opacities.detach(),
            sh0=torch.zeros(num_gaussians, 1, self.weight_dim, device=device),
            shN=torch.zeros(num_gaussians, 0, self.weight_dim, device=device),
        )

        logger.info(
            f"LangSplatV2Model initialized: {num_gaussians:,} Gaussians, "
            f"{vq_layer_num} VQ layers, codebook_size={codebook_size}, "
            f"topk={topk}, topk_refresh_every={topk_refresh_every}, "
            f"clip_dims={clip_n_dims}"
        )

    def refresh_topk_indices(self) -> None:
        """Recompute and cache the top-k indices from current logits.

        Called automatically by :meth:`get_render_weights` when the cached
        indices are stale, or can be called manually (e.g. before evaluation).
        """
        with torch.no_grad():
            idx_list: list[torch.Tensor] = []
            for i in range(self.vq_layer_num):
                layer_logits = self.logits[:, i * self.codebook_size : (i + 1) * self.codebook_size]
                _, layer_idx = torch.topk(layer_logits, self.topk, dim=1)  # [N, k]
                idx_list.append(layer_idx)
            self._cached_topk_idx = idx_list
        self._topk_step = 0

    def get_render_weights(self) -> torch.Tensor:
        """Compute sparse coefficient weights from logits.

        Uses cached top-k indices when available. The indices are refreshed
        every ``topk_refresh_every`` steps (or always in eval mode). Between
        refreshes, only ``gather`` + ``softmax`` + ``scatter`` on ``[N, k]``
        tensors are needed, avoiding the expensive ``topk`` kernel.

        Returns:
            Sparse weight tensor of shape ``[N, vq_layer_num * codebook_size]``.
        """
        # Refresh indices when stale or in eval mode
        needs_refresh = (
            self._cached_topk_idx is None
            or not self.training
            or self._topk_step >= self.topk_refresh_every
        )
        if needs_refresh:
            with nvtx.range("topk_refresh"):
                self.refresh_topk_indices()
        self._topk_step += 1

        # Cheap path: gather selected logits, softmax over k entries, scatter
        weights_list: list[torch.Tensor] = []
        for i in range(self.vq_layer_num):
            layer_logits = self.logits[:, i * self.codebook_size : (i + 1) * self.codebook_size]
            layer_idx = self._cached_topk_idx[i]  # type: ignore[index]

            selected = layer_logits.gather(1, layer_idx)  # [N, k]
            sparse_weights = selected.softmax(dim=1)       # [N, k]

            layer_result = torch.zeros_like(layer_logits)  # [N, codebook_size]
            layer_result.scatter_(1, layer_idx, sparse_weights)
            weights_list.append(layer_result)

        if self.vq_layer_num == 1:
            return weights_list[0]
        return torch.cat(weights_list, dim=-1)

    def render_weight_maps(
        self,
        world_to_camera: torch.Tensor,
        projection: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render per-pixel sparse weight maps.

        Computes sparse coefficients from logits, converts to SH format,
        and renders them using fVDB's differentiable rasterizer.

        Args:
            world_to_camera: World-to-camera matrices of shape ``[B, 4, 4]``.
            projection: Camera projection matrices of shape ``[B, 3, 3]``.
            image_width: Output image width.
            image_height: Output image height.

        Returns:
            Tuple of:
                - Weight maps of shape ``[B, H, W, weight_dim]``
                - Alpha maps of shape ``[B, H, W, 1]``
        """
        # Compute sparse weights from logits
        with nvtx.range("get_render_weights"):
            weights = self.get_render_weights()  # [N, weight_dim]

        sh0 = rgb_to_sh(weights).unsqueeze(1)  # [N, 1, weight_dim]
        self._gs_render.sh0 = sh0

        # Render weight maps
        rendered, alpha = self._gs_render.render_images(
            image_width=image_width,
            image_height=image_height,
            world_to_camera_matrices=world_to_camera,
            projection_matrices=projection,
            near=0.01,
            far=1e10,
            sh_degree_to_use=0,
        )

        return rendered, alpha  # [B, H, W, weight_dim], [B, H, W, 1]

    def decode_weight_maps(
        self,
        weight_maps: torch.Tensor,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """Decode per-pixel weight maps into CLIP features using codebooks.

        Performs matrix multiplication of weight maps with codebooks to produce
        CLIP features.

        Args:
            weight_maps: Rendered weight maps of shape ``[B, H, W, weight_dim]``.
            layer_idx: If specified, decode up to this layer index (inclusive).
                If None, decode all layers with residual connections.

        Returns:
            Decoded CLIP feature map of shape ``[B, H, W, clip_n_dims]``.
        """
        if layer_idx is None:
            layer_idx = self.vq_layer_num - 1

        feature_map = None
        for i in range(layer_idx + 1):
            # Extract weights for this layer: [B, H, W, codebook_size]
            layer_weights = weight_maps[..., i * self.codebook_size : (i + 1) * self.codebook_size]

            # [B, H, W, codebook_size] @ [codebook_size, clip_n_dims]
            layer_features = layer_weights @ self.codebooks[i] # [B, H, W, clip_n_dims]

            if i > 0 and feature_map is not None:
                # Residual connection (detach previous for gradient isolation,
                # matching the original LangSplatV2 implementation)
                layer_features = layer_features + feature_map.detach()

            feature_map = layer_features

        return feature_map  # [B, H, W, clip_n_dims] # type: ignore

    def forward(
        self,
        world_to_camera: torch.Tensor,
        projection: torch.Tensor,
        image_width: int,
        image_height: int,
        layer_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: render sparse weights and decode to CLIP features.

        Args:
            world_to_camera: World-to-camera matrices ``[B, 4, 4]``.
            projection: Projection matrices ``[B, 3, 3]``.
            image_width:  Image width.
            image_height: Image height.
            layer_idx: VQ layer to decode up to (None = all layers).

        Returns:
            Tuple of:
                - CLIP feature maps of shape ``[B, H, W, clip_n_dims]``
                - Alpha maps of shape ``[B, H, W, 1]``
        """
        weight_maps, alpha = self.render_weight_maps(
            world_to_camera, projection, image_width, image_height
        )
        feature_maps = self.decode_weight_maps(weight_maps, layer_idx)
        return feature_maps, alpha

    def initialize_codebooks(self, codebooks: torch.Tensor) -> None:
        """Initialize codebooks from pre-computed K-means centers.

        Args:
            codebooks: Codebook tensor of shape
                ``[vq_layer_num, codebook_size, clip_n_dims]``.
        """
        assert codebooks.shape == self.codebooks.shape, (
            f"Codebook shape mismatch: expected {self.codebooks.shape}, "
            f"got {codebooks.shape}"
        )
        with torch.no_grad():
            self.codebooks.data.copy_(codebooks.to(self.device))
        logger.info("Codebooks initialized from K-means clustering")

    def state_dict_with_config(self) -> dict[str, Any]:
        """Get state dict including model configuration.

        Returns:
            Dictionary containing model state and configuration.
        """
        return {
            "model_state": self.state_dict(),
            "config": {
                "vq_layer_num": self.vq_layer_num,
                "codebook_size": self.codebook_size,
                "clip_n_dims": self.clip_n_dims,
                "topk": self.topk,
                "topk_refresh_every": self.topk_refresh_every,
            },
        }

    @classmethod
    def from_state_dict_with_config(
        cls,
        state: dict[str, Any],
        gs_model: GaussianSplat3d,
        device: torch.device | str = "cuda",
    ) -> "LangSplatV2Model":
        """Create a model from a saved state dict with config.

        Args:
            state: Dictionary from ``state_dict_with_config()``.
            gs_model: The GaussianSplat3d model.
            device: Device for model parameters.

        Returns:
            Restored LangSplatV2Model instance.
        """
        config = state["config"]
        model = cls(
            gs_model=gs_model,
            vq_layer_num=config["vq_layer_num"],
            codebook_size=config["codebook_size"],
            clip_n_dims=config["clip_n_dims"],
            topk=config["topk"],
            topk_refresh_every=config.get("topk_refresh_every", 100),
            device=device,
        )
        model.load_state_dict(state["model_state"])
        return model
