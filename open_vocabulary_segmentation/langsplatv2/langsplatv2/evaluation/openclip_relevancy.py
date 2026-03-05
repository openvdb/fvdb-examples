# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""OpenCLIP relevancy computation for LERF evaluation.

Implements the relevancy scoring from the original LangSplatV2 evaluation
(``OpenCLIPNetwork.get_max_across_quick``).  For each pixel, the score is the
minimum across all negative prompts of ``softmax(10 * [pos_sim, neg_sim])[0]``.

"""
import logging
from typing import Sequence

import open_clip
import torch
import torchvision

logger = logging.getLogger(__name__)

# Default negative prompts (matching the original LangSplatV2 evaluation)
DEFAULT_NEGATIVES = ("object", "things", "stuff", "texture")


class OpenCLIPRelevancy:
    """Compute CLIP-based relevancy maps for open-vocabulary segmentation evaluation.

    This class encapsulates:

    * Loading an OpenCLIP model (ViT-B-16 by default)
    * Encoding positive (query) and negative (distractor) text prompts
    * Computing per-pixel relevancy scores from rendered CLIP feature maps

    The relevancy computation matches the original LangSplatV2
    ``get_max_across_quick`` exactly:

    1. Dot-product similarity between per-pixel features and all prompt embeddings
    2. For each positive prompt and each negative prompt, form a 2-way softmax
       with temperature 10
    3. Take the *minimum* positive probability across all negatives as the
       final relevancy score

    Args:
        clip_model_type: OpenCLIP model architecture (default ``"ViT-B-16"``).
        clip_model_pretrained: Pretrained weights identifier
            (default ``"laion2b_s34b_b88k"``).
        negatives: Tuple of negative distractor text prompts.
        device: Device for model and embeddings.
    """

    def __init__(
        self,
        clip_model_type: str = "ViT-B-16",
        clip_model_pretrained: str = "laion2b_s34b_b88k",
        negatives: Sequence[str] = DEFAULT_NEGATIVES,
        device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device)
        self.clip_model_type = clip_model_type
        self.clip_model_pretrained = clip_model_pretrained

        # Select precision based on device: use fp16 only on CUDA, fp32 otherwise
        precision = "fp16" if self.device.type == "cuda" else "fp32"

        # Load OpenCLIP model
        model, _, _ = open_clip.create_model_and_transforms(
            clip_model_type,
            pretrained=clip_model_pretrained,
            precision=precision,
        )
        model.eval()
        self.model = model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(clip_model_type)

        # Encode negative prompts
        self.negatives = tuple(negatives)
        with torch.no_grad():
            tok = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(self.device)
            self.neg_embeds = self.model.encode_text(tok)
        self.neg_embeds = self.neg_embeds / self.neg_embeds.norm(dim=-1, keepdim=True)

        # Positive embeddings (set per evaluation frame)
        self.positives: tuple[str, ...] = ()
        self.pos_embeds: torch.Tensor | None = None

        logger.info(
            f"OpenCLIPRelevancy initialized: model={clip_model_type}, "
            f"pretrained={clip_model_pretrained}, negatives={self.negatives}"
        )

    def set_positives(self, text_list: Sequence[str]) -> None:
        """Encode and cache positive (query) text prompts.

        Args:
            text_list: List of positive text prompts (e.g. category labels
                from the ground-truth annotations).
        """
        self.positives = tuple(text_list)
        with torch.no_grad():
            tok = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(self.device)
            self.pos_embeds = self.model.encode_text(tok)
        self.pos_embeds = self.pos_embeds / self.pos_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_relevancy_map(self, sem_map: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel relevancy for all positive prompts across all levels.

        This exactly replicates ``OpenCLIPNetwork.get_max_across_quick`` from the
        original LangSplatV2 evaluation.

        Args:
            sem_map: CLIP feature maps of shape ``[n_levels, H, W, 512]``.
                Each level corresponds to a different SAM scale model.

        Returns:
            Relevancy tensor of shape ``[n_levels, n_prompts, H, W]`` where each
            value is in ``[0, 1]``.
        """
        if self.pos_embeds is None:
            raise RuntimeError("Call set_positives() before computing relevancy maps.")

        n_levels, h, w, c = sem_map.shape
        n_phrases = len(self.positives)
        n_negatives = len(self.negatives)

        # Flatten spatial dims: [n_levels, H*W, 512]
        sem_map_flat = sem_map.reshape(n_levels, h * w, c).contiguous()

        # All prompt embeddings: [P+N, 512]
        phrase_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        phrase_embeds = phrase_embeds.to(sem_map.dtype).to(sem_map.device)

        # Dot-product similarity: [n_levels, H*W, P+N]
        sim = torch.einsum("nqc,pc->nqp", sem_map_flat, phrase_embeds)

        # Split into positive and negative similarities
        pos_vals = sim[:, :, :n_phrases]      # [n_levels, H*W, P]
        neg_vals = sim[:, :, n_phrases:]       # [n_levels, H*W, N]

        # For each (positive, negative) pair, compute 2-way softmax with temperature 10
        repeated_pos = pos_vals.unsqueeze(-1).expand(-1, -1, -1, n_negatives)   # [n_levels, H*W, P, N]
        neg_vals_exp = neg_vals.unsqueeze(2).expand(-1, -1, n_phrases, -1)      # [n_levels, H*W, P, N]

        sims = torch.stack([repeated_pos, neg_vals_exp], dim=-1)                # [n_levels, H*W, P, N, 2]
        softmax = torch.softmax(10 * sims, dim=-1)                             # [n_levels, H*W, P, N, 2]

        # Take minimum positive probability across all negatives
        min_pos_prob, _ = softmax[..., 0].min(dim=-1)                           # [n_levels, H*W, P]

        # Reshape back to spatial dimensions
        relev_map = min_pos_prob.permute(0, 2, 1).reshape(n_levels, n_phrases, h, w)

        return relev_map
