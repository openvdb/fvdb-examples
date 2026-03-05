# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from enum import Enum
from typing import Dict, Tuple

import torch
import torch.nn

from .backbone import MaskPLSEncoderDecoder
from .blocks import MLP
from .decoder import MaskedTransformerDecoder
from .utils import pad_batch


class MaskPLS(torch.nn.Module):
    class DecoderInputMode(Enum):
        XYZ = "xyz"
        GRID = "grid"

    def __init__(
        self,
        num_classes: int,
        dataset_extent: Tuple[float, float, float],
        decoder_input_mode: DecoderInputMode = DecoderInputMode.GRID,
        decoder_num_queries: int = 100,
        segmentation_only=False,
    ) -> None:
        """
        Mask-Based Panoptic LiDAR Segmentation for Autonomous Driving
        https://github.com/PRBonn/MaskPLS
        Args:
            num_classes (int): Number of classes for segmentation.
            dataset_extent (Tuple[float, float, float]): The magnitude of the spatial extents of the dataset.
            decoder_input_mode (DecoderInputMode, optional): Mode for decoder input. Defaults to DecoderInputMode.GRID.
            decoder_num_queries (int, optional): Number of queries for the decoder. Defaults to 100.
            segmentation_only (bool, optional): If True, only segmentation is performed, masked decoder not used. Defaults to False.
        Returns:
            None
        """
        super().__init__()
        self.decoder_input_mode = decoder_input_mode
        self.segmentation_only = segmentation_only

        self.backbone = MaskPLSEncoderDecoder(output_feature_levels=[3])

        self.sem_head = torch.nn.Linear(self.backbone.channels[-1], num_classes)

        self.semantic_embedding_distil = False
        if self.semantic_embedding_distil:
            semantic_embedding_hidden_dims = [512, 1024, 768]
            self.sem_embed = MLP(
                self.backbone.channels[-1],
                semantic_embedding_hidden_dims[:-1],
                semantic_embedding_hidden_dims[-1],
            )

        if not self.segmentation_only:
            self.decoder = MaskedTransformerDecoder(
                num_classes, dataset_extent, backbone_channels=self.backbone.channels, num_queries=decoder_num_queries
            )

    def forward(self, x: Dict):
        outputs = {}

        logits_sem_embed_grid = None

        ###### Backbone ######
        out_feats_grids = self.backbone(x)
        # out_feats_grids is a List[Tuple[JaggedTensor, GridBatch]]
        #    where each tuple corresponds to the `output_feature_levels`
        #    plus 1 additional entry which is the last/full-resolution feature level run through the conv mask projection

        ###### v2p ######
        # NOTE: Matching MaskPLS paper which performs v2p before sem_head
        #    In SAL, features are at voxel centers throughout, so we provide an option to try either
        if self.decoder_input_mode == MaskPLS.DecoderInputMode.XYZ:
            xyz = x["xyz"]
            feats = [grid.sample_trilinear(xyz, data).unbind() for data, grid in out_feats_grids]

            # pad batch
            feats, coords, pad_masks = pad_batch(feats, [xyz.unbind() for _ in feats])  # type: ignore

            logits = [self.sem_head(feats[-1])]
        else:
            # GRID mode: apply sem_head to the raw JaggedTensor features, then unpack into padded batches
            last_data, last_grid = out_feats_grids[-1]
            logits_jt = last_grid.jagged_like(self.sem_head(last_data.jdata))

            if self.semantic_embedding_distil:
                logits_sem_embed_grid = last_grid.jagged_like(self.sem_embed(last_data.jdata))

            coords = [grid.voxel_to_world(grid.ijk.float()).unbind() for data, grid in out_feats_grids]
            feats = [data.unbind() for data, grid in out_feats_grids]
            logits = [logits_jt.unbind()]
            feats, coords, pad_masks, logits = pad_batch(feats, coords, additional_feats=logits)  # type: ignore

        ###### Decoder ######
        if self.segmentation_only:
            padding = pad_masks.pop()
        else:
            outputs, padding = self.decoder(feats, coords, pad_masks)

        outputs["bb_sem_logits"] = logits[0]
        outputs["bb_sem_embed_logits"] = None if not self.semantic_embedding_distil else logits_sem_embed_grid
        outputs["padding"] = padding

        return outputs
