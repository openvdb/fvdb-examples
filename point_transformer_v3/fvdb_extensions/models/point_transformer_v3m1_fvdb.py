# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Point Transformer - V3 Mode1 FVDB Implementation
"""


from __future__ import annotations

import fvdb
import torch
from external.pointcept.pointcept.models.builder import MODELS
from external.pointcept.pointcept.models.modules import PointModule

# Import PTV3 FVDB implementation - use relative import since we're in the same package
from .ptv3_fvdb import PTV3


def tensor_hash_simple(tensor: torch.Tensor) -> int:
    """Simple Python hash - fastest but less robust"""
    return hash(tuple(tensor.detach().cpu().flatten().tolist()))


def create_grid_from_points(
    grid_coord: torch.Tensor,
    feat: torch.Tensor,
    offset: torch.Tensor,
    voxel_size: float,
    device: str = "cuda",
) -> tuple[fvdb.GridBatch, fvdb.JaggedTensor, fvdb.JaggedTensor]:
    """Create FVDB tensor from ScanNet-like point data with proper batching.

    Args:
        grid_coord: Batched grid coordinates [N, 3]
        feat: Batched features [N, C]
        offset: Tensor indicating batch boundaries [B]
        voxel_size: Voxel size for grid creation
        device: Device for tensor operations

    Returns:
        grid: fvdb.GridBatch
        jfeats: fvdb.JaggedTensor with features
        original_coord_to_voxel_idx: Mapping from original coords to voxel indices
    """

    offset_list = list(offset.cpu().numpy())
    # Convert offset to individual sample boundaries
    if len(offset_list) == 1:
        # Single sample case
        coords_list = [grid_coord.to(device=device, dtype=torch.int32)]
        feats_list = [feat.to(device=device, dtype=torch.float32)]
    else:
        # Multiple samples case - split using offset
        coords_list = []
        feats_list = []
        prev_offset = 0
        for curr_offset in offset_list:
            coords_list.append(grid_coord[prev_offset:curr_offset].to(device=device, dtype=torch.int32))
            feats_list.append(feat[prev_offset:curr_offset].to(device=device, dtype=torch.float32))
            prev_offset = curr_offset

    coords_jagged = fvdb.JaggedTensor(coords_list)

    grid = fvdb.GridBatch.from_ijk(
        coords_jagged,
        voxel_sizes=[[voxel_size, voxel_size, voxel_size]] * len(coords_list),
        origins=[0.0] * 3,
    )

    feats_jagged = fvdb.JaggedTensor(feats_list)
    feats_vdb_order = grid.inject_from_ijk(coords_jagged, feats_jagged)  #
    original_coord_to_voxel_idx = grid.ijk_to_index(coords_jagged, cumulative=True)

    return grid, feats_vdb_order, original_coord_to_voxel_idx


@MODELS.register_module("PT-v3fvdb")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels: int = 6,
        enc_depths: tuple[int, ...] = (2, 2, 2, 2),
        enc_channels: tuple[int, ...] = (32, 64, 128, 256),
        enc_num_heads: tuple[int, ...] = (1, 1, 1, 1),
        dec_depths: tuple[int, ...] = (2, 2, 2),
        dec_channels: tuple[int, ...] = (128, 64, 32),
        dec_num_heads: tuple[int, ...] = (1, 1, 1),
        patch_size: int = 1024,
        drop_path: float = 0.3,
        proj_drop: float = 0.0,
        qk_scale: float = 1.0,
        enable_batch_norm: bool = False,
        embedding_mode: str = "linear",
        no_conv_in_cpe: bool = False,
        cross_patch_attention: bool = False,
        cross_patch_pooling: str = "mean",
        sliding_window_attention: bool = False,
        pipelined_batch: bool = False,
        order_type: str | tuple[str, ...] = ("z", "z-trans"),
        shuffle_orders: bool = True,
    ):
        super().__init__()

        self.pipelined_batch = pipelined_batch
        self.order_type = order_type

        self.fvdb_ptv3_model = PTV3(
            num_classes=-1,
            input_dim=in_channels,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_heads=enc_num_heads,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_heads=dec_num_heads,
            patch_size=patch_size,
            drop_path=drop_path,
            proj_drop=proj_drop,
            qk_scale=qk_scale,
            enable_batch_norm=enable_batch_norm,
            embedding_mode=embedding_mode,
            no_conv_in_cpe=no_conv_in_cpe,
            # cross_patch_attention=cross_patch_attention,
            # cross_patch_pooling=cross_patch_pooling,
            sliding_window_attention=sliding_window_attention,
            order_type=order_type,
            shuffle_orders=shuffle_orders,
        )

    def __call__(self, data_dict: dict) -> torch.Tensor:
        """Override __call__ to preserve type hints from forward."""
        return super().__call__(data_dict)

    def forward(self, data_dict: dict) -> torch.Tensor:

        grid_coord = data_dict["grid_coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"]
        # import pdb; pdb.set_trace()
        # print(f"grid_coord.shape: {grid_coord.shape}, feat.shape: {feat.shape}, offset.shape: {offset.shape}")
        # exit()

        if self.pipelined_batch and len(offset) > 1:
            # Pipelined batch mode: process each point cloud individually
            # This mode splits the batch into individual point clouds, processes each
            # one separately through the FVDB model, and concatenates the results.
            # This can be useful for:
            # 1. Memory efficiency when individual processing uses less memory
            # 2. Debugging to isolate issues with specific point clouds
            # 3. Different processing requirements per sample
            outputs = []
            prev_offset = 0
            # catted_input_grid_ijk = []
            # catted_input_feat = []
            # catted_original_coord_to_voxel_idx = []
            for curr_offset in offset:
                # Extract data for current point cloud
                curr_grid_coord = grid_coord[prev_offset:curr_offset]
                curr_feat = feat[prev_offset:curr_offset]
                curr_num_points = curr_offset - prev_offset
                curr_offset_tensor = torch.tensor([curr_num_points], dtype=offset.dtype, device=offset.device)

                # Process single point cloud
                grid, jfeats, original_coord_to_voxel_idx = create_grid_from_points(
                    curr_grid_coord, curr_feat, curr_offset_tensor, voxel_size=0.02
                )
                assert (
                    grid.ijk.jdata.shape == curr_grid_coord.shape
                ), f"curr_grid_coord.shape: {curr_grid_coord.shape}, grid.ijk.jdata.shape: {grid.ijk.jdata.shape}"  #

                # catted_input_grid_ijk.append(grid.ijk.jdata)
                # catted_input_feat.append(jfeats.jdata)
                # catted_original_coord_to_voxel_idx.append(original_coord_to_voxel_idx.jdata)
                # grid shape and feats values match here.
                jfeats = self.fvdb_ptv3_model(jfeats, grid)
                # feats values does not match here.

                # Get output for this point cloud.
                curr_output = jfeats.jdata[original_coord_to_voxel_idx.jdata]
                outputs.append(curr_output)

                prev_offset = curr_offset

            # Concatenate all outputs
            output = torch.cat(outputs, dim=0)
            # import pdb; pdb.set_trace()

            # catted_input_grid_ijk = torch.cat(catted_input_grid_ijk, dim=0)
            # catted_input_feat = torch.cat(catted_input_feat, dim=0)
            # catted_original_coord_to_voxel_idx = torch.cat(catted_original_coord_to_voxel_idx, dim=0)

        else:
            # Standard batch mode (original implementation)
            grid, jfeats, original_coord_to_voxel_idx = create_grid_from_points(
                grid_coord, feat, offset, voxel_size=0.02
            )
            # import pdb; pdb.set_trace()
            # TODO: check the downsampling behavior is the same or not?
            assert (
                grid_coord.shape == grid.ijk.jdata.shape
            ), f"grid_coord.shape: {grid_coord.shape}, grid.ijk.jdata.shape: {grid.ijk.jdata.shape}"  # this is not always true, because mix-prob may duplicate points with the same coordinate.
            assert (
                grid_coord.shape[0] == original_coord_to_voxel_idx.jdata.shape[0]
            ), f"grid_coord.shape: {grid_coord.shape}, original_coord_to_voxel_idx.jdata.shape: {original_coord_to_voxel_idx.jdata.shape}"

            # import pdb; pdb.set_trace()
            if torch.is_autocast_enabled():
                with torch.autocast(device_type="cuda", enabled=False):
                    jfeats = self.fvdb_ptv3_model(jfeats, grid)
            else:
                jfeats = self.fvdb_ptv3_model(jfeats, grid)

            output = jfeats.jdata[original_coord_to_voxel_idx.jdata]
            # import pdb; pdb.set_trace()

        return output  # return logits in torch.tensor format
