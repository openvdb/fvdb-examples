# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import fvdb
import torch
import torch.nn as tnn
from fvdb.types import DeviceIdentifier, resolve_device
from torch_scatter import scatter_max, scatter_mean

from .resnet_block import ResnetBlockFC


class PointEncoder(tnn.Module):
    """PointNet-based encoder network with ResNet blocks for each point.

    This takes inputs which are batched according to the batching of the grid, and outputs
    results that are aligned to the voxels of the grid. Generally this will be a pooling, where
    the voxel count is lower than the input point count.

    ** IMPORTANT **
    Note that the point encoder deals with positions already in voxel space - no world to
    voxel conversion is done inside this class, it is assumed to be handled externally. The
    voxel space converts to ijks via floor.

    Args:
        size_feature: Feature dimension of the input points (e.g., 3 for position only,
            6 for position + normal).
        size_output: Feature dimension of the output. Defaults to 32.
        size_hidden: Feature dimension of the hidden layers. Defaults to 32.
        block_count: Number of ResNet blocks. Defaults to 3.
        dtype: Data type to use for the module. Defaults to torch.float32.
        device: Device to place the module on. Defaults to "cpu".
    """

    # Feature dimension of the input points. 3 for just pos, 6 for pos + norm, etc.
    size_feature: int

    # Feature dimension of the output. Defaults to 32
    size_output: int

    # Feature dimension of the hidden layers. Defaults to 32
    size_hidden: int

    # Number of ResNet blocks. Defaults to 3.
    block_count: int

    # The first fully connected layer, which goes from the input feature dimension to the hidden dimension.
    hidden_FC_input: tnn.Linear

    # The ResNet blocks.
    blocks: tnn.ModuleList

    # The final fully connected layer, which goes from the hidden dimension to the output dimension.
    output_FC_hidden: tnn.Linear

    def __init__(
        self,
        size_feature: int,
        *,
        size_output: int = 32,
        size_hidden: int = 32,
        block_count: int = 3,
        dtype: torch.dtype = torch.float32,
        device: DeviceIdentifier = "cpu",
    ):
        super().__init__()
        t_device: torch.device = resolve_device(device)

        self.size_feature = size_feature
        self.size_output = size_output
        self.size_hidden = size_hidden
        self.block_count = block_count

        self.hidden_FC_input = tnn.Linear(self.size_feature, 2 * self.size_hidden, dtype=dtype, device=t_device)
        self.blocks = tnn.ModuleList(
            [ResnetBlockFC(2 * size_hidden, size_hidden, dtype=dtype, device=device) for _ in range(block_count)]
        )
        self.output_FC_hidden = tnn.Linear(size_hidden, size_output, dtype=dtype, device=t_device)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize or reinitialize learnable parameters.

        Resets the input linear layer, all ResNet blocks, and the output linear layer
        using their respective initialization schemes.
        """
        self.hidden_FC_input.reset_parameters()
        for block in self.blocks:
            assert isinstance(block, ResnetBlockFC)
            block.reset_parameters()
        self.output_FC_hidden.reset_parameters()

    def extra_repr(self) -> str:
        return (
            f"size_feature={self.size_feature}, size_output={self.size_output}, "
            f"size_hidden={self.size_hidden}, block_count={self.block_count}"
        )

    def forward(
        self,
        jagged_positions_voxel: fvdb.JaggedTensor,
        jagged_extra_features: fvdb.JaggedTensor | None,
        grid: fvdb.GridBatch,
    ) -> fvdb.JaggedTensor:
        """Encode point cloud features into voxel-aligned features.

        This method implements a PointNet-style encoding where per-point features are
        processed through ResNet blocks with intermediate local max-pooling into voxels,
        then finally mean-pooled to produce one feature vector per voxel.

        Args:
            jagged_positions_voxel: Point positions in voxel-space coordinates, batched
                as a JaggedTensor matching the source point cloud topology.
            jagged_extra_features: Optional additional per-point features (e.g., normals).
                If provided, must have the same batch structure as jagged_positions_voxel.
                Shape of jdata: (total_points, size_feature - 3).
            grid: The sparse voxel grid defining the output structure. Points are pooled
                into the voxels of this grid.

        Returns:
            Per-voxel features as a JaggedTensor aligned to the grid's batch structure.
            Shape of jdata: (grid.total_voxels, size_output). Voxels with no overlapping
            input points will have zero features.

        Note:
            The input point count typically exceeds the voxel count, making this a
            many-to-one pooling operation. Points that don't overlap any grid voxel
            are ignored.

            Shape assertions throughout this method serve as executable documentation
            of tensor dimensions at each step. This is both pedagogical and ensures
            that this adaptation to the fVDB library doesn't violate any assumptions
            from the original architecture.
        """
        dtype: torch.dtype = jagged_positions_voxel.dtype
        device: torch.device = jagged_positions_voxel.device

        # The number of input positions, which is not the same as the number of voxels in the grid.
        input_count: int = jagged_positions_voxel.jdata.shape[0]
        if input_count == 0:
            # Our output is aligned to the grid, not the input.
            return grid.jagged_like(torch.zeros((grid.total_voxels, self.size_output), dtype=dtype, device=device))

        # Depending on the coarseness of the voxel transformation, many positions may
        # map to the same voxel position.
        jagged_ijks: fvdb.JaggedTensor = fvdb.floor(jagged_positions_voxel)

        # Get local voxel coords
        jagged_positions_voxlocal: fvdb.JaggedTensor = jagged_positions_voxel - jagged_ijks
        flat_positions_voxlocal: torch.Tensor = jagged_positions_voxlocal.jdata

        # Convert the ijks to the right data type.
        jagged_ijks = jagged_ijks.to(torch.int32)

        # Grid voxel indices, or -1, corresponding to each of the ijks of the positions. We expect
        # many positions to map to the same grid voxel index, often -1.
        jagged_gvoxel_indices: fvdb.JaggedTensor = grid.ijk_to_index(jagged_ijks, cumulative=True)
        flat_gvoxel_indices: torch.Tensor = jagged_gvoxel_indices.jdata

        # Get a mask (bool) tensor to indicate which feature indices contain data that
        # overlaps with the grid.
        overlap_mask: torch.Tensor = flat_gvoxel_indices != -1

        # Get only the valid gvoxel indices. These are the grid voxel indices of the
        # input positions that have overlap with the grid, extracted. Could be empty!!!
        # More interestingly, could be empty for some batch indices but not others. Flattened,
        # we won't see that directly unless there's no overlap in any batch index.
        valid_gvoxel_indices: torch.Tensor = flat_gvoxel_indices[overlap_mask]

        # Count the number of valid points. If it is zero, we can skip the rest of the forward pass.
        valid_count: int = valid_gvoxel_indices.shape[0]
        if valid_count == 0:
            return grid.jagged_like(torch.zeros((grid.total_voxels, self.size_output), dtype=dtype, device=device))

        # Get only the valid local voxel coords.
        valid_positions_voxlocal: torch.Tensor = flat_positions_voxlocal[overlap_mask]

        # Form the feature vector that goes through the network. It is only data for the
        # input positions that overlap with the grid.
        # The feature vector is [local_voxel_position (3D), extra_features (size_feature - 3)].
        # If size_feature == 3, we use only the local voxel position (no extra features).
        if jagged_extra_features is None:
            x = valid_positions_voxlocal
        else:
            valid_extra_features = jagged_extra_features.jdata[overlap_mask]
            x = torch.cat([valid_positions_voxlocal, valid_extra_features], dim=1)

        assert x.ndim == 2
        assert x.shape[0] == valid_count
        assert x.shape[1] == self.size_feature

        # The first step through the network is just the first linear layer.
        # This should just be a per-point expansion from size_feature to 2 * size_hidden,
        # so the ordering doesn't matter.
        x = self.hidden_FC_input(x)
        assert x.ndim == 2
        assert x.shape[0] == valid_count
        assert x.shape[1] == 2 * self.size_hidden

        # Then the first resnet block. This, again, is a per-point operation, so the ordering doesn't matter.
        x = self.blocks[0](x)
        assert x.ndim == 2
        assert x.shape[0] == valid_count
        assert x.shape[1] == self.size_hidden

        # For each of the remaining resnet blocks, we pool the features into their
        # corresponding voxel indices via scatter_max, then gather them back to points.
        # This pool-gather pattern lets each point "see" the max feature in its voxel,
        # enabling local context aggregation while maintaining per-point processing.
        for block in self.blocks[1:]:
            assert x.ndim == 2
            assert x.shape[0] == valid_count
            assert x.shape[1] == self.size_hidden

            # Take the values x, which are the network-transformed features only at the valid
            # feature indices, and scatter them via max (which removes order dependence) into an
            # output vector that's the same size as the original voxel grid.
            pooled, _ = scatter_max(x, valid_gvoxel_indices, dim=0, dim_size=grid.total_voxels)
            assert pooled.ndim == 2
            assert pooled.shape[0] == grid.total_voxels
            assert pooled.shape[1] == self.size_hidden

            # Though it isn't required, in general we'd expect the number of features to be
            # much larger than the number of grid voxels, assuming good overlap. So usually
            # the scatter max of the valid_count into the grid voxel indices is a pooling.

            # Gather the pooled features back to the valid feature indices.
            # usually, this will be a duplication of the pooled features back onto the input
            # positions that overlapped. This is usually an expansion.
            gathered = pooled[valid_gvoxel_indices]
            assert gathered.ndim == 2
            assert gathered.shape[0] == valid_count
            assert gathered.shape[1] == self.size_hidden

            # Then concatenate the gathered, pooled features with the original features
            x = torch.cat([x, gathered], dim=1)
            assert x.ndim == 2
            assert x.shape[0] == valid_count
            assert x.shape[1] == 2 * self.size_hidden

            # Pass the result through the next resnet block.
            x = block(x)
            assert x.ndim == 2
            assert x.shape[0] == valid_count
            assert x.shape[1] == self.size_hidden

        # convert the output from hidden channels to output channels.
        x = self.output_FC_hidden(x)
        assert x.ndim == 2
        assert x.shape[0] == valid_count
        assert x.shape[1] == self.size_output

        # Finally, we need to scatter the features back into their corresponding voxel indices.
        # Because this is a mean pooling, the ordering doesn't matter.
        # Note: Voxels with no overlapping input points will have zero features.
        x = scatter_mean(x, valid_gvoxel_indices, dim=0, dim_size=grid.total_voxels)
        assert x.ndim == 2
        assert x.shape[0] == grid.total_voxels
        assert x.shape[1] == self.size_output

        # Use the grid to re-jagged the output features. This means that while the input
        # data is an arbitrary point cloud, the output data is aligned to the batched voxel grid.
        return grid.jagged_like(x)
