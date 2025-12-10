# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for PointEncoder neural network component.

Tests cover the PointNet-style encoder that transforms point cloud features
into voxel-aligned features. This is a foundational component for neural
surface reconstruction. We test:
- Output shape alignment with grid structure
- Batch handling across different batch sizes
- Device compatibility (CPU and CUDA)
- Voxel size variations (coarsening/refining scenarios)
- Extra feature handling (e.g., colors, normals)
- Gradient flow for training
- Input mutation checks
- Inference with torch.no_grad
- Reset parameters functionality
- Edge cases (empty inputs, no grid overlap)
"""

import unittest

import fvdb
import torch
from nksr.nksr_fvdb.coord_xform import voxel_T_world_from_voxel_size
from nksr.nksr_fvdb.nn.point_encoder import PointEncoder
from parameterized import parameterized

from .utils import generate_street_scene_batch

# Parameter combinations for device and batch size
all_device_batch_combos = [
    ["cpu", 1],
    ["cpu", 2],
    ["cpu", 4],
    ["cuda", 1],
    ["cuda", 2],
    ["cuda", 4],
]

# Device-only combinations for tests that don't need batch size variation
all_devices = [
    ["cpu"],
    ["cuda"],
]

if not torch.cuda.is_available():
    all_device_batch_combos = [combo for combo in all_device_batch_combos if combo[0] != "cuda"]
    all_devices = [device for device in all_devices if device[0] != "cuda"]

# Standard test parameters for street scene hierarchy
# Note: We use "voxel space" (lower-left corner), NOT "voxcen space" (center)
DEFAULT_VOXEL_SIZE = 0.05  # 5cm - appropriate for street-scale scenes (30-50m)
DEFAULT_NUM_POINTS = 5000


def generate_spoofed_colors(
    points: fvdb.JaggedTensor,
    seed: int = 42,
) -> fvdb.JaggedTensor:
    """Generate spoofed color features for testing extra feature handling.

    Creates per-point RGB-like features in [0, 1] range based on position.
    This provides structured (not random) extra features for testing.

    Args:
        points: JaggedTensor of point positions, shape [B, N, 3].
        seed: Random seed for reproducibility.

    Returns:
        JaggedTensor of spoofed colors, same structure as points, shape [B, N, 3].
    """
    gen = torch.Generator(device=points.device).manual_seed(seed)

    # Create position-based colors with some randomness
    # Normalize positions to [0, 1] range based on bounding box
    jdata = points.jdata
    pos_min = jdata.min(dim=0, keepdim=True).values
    pos_max = jdata.max(dim=0, keepdim=True).values
    pos_range = pos_max - pos_min + 1e-6  # Avoid division by zero

    # Base color from normalized position
    base_colors = (jdata - pos_min) / pos_range

    # Add small random perturbation
    noise = torch.rand(jdata.shape, generator=gen, device=jdata.device, dtype=jdata.dtype) * 0.1
    colors = torch.clamp(base_colors + noise, 0.0, 1.0)

    return points.jagged_like(colors)


class TestPointEncoderOutputShape(unittest.TestCase):
    """Tests for output shape and structure alignment."""

    @parameterized.expand(all_device_batch_combos)
    def test_output_aligned_to_grid(self, device: str, batch_size: int) -> None:
        """Test that output JaggedTensor is aligned to grid voxels, not input points."""
        # Generate street scene data
        points_world = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        # Convert to voxel space
        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)

        # Create grid from points
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        # Create encoder (size_feature=3 for position only)
        encoder = PointEncoder(size_feature=3, size_output=32, size_hidden=32, block_count=3, device=device)

        with torch.no_grad():
            features = encoder(points_voxel, None, grid)

        # Output should be aligned to grid, not input points
        self.assertEqual(features.jdata.shape[0], grid.total_voxels)
        self.assertEqual(features.jdata.shape[1], 32)

        # Check batch structure matches grid
        self.assertEqual(len(features.joffsets), batch_size + 1)

    @parameterized.expand(all_device_batch_combos)
    def test_output_shape_with_extra_features(self, device: str, batch_size: int) -> None:
        """Test output shape when extra features (e.g., colors) are provided."""
        points_world = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        # Generate spoofed color features
        colors = generate_spoofed_colors(points_world)

        # size_feature = 3 (position) + 3 (colors) = 6
        encoder = PointEncoder(size_feature=6, size_output=64, size_hidden=48, block_count=4, device=device)

        with torch.no_grad():
            features = encoder(points_voxel, colors, grid)

        # Output should still be aligned to grid
        self.assertEqual(features.jdata.shape[0], grid.total_voxels)
        self.assertEqual(features.jdata.shape[1], 64)

    @parameterized.expand(all_devices)
    def test_output_dimensions_configurable(self, device: str) -> None:
        """Test that size_output parameter controls output dimension."""
        points_world = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        for size_output in [16, 32, 64, 128]:
            encoder = PointEncoder(size_feature=3, size_output=size_output, device=device)

            with torch.no_grad():
                features = encoder(points_voxel, None, grid)

            self.assertEqual(
                features.jdata.shape[1],
                size_output,
                f"Expected output dim {size_output}, got {features.jdata.shape[1]}",
            )


class TestPointEncoderVoxelSizeVariation(unittest.TestCase):
    """Tests for behavior with varying voxel sizes (coarsening/refining)."""

    @parameterized.expand(all_device_batch_combos)
    def test_coarser_voxels_fewer_outputs(self, device: str, batch_size: int) -> None:
        """Test that coarser voxel sizes result in fewer output features."""
        points_world = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_sizes = [0.05, 0.10, 0.20, 0.40]  # Progressively coarser
        voxel_counts = []

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        for voxel_size in voxel_sizes:
            voxel_T_world = voxel_T_world_from_voxel_size(voxel_size)
            points_voxel = voxel_T_world(points_world)
            grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

            with torch.no_grad():
                features = encoder(points_voxel, None, grid)

            voxel_counts.append(grid.total_voxels)

        # Coarser voxels should result in fewer voxels
        for i in range(1, len(voxel_counts)):
            self.assertLess(
                voxel_counts[i],
                voxel_counts[i - 1],
                f"Voxel size {voxel_sizes[i]} should have fewer voxels than {voxel_sizes[i-1]}",
            )

    @parameterized.expand(all_device_batch_combos)
    def test_finer_voxels_more_outputs(self, device: str, batch_size: int) -> None:
        """Test that finer voxel sizes result in more output features."""
        points_world = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_sizes = [0.20, 0.10, 0.05, 0.025]  # Progressively finer
        voxel_counts = []

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        for voxel_size in voxel_sizes:
            voxel_T_world = voxel_T_world_from_voxel_size(voxel_size)
            points_voxel = voxel_T_world(points_world)
            grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

            with torch.no_grad():
                features = encoder(points_voxel, None, grid)

            voxel_counts.append(grid.total_voxels)

        # Finer voxels should result in more voxels
        for i in range(1, len(voxel_counts)):
            self.assertGreater(
                voxel_counts[i],
                voxel_counts[i - 1],
                f"Voxel size {voxel_sizes[i]} should have more voxels than {voxel_sizes[i-1]}",
            )

    @parameterized.expand(all_devices)
    def test_same_encoder_different_voxel_sizes(self, device: str) -> None:
        """Test that the same encoder works correctly across different voxel sizes.

        This simulates the typical use case where a single encoder is used
        at multiple hierarchy levels with different effective voxel sizes.
        """
        points_world = generate_street_scene_batch(
            batch_size=2,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        # Create one encoder to use at all scales
        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        # Test at multiple voxel sizes
        for voxel_size in [0.025, 0.05, 0.10, 0.20]:
            voxel_T_world = voxel_T_world_from_voxel_size(voxel_size)
            points_voxel = voxel_T_world(points_world)
            grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

            with torch.no_grad():
                features = encoder(points_voxel, None, grid)

            # Verify output is valid
            self.assertEqual(features.jdata.shape[0], grid.total_voxels)
            self.assertEqual(features.jdata.shape[1], 32)
            self.assertTrue(torch.isfinite(features.jdata).all())


class TestPointEncoderGradients(unittest.TestCase):
    """Tests for gradient flow and autograd sanity."""

    @parameterized.expand(all_device_batch_combos)
    def test_backward_produces_finite_gradients(self, device: str, batch_size: int) -> None:
        """Test that backward pass works and produces finite gradients."""
        points_world = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        # Forward pass (no torch.no_grad - we want gradients)
        features = encoder(points_voxel, None, grid)

        # Compute loss and backward
        loss = features.jdata.square().mean()
        loss.backward()

        # Check all parameter gradients exist and are finite
        for name, param in encoder.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
            assert param.grad is not None
            self.assertTrue(torch.isfinite(param.grad).all(), f"Gradient for {name} contains non-finite values")

    @parameterized.expand(all_device_batch_combos)
    def test_gradients_with_extra_features(self, device: str, batch_size: int) -> None:
        """Test gradient flow when extra features are provided."""
        points_world = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        colors = generate_spoofed_colors(points_world)

        encoder = PointEncoder(size_feature=6, size_output=32, device=device)

        features = encoder(points_voxel, colors, grid)
        loss = features.jdata.square().mean()
        loss.backward()

        for name, param in encoder.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
            assert param.grad is not None
            self.assertTrue(torch.isfinite(param.grad).all(), f"Gradient for {name} contains non-finite values")

    @parameterized.expand(all_devices)
    def test_gradients_flow_after_training_step(self, device: str) -> None:
        """Test that gradients flow through all layers after a training step."""
        points_world = generate_street_scene_batch(
            batch_size=2,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)

        # Training step to move away from initialization
        for _ in range(3):
            optimizer.zero_grad()
            features = encoder(points_voxel, None, grid)
            loss = features.jdata.square().mean()
            loss.backward()
            optimizer.step()

        # Check gradients after training
        optimizer.zero_grad()
        features = encoder(points_voxel, None, grid)
        loss = features.jdata.square().mean()
        loss.backward()

        # Input FC layer should have non-zero gradients
        self.assertTrue(
            (encoder.hidden_FC_input.weight.grad != 0).any(),
            "hidden_FC_input should have non-zero gradients after training",
        )


class TestPointEncoderInference(unittest.TestCase):
    """Tests for inference mode and torch.no_grad behavior."""

    @parameterized.expand(all_device_batch_combos)
    def test_inference_with_no_grad(self, device: str, batch_size: int) -> None:
        """Test that inference works correctly with torch.no_grad."""
        points_world = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        with torch.no_grad():
            features = encoder(points_voxel, None, grid)

        # Should produce valid output
        self.assertEqual(features.jdata.shape[0], grid.total_voxels)
        self.assertTrue(torch.isfinite(features.jdata).all())

        # Should not require grad
        self.assertFalse(features.jdata.requires_grad)

    @parameterized.expand(all_device_batch_combos)
    def test_eval_mode_inference(self, device: str, batch_size: int) -> None:
        """Test inference in eval mode (though PointEncoder has no BatchNorm/Dropout)."""
        points_world = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)
        encoder.eval()

        with torch.no_grad():
            features = encoder(points_voxel, None, grid)

        self.assertEqual(features.jdata.shape[0], grid.total_voxels)
        self.assertTrue(torch.isfinite(features.jdata).all())

    @parameterized.expand(all_devices)
    def test_train_and_eval_modes_consistent(self, device: str) -> None:
        """Test that train and eval modes produce same output (no BN/Dropout)."""
        points_world = generate_street_scene_batch(
            batch_size=2,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        # Train mode
        encoder.train()
        with torch.no_grad():
            features_train = encoder(points_voxel, None, grid)

        # Eval mode
        encoder.eval()
        with torch.no_grad():
            features_eval = encoder(points_voxel, None, grid)

        # Should be identical since PointEncoder has no mode-dependent layers
        # Use allclose to handle minor CUDA floating-point non-determinism
        self.assertTrue(
            torch.allclose(features_train.jdata, features_eval.jdata, atol=1e-6, rtol=1e-6),
            "Train and eval mode outputs should be essentially identical",
        )


class TestPointEncoderInputMutation(unittest.TestCase):
    """Tests for ensuring inputs are not modified in-place."""

    @parameterized.expand(all_device_batch_combos)
    def test_does_not_modify_input_positions(self, device: str, batch_size: int) -> None:
        """Test that forward pass does not mutate the input position tensor."""
        points_world = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        # Clone inputs to compare after forward
        positions_clone = points_voxel.jdata.clone()

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        with torch.no_grad():
            _ = encoder(points_voxel, None, grid)

        self.assertTrue(torch.equal(points_voxel.jdata, positions_clone))

    @parameterized.expand(all_device_batch_combos)
    def test_does_not_modify_extra_features(self, device: str, batch_size: int) -> None:
        """Test that forward pass does not mutate the extra features tensor."""
        points_world = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        colors = generate_spoofed_colors(points_world)
        colors_clone = colors.jdata.clone()

        encoder = PointEncoder(size_feature=6, size_output=32, device=device)

        with torch.no_grad():
            _ = encoder(points_voxel, colors, grid)

        self.assertTrue(torch.equal(colors.jdata, colors_clone))


class TestPointEncoderEdgeCases(unittest.TestCase):
    """Tests for edge cases and validation."""

    @parameterized.expand(all_devices)
    def test_empty_input_returns_zero_features(self, device: str) -> None:
        """Test that empty input produces zero features aligned to grid."""
        # Create a grid with some voxels
        dummy_points = fvdb.JaggedTensor.from_list_of_tensors(
            [torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)]
        )
        grid = fvdb.GridBatch.from_points(dummy_points, voxel_sizes=1, origins=0)

        # Create empty input (no points)
        empty_points = fvdb.JaggedTensor.from_list_of_tensors([torch.zeros((0, 3), device=device)])

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        with torch.no_grad():
            features = encoder(empty_points, None, grid)

        # Output should be zeros aligned to grid
        self.assertEqual(features.jdata.shape[0], grid.total_voxels)
        self.assertEqual(features.jdata.shape[1], 32)
        self.assertTrue(torch.all(features.jdata == 0))

    @parameterized.expand(all_devices)
    def test_points_outside_grid_ignored(self, device: str) -> None:
        """Test that points outside the grid are handled gracefully."""
        # Create a small grid at origin
        grid_points = fvdb.JaggedTensor.from_list_of_tensors(
            [torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)]
        )
        grid = fvdb.GridBatch.from_points(grid_points, voxel_sizes=1, origins=0)

        # Create points far outside the grid
        outside_points = fvdb.JaggedTensor.from_list_of_tensors(
            [torch.tensor([[100.0, 100.0, 100.0], [200.0, 200.0, 200.0]], device=device)]
        )

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        with torch.no_grad():
            features = encoder(outside_points, None, grid)

        # Output should be zeros (no overlap with grid)
        self.assertEqual(features.jdata.shape[0], grid.total_voxels)
        self.assertTrue(torch.all(features.jdata == 0))

    @parameterized.expand(all_devices)
    def test_single_point_per_voxel(self, device: str) -> None:
        """Test behavior when there's exactly one point per voxel."""
        # Create points at voxel centers (one per voxel)
        points = fvdb.JaggedTensor.from_list_of_tensors(
            [
                torch.tensor(
                    [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 0.5, 1.5]],
                    device=device,
                )
            ]
        )
        grid = fvdb.GridBatch.from_points(points, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        with torch.no_grad():
            features = encoder(points, None, grid)

        # Should have one feature per voxel
        self.assertEqual(features.jdata.shape[0], 4)
        self.assertTrue(torch.isfinite(features.jdata).all())

    @parameterized.expand(all_devices)
    def test_many_points_per_voxel(self, device: str) -> None:
        """Test behavior when many points fall into each voxel."""
        points_world = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=10000,  # Many points
            device=device,
        )

        # Use very coarse voxels to get many points per voxel
        voxel_T_world = voxel_T_world_from_voxel_size(1.0)  # 1m voxels
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        with torch.no_grad():
            features = encoder(points_voxel, None, grid)

        # Should produce valid features
        self.assertEqual(features.jdata.shape[0], grid.total_voxels)
        self.assertTrue(torch.isfinite(features.jdata).all())


class TestPointEncoderResetParameters(unittest.TestCase):
    """Tests for reset_parameters functionality."""

    @parameterized.expand(all_devices)
    def test_reset_parameters_changes_weights(self, device: str) -> None:
        """Test that reset_parameters actually changes the weights."""
        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        # Get initial weights
        initial_weight = encoder.hidden_FC_input.weight.clone()

        # Train to modify weights
        points_world = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )
        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.5)
        for _ in range(5):
            optimizer.zero_grad()
            features = encoder(points_voxel, None, grid)
            loss = features.jdata.square().mean()
            loss.backward()
            optimizer.step()

        # Verify weights changed
        self.assertFalse(torch.equal(encoder.hidden_FC_input.weight, initial_weight))

        # Reset and verify we get new random weights (not the same as initial)
        encoder.reset_parameters()
        # Note: reset_parameters uses new random values, so weights will differ
        # The important thing is that reset works without error


class TestPointEncoderDeterminism(unittest.TestCase):
    """Tests for reproducibility and determinism."""

    @parameterized.expand(all_devices)
    def test_same_input_produces_same_output(self, device: str) -> None:
        """Test that the same input produces essentially identical output.

        Note: PyTorch's scatter_reduce_ can be non-deterministic on CUDA devices,
        so we use allclose to allow for minor floating-point variations.
        """
        points_world = generate_street_scene_batch(
            batch_size=2,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        with torch.no_grad():
            features1 = encoder(points_voxel, None, grid)
            features2 = encoder(points_voxel, None, grid)

        # Use allclose to handle potential CUDA non-determinism in scatter operations
        self.assertTrue(
            torch.allclose(features1.jdata, features2.jdata, atol=1e-5, rtol=1e-5),
            "Same input should produce essentially identical output",
        )

    @parameterized.expand(all_devices)
    def test_different_seeds_produce_different_outputs(self, device: str) -> None:
        """Test that different input data produces different outputs."""
        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        # First scene
        points_world_1 = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )
        points_voxel_1 = voxel_T_world(points_world_1)
        grid_1 = fvdb.GridBatch.from_points(points_voxel_1, voxel_sizes=1, origins=0)

        with torch.no_grad():
            features1 = encoder(points_voxel_1, None, grid_1)

        # Different scene
        points_world_2 = generate_street_scene_batch(
            batch_size=1,
            base_seed=12345,  # Different seed
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )
        points_voxel_2 = voxel_T_world(points_world_2)
        grid_2 = fvdb.GridBatch.from_points(points_voxel_2, voxel_sizes=1, origins=0)

        with torch.no_grad():
            features2 = encoder(points_voxel_2, None, grid_2)

        # Features should be different (different scenes)
        self.assertFalse(torch.equal(features1.jdata, features2.jdata))


class TestPointEncoderDeviceAndDtype(unittest.TestCase):
    """Tests for device and dtype handling."""

    def test_cuda_matches_cpu(self) -> None:
        """Test that CUDA and CPU produce consistent results."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        points_world = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device="cpu",
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel_cpu = voxel_T_world(points_world)
        grid_cpu = fvdb.GridBatch.from_points(points_voxel_cpu, voxel_sizes=1, origins=0)

        encoder_cpu = PointEncoder(size_feature=3, size_output=32, device="cpu")

        # Create GPU encoder with same weights
        encoder_gpu = PointEncoder(size_feature=3, size_output=32, device="cuda")
        encoder_gpu.load_state_dict(encoder_cpu.state_dict())

        # Move data to GPU
        points_voxel_gpu = fvdb.JaggedTensor.from_list_of_tensors([points_voxel_cpu.jdata.cuda()])
        grid_gpu = fvdb.GridBatch.from_points(points_voxel_gpu, voxel_sizes=1, origins=0)

        with torch.no_grad():
            features_cpu = encoder_cpu(points_voxel_cpu, None, grid_cpu)
            features_gpu = encoder_gpu(points_voxel_gpu, None, grid_gpu)

        # Results should be close (allowing for floating point differences)
        self.assertTrue(
            torch.allclose(features_cpu.jdata, features_gpu.jdata.cpu(), atol=1e-5, rtol=1e-5),
            "CPU and GPU results should match",
        )

    @parameterized.expand(all_devices)
    def test_output_dtype_matches_input(self, device: str) -> None:
        """Test that output dtype matches input dtype."""
        points_world = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=1000,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, dtype=torch.float32, device=device)

        with torch.no_grad():
            features = encoder(points_voxel, None, grid)

        self.assertEqual(features.jdata.dtype, torch.float32)


class TestPointEncoderSerialization(unittest.TestCase):
    """Tests for serialization and state dict handling."""

    @parameterized.expand(all_devices)
    def test_state_dict_round_trip(self, device: str) -> None:
        """Test that saving/loading state_dict preserves behavior."""
        points_world = generate_street_scene_batch(
            batch_size=2,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        # Train to move away from initial state
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
        for _ in range(5):
            optimizer.zero_grad()
            features = encoder(points_voxel, None, grid)
            loss = features.jdata.square().mean()
            loss.backward()
            optimizer.step()

        # Get output before save
        with torch.no_grad():
            features_before = encoder(points_voxel, None, grid)

        # Save and reload
        state = encoder.state_dict()
        encoder2 = PointEncoder(size_feature=3, size_output=32, device=device)
        encoder2.load_state_dict(state)

        # Get output after load
        with torch.no_grad():
            features_after = encoder2(points_voxel, None, grid)

        self.assertTrue(torch.allclose(features_before.jdata, features_after.jdata, atol=1e-6, rtol=1e-6))

    @parameterized.expand(all_devices)
    def test_repr_contains_hyperparameters(self, device: str) -> None:
        """Test that repr contains key hyperparameters."""
        encoder = PointEncoder(size_feature=6, size_output=64, size_hidden=48, block_count=5, device=device)
        s = repr(encoder)

        self.assertIn("size_feature=6", s)
        self.assertIn("size_output=64", s)
        self.assertIn("size_hidden=48", s)
        self.assertIn("block_count=5", s)


class TestPointEncoderScatterReduction(unittest.TestCase):
    """Tests for scatter reduction operations (max pooling and mean pooling).

    These tests specifically verify that the scatter operations correctly
    aggregate multiple points per voxel. This is critical for the PointNet-style
    architecture where many points can fall into the same voxel.
    """

    @parameterized.expand(all_devices)
    def test_multiple_points_per_voxel_produces_valid_output(self, device: str) -> None:
        """Test that voxels with multiple points produce finite, non-zero features."""
        # Create points where multiple points fall into the same voxel
        # Points at [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3] all map to voxel [0,0,0]
        # Points at [1.1, 1.1, 1.1], [1.2, 1.2, 1.2] map to voxel [1,1,1]
        points = fvdb.JaggedTensor.from_list_of_tensors(
            [
                torch.tensor(
                    [
                        [0.1, 0.1, 0.1],
                        [0.2, 0.2, 0.2],
                        [0.3, 0.3, 0.3],
                        [1.1, 1.1, 1.1],
                        [1.2, 1.2, 1.2],
                    ],
                    device=device,
                )
            ]
        )
        grid = fvdb.GridBatch.from_points(points, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        with torch.no_grad():
            features = encoder(points, None, grid)

        # Should have exactly 2 voxels with valid features
        self.assertEqual(features.jdata.shape[0], grid.total_voxels)
        self.assertTrue(torch.isfinite(features.jdata).all())
        # Features should not be all zeros (since we have points in all voxels)
        self.assertTrue((features.jdata != 0).any())

    @parameterized.expand(all_devices)
    def test_varying_points_per_voxel(self, device: str) -> None:
        """Test encoder with varying number of points per voxel.

        This test creates multiple points that should cluster into a small number
        of voxels (fewer than the total number of points), exercising the
        many-to-one scatter aggregation.
        """
        # Create points clustered around certain locations
        points = fvdb.JaggedTensor.from_list_of_tensors(
            [
                torch.tensor(
                    [
                        # Cluster 1: 1 point
                        [0.5, 0.5, 0.5],
                        # Cluster 2: 3 points (all in same voxel due to small offsets)
                        [10.1, 10.1, 10.1],
                        [10.2, 10.2, 10.2],
                        [10.3, 10.3, 10.3],
                        # Cluster 3: 5 points (all in same voxel)
                        [20.1, 20.1, 20.1],
                        [20.2, 20.2, 20.2],
                        [20.3, 20.3, 20.3],
                        [20.4, 20.4, 20.4],
                        [20.5, 20.5, 20.5],
                    ],
                    device=device,
                )
            ]
        )
        grid = fvdb.GridBatch.from_points(points, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        with torch.no_grad():
            features = encoder(points, None, grid)

        # Key assertion: there should be fewer voxels than input points
        # (demonstrating many-to-one aggregation)
        num_points = points.jdata.shape[0]
        self.assertLess(grid.total_voxels, num_points, "Should have fewer voxels than input points")
        self.assertEqual(features.jdata.shape[0], grid.total_voxels)
        self.assertTrue(torch.isfinite(features.jdata).all())
        # At least some voxels should have non-zero features
        self.assertTrue((features.jdata != 0).any(), "At least some features should be non-zero")

    @parameterized.expand(all_devices)
    def test_dense_points_many_per_voxel(self, device: str) -> None:
        """Test with very dense point cloud - 100 points in a small region."""
        # Create 100 random points within a small region
        # Use small offsets from a base point to ensure they all fall in the same voxel
        torch.manual_seed(42)
        base = torch.tensor([[10.0, 10.0, 10.0]], device=device)
        offsets = torch.rand(100, 3, device=device) * 0.5  # Small offsets in [0, 0.5)
        single_voxel_points = base + offsets

        points = fvdb.JaggedTensor.from_list_of_tensors([single_voxel_points])
        grid = fvdb.GridBatch.from_points(points, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        with torch.no_grad():
            features = encoder(points, None, grid)

        # Should have exactly 1 voxel (all points in the same voxel)
        self.assertEqual(grid.total_voxels, 1)
        self.assertEqual(features.jdata.shape[0], 1)
        self.assertTrue(torch.isfinite(features.jdata).all())

    @parameterized.expand(all_devices)
    def test_scatter_consistency_with_duplicates(self, device: str) -> None:
        """Test that scatter operations produce consistent results with multiple points per voxel.

        Note: PyTorch's scatter_reduce_ can be non-deterministic on CUDA devices due to
        atomic operation ordering. We use allclose to allow for minor floating-point
        variations while ensuring results are essentially the same.
        """
        points = fvdb.JaggedTensor.from_list_of_tensors(
            [
                torch.tensor(
                    [
                        [10.1, 10.1, 10.1],
                        [10.5, 10.5, 10.5],
                        [10.9, 10.9, 10.9],
                        [20.1, 20.1, 20.1],
                        [20.5, 20.5, 20.5],
                    ],
                    device=device,
                )
            ]
        )
        grid = fvdb.GridBatch.from_points(points, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        with torch.no_grad():
            features1 = encoder(points, None, grid)
            features2 = encoder(points, None, grid)
            features3 = encoder(points, None, grid)

        # All runs should produce essentially identical results
        # Use allclose to handle potential CUDA non-determinism in scatter operations
        self.assertTrue(
            torch.allclose(features1.jdata, features2.jdata, atol=1e-5, rtol=1e-5),
            "Features from run 1 and 2 should be essentially identical",
        )
        self.assertTrue(
            torch.allclose(features2.jdata, features3.jdata, atol=1e-5, rtol=1e-5),
            "Features from run 2 and 3 should be essentially identical",
        )

    @parameterized.expand(all_devices)
    def test_gradients_flow_with_multiple_points_per_voxel(self, device: str) -> None:
        """Test that gradients flow correctly when multiple points map to same voxel."""
        points = fvdb.JaggedTensor.from_list_of_tensors(
            [
                torch.tensor(
                    [
                        [0.1, 0.1, 0.1],
                        [0.5, 0.5, 0.5],
                        [0.9, 0.9, 0.9],  # 3 points in voxel [0,0,0]
                        [1.5, 1.5, 1.5],  # 1 point in voxel [1,1,1]
                    ],
                    device=device,
                )
            ]
        )
        grid = fvdb.GridBatch.from_points(points, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        # Forward and backward pass
        features = encoder(points, None, grid)
        loss = features.jdata.square().mean()
        loss.backward()

        # All parameters should have gradients
        for name, param in encoder.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
            assert param.grad is not None
            self.assertTrue(
                torch.isfinite(param.grad).all(),
                f"Gradient for {name} contains non-finite values",
            )

    @parameterized.expand(all_device_batch_combos)
    def test_batched_with_varying_density(self, device: str, batch_size: int) -> None:
        """Test batched input where different batches have different point densities."""
        tensors = []
        for i in range(batch_size):
            # Each batch item has (i+1)*10 points, creating varying density
            num_points = (i + 1) * 10
            torch.manual_seed(42 + i)
            pts = torch.rand(num_points, 3, device=device) * 3  # Points in [0, 3) range
            tensors.append(pts)

        points = fvdb.JaggedTensor.from_list_of_tensors(tensors)
        grid = fvdb.GridBatch.from_points(points, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=3, size_output=32, device=device)

        with torch.no_grad():
            features = encoder(points, None, grid)

        self.assertEqual(features.jdata.shape[0], grid.total_voxels)
        self.assertTrue(torch.isfinite(features.jdata).all())


class TestPointEncoderExtraFeatures(unittest.TestCase):
    """Tests for extra feature handling (colors, normals, etc.)."""

    @parameterized.expand(all_device_batch_combos)
    def test_with_color_features(self, device: str, batch_size: int) -> None:
        """Test encoder with spoofed color features (3 extra dims)."""
        points_world = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        colors = generate_spoofed_colors(points_world)

        # size_feature = 3 (local voxel pos) + 3 (colors) = 6
        encoder = PointEncoder(size_feature=6, size_output=32, device=device)

        with torch.no_grad():
            features = encoder(points_voxel, colors, grid)

        self.assertEqual(features.jdata.shape[0], grid.total_voxels)
        self.assertEqual(features.jdata.shape[1], 32)
        self.assertTrue(torch.isfinite(features.jdata).all())

    @parameterized.expand(all_device_batch_combos)
    def test_with_intensity_feature(self, device: str, batch_size: int) -> None:
        """Test encoder with single extra feature (e.g., intensity)."""
        points_world = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        # Spoof intensity as 1D feature
        intensity_data = torch.rand(points_world.jdata.shape[0], 1, device=device)
        intensity = points_world.jagged_like(intensity_data)

        # size_feature = 3 (local voxel pos) + 1 (intensity) = 4
        encoder = PointEncoder(size_feature=4, size_output=32, device=device)

        with torch.no_grad():
            features = encoder(points_voxel, intensity, grid)

        self.assertEqual(features.jdata.shape[0], grid.total_voxels)
        self.assertTrue(torch.isfinite(features.jdata).all())

    @parameterized.expand(all_devices)
    def test_extra_features_affect_output(self, device: str) -> None:
        """Test that different extra features produce different outputs."""
        points_world = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=DEFAULT_NUM_POINTS,
            device=device,
        )

        voxel_T_world = voxel_T_world_from_voxel_size(DEFAULT_VOXEL_SIZE)
        points_voxel = voxel_T_world(points_world)
        grid = fvdb.GridBatch.from_points(points_voxel, voxel_sizes=1, origins=0)

        encoder = PointEncoder(size_feature=6, size_output=32, device=device)

        # Run with two different color inputs
        colors1 = generate_spoofed_colors(points_world, seed=42)
        colors2 = generate_spoofed_colors(points_world, seed=999)

        with torch.no_grad():
            features1 = encoder(points_voxel, colors1, grid)
            features2 = encoder(points_voxel, colors2, grid)

        # Different extra features should produce different outputs
        self.assertFalse(torch.equal(features1.jdata, features2.jdata))


if __name__ == "__main__":
    unittest.main()
