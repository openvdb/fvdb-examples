# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ResnetBlockFC fully connected ResNet block.

Tests cover:
- Shape behavior for point cloud inputs
- Initialization behavior (identity vs projection)
- Gradient flow
- Input mutation checks
- Hidden dimension configuration
- Device compatibility (CPU and CUDA)
- Serialization and reset_parameters
"""

import unittest

import torch
from nksr.nksr_fvdb.nn.resnet_block import ResnetBlockFC
from parameterized import parameterized

all_devices = [["cpu"], ["cuda"]]
if not torch.cuda.is_available():
    all_devices.remove(["cuda"])


class TestResnetBlockFCShapes(unittest.TestCase):
    """Tests for output shape behavior."""

    @parameterized.expand(all_devices)
    def test_shape_same_dim(self, device: str) -> None:
        """Test [N, D] -> [N, D] (identity-shaped block)."""
        block = ResnetBlockFC(6, 6, device=device)
        x = torch.randn(128, 6, device=device)
        y = block(x)
        self.assertEqual(y.shape, (128, 6))

    @parameterized.expand(all_devices)
    def test_shape_different_dim(self, device: str) -> None:
        """Test [N, D_in] -> [N, D_out] (dimension change)."""
        block = ResnetBlockFC(3, 64, device=device)
        x = torch.randn(128, 3, device=device)
        y = block(x)
        self.assertEqual(y.shape, (128, 64))

    @parameterized.expand(all_devices)
    def test_shape_batched(self, device: str) -> None:
        """Test [B, N, D] batched input."""
        block = ResnetBlockFC(6, 10, device=device)
        x = torch.randn(4, 1024, 6, device=device)
        y = block(x)
        self.assertEqual(y.shape, (4, 1024, 10))

    @parameterized.expand(all_devices)
    def test_shape_default_size_out(self, device: str) -> None:
        """Test that size_out defaults to size_in when not specified."""
        block = ResnetBlockFC(6, device=device)
        x = torch.randn(128, 6, device=device)
        y = block(x)
        self.assertEqual(y.shape, (128, 6))
        self.assertEqual(block.size_out, 6)


class TestResnetBlockFCInitialization(unittest.TestCase):
    """Tests for initialization behavior (identity vs projection)."""

    @parameterized.expand(all_devices)
    def test_initially_identity_when_same_dim(self, device: str) -> None:
        """Test that initial output equals input when size_in == size_out.

        Since fully_connected_1 is zero-initialized, the residual branch outputs
        zero at initialization, making the block behave as identity.
        """
        block = ResnetBlockFC(6, device=device)
        x = torch.randn(128, 6, device=device)
        y = block(x)
        self.assertTrue(torch.allclose(y, x, atol=0, rtol=0))

    @parameterized.expand(all_devices)
    def test_initially_projection_when_dim_differs(self, device: str) -> None:
        """Test that initial output equals shortcut(x) when size_in != size_out.

        Since fully_connected_1 is zero-initialized, the residual branch outputs
        zero at initialization, leaving only the shortcut projection.
        """
        block = ResnetBlockFC(3, 16, device=device)
        x = torch.randn(128, 3, device=device)
        y = block(x)
        assert block.shortcut is not None
        shortcut_y = block.shortcut(x)
        self.assertTrue(torch.allclose(y, shortcut_y, atol=0, rtol=0))


class TestResnetBlockFCGradients(unittest.TestCase):
    """Tests for gradient and autograd sanity."""

    @parameterized.expand(all_devices)
    def test_backward_produces_finite_gradients(self, device: str) -> None:
        """Test that backward pass works and produces finite gradients."""
        block = ResnetBlockFC(6, 16, device=device)
        x = torch.randn(64, 6, device=device, requires_grad=True)

        y = block(x)
        loss = y.square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertTrue(torch.isfinite(x.grad).all())

        # Check all parameter gradients exist and are finite
        for param in block.parameters():
            self.assertIsNotNone(param.grad)
            assert param.grad is not None
            self.assertTrue(torch.isfinite(param.grad).all())

    @parameterized.expand(all_devices)
    def test_gradients_flow_through_residual_after_training(self, device: str) -> None:
        """Test that gradients flow through fc0 after training activates residual branch.

        At initialization, fully_connected_1 is zero, so gradients don't flow
        through fully_connected_0. After one training step, the residual branch
        becomes active.
        """
        block = ResnetBlockFC(3, 8, device=device)

        # Training step to activate residual branch
        optimizer = torch.optim.SGD(block.parameters(), lr=0.1)
        x_train = torch.randn(64, 3, device=device)
        loss = block(x_train).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Test gradient flow
        optimizer.zero_grad()
        x = torch.randn(64, 3, device=device, requires_grad=True)
        loss = block(x).square().mean()
        loss.backward()

        assert block.fully_connected_0.weight.grad is not None
        self.assertTrue((block.fully_connected_0.weight.grad != 0).any())


class TestResnetBlockFCInputMutation(unittest.TestCase):
    """Tests for ensuring inputs are not modified in-place."""

    @parameterized.expand(all_devices)
    def test_does_not_modify_input(self, device: str) -> None:
        """Test that forward pass does not mutate the input tensor."""
        block = ResnetBlockFC(3, 8, device=device)
        x = torch.randn(128, 3, device=device)
        x_clone = x.clone()
        _ = block(x)
        self.assertTrue(torch.allclose(x, x_clone))


class TestResnetBlockFCHiddenDimension(unittest.TestCase):
    """Tests for hidden dimension configuration."""

    def test_hidden_dim_is_min_of_in_out(self) -> None:
        """Test size_hidden = min(size_in, size_out)."""
        # Lifting: hidden = min(3, 16) = 3
        block_lift = ResnetBlockFC(3, 16)
        self.assertEqual(block_lift.size_hidden, 3)

        # Reducing: hidden = min(16, 3) = 3
        block_reduce = ResnetBlockFC(16, 3)
        self.assertEqual(block_reduce.size_hidden, 3)

        # Same: hidden = min(8, 8) = 8
        block_same = ResnetBlockFC(8, 8)
        self.assertEqual(block_same.size_hidden, 8)


class TestResnetBlockFCDeviceAndDtype(unittest.TestCase):
    """Tests for device and dtype handling."""

    def test_cuda_matches_cpu(self) -> None:
        """Test that CUDA and CPU produce consistent results."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        block_cpu = ResnetBlockFC(6, 16)
        block_gpu = ResnetBlockFC(6, 16).cuda()
        block_gpu.load_state_dict(block_cpu.state_dict())

        x_cpu = torch.randn(128, 6)
        x_gpu = x_cpu.cuda()

        y_cpu = block_cpu(x_cpu)
        y_gpu = block_gpu(x_gpu).cpu()

        self.assertTrue(torch.allclose(y_cpu, y_gpu, atol=1e-6, rtol=1e-6))

    @parameterized.expand(all_devices)
    def test_dtype_preserved(self, device: str) -> None:
        """Test that module dtype produces output of the same dtype."""
        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            block = ResnetBlockFC(6, 16, dtype=dtype, device=device)
            x = torch.randn(64, 6, dtype=dtype, device=device)
            y = block(x)
            self.assertEqual(y.dtype, dtype)


class TestResnetBlockFCSerialization(unittest.TestCase):
    """Tests for serialization and representation."""

    def test_repr_contains_sizes(self) -> None:
        """Test that repr contains key hyperparameters."""
        block = ResnetBlockFC(3, 16)
        s = repr(block)
        self.assertIn("size_in=3", s)
        self.assertIn("size_out=16", s)
        self.assertIn("size_hidden=3", s)

    @parameterized.expand(all_devices)
    def test_state_dict_round_trip(self, device: str) -> None:
        """Test that saving/loading state_dict preserves behavior."""
        block = ResnetBlockFC(6, 10, device=device)

        # Train to move away from initial state
        optimizer = torch.optim.SGD(block.parameters(), lr=0.1)
        for _ in range(5):
            x_train = torch.randn(32, 6, device=device)
            loss = block(x_train).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x = torch.randn(32, 6, device=device)
        y_before = block(x)

        # Save and reload
        state = block.state_dict()
        block2 = ResnetBlockFC(6, 10, device=device)
        block2.load_state_dict(state)
        y_after = block2(x)

        self.assertTrue(torch.allclose(y_before, y_after, atol=1e-6, rtol=1e-6))


class TestResnetBlockFCResetParameters(unittest.TestCase):
    """Tests for reset_parameters functionality."""

    @parameterized.expand(all_devices)
    def test_reset_parameters_restores_identity(self, device: str) -> None:
        """Test that reset_parameters restores initial identity behavior."""
        block = ResnetBlockFC(6, device=device)

        # Verify initial identity behavior
        x = torch.randn(64, 6, device=device)
        self.assertTrue(torch.allclose(block(x), x))

        # Train to change weights
        optimizer = torch.optim.SGD(block.parameters(), lr=0.5)
        for _ in range(10):
            x_train = torch.randn(32, 6, device=device)
            loss = (block(x_train) - x_train * 2).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verify weights changed
        self.assertFalse(torch.allclose(block(x), x, atol=1e-5, rtol=1e-5))

        # Reset and verify identity restored
        block.reset_parameters()
        self.assertTrue(torch.allclose(block(x), x))


if __name__ == "__main__":
    unittest.main()
