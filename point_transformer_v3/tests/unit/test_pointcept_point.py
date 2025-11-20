# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Pointcept Point class.

These tests validate how Point objects are created and used in Point Transformer V3,
focusing on the data structures and transformations that occur.
"""

import unittest
from typing import TypedDict

import pytest
import torch
from pointcept.models.utils.structure import Point


class SyntheticPointCloudData(TypedDict):
    coord: torch.Tensor
    feat: torch.Tensor
    offset: torch.Tensor
    grid_size: float


def create_synthetic_pointcloud_data(
    num_points: int = 100,
    num_features: int = 3,
    grid_size: float = 0.1,
    device: str = "cpu",
) -> SyntheticPointCloudData:
    """
    Create synthetic point cloud data similar to ScanNet format.

    Args:
        num_points: Number of points in the point cloud
        num_features: Number of feature channels (typically 3 for RGB)
        grid_size: Voxel grid size
        device: Device for tensors

    Returns:
        Dictionary with keys: coord, feat, offset, grid_size
    """
    # Generate random 3D coordinates in a reasonable range (e.g., 0-10 meters)
    coord = torch.rand(num_points, 3, device=device) * 10.0

    # Generate random features (e.g., RGB colors)
    feat = torch.rand(num_points, num_features, device=device)

    # For single batch, offset is just [num_points]
    offset = torch.tensor([num_points], device=device, dtype=torch.int64)

    return {
        "coord": coord,
        "feat": feat,
        "offset": offset,
        "grid_size": grid_size,
    }


class TestPointCreation(unittest.TestCase):
    """Test basic Point object creation from data dictionary."""

    def test_create_point_from_minimal_data(self):
        """Test creating Point with minimal required fields."""
        data_dict = create_synthetic_pointcloud_data(num_points=50)

        point = Point(data_dict)

        # Verify Point is created
        self.assertIsInstance(point, Point)

        # Verify required fields exist
        self.assertIn("coord", point)
        self.assertIn("feat", point)
        self.assertIn("offset", point)
        self.assertIn("grid_size", point)

        # Verify shapes
        self.assertEqual(point.coord.shape, (50, 3))
        self.assertEqual(point.feat.shape, (50, 3))
        self.assertEqual(point.offset.shape, (1,))
        self.assertEqual(point.offset[0], 50)

    def test_batch_offset_auto_generation(self):
        """Test that batch is auto-generated from offset."""
        data_dict = create_synthetic_pointcloud_data(num_points=50)
        # Don't provide batch, let it be auto-generated
        point = Point(data_dict)

        # Verify batch was created
        self.assertIn("batch", point)
        # For single batch, all batch indices should be 0
        self.assertTrue(torch.all(point.batch == 0))
        self.assertEqual(point.batch.shape, (50,))

    def test_offset_auto_generation_from_batch(self):
        """Test that offset is auto-generated from batch."""
        # Create data with batch but no offset
        coord = torch.rand(50, 3)
        feat = torch.rand(50, 3)
        batch = torch.zeros(50, dtype=torch.long)  # All points in batch 0

        point = Point({"coord": coord, "feat": feat, "batch": batch, "grid_size": 0.1})

        # Verify offset was created
        self.assertIn("offset", point)
        self.assertEqual(point.offset.shape, (1,))
        self.assertEqual(point.offset[0], 50)

    def test_multi_batch_point_creation(self):
        """Test creating Point with multiple batches."""
        # Create data with 2 batches: 30 points in batch 0, 20 points in batch 1
        coord = torch.rand(50, 3)
        feat = torch.rand(50, 3)
        batch = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        offset = torch.tensor([30, 50], dtype=torch.int64)

        point = Point({"coord": coord, "feat": feat, "batch": batch, "offset": offset, "grid_size": 0.1})

        # Verify batch and offset are consistent
        self.assertEqual(point.batch.shape, (50,))
        self.assertTrue(torch.all(point.batch[:30] == 0))
        self.assertTrue(torch.all(point.batch[30:] == 1))
        self.assertTrue(torch.all(point.offset == torch.tensor([30, 50])))


class TestPointBatchesAndGridCoords(unittest.TestCase):
    """Additional tests to validate Point batches and grid coordinates."""

    def test_multi_batch_batch_auto_generated_from_offset(self):
        """Given multi-batch offsets, verify that batch indices are correctly generated."""
        # 3 batches with sizes 2, 3, 4 -> cumulative offsets [2, 5, 9]
        num_points = 9
        coord = torch.rand(num_points, 3)
        feat = torch.rand(num_points, 3)
        offset = torch.tensor([2, 5, 9], dtype=torch.int64)

        point = Point({"coord": coord, "feat": feat, "offset": offset, "grid_size": 0.1})

        # Expected batch vector: [0, 0, 1, 1, 1, 2, 2, 2, 2]
        expected_batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
        self.assertIn("batch", point)
        self.assertTrue(torch.equal(point.batch, expected_batch))
        self.assertTrue(torch.equal(point.offset, offset))

    def test_multi_batch_offset_auto_generated_from_batch(self):
        """Given multi-batch batch indices, verify that offsets are correctly generated."""
        # 3 batches with sizes 2, 3, 4 -> cumulative offsets [2, 5, 9]
        coord = torch.rand(9, 3)
        feat = torch.rand(9, 3)
        batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)

        point = Point({"coord": coord, "feat": feat, "batch": batch, "grid_size": 0.1})

        expected_offset = torch.tensor([2, 5, 9], dtype=torch.int64)
        self.assertIn("offset", point)
        self.assertTrue(torch.equal(point.batch, batch))
        self.assertTrue(torch.equal(point.offset, expected_offset))

    def test_grid_coord_generation_single_batch(self):
        """Verify that grid_coord is generated as expected for a single batch."""
        # Construct a small, deterministic point cloud
        coord = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        )
        feat = torch.rand(4, 3)
        offset = torch.tensor([4], dtype=torch.int64)
        grid_size = 1.0

        point = Point({"coord": coord, "feat": feat, "offset": offset, "grid_size": grid_size})

        # Trigger grid_coord creation via serialization
        point.serialization(order="z")

        # Expected grid coordinates:
        # coord_min = [0.0, 0.0, 0.0]
        # (coord - coord_min) / grid_size, truncated to int
        expected_grid_coord = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 2, 0],
                [0, 0, 3],
            ],
            dtype=torch.int32,
        )

        self.assertIn("grid_coord", point)
        self.assertTrue(torch.equal(point.grid_coord, expected_grid_coord))

    def test_grid_coord_generation_multi_batch_uses_global_min(self):
        """Verify that grid_coord computation uses the global minimum across batches."""
        # Two batches of 2 points each; second batch is shifted in space
        coord = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # batch 0
                [1.0, 0.0, 0.0],  # batch 0
                [9.0, 0.0, 0.0],  # batch 1
                [11.0, 2.0, 3.0],  # batch 1
            ],
            dtype=torch.float32,
        )
        feat = torch.rand(4, 3)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        grid_size = 1.0

        point = Point({"coord": coord, "feat": feat, "batch": batch, "grid_size": grid_size})

        # Trigger grid_coord creation via serialization
        point.serialization(order="z")

        # Global coord_min = [0.0, 0.0, 0.0]
        # (coord - coord_min) / grid_size, truncated to int
        expected_grid_coord = torch.tensor(
            [
                [0, 0, 0],  # (0.0, 0.0, 0.0)
                [1, 0, 0],  # (1.0, 0.0, 0.0)
                [9, 0, 0],  # (9.0, 0.0, 0.0)
                [11, 2, 3],  # (11.0, 2.0, 3.0)
            ],
            dtype=torch.int32,
        )

        self.assertIn("grid_coord", point)
        self.assertTrue(torch.equal(point.grid_coord, expected_grid_coord))

    def test_grid_coord_non_unit_grid_size_single_batch(self):
        """grid_coord generation should respect a non-unit, uniform grid size (single batch)."""
        coord = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.15, 0.0, 0.0],
                [0.0, 0.30, 0.0],
                [0.0, 0.0, 0.45],
            ],
            dtype=torch.float32,
        )
        feat = torch.rand(4, 3)
        offset = torch.tensor([4], dtype=torch.int64)
        grid_size = 0.15

        point = Point({"coord": coord, "feat": feat, "offset": offset, "grid_size": grid_size})
        point.serialization(order="z")

        # Expected: same formula as Point.serialization uses
        coord_min = coord.min(0)[0]
        expected_grid_coord = torch.div(coord - coord_min, grid_size, rounding_mode="trunc").int()

        self.assertIn("grid_coord", point)
        self.assertTrue(torch.equal(point.grid_coord, expected_grid_coord))

    def test_grid_coord_non_unit_grid_size_multi_batch_global_min(self):
        """With non-unit grid size, grid_coord still uses the global min across batches."""
        coord = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # batch 0
                [0.15, 0.0, 0.0],  # batch 0
                [0.75, 0.0, 0.0],  # batch 1
                [1.05, 0.30, 0.45],  # batch 1
            ],
            dtype=torch.float32,
        )
        feat = torch.rand(4, 3)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        grid_size = 0.15

        point = Point({"coord": coord, "feat": feat, "batch": batch, "grid_size": grid_size})
        point.serialization(order="z")

        coord_min = coord.min(0)[0]
        expected_grid_coord = torch.div(coord - coord_min, grid_size, rounding_mode="trunc").int()

        self.assertIn("grid_coord", point)
        self.assertTrue(torch.equal(point.grid_coord, expected_grid_coord))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
