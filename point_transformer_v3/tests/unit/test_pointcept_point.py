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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
