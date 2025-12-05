# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for SparseFeatureHierarchy and related functionality.

Tests cover the core sparse feature hierarchy class which is a foundational
component for neural surface reconstruction. We test:
- Construction via different class methods (iterative coarsening, point splatting, refinement)
- Structural properties (depth, voxel counts, transform consistency)
- Batch handling across different batch sizes
- Device compatibility (CPU and CUDA)
"""

import unittest

import torch
from fvdb import GridBatch, JaggedTensor
from nksr.nksr_fvdb.coord_xform import world_T_voxcen_from_voxel_size
from nksr.nksr_fvdb.sparse_feature_hierarchy import (
    SparseFeatureHierarchy,
    SparseFeatureLevel,
    VoxelStatus,
    evaluate_voxel_status,
)
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

# Standard test parameters for street scene hierarchy
DEFAULT_VOXEL_SIZE = 0.05  # 5cm - appropriate for street-scale scenes (30-50m)
DEFAULT_DEPTH = 4  # Gives levels at 5cm, 10cm, 20cm, 40cm
DEFAULT_COARSENING_FACTOR = 2


class TestSparseFeatureHierarchyConstruction(unittest.TestCase):
    """Tests for SparseFeatureHierarchy construction methods."""

    @parameterized.expand(all_device_batch_combos)
    def test_from_iterative_coarsening_basic(self, device: str, batch_size: int) -> None:
        """Test basic construction via iterative coarsening."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Generate street scene data
        world_points = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        # Create hierarchy
        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        # Basic structural checks
        self.assertEqual(hierarchy.depth, DEFAULT_DEPTH)
        self.assertEqual(len(hierarchy.levels), DEFAULT_DEPTH)
        self.assertEqual(len(hierarchy.grids), DEFAULT_DEPTH)

    @parameterized.expand(all_device_batch_combos)
    def test_from_point_splatting_basic(self, device: str, batch_size: int) -> None:
        """Test basic construction via point splatting."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_point_splatting(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        self.assertEqual(hierarchy.depth, DEFAULT_DEPTH)
        self.assertEqual(len(hierarchy.levels), DEFAULT_DEPTH)
        self.assertEqual(len(hierarchy.grids), DEFAULT_DEPTH)

    @parameterized.expand(all_device_batch_combos)
    def test_from_refinement_basic(self, device: str, batch_size: int) -> None:
        """Test construction via from_refinement by extending a base hierarchy."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        # Create a base hierarchy with depth 3
        base = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=3,
        )

        # Create a new finest level (half the voxel size of the base finest)
        finer_voxel_size = DEFAULT_VOXEL_SIZE / 2
        finer_world_T_voxcen = world_T_voxcen_from_voxel_size(finer_voxel_size)
        voxcen_T_world = finer_world_T_voxcen.inverse()
        voxcen_points = voxcen_T_world @ world_points
        finer_grid = GridBatch.from_points(voxcen_points)
        finest_level = SparseFeatureLevel(finer_grid, finer_world_T_voxcen)

        # Extend the hierarchy
        extended = SparseFeatureHierarchy.from_refinement(finest_level, base)

        # Should have one more level
        self.assertEqual(extended.depth, base.depth + 1)
        self.assertEqual(extended.levels[0], finest_level)
        # Base levels should be preserved
        for i, level in enumerate(base.levels):
            self.assertEqual(extended.levels[i + 1].grid.total_voxels, level.grid.total_voxels)


class TestSparseFeatureHierarchyStructure(unittest.TestCase):
    """Tests for hierarchy structural properties."""

    @parameterized.expand(all_device_batch_combos)
    def test_voxel_counts_decrease_with_depth(self, device: str, batch_size: int) -> None:
        """Test that voxel counts decrease (or stay same) at coarser levels."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=10000,  # More points for clearer hierarchy
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        # Coarser levels should have fewer or equal voxels
        for d in range(1, hierarchy.depth):
            fine_voxels = hierarchy.levels[d - 1].grid.total_voxels
            coarse_voxels = hierarchy.levels[d].grid.total_voxels
            self.assertLessEqual(
                coarse_voxels,
                fine_voxels,
                f"Level {d} ({coarse_voxels} voxels) should have <= voxels than level {d-1} ({fine_voxels} voxels)",
            )

    @parameterized.expand(all_device_batch_combos)
    def test_pseudo_voxel_sizes_double_with_depth(self, device: str, batch_size: int) -> None:
        """Test that pseudo voxel sizes double at each coarser level (for factor=2)."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
            coarsening_factor=2,
        )

        # Check finest level voxel size
        self.assertAlmostEqual(
            hierarchy.pseudo_voxel_size,
            DEFAULT_VOXEL_SIZE,
            places=6,
            msg="Finest level voxel size should match the input transform",
        )

        # Check that voxel sizes double at each level
        for d in range(hierarchy.depth):
            expected_size = DEFAULT_VOXEL_SIZE * (2**d)
            actual_size = hierarchy.levels[d].pseudo_voxel_size
            self.assertAlmostEqual(
                actual_size,
                expected_size,
                places=6,
                msg=f"Level {d} voxel size should be {expected_size}, got {actual_size}",
            )

    @parameterized.expand(all_device_batch_combos)
    def test_grid_counts_match_batch_size(self, device: str, batch_size: int) -> None:
        """Test that all grids have the correct batch size (grid_count)."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        for d, level in enumerate(hierarchy.levels):
            self.assertEqual(
                level.grid.grid_count,
                batch_size,
                f"Level {d} grid_count should be {batch_size}",
            )

    @parameterized.expand(all_devices)
    def test_device_property(self, device: str) -> None:
        """Test that the device property returns the correct device."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        self.assertEqual(hierarchy.device.type, device)
        for level in hierarchy.levels:
            self.assertEqual(level.device.type, device)


class TestSparseFeatureHierarchyBounds(unittest.TestCase):
    """Tests for hierarchy bounds computation."""

    @parameterized.expand(all_device_batch_combos)
    def test_bounds_world_contains_input_points(self, device: str, batch_size: int) -> None:
        """Test that world bounds contain the input point cloud."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        bounds_world = hierarchy.bounds_world

        # Check that bounds contain the input points for each batch
        # Access JaggedTensor data via jdata and joffsets
        point_offsets = world_points.joffsets
        bound_offsets = bounds_world.joffsets

        for b in range(batch_size):
            # Get points for this batch element
            pt_start = point_offsets[b].item()
            pt_end = point_offsets[b + 1].item()
            batch_points = world_points.jdata[pt_start:pt_end]  # [N_b, 3]

            # Get bounds for this batch element (one bound per batch)
            bd_start = bound_offsets[b].item()
            bd_end = bound_offsets[b + 1].item()
            batch_bounds = bounds_world.jdata[bd_start:bd_end]  # [1, 2, 3]
            batch_bounds = batch_bounds.squeeze(0)  # [2, 3]

            # bounds is [2, 3] where [0, :] is min and [1, :] is max
            bound_min = batch_bounds[0]
            bound_max = batch_bounds[1]
            point_min = batch_points.min(dim=0).values
            point_max = batch_points.max(dim=0).values

            # Bounds should contain all points (with small tolerance for voxel quantization)
            voxel_tolerance = DEFAULT_VOXEL_SIZE  # One voxel tolerance
            for dim in range(3):
                self.assertLessEqual(
                    bound_min[dim].item() - voxel_tolerance,
                    point_min[dim].item(),
                    f"Batch {b}, dim {dim}: bound min should be <= point min",
                )
                self.assertGreaterEqual(
                    bound_max[dim].item() + voxel_tolerance,
                    point_max[dim].item(),
                    f"Batch {b}, dim {dim}: bound max should be >= point max",
                )

    @parameterized.expand(all_device_batch_combos)
    def test_bounds_voxcen_are_integers_at_finest_level(self, device: str, batch_size: int) -> None:
        """Test that voxcen bounds at finest level are integer-ish (from grid ijk)."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        bounds_voxcen = hierarchy.bounds_voxcen

        # Voxcen bounds should be close to integers
        bound_offsets = bounds_voxcen.joffsets
        for b in range(batch_size):
            bd_start = bound_offsets[b].item()
            bd_end = bound_offsets[b + 1].item()
            batch_bounds = bounds_voxcen.jdata[bd_start:bd_end].squeeze(0)  # [2, 3]
            for corner_idx in range(2):
                for dim in range(3):
                    val = batch_bounds[corner_idx, dim].item()
                    self.assertAlmostEqual(
                        val,
                        round(val),
                        places=5,
                        msg=f"Batch {b}, corner {corner_idx}, dim {dim}: voxcen bound should be integer",
                    )


class TestSparseFeatureHierarchyComparisons(unittest.TestCase):
    """Tests comparing different construction methods."""

    @parameterized.expand(all_device_batch_combos)
    def test_iterative_vs_splatting_coverage(self, device: str, batch_size: int) -> None:
        """Test that point splatting creates at least as many voxels as iterative coarsening.

        Point splatting uses from_nearest_voxels_to_points which realizes the voxel
        neighborhood, so it should create >= voxels compared to from_points.
        """
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy_iter = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        hierarchy_splat = SparseFeatureHierarchy.from_point_splatting(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        # Point splatting should create >= voxels at each level
        for d in range(DEFAULT_DEPTH):
            iter_voxels = hierarchy_iter.levels[d].grid.total_voxels
            splat_voxels = hierarchy_splat.levels[d].grid.total_voxels
            self.assertGreaterEqual(
                splat_voxels,
                iter_voxels,
                f"Level {d}: splatting ({splat_voxels}) should have >= voxels than iterative ({iter_voxels})",
            )


class TestSparseFeatureHierarchyEdgeCases(unittest.TestCase):
    """Tests for edge cases and validation."""

    @parameterized.expand(all_devices)
    def test_depth_one_hierarchy(self, device: str) -> None:
        """Test hierarchy with depth=1 (single level)."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=1,
        )

        self.assertEqual(hierarchy.depth, 1)
        self.assertEqual(len(hierarchy.levels), 1)
        self.assertAlmostEqual(hierarchy.pseudo_voxel_size, DEFAULT_VOXEL_SIZE, places=6)

    @parameterized.expand(all_devices)
    def test_different_coarsening_factors(self, device: str) -> None:
        """Test hierarchy with different coarsening factors."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        for factor in [2, 3, 4]:
            hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
                world_points=world_points,
                world_T_voxcen=world_T_voxcen,
                depth=3,
                coarsening_factor=factor,
            )

            # Check voxel sizes grow by the factor
            for d in range(hierarchy.depth):
                expected_size = DEFAULT_VOXEL_SIZE * (factor**d)
                actual_size = hierarchy.levels[d].pseudo_voxel_size
                self.assertAlmostEqual(
                    actual_size,
                    expected_size,
                    places=5,
                    msg=f"Factor {factor}, level {d}: expected {expected_size}, got {actual_size}",
                )

    def test_empty_hierarchy_raises(self) -> None:
        """Test that creating an empty hierarchy raises ValueError."""
        with self.assertRaises(ValueError):
            SparseFeatureHierarchy(levels=[])

    @parameterized.expand(all_devices)
    def test_hierarchy_str_representation(self, device: str) -> None:
        """Test that __str__ produces a reasonable representation."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=2,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        str_repr = str(hierarchy)
        # Should contain depth info
        self.assertIn(str(DEFAULT_DEPTH), str_repr)
        # Should contain voxel size info
        self.assertIn("pseudo_voxel_size", str_repr)


class TestSparseFeatureLevelTestGrid(unittest.TestCase):
    """Tests for SparseFeatureLevel.get_test_grid functionality."""

    @parameterized.expand(all_device_batch_combos)
    def test_get_test_grid_basic(self, device: str, batch_size: int) -> None:
        """Test basic test grid generation."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        resolution = 2
        query_world, primal_coords = hierarchy.levels[0].get_test_grid(resolution=resolution)

        # Check shapes
        num_voxels = hierarchy.levels[0].grid.total_voxels
        expected_queries = num_voxels * (resolution**3)
        self.assertEqual(query_world.jdata.shape[0], expected_queries)
        self.assertEqual(query_world.jdata.shape[1], 3)
        self.assertEqual(primal_coords.jdata.shape[0], num_voxels)
        self.assertEqual(primal_coords.jdata.shape[1], 3)

    @parameterized.expand(all_devices)
    def test_get_test_grid_resolution_variants(self, device: str) -> None:
        """Test test grid with different resolutions."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=2,
        )

        level = hierarchy.levels[0]
        num_voxels = level.grid.total_voxels

        for resolution in [1, 2, 3, 4]:
            query_world, _ = level.get_test_grid(resolution=resolution)
            expected_queries = num_voxels * (resolution**3)
            self.assertEqual(
                query_world.jdata.shape[0],
                expected_queries,
                f"Resolution {resolution}: expected {expected_queries} queries, got {query_world.jdata.shape[0]}",
            )


class TestVoxelStatusEvaluation(unittest.TestCase):
    """Tests for voxel status evaluation functionality."""

    @parameterized.expand(all_device_batch_combos)
    def test_evaluate_voxel_status_basic(self, device: str, batch_size: int) -> None:
        """Test basic voxel status evaluation."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        # Evaluate at coarsest level
        target_grid = hierarchy.levels[-1].grid
        jagged_status = hierarchy.evaluate_voxel_status(target_grid, coarse_depth=DEFAULT_DEPTH - 1)
        status = jagged_status.jdata

        # Should return tensor of correct shape
        self.assertEqual(status.shape[0], target_grid.total_voxels)
        self.assertEqual(status.dtype, torch.uint8)

        # All values should be valid VoxelStatus values
        valid_values = {
            VoxelStatus.VS_NON_EXIST.value,
            VoxelStatus.VS_EXIST_STOP.value,
            VoxelStatus.VS_EXIST_CONTINUE.value,
        }
        unique_values = set(status.unique().tolist())
        self.assertTrue(unique_values.issubset(valid_values))

    @parameterized.expand(all_device_batch_combos)
    def test_evaluate_voxel_status_finest_level(self, device: str, batch_size: int) -> None:
        """Test voxel status at finest level (no finer level to continue to)."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        # Evaluate at finest level (depth=0)
        target_grid = hierarchy.levels[0].grid
        jagged_status = hierarchy.evaluate_voxel_status(target_grid, coarse_depth=0)
        status = jagged_status.jdata

        # At finest level, all existing voxels should be EXIST_STOP (no finer level)
        # Non-existing voxels would be NON_EXIST
        unique_values = set(status.unique().tolist())
        # Since target_grid == hierarchy.levels[0].grid, all should be EXIST_STOP
        self.assertIn(VoxelStatus.VS_EXIST_STOP.value, unique_values)

    @parameterized.expand(all_devices)
    def test_evaluate_voxel_status_invalid_depth(self, device: str) -> None:
        """Test that invalid coarse_depth raises ValueError."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        target_grid = hierarchy.levels[0].grid

        # Invalid depths should raise
        with self.assertRaises(ValueError):
            hierarchy.evaluate_voxel_status(target_grid, coarse_depth=-1)

        with self.assertRaises(ValueError):
            hierarchy.evaluate_voxel_status(target_grid, coarse_depth=DEFAULT_DEPTH)


class TestModuleLevelEvaluateVoxelStatus(unittest.TestCase):
    """Tests for the module-level evaluate_voxel_status function."""

    @parameterized.expand(all_devices)
    def test_evaluate_voxel_status_all_exist(self, device: str) -> None:
        """Test status when target grid exactly matches coarse grid."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_points = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points,
            world_T_voxcen=world_T_voxcen,
            depth=2,
        )

        # Target grid equals coarse grid, fine grid is None
        coarse_grid = hierarchy.levels[0].grid
        jagged_status = evaluate_voxel_status(coarse_grid, coarse_grid, fine_grid=None)
        status = jagged_status.jdata

        # All voxels should be EXIST_STOP when target == coarse and no fine grid
        self.assertTrue(torch.all(status == VoxelStatus.VS_EXIST_STOP.value))

    @parameterized.expand(all_devices)
    def test_evaluate_voxel_status_device_mismatch(self, device: str) -> None:
        """Test that device mismatch raises ValueError."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        if device == "cpu":
            # Need CUDA to test device mismatch with CPU
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")
            other_device = "cuda"
        else:
            other_device = "cpu"

        # Create grids on different devices
        world_points_1 = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        world_points_2 = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=5000,
            device=other_device,
        )
        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        hierarchy_1 = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points_1,
            world_T_voxcen=world_T_voxcen,
            depth=2,
        )
        hierarchy_2 = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points_2,
            world_T_voxcen=world_T_voxcen,
            depth=2,
        )

        # Should raise due to device mismatch
        with self.assertRaises(ValueError):
            evaluate_voxel_status(
                hierarchy_1.levels[0].grid,
                hierarchy_2.levels[0].grid,
                fine_grid=None,
            )


class TestSparseFeatureHierarchyDeterminism(unittest.TestCase):
    """Tests for reproducibility and determinism."""

    @parameterized.expand(all_devices)
    def test_same_seed_produces_same_hierarchy(self, device: str) -> None:
        """Test that the same seed produces identical hierarchies."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        # Create two hierarchies with same seed
        world_points_1 = generate_street_scene_batch(
            batch_size=2,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        hierarchy_1 = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points_1,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        world_points_2 = generate_street_scene_batch(
            batch_size=2,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        hierarchy_2 = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points_2,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        # Should have identical voxel counts
        for d in range(DEFAULT_DEPTH):
            self.assertEqual(
                hierarchy_1.levels[d].grid.total_voxels,
                hierarchy_2.levels[d].grid.total_voxels,
                f"Level {d} voxel counts should match",
            )

    @parameterized.expand(all_devices)
    def test_different_seeds_produce_different_hierarchies(self, device: str) -> None:
        """Test that different seeds produce different hierarchies."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_T_voxcen = world_T_voxcen_from_voxel_size(DEFAULT_VOXEL_SIZE)

        world_points_1 = generate_street_scene_batch(
            batch_size=1,
            base_seed=42,
            num_points=5000,
            device=device,
        )
        hierarchy_1 = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points_1,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        world_points_2 = generate_street_scene_batch(
            batch_size=1,
            base_seed=12345,  # Different seed
            num_points=5000,
            device=device,
        )
        hierarchy_2 = SparseFeatureHierarchy.from_iterative_coarsening(
            world_points=world_points_2,
            world_T_voxcen=world_T_voxcen,
            depth=DEFAULT_DEPTH,
        )

        # Should have different voxel counts (very likely with different random scenes)
        # Note: There's a tiny chance they could be equal, but extremely unlikely
        at_least_one_different = False
        for d in range(DEFAULT_DEPTH):
            if hierarchy_1.levels[d].grid.total_voxels != hierarchy_2.levels[d].grid.total_voxels:
                at_least_one_different = True
                break
        self.assertTrue(at_least_one_different, "Different seeds should produce different hierarchies")


if __name__ == "__main__":
    unittest.main()
