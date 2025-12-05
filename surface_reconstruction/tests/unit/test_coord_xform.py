# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for coordinate frame transformations in coord_xform.py.

Tests the basic transformation classes: IdentityXform and UniformScaleThenTranslate.
"""

import unittest

import torch
from nksr.nksr_fvdb.coord_xform import IdentityXform, UniformScaleThenTranslate


class TestIdentityXform(unittest.TestCase):
    """Test cases for IdentityXform."""

    def test_identity_returns_same_coords(self):
        """Test that IdentityXform returns coordinates unchanged."""
        xform = IdentityXform()
        coords = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = xform.apply_tensor(coords)

        self.assertTrue(torch.equal(result, coords))

    def test_identity_is_invertible(self):
        """Test that IdentityXform is invertible."""
        xform = IdentityXform()
        self.assertTrue(xform.invertible)

    def test_identity_inverse_is_identity(self):
        """Test that inverse of IdentityXform is itself."""
        xform = IdentityXform()
        inverse = xform.inverse()
        self.assertIsInstance(inverse, IdentityXform)


class TestUniformScaleThenTranslate(unittest.TestCase):
    """Test cases for UniformScaleThenTranslate."""

    def test_scale_only(self):
        """Test scaling with scale only."""
        xform = UniformScaleThenTranslate(scale=2.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]])

        result = xform.apply_tensor(coords)
        expected = torch.tensor([[2.0, 4.0, 6.0]])

        self.assertTrue(torch.allclose(result, expected))

    def test_translation_only(self):
        """Test translation with translation only."""
        xform = UniformScaleThenTranslate(translation=1.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]])

        result = xform.apply_tensor(coords)
        expected = torch.tensor([[2.0, 3.0, 4.0]])

        self.assertTrue(torch.allclose(result, expected))

    def test_scale_and_translation(self):
        """Test combined scaling and translation."""
        xform = UniformScaleThenTranslate(scale=2.0, translation=1.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]])

        result = xform.apply_tensor(coords)
        expected = torch.tensor([[3.0, 5.0, 7.0]])  # coords * 2 + 1

        self.assertTrue(torch.allclose(result, expected))

    def test_inverse(self):
        """Test that inverse correctly reverses the transformation."""
        xform = UniformScaleThenTranslate(scale=2.0, translation=1.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]])

        transformed = xform.apply_tensor(coords)
        recovered = xform.inverse().apply_tensor(transformed)

        self.assertTrue(torch.allclose(recovered, coords))

    def test_invertible_when_scale_nonzero(self):
        """Test that UniformScaleThenTranslate is invertible when scale != 0."""
        xform = UniformScaleThenTranslate(scale=2.0, translation=1.0)
        self.assertTrue(xform.invertible)

    def test_construction_fails_when_scale_zero(self):
        """Test that UniformScaleThenTranslate cannot be constructed with scale == 0."""
        with self.assertRaises(ValueError):
            UniformScaleThenTranslate(scale=0.0, translation=1.0)

    def test_compose_fuses_two_uniform_scale_translates(self):
        """Test that composing two UniformScaleThenTranslate fuses them."""
        # First: y = x * 2 + 1
        # Second: z = y * 3 + 5
        # Composed: z = x * (2*3) + (1*3 + 5) = x * 6 + 8
        first = UniformScaleThenTranslate(scale=2.0, translation=1.0)
        second = UniformScaleThenTranslate(scale=3.0, translation=5.0)

        composed = first.compose(second)

        self.assertIsInstance(composed, UniformScaleThenTranslate)
        assert isinstance(composed, UniformScaleThenTranslate)  # for type narrowing
        self.assertEqual(composed.scale, 6.0)
        self.assertEqual(composed.translation, 8.0)

        # Verify the result matches applying them sequentially
        coords = torch.tensor([[1.0, 2.0, 3.0]])
        sequential = second.apply_tensor(first.apply_tensor(coords))
        fused = composed.apply_tensor(coords)
        self.assertTrue(torch.allclose(sequential, fused))


class TestVoxelCenterAlignedCoarsening(unittest.TestCase):
    """Tests for center-aligned coarsening transforms.

    When tracking voxel centers, ijk=(0,0,0) maps to the center of voxel 0 in world space,
    which is at (voxel_size/2, voxel_size/2, voxel_size/2).

    For center-aligned coarsening, we need the coarse ijk=(0,0,0) to map to the center
    of the coarse voxel, not the corner. This requires a half-voxel offset in the
    fine-to-coarse relationship.
    """

    def test_voxel_center_transform_origin_maps_to_half_voxel(self):
        """Verify that a voxel-center transform maps ijk=(0,0,0) to (voxel_size/2, ...)."""
        voxel_size = 0.1
        # Voxel-center transform: world = ijk * voxel_size + voxel_size/2
        xform = UniformScaleThenTranslate(scale=voxel_size, translation=voxel_size / 2)

        origin_ijk = torch.tensor([[0.0, 0.0, 0.0]])
        world_pos = xform.apply_tensor(origin_ijk)

        expected = torch.tensor([[voxel_size / 2, voxel_size / 2, voxel_size / 2]])
        self.assertTrue(torch.allclose(world_pos, expected))

    def test_center_aligned_coarsening_maps_to_coarse_voxel_center(self):
        """Test that center-aligned coarsening maps coarse ijk=0 to the coarse voxel center.

        For a fine grid with voxel_size=0.1 and voxel-center tracking:
        - Fine voxel 0 center: 0.05
        - Fine voxel 1 center: 0.15
        - Coarse voxel 0 covers fine voxels 0 and 1
        - Coarse voxel 0 center: (0.05 + 0.15) / 2 = 0.10

        The center-aligned coarsening transform is:
            fine_T_coarse = scale(factor) + translate((factor-1)/2)

        For factor=2: fine_ijk = coarse_ijk * 2 + 0.5
        """
        from nksr.nksr_fvdb.sparse_feature_hierarchy import (
            voxel_center_aligned_coarsening_xform,
        )

        voxel_size = 0.1
        world_T_fine = UniformScaleThenTranslate(scale=voxel_size, translation=voxel_size / 2)

        # Create center-aligned coarse transform
        fine_T_coarse = voxel_center_aligned_coarsening_xform(2)
        world_T_coarse = world_T_fine.compose(fine_T_coarse)

        origin_ijk = torch.tensor([[0.0, 0.0, 0.0]])
        coarse_world_pos = world_T_coarse.apply_tensor(origin_ijk)

        # Coarse ijk=0 should map to the center of the coarse voxel: voxel_size
        expected_center = torch.tensor([[voxel_size, voxel_size, voxel_size]])
        self.assertTrue(torch.allclose(coarse_world_pos, expected_center))

    def test_center_aligned_coarsening_doubles_voxel_size(self):
        """Test that center-aligned coarsening by 2 doubles the effective voxel size."""
        from nksr.nksr_fvdb.sparse_feature_hierarchy import (
            voxel_center_aligned_coarsening_xform,
        )

        voxel_size = 0.1
        world_T_fine = UniformScaleThenTranslate(scale=voxel_size, translation=voxel_size / 2)

        fine_T_coarse = voxel_center_aligned_coarsening_xform(2)
        world_T_coarse = world_T_fine.compose(fine_T_coarse)

        # The pseudo_scaling_factor should be 2x the original voxel size
        self.assertAlmostEqual(world_T_coarse.pseudo_scaling_factor, voxel_size * 2, places=10)

    def test_center_aligned_coarsening_multiple_voxels(self):
        """Test center-aligned coarsening for multiple voxel coordinates."""
        from nksr.nksr_fvdb.sparse_feature_hierarchy import (
            voxel_center_aligned_coarsening_xform,
        )

        voxel_size = 0.1
        world_T_fine = UniformScaleThenTranslate(scale=voxel_size, translation=voxel_size / 2)

        fine_T_coarse = voxel_center_aligned_coarsening_xform(2)
        world_T_coarse = world_T_fine.compose(fine_T_coarse)

        # Test multiple coarse ijk coordinates
        coarse_ijk = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )

        coarse_world = world_T_coarse.apply_tensor(coarse_ijk)

        # For center-aligned: world = (coarse_ijk * 2 + 0.5) * voxel_size + voxel_size/2
        #                           = coarse_ijk * 2 * voxel_size + 0.5 * voxel_size + voxel_size/2
        #                           = coarse_ijk * 2 * voxel_size + voxel_size
        expected = coarse_ijk * 2 * voxel_size + voxel_size

        self.assertTrue(torch.allclose(coarse_world, expected))

    def test_center_aligned_coarsening_factor_4(self):
        """Test center-aligned coarsening with factor=4."""
        from nksr.nksr_fvdb.sparse_feature_hierarchy import (
            voxel_center_aligned_coarsening_xform,
        )

        voxel_size = 0.1
        world_T_fine = UniformScaleThenTranslate(scale=voxel_size, translation=voxel_size / 2)

        fine_T_coarse = voxel_center_aligned_coarsening_xform(4)
        world_T_coarse = world_T_fine.compose(fine_T_coarse)

        origin_ijk = torch.tensor([[0.0, 0.0, 0.0]])
        coarse_world_pos = world_T_coarse.apply_tensor(origin_ijk)

        # For factor=4: fine_ijk = coarse_ijk * 4 + 1.5 (since (4-1)/2 = 1.5)
        # world = (0 * 4 + 1.5) * 0.1 + 0.05 = 0.15 + 0.05 = 0.20
        # which is 2 * voxel_size = center of a 4x coarser voxel
        expected = torch.tensor([[voxel_size * 2, voxel_size * 2, voxel_size * 2]])
        self.assertTrue(torch.allclose(coarse_world_pos, expected))

        # Voxel size should be 4x
        self.assertAlmostEqual(world_T_coarse.pseudo_scaling_factor, voxel_size * 4, places=10)

    def test_iterative_center_aligned_coarsening(self):
        """Test that iterative center-aligned coarsening works correctly.

        This simulates what SparseFeatureHierarchy.from_iterative_coarsening does.
        """
        from nksr.nksr_fvdb.sparse_feature_hierarchy import (
            voxel_center_aligned_coarsening_xform,
        )

        voxel_size = 0.1
        world_T_level0 = UniformScaleThenTranslate(scale=voxel_size, translation=voxel_size / 2)

        fine_T_coarse = voxel_center_aligned_coarsening_xform(2)

        # Level 1: coarsen once
        world_T_level1 = world_T_level0.compose(fine_T_coarse)

        # Level 2: coarsen again
        world_T_level2 = world_T_level1.compose(fine_T_coarse)

        origin_ijk = torch.tensor([[0.0, 0.0, 0.0]])

        # Level 0: center at voxel_size/2 = 0.05
        level0_world = world_T_level0.apply_tensor(origin_ijk)
        self.assertTrue(torch.allclose(level0_world, torch.tensor([[0.05, 0.05, 0.05]])))

        # Level 1: center at voxel_size = 0.10 (2x coarser)
        level1_world = world_T_level1.apply_tensor(origin_ijk)
        self.assertTrue(torch.allclose(level1_world, torch.tensor([[0.10, 0.10, 0.10]])))

        # Level 2: center at 2*voxel_size = 0.20 (4x coarser)
        level2_world = world_T_level2.apply_tensor(origin_ijk)
        self.assertTrue(torch.allclose(level2_world, torch.tensor([[0.20, 0.20, 0.20]])))

        # Check voxel sizes
        self.assertAlmostEqual(world_T_level0.pseudo_scaling_factor, 0.1, places=10)
        self.assertAlmostEqual(world_T_level1.pseudo_scaling_factor, 0.2, places=10)
        self.assertAlmostEqual(world_T_level2.pseudo_scaling_factor, 0.4, places=10)


if __name__ == "__main__":
    unittest.main()
