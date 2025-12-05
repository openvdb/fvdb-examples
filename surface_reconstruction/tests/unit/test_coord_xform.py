# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for coordinate frame transformations in coord_xform.py.

Tests the basic transformation classes: IdentityXform and UniformScaleThenTranslate.
"""

import unittest

import torch
from nksr.nksr_fvdb.coord_xform import IdentityXform, UniformScaleThenTranslate
from parameterized import parameterized

all_devices = [
    "cpu",
    "cuda",
]


class TestMatmulOperator(unittest.TestCase):
    """Test cases for the @ (matmul) operator on CoordXform.

    The @ operator is overloaded to support both composition and application:
    - xform_a @ xform_b: Composition, returns a transform where xform_b is applied first, then xform_a.
    - xform @ coords: Application, transforms the coordinates.

    These tests validate all usages of the @ operator to ensure correct behavior.
    """

    @parameterized.expand(all_devices)
    def test_matmul_applies_tensor(self, device: str):
        """Test that xform @ tensor applies the transform to coordinates."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        xform = UniformScaleThenTranslate(scale=2.0, translation=1.0)
        coords = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)

        result = xform @ coords

        expected = torch.tensor([[3.0, 5.0, 7.0], [9.0, 11.0, 13.0]], device=device)  # coords * 2 + 1
        self.assertTrue(torch.allclose(result, expected))

    @parameterized.expand(all_devices)
    def test_matmul_applies_tensor_matches_apply(self, device: str):
        """Test that xform @ tensor gives same result as xform.apply_tensor()."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        xform = UniformScaleThenTranslate(scale=3.0, translation=-2.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]], device=device)

        matmul_result = xform @ coords
        apply_result = xform.apply_tensor(coords)

        self.assertTrue(torch.equal(matmul_result, apply_result))

    def test_matmul_composes_transforms(self):
        """Test that xform_a @ xform_b composes transforms correctly.

        Composition semantics: (A @ B)(x) = A(B(x)), so B is applied first, then A.
        """
        # B: y = x * 3 + 5  (applied first)
        # A: z = y * 2 + 1  (applied second)
        # Composed: z = (x * 3 + 5) * 2 + 1 = x * 6 + 11
        xform_a = UniformScaleThenTranslate(scale=2.0, translation=1.0)
        xform_b = UniformScaleThenTranslate(scale=3.0, translation=5.0)

        composed = xform_a @ xform_b

        self.assertIsInstance(composed, UniformScaleThenTranslate)
        assert isinstance(composed, UniformScaleThenTranslate)  # for type narrowing
        self.assertEqual(composed.scale, 6.0)
        self.assertEqual(composed.translation, 11.0)

    def test_matmul_compose_matches_compose_method(self):
        """Test that xform_a @ xform_b gives same result as xform_a.compose(xform_b)."""
        xform_a = UniformScaleThenTranslate(scale=2.0, translation=1.0)
        xform_b = UniformScaleThenTranslate(scale=3.0, translation=5.0)

        matmul_result = xform_a @ xform_b
        compose_result = xform_a.compose(xform_b)

        # Both should be equivalent UniformScaleThenTranslate
        self.assertIsInstance(matmul_result, UniformScaleThenTranslate)
        self.assertIsInstance(compose_result, UniformScaleThenTranslate)
        assert isinstance(matmul_result, UniformScaleThenTranslate)
        assert isinstance(compose_result, UniformScaleThenTranslate)
        self.assertEqual(matmul_result.scale, compose_result.scale)
        self.assertEqual(matmul_result.translation, compose_result.translation)

    @parameterized.expand(all_devices)
    def test_matmul_composition_order(self, device: str):
        """Test that (A @ B)(x) == A(B(x)) - B is applied first, then A."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        xform_a = UniformScaleThenTranslate(scale=2.0, translation=1.0)
        xform_b = UniformScaleThenTranslate(scale=3.0, translation=5.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]], device=device)

        # Composed transform applied to coords
        composed = xform_a @ xform_b
        composed_result = composed @ coords

        # Sequential application: A(B(x))
        sequential_result = xform_a @ (xform_b @ coords)

        self.assertTrue(torch.allclose(composed_result, sequential_result))

    @parameterized.expand(all_devices)
    def test_matmul_chain_of_three(self, device: str):
        """Test chaining three transforms: (A @ B @ C)(x) = A(B(C(x)))."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        xform_a = UniformScaleThenTranslate(scale=2.0, translation=0.0)
        xform_b = UniformScaleThenTranslate(scale=1.0, translation=3.0)
        xform_c = UniformScaleThenTranslate(scale=0.5, translation=1.0)
        coords = torch.tensor([[2.0, 4.0, 6.0]], device=device)

        # Compose all three
        composed = xform_a @ xform_b @ xform_c

        # Apply composed
        composed_result = composed @ coords

        # Sequential: A(B(C(x)))
        # C(x) = x * 0.5 + 1 = [2.0, 3.0, 4.0]
        # B(C(x)) = C(x) * 1.0 + 3.0 = [5.0, 6.0, 7.0]
        # A(B(C(x))) = B(C(x)) * 2.0 + 0.0 = [10.0, 12.0, 14.0]
        expected = torch.tensor([[10.0, 12.0, 14.0]], device=device)

        self.assertTrue(torch.allclose(composed_result, expected))

    @parameterized.expand(all_devices)
    def test_matmul_with_identity(self, device: str):
        """Test that identity @ xform and xform @ identity both equal xform."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        identity = IdentityXform()
        xform = UniformScaleThenTranslate(scale=2.0, translation=3.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]], device=device)

        # identity @ xform should behave like xform
        left_composed = identity @ xform
        left_result = left_composed @ coords

        # xform @ identity should behave like xform
        right_composed = xform @ identity
        right_result = right_composed @ coords

        # Direct application
        direct_result = xform @ coords

        self.assertTrue(torch.allclose(left_result, direct_result))
        self.assertTrue(torch.allclose(right_result, direct_result))

    @parameterized.expand(all_devices)
    def test_matmul_with_inverse(self, device: str):
        """Test that xform @ xform.inverse() is equivalent to identity."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        xform = UniformScaleThenTranslate(scale=2.0, translation=3.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]], device=device)

        # xform @ inverse should give us back the original coords
        composed = xform @ xform.inverse()
        result = composed @ coords

        self.assertTrue(torch.allclose(result, coords))

    @parameterized.expand(all_devices)
    def test_matmul_inverse_order(self, device: str):
        """Test that inverse @ xform also gives identity-like behavior."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        xform = UniformScaleThenTranslate(scale=2.0, translation=3.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]], device=device)

        # inverse @ xform: inverse is applied first, then xform
        # This should NOT give identity behavior (order matters)
        # But (inverse @ xform) @ (xform @ coords) should give coords back
        transformed = xform @ coords
        recovered = xform.inverse() @ transformed

        self.assertTrue(torch.allclose(recovered, coords))

    @parameterized.expand(all_devices)
    def test_matmul_mixed_types_compose_then_apply(self, device: str):
        """Test mixed usage: compose transforms, then apply to coords."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        world_T_camera = UniformScaleThenTranslate(scale=1.0, translation=10.0)
        camera_T_object = UniformScaleThenTranslate(scale=2.0, translation=0.0)
        object_coords = torch.tensor([[1.0, 2.0, 3.0]], device=device)

        # Compose first, then apply
        world_T_object = world_T_camera @ camera_T_object
        world_coords = world_T_object @ object_coords

        # Should equal sequential application
        expected = world_T_camera @ (camera_T_object @ object_coords)

        self.assertTrue(torch.allclose(world_coords, expected))


class TestIdentityXform(unittest.TestCase):
    """Test cases for IdentityXform."""

    @parameterized.expand(all_devices)
    def test_identity_returns_same_coords(self, device: str):
        """Test that IdentityXform returns coordinates unchanged."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        xform = IdentityXform()
        coords = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)

        result = xform @ coords

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

    @parameterized.expand(all_devices)
    def test_scale_only(self, device: str):
        """Test scaling with scale only."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        xform = UniformScaleThenTranslate(scale=2.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]], device=device)

        result = xform @ coords
        expected = torch.tensor([[2.0, 4.0, 6.0]], device=device)

        self.assertTrue(torch.allclose(result, expected))

    @parameterized.expand(all_devices)
    def test_translation_only(self, device: str):
        """Test translation with translation only."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        xform = UniformScaleThenTranslate(translation=1.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]], device=device)

        result = xform @ coords
        expected = torch.tensor([[2.0, 3.0, 4.0]], device=device)

        self.assertTrue(torch.allclose(result, expected))

    @parameterized.expand(all_devices)
    def test_scale_and_translation(self, device: str):
        """Test combined scaling and translation."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        xform = UniformScaleThenTranslate(scale=2.0, translation=1.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]], device=device)

        result = xform @ coords
        expected = torch.tensor([[3.0, 5.0, 7.0]], device=device)  # coords * 2 + 1

        self.assertTrue(torch.allclose(result, expected))

    @parameterized.expand(all_devices)
    def test_inverse(self, device: str):
        """Test that inverse correctly reverses the transformation."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        xform = UniformScaleThenTranslate(scale=2.0, translation=1.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]], device=device)

        transformed = xform @ coords
        recovered = xform.inverse() @ transformed

        self.assertTrue(torch.allclose(recovered, coords))

    def test_invertible_when_scale_nonzero(self):
        """Test that UniformScaleThenTranslate is invertible when scale != 0."""
        xform = UniformScaleThenTranslate(scale=2.0, translation=1.0)
        self.assertTrue(xform.invertible)

    def test_construction_fails_when_scale_zero(self):
        """Test that UniformScaleThenTranslate cannot be constructed with scale == 0."""
        with self.assertRaises(ValueError):
            UniformScaleThenTranslate(scale=0.0, translation=1.0)

    @parameterized.expand(all_devices)
    def test_compose_fuses_two_uniform_scale_translates(self, device: str):
        """Test that composing two UniformScaleThenTranslate fuses them.

        Composition semantics: first.compose(second) returns a transform where
        second is applied first, then first. i.e., (first.compose(second))(x) = first(second(x)).
        """
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # second: y = x * 3 + 5  (applied first)
        # first:  z = y * 2 + 1  (applied second)
        # Composed: z = (x * 3 + 5) * 2 + 1 = x * 6 + 11
        first = UniformScaleThenTranslate(scale=2.0, translation=1.0)
        second = UniformScaleThenTranslate(scale=3.0, translation=5.0)

        composed = first.compose(second)

        self.assertIsInstance(composed, UniformScaleThenTranslate)
        assert isinstance(composed, UniformScaleThenTranslate)  # for type narrowing
        self.assertEqual(composed.scale, 6.0)
        self.assertEqual(composed.translation, 11.0)

        # Verify the result matches applying them sequentially: first(second(x))
        coords = torch.tensor([[1.0, 2.0, 3.0]], device=device)
        sequential = first.apply_tensor(second.apply_tensor(coords))
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

    @parameterized.expand(all_devices)
    def test_voxel_center_transform_origin_maps_to_half_voxel(self, device: str):
        """Verify that a voxel-center transform maps ijk=(0,0,0) to (voxel_size/2, ...)."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        voxel_size = 0.1
        # Voxel-center transform: world = ijk * voxel_size + voxel_size/2
        xform = UniformScaleThenTranslate(scale=voxel_size, translation=voxel_size / 2)

        origin_ijk = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        world_pos = xform @ origin_ijk

        expected = torch.tensor([[voxel_size / 2, voxel_size / 2, voxel_size / 2]], device=device)
        self.assertTrue(torch.allclose(world_pos, expected))

    @parameterized.expand(all_devices)
    def test_center_aligned_coarsening_maps_to_coarse_voxel_center(self, device: str):
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
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        from nksr.nksr_fvdb.sparse_feature_hierarchy import (
            voxel_center_aligned_coarsening_xform,
        )

        voxel_size = 0.1
        world_T_fine = UniformScaleThenTranslate(scale=voxel_size, translation=voxel_size / 2)

        # Create center-aligned coarse transform using @ for composition
        fine_T_coarse = voxel_center_aligned_coarsening_xform(2)
        world_T_coarse = world_T_fine @ fine_T_coarse

        origin_ijk = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        coarse_world_pos = world_T_coarse @ origin_ijk

        # Coarse ijk=0 should map to the center of the coarse voxel: voxel_size
        expected_center = torch.tensor([[voxel_size, voxel_size, voxel_size]], device=device)
        self.assertTrue(torch.allclose(coarse_world_pos, expected_center))

    def test_center_aligned_coarsening_doubles_voxel_size(self):
        """Test that center-aligned coarsening by 2 doubles the effective voxel size."""
        from nksr.nksr_fvdb.sparse_feature_hierarchy import (
            voxel_center_aligned_coarsening_xform,
        )

        voxel_size = 0.1
        world_T_fine = UniformScaleThenTranslate(scale=voxel_size, translation=voxel_size / 2)

        fine_T_coarse = voxel_center_aligned_coarsening_xform(2)
        world_T_coarse = world_T_fine @ fine_T_coarse

        # The pseudo_scaling_factor should be 2x the original voxel size
        self.assertAlmostEqual(world_T_coarse.pseudo_scaling_factor, voxel_size * 2, places=5)

    @parameterized.expand(all_devices)
    def test_center_aligned_coarsening_multiple_voxels(self, device: str):
        """Test center-aligned coarsening for multiple voxel coordinates."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        from nksr.nksr_fvdb.sparse_feature_hierarchy import (
            voxel_center_aligned_coarsening_xform,
        )

        voxel_size = 0.1
        world_T_fine = UniformScaleThenTranslate(scale=voxel_size, translation=voxel_size / 2)

        fine_T_coarse = voxel_center_aligned_coarsening_xform(2)
        world_T_coarse = world_T_fine @ fine_T_coarse

        # Test multiple coarse ijk coordinates
        coarse_ijk = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            device=device,
        )

        coarse_world = world_T_coarse @ coarse_ijk

        # For center-aligned: world = (coarse_ijk * 2 + 0.5) * voxel_size + voxel_size/2
        #                           = coarse_ijk * 2 * voxel_size + 0.5 * voxel_size + voxel_size/2
        #                           = coarse_ijk * 2 * voxel_size + voxel_size
        expected = coarse_ijk * 2 * voxel_size + voxel_size

        self.assertTrue(torch.allclose(coarse_world, expected))

    @parameterized.expand(all_devices)
    def test_center_aligned_coarsening_factor_4(self, device: str):
        """Test center-aligned coarsening with factor=4."""
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        from nksr.nksr_fvdb.sparse_feature_hierarchy import (
            voxel_center_aligned_coarsening_xform,
        )

        voxel_size = 0.1
        world_T_fine = UniformScaleThenTranslate(scale=voxel_size, translation=voxel_size / 2)

        fine_T_coarse = voxel_center_aligned_coarsening_xform(4)
        world_T_coarse = world_T_fine @ fine_T_coarse

        origin_ijk = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        coarse_world_pos = world_T_coarse @ origin_ijk

        # For factor=4: fine_ijk = coarse_ijk * 4 + 1.5 (since (4-1)/2 = 1.5)
        # world = (0 * 4 + 1.5) * 0.1 + 0.05 = 0.15 + 0.05 = 0.20
        # which is 2 * voxel_size = center of a 4x coarser voxel
        expected = torch.tensor([[voxel_size * 2, voxel_size * 2, voxel_size * 2]], device=device)
        self.assertTrue(torch.allclose(coarse_world_pos, expected))

        # Voxel size should be 4x
        self.assertAlmostEqual(world_T_coarse.pseudo_scaling_factor, voxel_size * 4, places=5)

    @parameterized.expand(all_devices)
    def test_iterative_center_aligned_coarsening(self, device: str):
        """Test that iterative center-aligned coarsening works correctly.

        This simulates what SparseFeatureHierarchy.from_iterative_coarsening does.
        """
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        from nksr.nksr_fvdb.sparse_feature_hierarchy import (
            voxel_center_aligned_coarsening_xform,
        )

        voxel_size = 0.1
        world_T_level0 = UniformScaleThenTranslate(scale=voxel_size, translation=voxel_size / 2)

        fine_T_coarse = voxel_center_aligned_coarsening_xform(2)

        # Level 1: coarsen once using @ for composition
        world_T_level1 = world_T_level0 @ fine_T_coarse

        # Level 2: coarsen again
        world_T_level2 = world_T_level1 @ fine_T_coarse

        origin_ijk = torch.tensor([[0.0, 0.0, 0.0]], device=device)

        # Level 0: center at voxel_size/2 = 0.05
        level0_world = world_T_level0 @ origin_ijk
        self.assertTrue(torch.allclose(level0_world, torch.tensor([[0.05, 0.05, 0.05]], device=device)))

        # Level 1: center at voxel_size = 0.10 (2x coarser)
        level1_world = world_T_level1 @ origin_ijk
        self.assertTrue(torch.allclose(level1_world, torch.tensor([[0.10, 0.10, 0.10]], device=device)))

        # Level 2: center at 2*voxel_size = 0.20 (4x coarser)
        level2_world = world_T_level2 @ origin_ijk
        self.assertTrue(torch.allclose(level2_world, torch.tensor([[0.20, 0.20, 0.20]], device=device)))

        # Check voxel sizes
        self.assertAlmostEqual(world_T_level0.pseudo_scaling_factor, 0.1, places=5)
        self.assertAlmostEqual(world_T_level1.pseudo_scaling_factor, 0.2, places=5)
        self.assertAlmostEqual(world_T_level2.pseudo_scaling_factor, 0.4, places=5)


if __name__ == "__main__":
    unittest.main()
