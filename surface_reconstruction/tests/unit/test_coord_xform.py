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


if __name__ == "__main__":
    unittest.main()
