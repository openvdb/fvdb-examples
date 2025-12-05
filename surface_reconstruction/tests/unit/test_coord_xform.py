# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for coordinate frame transformations in coord_xform.py.

Tests the basic transformation classes: IdentityXform and ScalarGainBiasXform.
"""

import unittest

import torch
from nksr.nksr_fvdb.coord_xform import IdentityXform, ScalarGainBiasXform


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


class TestScalarGainBiasXform(unittest.TestCase):
    """Test cases for ScalarGainBiasXform."""

    def test_gain_only(self):
        """Test scaling with gain only."""
        xform = ScalarGainBiasXform(gain=2.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]])

        result = xform.apply_tensor(coords)
        expected = torch.tensor([[2.0, 4.0, 6.0]])

        self.assertTrue(torch.allclose(result, expected))

    def test_bias_only(self):
        """Test translation with bias only."""
        xform = ScalarGainBiasXform(bias=1.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]])

        result = xform.apply_tensor(coords)
        expected = torch.tensor([[2.0, 3.0, 4.0]])

        self.assertTrue(torch.allclose(result, expected))

    def test_gain_and_bias(self):
        """Test combined scaling and translation."""
        xform = ScalarGainBiasXform(gain=2.0, bias=1.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]])

        result = xform.apply_tensor(coords)
        expected = torch.tensor([[3.0, 5.0, 7.0]])  # coords * 2 + 1

        self.assertTrue(torch.allclose(result, expected))

    def test_inverse(self):
        """Test that inverse correctly reverses the transformation."""
        xform = ScalarGainBiasXform(gain=2.0, bias=1.0)
        coords = torch.tensor([[1.0, 2.0, 3.0]])

        transformed = xform.apply_tensor(coords)
        recovered = xform.inverse().apply_tensor(transformed)

        self.assertTrue(torch.allclose(recovered, coords))

    def test_invertible_when_gain_nonzero(self):
        """Test that ScalarGainBiasXform is invertible when gain != 0."""
        xform = ScalarGainBiasXform(gain=2.0, bias=1.0)
        self.assertTrue(xform.invertible)

    def test_not_invertible_when_gain_zero(self):
        """Test that ScalarGainBiasXform is not invertible when gain == 0."""
        xform = ScalarGainBiasXform(gain=0.0, bias=1.0)
        self.assertFalse(xform.invertible)


if __name__ == "__main__":
    unittest.main()
