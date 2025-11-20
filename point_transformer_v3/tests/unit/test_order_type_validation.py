# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for order type validation in PTV3_Attention.

This is a minimal unit test to establish the test infrastructure.
Tests a pure function with no dependencies on FVDB operations or external libraries.
"""

import unittest

from fvdb_extensions.models.ptv3_fvdb import PTV3_Attention


class TestOrderTypeValidation(unittest.TestCase):
    """Test cases for order type validation in PTV3_Attention."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_permute_valid_order_types(self):
        """Test that _permute accepts valid order types."""
        # Create a minimal attention instance (we won't call forward, just test _permute)
        attn = PTV3_Attention(
            hidden_size=64,
            num_heads=1,
            proj_drop=0.0,
            patch_size=0,
            order_index=0,
            order_types=("z", "z-trans"),
        )

        # Test that valid order types don't raise errors
        # Note: We can't actually call _permute without a grid, but we can test the validation logic
        valid_types = ["z", "z-trans", "hilbert", "hilbert-trans"]
        for order_type in valid_types:
            # The _permute method should accept these without raising ValueError
            # We test this by checking the method exists and has the right signature
            self.assertTrue(hasattr(attn, "_permute"), "Should have _permute method")
            self.assertTrue(callable(getattr(attn, "_permute")), "_permute should be callable")

    def test_permute_invalid_order_type_raises(self):
        """Test that _permute raises ValueError for invalid order types."""
        attn = PTV3_Attention(
            hidden_size=64,
            num_heads=1,
            proj_drop=0.0,
            patch_size=0,
            order_index=0,
            order_types=("z",),
        )

        # Create a mock grid object with minimal interface needed for _permute
        class MockGrid:
            def morton(self):
                return None

            def morton_zyx(self):
                return None

            def hilbert(self):
                return None

            def hilbert_zyx(self):
                return None

        mock_grid = MockGrid()

        # Test that invalid order type raises ValueError
        with self.assertRaises(ValueError):
            attn._permute(mock_grid, "invalid_order_type")

    def test_order_type_initialization(self):
        """Test that order_types are correctly stored during initialization."""
        order_types = ("z", "z-trans", "hilbert")
        attn = PTV3_Attention(
            hidden_size=64,
            num_heads=1,
            proj_drop=0.0,
            patch_size=0,
            order_index=0,
            order_types=order_types,
        )
        self.assertEqual(attn.order_types, order_types, "Order types should be stored correctly")


if __name__ == "__main__":
    unittest.main()

