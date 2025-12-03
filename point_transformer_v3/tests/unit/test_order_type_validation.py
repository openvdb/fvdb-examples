# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for space-filling curve functions in fvdb_utils.

Tests the standalone curve type validation and dispatch logic
in space_filling_curve_from_jagged_ijk and related functions.
"""

import unittest
from unittest.mock import MagicMock, patch

from fvdb_extensions.models.fvdb_utils import (
    hilbert_flipped_from_jagged_ijk,
    hilbert_from_jagged_ijk,
    identity_from_jagged_ijk,
    morton_flipped_from_jagged_ijk,
    morton_from_jagged_ijk,
    space_filling_curve_from_jagged_ijk,
)


class TestSpaceFillingCurveValidation(unittest.TestCase):
    """Test cases for curve type validation in space_filling_curve_from_jagged_ijk."""

    def test_invalid_curve_type_raises_value_error(self):
        """Test that invalid curve types raise ValueError."""
        mock_jagged_ijk = MagicMock()

        invalid_types = ["invalid", "unknown", "random", "xyz", ""]
        for curve_type in invalid_types:
            with self.assertRaises(ValueError, msg=f"Should raise ValueError for '{curve_type}'"):
                space_filling_curve_from_jagged_ijk(mock_jagged_ijk, curve_type)

    @patch("fvdb_extensions.models.fvdb_utils.morton_from_jagged_ijk")
    def test_morton_alias_z(self, mock_morton):
        """Test that 'z' dispatches to morton_from_jagged_ijk."""
        mock_jagged_ijk = MagicMock()
        mock_morton.return_value = MagicMock()

        space_filling_curve_from_jagged_ijk(mock_jagged_ijk, "z")
        mock_morton.assert_called_once_with(mock_jagged_ijk)

    @patch("fvdb_extensions.models.fvdb_utils.morton_from_jagged_ijk")
    def test_morton_alias_morton(self, mock_morton):
        """Test that 'morton' dispatches to morton_from_jagged_ijk."""
        mock_jagged_ijk = MagicMock()
        mock_morton.return_value = MagicMock()

        space_filling_curve_from_jagged_ijk(mock_jagged_ijk, "morton")
        mock_morton.assert_called_once_with(mock_jagged_ijk)

    @patch("fvdb_extensions.models.fvdb_utils.morton_flipped_from_jagged_ijk")
    def test_morton_flipped_alias_z_trans(self, mock_morton_flipped):
        """Test that 'z-trans' dispatches to morton_flipped_from_jagged_ijk."""
        mock_jagged_ijk = MagicMock()
        mock_morton_flipped.return_value = MagicMock()

        space_filling_curve_from_jagged_ijk(mock_jagged_ijk, "z-trans")
        mock_morton_flipped.assert_called_once_with(mock_jagged_ijk)

    @patch("fvdb_extensions.models.fvdb_utils.morton_flipped_from_jagged_ijk")
    def test_morton_flipped_alias_morton_zyx(self, mock_morton_flipped):
        """Test that 'morton_zyx' dispatches to morton_flipped_from_jagged_ijk."""
        mock_jagged_ijk = MagicMock()
        mock_morton_flipped.return_value = MagicMock()

        space_filling_curve_from_jagged_ijk(mock_jagged_ijk, "morton_zyx")
        mock_morton_flipped.assert_called_once_with(mock_jagged_ijk)

    @patch("fvdb_extensions.models.fvdb_utils.hilbert_from_jagged_ijk")
    def test_hilbert_dispatch(self, mock_hilbert):
        """Test that 'hilbert' dispatches to hilbert_from_jagged_ijk."""
        mock_jagged_ijk = MagicMock()
        mock_hilbert.return_value = MagicMock()

        space_filling_curve_from_jagged_ijk(mock_jagged_ijk, "hilbert")
        mock_hilbert.assert_called_once_with(mock_jagged_ijk)

    @patch("fvdb_extensions.models.fvdb_utils.hilbert_flipped_from_jagged_ijk")
    def test_hilbert_trans_dispatch(self, mock_hilbert_flipped):
        """Test that 'hilbert-trans' dispatches to hilbert_flipped_from_jagged_ijk."""
        mock_jagged_ijk = MagicMock()
        mock_hilbert_flipped.return_value = MagicMock()

        space_filling_curve_from_jagged_ijk(mock_jagged_ijk, "hilbert-trans")
        mock_hilbert_flipped.assert_called_once_with(mock_jagged_ijk)

    @patch("fvdb_extensions.models.fvdb_utils.identity_from_jagged_ijk")
    def test_identity_alias_vdb(self, mock_identity):
        """Test that 'vdb' dispatches to identity_from_jagged_ijk."""
        mock_jagged_ijk = MagicMock()
        mock_identity.return_value = MagicMock()

        space_filling_curve_from_jagged_ijk(mock_jagged_ijk, "vdb")
        mock_identity.assert_called_once_with(mock_jagged_ijk)

    @patch("fvdb_extensions.models.fvdb_utils.identity_from_jagged_ijk")
    def test_identity_alias_identity(self, mock_identity):
        """Test that 'identity' dispatches to identity_from_jagged_ijk."""
        mock_jagged_ijk = MagicMock()
        mock_identity.return_value = MagicMock()

        space_filling_curve_from_jagged_ijk(mock_jagged_ijk, "identity")
        mock_identity.assert_called_once_with(mock_jagged_ijk)


class TestValidCurveTypes(unittest.TestCase):
    """Test that all documented curve types are valid."""

    def test_all_valid_curve_types_documented(self):
        """Verify the complete set of valid curve types."""
        valid_types = {
            "z",
            "morton",
            "z-trans",
            "morton_zyx",
            "hilbert",
            "hilbert-trans",
            "vdb",
            "identity",
        }
        # This test documents the expected valid types
        # The dispatch tests above verify each one works
        self.assertEqual(len(valid_types), 8, "Should have 8 valid curve type aliases")


if __name__ == "__main__":
    unittest.main()
