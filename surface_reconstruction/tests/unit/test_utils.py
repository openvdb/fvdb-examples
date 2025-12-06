# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for synthetic point cloud generation utilities.

These tests validate that the point cloud generators produce geometrically
correct outputs suitable for testing the sparse feature hierarchy.
"""

import unittest

import torch
from fvdb import JaggedTensor
from parameterized import parameterized

from .utils import (
    PointCloudConfig,
    generate_box_surface,
    generate_cylinder_patch,
    generate_plane_patch,
    generate_scene,
    generate_sphere_patch,
    generate_street_scene_batch,
    generate_test_jagged_tensor,
    generate_test_point_clouds,
    point_clouds_to_jagged_tensor,
)

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


class TestPointCloudConfig(unittest.TestCase):
    """Test cases for PointCloudConfig dataclass."""

    def test_default_config(self):
        """Test that default configuration has sensible values."""
        config = PointCloudConfig()

        self.assertEqual(config.num_points, 1000)
        self.assertEqual(config.noise_std, 0.01)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.device, "cpu")

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PointCloudConfig(num_points=500, noise_std=0.05, seed=123, device="cpu")

        self.assertEqual(config.num_points, 500)
        self.assertEqual(config.noise_std, 0.05)
        self.assertEqual(config.seed, 123)


class TestGeneratePlanePatch(unittest.TestCase):
    """Test cases for plane patch generation."""

    def test_output_shape(self):
        """Test that output has correct shape (N, 3)."""
        config = PointCloudConfig(num_points=100, noise_std=0.0)
        points = generate_plane_patch(config=config)

        self.assertEqual(points.shape, (100, 3))

    def test_points_near_plane(self):
        """Test that generated points lie on/near the specified plane."""
        config = PointCloudConfig(num_points=500, noise_std=0.0, seed=42)
        center = (1.0, 2.0, 3.0)
        normal = (0.0, 0.0, 1.0)

        points = generate_plane_patch(center=center, normal=normal, config=config)

        # All points should have z ≈ 3.0 (on the plane)
        z_coords = points[:, 2]
        self.assertTrue(torch.allclose(z_coords, torch.full_like(z_coords, 3.0)))

    def test_points_within_bounds(self):
        """Test that points are within the specified width/height bounds."""
        config = PointCloudConfig(num_points=1000, noise_std=0.0, seed=42)
        width, height = 2.0, 1.5

        points = generate_plane_patch(
            center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), width=width, height=height, config=config
        )

        # For a z-normal plane, x extent is width, y extent is height
        self.assertTrue(torch.all(points[:, 0] >= -width / 2))
        self.assertTrue(torch.all(points[:, 0] <= width / 2))
        self.assertTrue(torch.all(points[:, 1] >= -height / 2))
        self.assertTrue(torch.all(points[:, 1] <= height / 2))

    def test_noise_adds_variation(self):
        """Test that noise adds Z-variation to plane points."""
        config_no_noise = PointCloudConfig(num_points=100, noise_std=0.0, seed=42)
        config_with_noise = PointCloudConfig(num_points=100, noise_std=0.1, seed=42)

        points_clean = generate_plane_patch(config=config_no_noise)
        points_noisy = generate_plane_patch(config=config_with_noise)

        # Clean points should have zero z-variance (all on plane)
        z_std_clean = points_clean[:, 2].std()
        z_std_noisy = points_noisy[:, 2].std()

        self.assertAlmostEqual(z_std_clean.item(), 0.0, places=5)
        self.assertGreater(z_std_noisy.item(), 0.01)  # Should have noise

    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical results."""
        config = PointCloudConfig(num_points=100, seed=42)

        points1 = generate_plane_patch(config=config)
        points2 = generate_plane_patch(config=config)

        self.assertTrue(torch.equal(points1, points2))


class TestGenerateSpherePatch(unittest.TestCase):
    """Test cases for sphere patch generation."""

    def test_output_shape(self):
        """Test that output has correct shape (N, 3)."""
        config = PointCloudConfig(num_points=100, noise_std=0.0)
        points = generate_sphere_patch(config=config)

        self.assertEqual(points.shape, (100, 3))

    def test_points_at_correct_radius(self):
        """Test that generated points lie at the specified radius."""
        config = PointCloudConfig(num_points=500, noise_std=0.0, seed=42)
        center = (1.0, 2.0, 3.0)
        radius = 0.5

        points = generate_sphere_patch(center=center, radius=radius, config=config)

        # Compute distance from center
        center_t = torch.tensor(center)
        distances = torch.linalg.norm(points - center_t, dim=1)

        self.assertTrue(torch.allclose(distances, torch.full_like(distances, radius), atol=1e-5))

    def test_hemisphere_only_positive_z(self):
        """Test that hemisphere (elevation 0-90) produces only positive Z."""
        config = PointCloudConfig(num_points=500, noise_std=0.0, seed=42)

        points = generate_sphere_patch(center=(0, 0, 0), elevation_range=(0.0, 90.0), config=config)

        # All z-coordinates should be >= 0
        self.assertTrue(torch.all(points[:, 2] >= -1e-5))

    def test_quarter_sphere_bounds(self):
        """Test that quarter sphere is in expected quadrant."""
        config = PointCloudConfig(num_points=500, noise_std=0.0, seed=42)

        # Azimuth 0-90 should give positive x and y
        points = generate_sphere_patch(
            center=(0, 0, 0), azimuth_range=(0.0, 90.0), elevation_range=(0.0, 90.0), config=config
        )

        self.assertTrue(torch.all(points[:, 0] >= -1e-5))  # x >= 0
        self.assertTrue(torch.all(points[:, 1] >= -1e-5))  # y >= 0
        self.assertTrue(torch.all(points[:, 2] >= -1e-5))  # z >= 0


class TestGenerateCylinderPatch(unittest.TestCase):
    """Test cases for cylinder patch generation."""

    def test_output_shape(self):
        """Test that output has correct shape (N, 3)."""
        config = PointCloudConfig(num_points=100, noise_std=0.0)
        points = generate_cylinder_patch(config=config)

        self.assertEqual(points.shape, (100, 3))

    def test_points_at_correct_radius_z_axis(self):
        """Test that points lie at correct radial distance from Z axis."""
        config = PointCloudConfig(num_points=500, noise_std=0.0, seed=42)
        radius = 0.3
        center = (1.0, 2.0, 0.0)

        points = generate_cylinder_patch(center=center, radius=radius, axis="z", config=config)

        # Radial distance in XY plane from axis
        center_xy = torch.tensor([center[0], center[1]])
        radial_dist = torch.linalg.norm(points[:, :2] - center_xy, dim=1)

        self.assertTrue(torch.allclose(radial_dist, torch.full_like(radial_dist, radius), atol=1e-5))

    def test_height_range(self):
        """Test that points are within specified height range."""
        config = PointCloudConfig(num_points=500, noise_std=0.0, seed=42)
        height = 2.0

        points = generate_cylinder_patch(
            center=(0, 0, 0), height=height, height_range=(0.25, 0.75), axis="z", config=config
        )

        # Height range 0.25-0.75 of 2.0 height centered at 0 means z in [-0.5, 0.5]
        # (0.25 - 0.5) * 2 = -0.5, (0.75 - 0.5) * 2 = 0.5
        self.assertTrue(torch.all(points[:, 2] >= -0.5 - 1e-5))
        self.assertTrue(torch.all(points[:, 2] <= 0.5 + 1e-5))

    def test_different_axes(self):
        """Test that axis parameter rotates the cylinder correctly."""
        config = PointCloudConfig(num_points=100, noise_std=0.0, seed=42)

        points_z = generate_cylinder_patch(axis="z", config=config)
        points_y = generate_cylinder_patch(axis="y", config=config)
        points_x = generate_cylinder_patch(axis="x", config=config)

        # Z-axis cylinder should have largest variance in Z
        # Y-axis cylinder should have largest variance in Y
        # X-axis cylinder should have largest variance in X
        z_var = points_z.var(dim=0)
        y_var = points_y.var(dim=0)
        x_var = points_x.var(dim=0)

        self.assertEqual(torch.argmax(z_var).item(), 2)  # Z has max variance
        self.assertEqual(torch.argmax(y_var).item(), 1)  # Y has max variance
        self.assertEqual(torch.argmax(x_var).item(), 0)  # X has max variance


class TestGenerateBoxSurface(unittest.TestCase):
    """Test cases for box surface generation."""

    def test_output_shape(self):
        """Test that output has correct shape (N, 3)."""
        config = PointCloudConfig(num_points=100, noise_std=0.0)
        points = generate_box_surface(config=config)

        self.assertEqual(points.shape, (100, 3))

    def test_single_face_planar(self):
        """Test that a single face produces planar points."""
        config = PointCloudConfig(num_points=100, noise_std=0.0, seed=42)
        size = (1.0, 1.0, 1.0)

        points = generate_box_surface(center=(0, 0, 0), size=size, faces=("+z",), config=config)

        # +z face should have z = 0.5 (half of size)
        self.assertTrue(torch.allclose(points[:, 2], torch.full((100,), 0.5)))

    def test_points_on_correct_faces(self):
        """Test that points lie on the specified faces."""
        config = PointCloudConfig(num_points=600, noise_std=0.0, seed=42)
        size = (2.0, 2.0, 2.0)

        points = generate_box_surface(center=(0, 0, 0), size=size, faces=("+x", "-x"), config=config)

        # All points should have x = ±1.0
        x_vals = points[:, 0].abs()
        self.assertTrue(torch.allclose(x_vals, torch.ones_like(x_vals)))

    def test_empty_faces_raises(self):
        """Test that empty faces tuple raises ValueError."""
        config = PointCloudConfig(num_points=100)

        with self.assertRaises(ValueError):
            generate_box_surface(faces=(), config=config)


class TestGenerateScene(unittest.TestCase):
    """Test cases for composite scene generation."""

    def test_combines_multiple_primitives(self):
        """Test that scene combines points from multiple primitives."""
        config = PointCloudConfig(num_points=50, noise_std=0.0, seed=42)

        primitives = [
            {"type": "plane", "center": (0, 0, 0)},
            {"type": "sphere", "center": (2, 0, 0), "radius": 0.5},
        ]

        points = generate_scene(primitives, config=config)

        # Should have 2 * 50 = 100 points
        self.assertEqual(points.shape[0], 100)

        # Some points should be near z=0 (plane), others near sphere
        near_plane = torch.sum(torch.abs(points[:, 2]) < 0.1)
        near_sphere_center = torch.sum(torch.abs(points[:, 0] - 2.0) < 0.6)

        self.assertGreater(near_plane, 0)
        self.assertGreater(near_sphere_center, 0)

    def test_invalid_primitive_type_raises(self):
        """Test that invalid primitive type raises ValueError."""
        primitives = [{"type": "invalid_type"}]

        with self.assertRaises(ValueError):
            generate_scene(primitives)


class TestGenerateTestPointClouds(unittest.TestCase):
    """Test cases for batch point cloud generation."""

    def test_batch_size_1(self):
        """Test generation with batch size 1."""
        clouds = generate_test_point_clouds(batch_size=1, points_per_cloud=100, seed=42)

        self.assertEqual(len(clouds), 1)
        self.assertEqual(clouds[0].shape[1], 3)
        # Should have approximately 100 points
        self.assertGreater(clouds[0].shape[0], 50)

    def test_batch_size_2(self):
        """Test generation with batch size 2."""
        clouds = generate_test_point_clouds(batch_size=2, points_per_cloud=100, seed=42)

        self.assertEqual(len(clouds), 2)
        for cloud in clouds:
            self.assertEqual(cloud.shape[1], 3)

    def test_batch_size_4(self):
        """Test generation with batch size 4."""
        clouds = generate_test_point_clouds(batch_size=4, points_per_cloud=100, seed=42)

        self.assertEqual(len(clouds), 4)
        for cloud in clouds:
            self.assertEqual(cloud.shape[1], 3)

    def test_different_seeds_produce_different_clouds(self):
        """Test that different seeds produce different point clouds."""
        clouds1 = generate_test_point_clouds(batch_size=1, seed=42)
        clouds2 = generate_test_point_clouds(batch_size=1, seed=43)

        self.assertFalse(torch.equal(clouds1[0], clouds2[0]))

    def test_same_seed_reproducible(self):
        """Test that same seed produces identical results."""
        clouds1 = generate_test_point_clouds(batch_size=2, seed=42)
        clouds2 = generate_test_point_clouds(batch_size=2, seed=42)

        for c1, c2 in zip(clouds1, clouds2):
            self.assertTrue(torch.equal(c1, c2))

    def test_clouds_are_distinct(self):
        """Test that different clouds in a batch are geometrically distinct."""
        clouds = generate_test_point_clouds(batch_size=4, points_per_cloud=200, seed=42)

        # Each cloud should have a different centroid (different scenes)
        centroids = [cloud.mean(dim=0) for cloud in clouds]

        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                # Centroids should not be identical
                dist = torch.linalg.norm(centroids[i] - centroids[j])
                # Allow for some overlap but expect meaningful differences
                # (different scene templates should have different centroids)
                self.assertGreater(dist, 0.01)


class TestPointCloudsToJaggedTensor(unittest.TestCase):
    """Test cases for JaggedTensor conversion."""

    def test_single_cloud(self):
        """Test conversion of single point cloud."""
        clouds = [torch.randn(100, 3)]
        jt = point_clouds_to_jagged_tensor(clouds)

        self.assertIsInstance(jt, JaggedTensor)
        self.assertEqual(jt.jdata.shape, (100, 3))

    def test_multiple_clouds(self):
        """Test conversion of multiple point clouds."""
        clouds = [torch.randn(100, 3), torch.randn(150, 3), torch.randn(80, 3)]
        jt = point_clouds_to_jagged_tensor(clouds)

        self.assertIsInstance(jt, JaggedTensor)
        # Total points should be 100 + 150 + 80 = 330
        self.assertEqual(jt.jdata.shape, (330, 3))

    def test_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with self.assertRaises(ValueError):
            point_clouds_to_jagged_tensor([])

    def test_preserves_data(self):
        """Test that conversion preserves point data."""
        cloud1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cloud2 = torch.tensor([[7.0, 8.0, 9.0]])

        jt = point_clouds_to_jagged_tensor([cloud1, cloud2])

        # First cloud points
        self.assertTrue(torch.equal(jt.jdata[:2], cloud1))
        # Second cloud points
        self.assertTrue(torch.equal(jt.jdata[2:], cloud2))


class TestGenerateTestJaggedTensor(unittest.TestCase):
    """Test cases for the convenience JaggedTensor generator."""

    def test_returns_jagged_tensor(self):
        """Test that function returns a JaggedTensor."""
        jt = generate_test_jagged_tensor(batch_size=2, points_per_cloud=100, seed=42)

        self.assertIsInstance(jt, JaggedTensor)

    def test_batch_sizes(self):
        """Test various batch sizes produce correct structure."""
        for batch_size in [1, 2, 4]:
            jt = generate_test_jagged_tensor(batch_size=batch_size, points_per_cloud=100, seed=42)

            # Check that we have the right number of offsets (batch_size + 1)
            self.assertEqual(len(jt.joffsets), batch_size + 1)

    def test_reproducibility(self):
        """Test that same seed produces identical results."""
        jt1 = generate_test_jagged_tensor(batch_size=2, seed=42)
        jt2 = generate_test_jagged_tensor(batch_size=2, seed=42)

        self.assertTrue(torch.equal(jt1.jdata, jt2.jdata))


# =============================================================================
# Street scene batch generation tests
# =============================================================================


class TestGenerateStreetSceneBatch(unittest.TestCase):
    """Test cases for street scene batch JaggedTensor generation."""

    @parameterized.expand(all_device_batch_combos)
    def test_batch_generation(self, device: str, batch_size: int) -> None:
        """Smoke test: verify batch generation succeeds and returns valid JaggedTensor."""
        jt = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=42,
            num_points=1000,  # Smaller for faster tests
            device=device,
        )

        # Verify it's a JaggedTensor
        self.assertIsInstance(jt, JaggedTensor)

        # Verify shape: data should be (N, 3)
        self.assertEqual(jt.jdata.ndim, 2)
        self.assertEqual(jt.jdata.shape[1], 3)

        # Verify batch structure: offsets should have batch_size + 1 elements
        self.assertEqual(len(jt.joffsets), batch_size + 1)

        # Verify offsets start at 0 and end at total points
        self.assertEqual(jt.joffsets[0], 0)
        self.assertEqual(jt.joffsets[-1], jt.jdata.shape[0])

        # Verify device
        self.assertEqual(jt.jdata.device.type, device)

    @parameterized.expand(all_device_batch_combos)
    def test_batch_has_points(self, device: str, batch_size: int) -> None:
        """Verify each batch element has a reasonable number of points."""
        jt = generate_street_scene_batch(
            batch_size=batch_size,
            base_seed=123,
            num_points=1000,
            device=device,
        )

        # Check each batch element has points
        for i in range(batch_size):
            start = jt.joffsets[i].item()
            end = jt.joffsets[i + 1].item()
            num_points = end - start

            # Each scene should have a reasonable number of points
            # (at least 100, since we requested 1000)
            self.assertGreaterEqual(num_points, 100, f"Batch element {i} has too few points: {num_points}")

    @parameterized.expand(all_devices)
    def test_different_seeds_produce_different_scenes(self, device: str) -> None:
        """Verify that different base seeds produce different point clouds."""
        jt1 = generate_street_scene_batch(batch_size=1, base_seed=42, num_points=500, device=device)
        jt2 = generate_street_scene_batch(batch_size=1, base_seed=43, num_points=500, device=device)

        # The point clouds should be different
        self.assertFalse(torch.equal(jt1.jdata, jt2.jdata))

    @parameterized.expand(all_devices)
    def test_same_seed_is_reproducible(self, device: str) -> None:
        """Verify that same seed produces identical results."""
        jt1 = generate_street_scene_batch(batch_size=2, base_seed=42, num_points=500, device=device)
        jt2 = generate_street_scene_batch(batch_size=2, base_seed=42, num_points=500, device=device)

        # The point clouds should be identical
        self.assertTrue(torch.equal(jt1.jdata, jt2.jdata))
        self.assertTrue(torch.equal(jt1.joffsets, jt2.joffsets))

    @parameterized.expand(all_devices)
    def test_batch_elements_are_distinct(self, device: str) -> None:
        """Verify that different batch elements have different geometry."""
        jt = generate_street_scene_batch(batch_size=4, base_seed=42, num_points=1000, device=device)

        # Compute centroids for each batch element
        centroids = []
        for i in range(4):
            start = jt.joffsets[i].item()
            end = jt.joffsets[i + 1].item()
            batch_points = jt.jdata[start:end]
            centroid = batch_points.mean(dim=0)
            centroids.append(centroid)

        # Each pair of centroids should be different
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = torch.linalg.norm(centroids[i] - centroids[j])
                # Different scenes should have different centroids
                self.assertGreater(dist, 0.01, f"Batch elements {i} and {j} have nearly identical centroids")

    @parameterized.expand(all_devices)
    def test_points_in_reasonable_range(self, device: str) -> None:
        """Verify that generated points are in a reasonable coordinate range."""
        jt = generate_street_scene_batch(batch_size=2, base_seed=42, num_points=1000, device=device)

        # Street scenes should be roughly 30-50m in length, 8-12m wide
        # Points should generally be within a reasonable bounding box
        min_coords = jt.jdata.min(dim=0).values
        max_coords = jt.jdata.max(dim=0).values
        extent = max_coords - min_coords

        # The scene should have some spatial extent
        self.assertGreater(extent[0], 1.0, "Scene X extent too small")
        self.assertGreater(extent[1], 1.0, "Scene Y extent too small")

        # But not be astronomically large (sanity check)
        self.assertLess(extent.max(), 200.0, "Scene extent unexpectedly large")


if __name__ == "__main__":
    unittest.main()
