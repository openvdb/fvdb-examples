# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Synthetic point cloud generation utilities for testing sparse feature hierarchy.

These generators create point clouds that mimic real-world surface reconstruction
data from LiDAR or image-based reconstruction (e.g., COLMAP/Ceres). Real scans
typically capture partial surfaces with sensor noise, so these utilities generate
points sampled from canonical geometric primitives with configurable:
  - Noise levels (Gaussian perturbation)
  - Partial visibility (azimuthal/elevation ranges for spheres/cylinders)
  - Point density

All generators return tensors of shape (N, 3) representing 3D point coordinates.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from fvdb import JaggedTensor


@dataclass
class PointCloudConfig:
    """Configuration for synthetic point cloud generation.

    Attributes:
        num_points: Target number of points to generate.
        noise_std: Standard deviation of Gaussian noise added to points (in same units as geometry).
        seed: Random seed for reproducibility. If None, uses non-deterministic generation.
        device: Torch device for generated tensors.
    """

    num_points: int = 1000
    noise_std: float = 0.01
    seed: int | None = 42
    device: torch.device | str = "cpu"


def _get_generator(config: PointCloudConfig) -> torch.Generator:
    """Create a torch Generator with optional seeding."""
    gen = torch.Generator(device=config.device)
    if config.seed is not None:
        gen.manual_seed(config.seed)
    return gen


def _add_noise(points: torch.Tensor, config: PointCloudConfig, generator: torch.Generator) -> torch.Tensor:
    """Add Gaussian noise to points."""
    if config.noise_std > 0:
        noise = torch.randn(points.shape, device=points.device, generator=generator) * config.noise_std
        points = points + noise
    return points


# =============================================================================
# Plane patch generator
# =============================================================================


def generate_plane_patch(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    width: float = 1.0,
    height: float = 1.0,
    config: PointCloudConfig | None = None,
) -> torch.Tensor:
    """Generate points on a rectangular plane patch.

    Creates a planar point cloud useful for testing flat surfaces like walls,
    floors, and tables. The plane is defined by its center, normal direction,
    and rectangular extent.

    Args:
        center: Center point of the plane in world coordinates.
        normal: Normal vector of the plane (will be normalized).
        width: Extent along the plane's local X axis.
        height: Extent along the plane's local Y axis.
        config: Point cloud generation configuration.

    Returns:
        Tensor of shape (N, 3) with points lying on the plane (plus noise).
    """
    if config is None:
        config = PointCloudConfig()

    gen = _get_generator(config)
    device = config.device

    # Generate random 2D coordinates in [-0.5, 0.5] range
    u = torch.rand(config.num_points, device=device, generator=gen) - 0.5  # [-0.5, 0.5]
    v = torch.rand(config.num_points, device=device, generator=gen) - 0.5

    # Build orthonormal basis from normal
    normal_t = torch.tensor(normal, dtype=torch.float32, device=device)
    normal_t = normal_t / torch.linalg.norm(normal_t)

    # Find a vector not parallel to normal
    if abs(normal_t[0]) < 0.9:
        arbitrary = torch.tensor([1.0, 0.0, 0.0], device=device)
    else:
        arbitrary = torch.tensor([0.0, 1.0, 0.0], device=device)

    # Gram-Schmidt to get orthonormal basis
    tangent_u = arbitrary - torch.dot(arbitrary, normal_t) * normal_t
    tangent_u = tangent_u / torch.linalg.norm(tangent_u)
    tangent_v = torch.cross(normal_t, tangent_u)

    # Generate points on plane
    center_t = torch.tensor(center, dtype=torch.float32, device=device)
    points = (
        center_t.unsqueeze(0)
        + (u * width).unsqueeze(1) * tangent_u.unsqueeze(0)
        + (v * height).unsqueeze(1) * tangent_v.unsqueeze(0)
    )

    return _add_noise(points, config, gen)


# =============================================================================
# Sphere patch generator
# =============================================================================


def generate_sphere_patch(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
    azimuth_range: tuple[float, float] = (0.0, 360.0),
    elevation_range: tuple[float, float] = (-90.0, 90.0),
    config: PointCloudConfig | None = None,
) -> torch.Tensor:
    """Generate points on a partial sphere surface.

    Creates a spherical point cloud patch, useful for testing curved surfaces.
    The patch extent is controlled by azimuth (horizontal) and elevation
    (vertical) angle ranges, allowing simulation of partial scans.

    Common configurations:
      - Full sphere: azimuth=(0, 360), elevation=(-90, 90)
      - Hemisphere (top): azimuth=(0, 360), elevation=(0, 90)
      - Quarter sphere: azimuth=(0, 90), elevation=(0, 90)
      - Front-facing patch: azimuth=(-45, 45), elevation=(-30, 30)

    Args:
        center: Center of the sphere.
        radius: Radius of the sphere.
        azimuth_range: (min, max) azimuth angles in degrees [0, 360].
        elevation_range: (min, max) elevation angles in degrees [-90, 90].
        config: Point cloud generation configuration.

    Returns:
        Tensor of shape (N, 3) with points on the sphere surface (plus noise).
    """
    if config is None:
        config = PointCloudConfig()

    gen = _get_generator(config)
    device = config.device

    # Convert angles to radians
    az_min, az_max = torch.deg2rad(torch.tensor(azimuth_range, device=device))
    el_min, el_max = torch.deg2rad(torch.tensor(elevation_range, device=device))

    # Generate random angles within range
    azimuth = torch.rand(config.num_points, device=device, generator=gen) * (az_max - az_min) + az_min
    elevation = torch.rand(config.num_points, device=device, generator=gen) * (el_max - el_min) + el_min

    # Convert spherical to Cartesian
    cos_el = torch.cos(elevation)
    x = radius * cos_el * torch.cos(azimuth)
    y = radius * cos_el * torch.sin(azimuth)
    z = radius * torch.sin(elevation)

    center_t = torch.tensor(center, dtype=torch.float32, device=device)
    points = torch.stack([x, y, z], dim=1) + center_t.unsqueeze(0)

    return _add_noise(points, config, gen)


# =============================================================================
# Cylinder patch generator
# =============================================================================


def generate_cylinder_patch(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 0.5,
    height: float = 2.0,
    axis: Literal["x", "y", "z"] = "z",
    azimuth_range: tuple[float, float] = (0.0, 360.0),
    height_range: tuple[float, float] = (0.0, 1.0),
    config: PointCloudConfig | None = None,
) -> torch.Tensor:
    """Generate points on a partial cylinder surface.

    Creates a cylindrical point cloud patch, useful for testing elongated curved
    surfaces like pipes, pillars, or tree trunks. The patch extent is controlled
    by azimuth angle and height fraction ranges.

    Args:
        center: Center of the cylinder (at mid-height).
        radius: Radius of the cylinder.
        height: Total height of the cylinder.
        axis: Axis along which the cylinder extends ("x", "y", or "z").
        azimuth_range: (min, max) azimuth angles in degrees [0, 360].
        height_range: (min, max) height as fraction of total height [0, 1].
        config: Point cloud generation configuration.

    Returns:
        Tensor of shape (N, 3) with points on the cylinder surface (plus noise).
    """
    if config is None:
        config = PointCloudConfig()

    gen = _get_generator(config)
    device = config.device

    # Convert angles to radians
    az_min, az_max = torch.deg2rad(torch.tensor(azimuth_range, device=device))

    # Generate random parameters
    azimuth = torch.rand(config.num_points, device=device, generator=gen) * (az_max - az_min) + az_min
    h_frac = (
        torch.rand(config.num_points, device=device, generator=gen) * (height_range[1] - height_range[0])
        + height_range[0]
    )
    h = (h_frac - 0.5) * height  # Centered at 0

    # Generate points in local coordinates (cylinder along Z)
    local_x = radius * torch.cos(azimuth)
    local_y = radius * torch.sin(azimuth)
    local_z = h

    # Rotate to align with specified axis
    if axis == "z":
        points = torch.stack([local_x, local_y, local_z], dim=1)
    elif axis == "y":
        points = torch.stack([local_x, local_z, local_y], dim=1)
    elif axis == "x":
        points = torch.stack([local_z, local_x, local_y], dim=1)
    else:
        raise ValueError(f"Invalid axis '{axis}', must be 'x', 'y', or 'z'")

    center_t = torch.tensor(center, dtype=torch.float32, device=device)
    points = points + center_t.unsqueeze(0)

    return _add_noise(points, config, gen)


# =============================================================================
# Box surface generator
# =============================================================================


def generate_box_surface(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    faces: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z", "-z"),
    config: PointCloudConfig | None = None,
) -> torch.Tensor:
    """Generate points on selected faces of a box surface.

    Creates a box surface point cloud, useful for testing axis-aligned flat
    surfaces like furniture, buildings, or room interiors. Individual faces
    can be selected to simulate partial visibility.

    Args:
        center: Center of the box.
        size: (width, depth, height) dimensions of the box.
        faces: Tuple of face identifiers to include. Options: "+x", "-x", "+y", "-y", "+z", "-z".
        config: Point cloud generation configuration.

    Returns:
        Tensor of shape (N, 3) with points on the box faces (plus noise).
    """
    if config is None:
        config = PointCloudConfig()

    gen = _get_generator(config)
    device = config.device

    if not faces:
        raise ValueError("At least one face must be specified")

    # Calculate points per face (roughly equal distribution)
    points_per_face = config.num_points // len(faces)
    remainder = config.num_points % len(faces)

    all_points = []
    half_size = (size[0] / 2, size[1] / 2, size[2] / 2)
    center_t = torch.tensor(center, dtype=torch.float32, device=device)

    for i, face in enumerate(faces):
        n_pts = points_per_face + (1 if i < remainder else 0)
        if n_pts == 0:
            continue

        # Generate random UV coordinates for this face
        u = torch.rand(n_pts, device=device, generator=gen) - 0.5
        v = torch.rand(n_pts, device=device, generator=gen) - 0.5

        if face == "+x":
            pts = torch.stack([torch.full((n_pts,), half_size[0], device=device), u * size[1], v * size[2]], dim=1)
        elif face == "-x":
            pts = torch.stack([torch.full((n_pts,), -half_size[0], device=device), u * size[1], v * size[2]], dim=1)
        elif face == "+y":
            pts = torch.stack([u * size[0], torch.full((n_pts,), half_size[1], device=device), v * size[2]], dim=1)
        elif face == "-y":
            pts = torch.stack([u * size[0], torch.full((n_pts,), -half_size[1], device=device), v * size[2]], dim=1)
        elif face == "+z":
            pts = torch.stack([u * size[0], v * size[1], torch.full((n_pts,), half_size[2], device=device)], dim=1)
        elif face == "-z":
            pts = torch.stack([u * size[0], v * size[1], torch.full((n_pts,), -half_size[2], device=device)], dim=1)
        else:
            raise ValueError(f"Invalid face '{face}'")

        all_points.append(pts + center_t)

    points = torch.cat(all_points, dim=0)
    return _add_noise(points, config, gen)


# =============================================================================
# Composite scene generator
# =============================================================================


def generate_scene(
    primitives: list[dict],
    config: PointCloudConfig | None = None,
) -> torch.Tensor:
    """Generate a composite point cloud from multiple primitives.

    Combines multiple geometric primitives into a single point cloud, useful
    for testing more complex scenes. Each primitive is specified as a dict
    with a "type" key and the corresponding generator arguments.

    Args:
        primitives: List of primitive specifications. Each dict must have a "type"
            key ("plane", "sphere", "cylinder", or "box") and can include any
            arguments accepted by the corresponding generator function. The "config"
            key is handled specially to allow per-primitive point counts.
        config: Default configuration for primitives that don't specify their own.

    Returns:
        Tensor of shape (total_N, 3) with combined points from all primitives.

    Example:
        >>> primitives = [
        ...     {"type": "plane", "center": (0, 0, 0), "width": 2.0},
        ...     {"type": "sphere", "center": (1, 0, 0.5), "radius": 0.3},
        ... ]
        >>> points = generate_scene(primitives)
    """
    if config is None:
        config = PointCloudConfig()

    generators = {
        "plane": generate_plane_patch,
        "sphere": generate_sphere_patch,
        "cylinder": generate_cylinder_patch,
        "box": generate_box_surface,
    }

    all_points = []
    for i, prim in enumerate(primitives):
        prim_copy = prim.copy()
        prim_type = prim_copy.pop("type")

        if prim_type not in generators:
            raise ValueError(f"Unknown primitive type '{prim_type}'")

        # Handle per-primitive config
        if "config" in prim_copy:
            prim_config = prim_copy.pop("config")
        else:
            # Create a new config with offset seed to ensure variety
            prim_config = PointCloudConfig(
                num_points=config.num_points,
                noise_std=config.noise_std,
                seed=(config.seed + i * 1000) if config.seed is not None else None,
                device=config.device,
            )

        points = generators[prim_type](**prim_copy, config=prim_config)
        all_points.append(points)

    return torch.cat(all_points, dim=0)


# =============================================================================
# Batch generation utilities for testing SparseFeatureHierarchy
# =============================================================================


def generate_test_point_clouds(
    batch_size: int = 1,
    points_per_cloud: int = 500,
    noise_std: float = 0.01,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    """Generate a batch of diverse test point clouds.

    Creates multiple distinct point clouds suitable for batch testing of
    SparseFeatureHierarchy. Each cloud is a different geometric configuration
    to ensure the hierarchy handles diverse inputs correctly.

    The generated configurations are deterministic for a given seed, enabling
    reproducible tests.

    Args:
        batch_size: Number of point clouds to generate (1, 2, 4, etc.).
        points_per_cloud: Approximate number of points per cloud.
        noise_std: Noise standard deviation.
        seed: Random seed for reproducibility.
        device: Torch device.

    Returns:
        List of tensors, each of shape (N, 3), one per cloud.
    """
    # Define a variety of scenes to cycle through
    scene_templates = [
        # Scene 0: Simple plane (floor/wall)
        [{"type": "plane", "center": (0.0, 0.0, 0.0), "normal": (0.0, 0.0, 1.0), "width": 1.0, "height": 1.0}],
        # Scene 1: Hemisphere (dome/ball)
        [{"type": "sphere", "center": (0.5, 0.5, 0.0), "radius": 0.4, "elevation_range": (0.0, 90.0)}],
        # Scene 2: Vertical cylinder (pillar/pipe)
        [
            {
                "type": "cylinder",
                "center": (0.0, 0.0, 0.5),
                "radius": 0.2,
                "height": 1.0,
                "axis": "z",
                "azimuth_range": (0.0, 270.0),
            }
        ],
        # Scene 3: Box with 3 visible faces (corner view)
        [{"type": "box", "center": (0.3, 0.3, 0.3), "size": (0.5, 0.5, 0.5), "faces": ("+x", "+y", "+z")}],
        # Scene 4: Plane + sphere (table with object)
        [
            {"type": "plane", "center": (0.0, 0.0, 0.0), "width": 1.5, "height": 1.5},
            {"type": "sphere", "center": (0.2, -0.1, 0.15), "radius": 0.15, "elevation_range": (0.0, 90.0)},
        ],
        # Scene 5: L-shaped walls (room corner)
        [
            {"type": "plane", "center": (0.0, 0.5, 0.5), "normal": (0.0, -1.0, 0.0), "width": 1.0, "height": 1.0},
            {"type": "plane", "center": (0.5, 0.0, 0.5), "normal": (-1.0, 0.0, 0.0), "width": 1.0, "height": 1.0},
        ],
        # Scene 6: Cylinder on plane (pipe on floor)
        [
            {"type": "plane", "center": (0.0, 0.0, 0.0), "width": 1.2, "height": 1.2},
            {"type": "cylinder", "center": (0.0, 0.0, 0.1), "radius": 0.1, "height": 0.8, "axis": "y"},
        ],
        # Scene 7: Multiple small spheres (scattered objects)
        [
            {"type": "sphere", "center": (-0.3, -0.3, 0.1), "radius": 0.1},
            {"type": "sphere", "center": (0.3, -0.2, 0.1), "radius": 0.12},
            {"type": "sphere", "center": (0.0, 0.3, 0.1), "radius": 0.08},
        ],
    ]

    clouds = []
    for i in range(batch_size):
        # Cycle through scene templates
        scene_idx = i % len(scene_templates)
        scene = scene_templates[scene_idx]

        # Calculate points per primitive
        num_primitives = len(scene)
        pts_per_prim = points_per_cloud // num_primitives

        # Build scene with per-primitive configs
        primitives = []
        for j, prim in enumerate(scene):
            prim_copy = prim.copy()
            prim_copy["config"] = PointCloudConfig(
                num_points=pts_per_prim,
                noise_std=noise_std,
                seed=seed + i * 1000 + j * 100,
                device=device,
            )
            primitives.append(prim_copy)

        cloud = generate_scene(primitives, config=PointCloudConfig(device=device))
        clouds.append(cloud)

    return clouds


def point_clouds_to_jagged_tensor(clouds: list[torch.Tensor]) -> JaggedTensor:
    """Convert a list of point cloud tensors to a JaggedTensor.

    This is the format expected by SparseFeatureHierarchy.from_iterative_coarsening
    and related methods.

    Args:
        clouds: List of point cloud tensors, each of shape (N_i, 3).

    Returns:
        JaggedTensor combining all clouds with proper batch structure.
    """
    if not clouds:
        raise ValueError("Cannot create JaggedTensor from empty list")

    # Build offsets from cumulative point counts
    offsets = [0]
    for cloud in clouds:
        offsets.append(offsets[-1] + cloud.shape[0])

    # Concatenate all point data
    data = torch.cat(clouds, dim=0)

    # Build offsets tensor
    device = clouds[0].device
    offsets_tensor = torch.tensor(offsets, dtype=torch.int32, device=device)

    return JaggedTensor.from_data_and_offsets(data, offsets_tensor)


def generate_test_jagged_tensor(
    batch_size: int = 1,
    points_per_cloud: int = 500,
    noise_std: float = 0.01,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> JaggedTensor:
    """Generate a JaggedTensor of test point clouds for hierarchy testing.

    Convenience function that combines generate_test_point_clouds and
    point_clouds_to_jagged_tensor.

    Args:
        batch_size: Number of point clouds in the batch.
        points_per_cloud: Approximate number of points per cloud.
        noise_std: Noise standard deviation.
        seed: Random seed for reproducibility.
        device: Torch device.

    Returns:
        JaggedTensor suitable for SparseFeatureHierarchy construction.
    """
    clouds = generate_test_point_clouds(
        batch_size=batch_size,
        points_per_cloud=points_per_cloud,
        noise_std=noise_std,
        seed=seed,
        device=device,
    )
    return point_clouds_to_jagged_tensor(clouds)


# =============================================================================
# High-level scene generators by scale
# =============================================================================


def generate_object_scene(
    seed: int = 42,
    num_points: int = 2000,
    noise_std: float = 0.005,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate a small-scale object scene (e.g., tabletop objects).

    Creates a scene with a few objects on a surface, typical of object scanning
    or tabletop manipulation scenarios. Scale is approximately 0.5-1.0 meters.

    Args:
        seed: Random seed for reproducibility.
        num_points: Total number of points in the scene.
        noise_std: Noise standard deviation.
        device: Torch device.

    Returns:
        Tensor of shape (N, 3) with scene points.
    """
    gen = torch.Generator().manual_seed(seed)

    # Randomize object placement and types
    num_objects = 2 + int(torch.randint(0, 3, (1,), generator=gen).item())  # 2-4 objects

    primitives = [
        # Base surface (small table/platform)
        {
            "type": "plane",
            "center": (0.0, 0.0, 0.0),
            "normal": (0.0, 0.0, 1.0),
            "width": 0.6,
            "height": 0.6,
        },
    ]

    object_types = ["sphere", "cylinder", "box"]
    for i in range(num_objects):
        obj_type = object_types[int(torch.randint(0, len(object_types), (1,), generator=gen).item())]
        # Random position on the table surface
        x = (torch.rand(1, generator=gen).item() - 0.5) * 0.4
        y = (torch.rand(1, generator=gen).item() - 0.5) * 0.4
        scale = 0.05 + torch.rand(1, generator=gen).item() * 0.1  # 0.05-0.15

        if obj_type == "sphere":
            primitives.append(
                {
                    "type": "sphere",
                    "center": (x, y, scale),
                    "radius": scale,
                    "elevation_range": (0.0, 90.0),  # Hemisphere visible from above
                }
            )
        elif obj_type == "cylinder":
            height = scale * 2 + torch.rand(1, generator=gen).item() * 0.1
            primitives.append(
                {
                    "type": "cylinder",
                    "center": (x, y, height / 2),
                    "radius": scale * 0.5,
                    "height": height,
                    "axis": "z",
                    "azimuth_range": (0.0, 300.0),
                }
            )
        else:  # box
            primitives.append(
                {
                    "type": "box",
                    "center": (x, y, scale),
                    "size": (scale * 1.5, scale, scale * 2),
                    "faces": ("+x", "+y", "+z", "-x", "-y"),  # All but bottom
                }
            )

    # Distribute points across primitives
    pts_per_prim = num_points // len(primitives)
    for j, prim in enumerate(primitives):
        prim["config"] = PointCloudConfig(
            num_points=pts_per_prim,
            noise_std=noise_std,
            seed=seed + j * 100,
            device=device,
        )

    return generate_scene(primitives)


def generate_room_scene(
    seed: int = 42,
    num_points: int = 5000,
    noise_std: float = 0.01,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate a room-scale indoor scene (e.g., office, living room).

    Creates a scene with walls, floor, and furniture-like objects, typical of
    indoor scanning scenarios. Scale is approximately 3-5 meters.

    Args:
        seed: Random seed for reproducibility.
        num_points: Total number of points in the scene.
        noise_std: Noise standard deviation.
        device: Torch device.

    Returns:
        Tensor of shape (N, 3) with scene points.
    """
    gen = torch.Generator().manual_seed(seed)

    room_width = 3.0 + torch.rand(1, generator=gen).item() * 2.0  # 3-5m
    room_depth = 3.0 + torch.rand(1, generator=gen).item() * 2.0
    room_height = 2.5 + torch.rand(1, generator=gen).item() * 0.5  # 2.5-3m

    primitives = [
        # Floor
        {
            "type": "plane",
            "center": (0.0, 0.0, 0.0),
            "normal": (0.0, 0.0, 1.0),
            "width": room_width,
            "height": room_depth,
        },
        # Back wall
        {
            "type": "plane",
            "center": (0.0, room_depth / 2, room_height / 2),
            "normal": (0.0, -1.0, 0.0),
            "width": room_width,
            "height": room_height,
        },
        # Left wall
        {
            "type": "plane",
            "center": (-room_width / 2, 0.0, room_height / 2),
            "normal": (1.0, 0.0, 0.0),
            "width": room_depth,
            "height": room_height,
        },
    ]

    # Add furniture-like objects
    num_furniture = 2 + int(torch.randint(0, 3, (1,), generator=gen).item())

    for i in range(num_furniture):
        furn_type = int(torch.randint(0, 3, (1,), generator=gen).item())
        x = (torch.rand(1, generator=gen).item() - 0.5) * room_width * 0.6
        y = (torch.rand(1, generator=gen).item() - 0.5) * room_depth * 0.6

        if furn_type == 0:  # Table/desk
            w = 0.8 + torch.rand(1, generator=gen).item() * 0.4
            d = 0.5 + torch.rand(1, generator=gen).item() * 0.3
            h = 0.7 + torch.rand(1, generator=gen).item() * 0.1
            primitives.append(
                {
                    "type": "box",
                    "center": (x, y, h / 2),
                    "size": (w, d, h),
                    "faces": ("+x", "-x", "+y", "-y", "+z"),
                }
            )
        elif furn_type == 1:  # Chair/stool (cylinder)
            r = 0.2 + torch.rand(1, generator=gen).item() * 0.1
            h = 0.4 + torch.rand(1, generator=gen).item() * 0.2
            primitives.append(
                {
                    "type": "cylinder",
                    "center": (x, y, h / 2),
                    "radius": r,
                    "height": h,
                    "axis": "z",
                }
            )
        else:  # Lamp/vase (sphere on cylinder)
            r = 0.1 + torch.rand(1, generator=gen).item() * 0.1
            primitives.append(
                {
                    "type": "sphere",
                    "center": (x, y, 0.8 + r),
                    "radius": r,
                }
            )

    # Distribute points across primitives
    pts_per_prim = num_points // len(primitives)
    for j, prim in enumerate(primitives):
        prim["config"] = PointCloudConfig(
            num_points=pts_per_prim,
            noise_std=noise_std,
            seed=seed + j * 100,
            device=device,
        )

    return generate_scene(primitives)


def generate_street_scene(
    seed: int = 42,
    num_points: int = 10000,
    noise_std: float = 0.02,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate a street-scale outdoor scene (e.g., urban environment).

    Creates a scene with ground plane, building facades, and street furniture,
    typical of autonomous driving or urban mapping scenarios. Scale is
    approximately 20-50 meters.

    Args:
        seed: Random seed for reproducibility.
        num_points: Total number of points in the scene.
        noise_std: Noise standard deviation.
        device: Torch device.

    Returns:
        Tensor of shape (N, 3) with scene points.
    """
    gen = torch.Generator().manual_seed(seed)

    street_length = 30.0 + torch.rand(1, generator=gen).item() * 20.0  # 30-50m
    street_width = 8.0 + torch.rand(1, generator=gen).item() * 4.0  # 8-12m

    primitives = [
        # Ground plane (street + sidewalks)
        {
            "type": "plane",
            "center": (0.0, 0.0, 0.0),
            "normal": (0.0, 0.0, 1.0),
            "width": street_length,
            "height": street_width + 4.0,  # Include sidewalks
        },
    ]

    # Add buildings on both sides
    num_buildings = 3 + int(torch.randint(0, 3, (1,), generator=gen).item())
    building_x = -street_length / 2 + 5.0

    for i in range(num_buildings):
        side = 1 if i % 2 == 0 else -1
        building_width = 5.0 + torch.rand(1, generator=gen).item() * 8.0
        building_height = 8.0 + torch.rand(1, generator=gen).item() * 12.0
        building_depth = 4.0 + torch.rand(1, generator=gen).item() * 3.0

        y_pos = side * (street_width / 2 + building_depth / 2 + 2.0)

        # Building facade (facing street)
        primitives.append(
            {
                "type": "plane",
                "center": (building_x + building_width / 2, y_pos - side * building_depth / 2, building_height / 2),
                "normal": (0.0, -side, 0.0),
                "width": building_width,
                "height": building_height,
            }
        )

        building_x += building_width + 2.0 + torch.rand(1, generator=gen).item() * 3.0

    # Add street furniture (poles, signs)
    num_poles = 4 + int(torch.randint(0, 4, (1,), generator=gen).item())
    for i in range(num_poles):
        x = (torch.rand(1, generator=gen).item() - 0.5) * street_length * 0.8
        side = 1 if torch.rand(1, generator=gen).item() > 0.5 else -1
        y = side * (street_width / 2 + 1.0)
        pole_height = 3.0 + torch.rand(1, generator=gen).item() * 2.0

        primitives.append(
            {
                "type": "cylinder",
                "center": (x, y, pole_height / 2),
                "radius": 0.1,
                "height": pole_height,
                "axis": "z",
            }
        )

    # Add some parked cars (boxes)
    num_cars = 2 + int(torch.randint(0, 3, (1,), generator=gen).item())
    for i in range(num_cars):
        x = (torch.rand(1, generator=gen).item() - 0.5) * street_length * 0.6
        side = 1 if i % 2 == 0 else -1
        y = side * (street_width / 2 - 1.5)

        primitives.append(
            {
                "type": "box",
                "center": (x, y, 0.7),
                "size": (4.5, 1.8, 1.4),
                "faces": ("+x", "-x", "+y", "-y", "+z"),
            }
        )

    # Distribute points across primitives
    pts_per_prim = num_points // len(primitives)
    for j, prim in enumerate(primitives):
        prim["config"] = PointCloudConfig(
            num_points=pts_per_prim,
            noise_std=noise_std,
            seed=seed + j * 100,
            device=device,
        )

    return generate_scene(primitives)


def generate_default_scenes(seed: int = 42, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Generate all default scene types for visualization/testing.

    Convenience function that generates one scene at each scale (object, room,
    street) using sensible defaults. Useful for quick visualization and testing.

    Args:
        seed: Random seed for reproducibility.
        device: Torch device.

    Returns:
        Dict mapping scene names to point cloud tensors.
    """
    return {
        "object": generate_object_scene(seed=seed, device=device),
        "room": generate_room_scene(seed=seed + 1000, device=device),
        "street": generate_street_scene(seed=seed + 2000, device=device),
    }


# =============================================================================
# Visualization utilities
# =============================================================================


def visualize_scenes(scenes: dict[str, torch.Tensor] | None = None, seed: int = 42) -> None:
    """Visualize point cloud scenes using polyscope.

    This function provides interactive 3D visualization of synthetic point clouds.
    If polyscope is not installed, the function silently returns without error.

    Args:
        scenes: Dict mapping scene names to point cloud tensors. If None, generates
            default scenes using generate_default_scenes().
        seed: Random seed used when generating default scenes (ignored if scenes provided).
    """
    # Try to import polyscope - it's an optional dependency
    try:
        import polyscope as ps
    except ImportError:
        # polyscope not available, silently skip visualization
        return

    # Generate default scenes if none provided
    if scenes is None:
        scenes = generate_default_scenes(seed=seed)

    # Initialize polyscope
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Color palette for different scenes
    colors = [
        (0.267, 0.467, 0.671),  # Steel blue
        (0.929, 0.486, 0.192),  # Orange
        (0.392, 0.710, 0.529),  # Green
        (0.839, 0.373, 0.373),  # Red
        (0.584, 0.459, 0.702),  # Purple
    ]

    # Register each scene as a point cloud
    # Position scenes side by side with appropriate spacing
    x_offset = 0.0
    for i, (name, points) in enumerate(scenes.items()):
        # Convert to numpy for polyscope
        pts_np = points.cpu().numpy()

        # Calculate bounding box for spacing
        bbox_min = pts_np.min(axis=0)
        bbox_max = pts_np.max(axis=0)
        scene_width = bbox_max[0] - bbox_min[0]
        scene_center_x = (bbox_max[0] + bbox_min[0]) / 2

        # Offset points to position scenes side by side
        pts_np[:, 0] += x_offset - scene_center_x

        # Register with polyscope
        cloud = ps.register_point_cloud(f"{name}_scene", pts_np)
        cloud.set_color(colors[i % len(colors)])
        # cloud.set_radius(0.003, relative=False)

        # Update offset for next scene
        x_offset += scene_width + max(scene_width * 0.2, 2.0)

    # Show the visualization
    ps.show()


def main() -> None:
    """Main entry point for visualizing synthetic test scenes."""
    print("Generating synthetic test scenes...")
    print("  - Object scale (tabletop)")
    print("  - Room scale (indoor)")
    print("  - Street scale (outdoor)")
    print()

    # Generate and visualize all default scenes
    scenes = generate_default_scenes(seed=42)

    for name, points in scenes.items():
        bbox_min = points.min(dim=0).values
        bbox_max = points.max(dim=0).values
        print(f"{name:8s}: {points.shape[0]:6d} points, bbox: [{bbox_min.tolist()}] to [{bbox_max.tolist()}]")

    print()
    print("Launching polyscope visualizer...")
    visualize_scenes(scenes)
    print("Done.")


if __name__ == "__main__":
    main()
