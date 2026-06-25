#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
fVDB-examples provide several methods for Gaussian splat segmentation.
From these method you can get a new Gaussian splat PLY file containing an object or region of interest from the original scene.
A corresponding mesh is commonly used in Isaac Sim and other simulation tools when working with splats.
This method allows for ripping out submeshes corresponding to Gaussian splat segments in PLY format.
Both of which can be used to create a USDZ for downstream simulation in Isaac Sim or other similar tools.
When working with many segments it's faster to rip from one larger mesh corresponding to the original scene mesh than it is to create a new mesh for each segment.

The segment splat PLY is segmentation-method agnostic (GarfVDB, manual crop, etc.).
The full-scene mesh is typically from an offline reconstruction (for example frgs mesh-dlnr).

By default, boundary holes are closed with harmonic Laplacian fill and the patch is made
watertight for physics simulation. Use --no-gap-fill to export the raw vertex/face subset.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import igl
import numpy as np
import point_cloud_utils as pcu
from fvdb import GaussianSplat3d
from scipy.spatial import cKDTree


def _reproject_vertex_colors(
    target_vertices: np.ndarray,
    source_vertices: np.ndarray,
    source_colors: np.ndarray,
) -> np.ndarray:
    """
    Copy vertex colors onto a new mesh via nearest-neighbor lookup.

    Args:
        target_vertices (np.ndarray): Vertex positions of the destination mesh, shape ``(#V, 3)``.
        source_vertices (np.ndarray): Vertex positions of the source mesh, shape ``(#V, 3)``.
        source_colors (np.ndarray): Per-vertex colors on the source mesh, shape ``(#V, 3)`` or ``(#V, 4)``.

    Returns:
        np.ndarray: Colors for ``target_vertices``, same channel count as ``source_colors``.
    """
    finite_mask = np.isfinite(source_vertices).all(axis=1)
    if not finite_mask.all():
        source_vertices = source_vertices[finite_mask]
        source_colors = source_colors[finite_mask]
    if source_vertices.shape[0] == 0:
        raise ValueError("No finite source vertices available for color reprojection")

    tree = cKDTree(source_vertices)
    query_vertices = np.asarray(target_vertices, dtype=np.float64)
    bad = ~np.isfinite(query_vertices).all(axis=1)
    if bad.any():
        query_vertices = query_vertices.copy()
        query_vertices[bad] = source_vertices.mean(axis=0)
    _, indices = tree.query(query_vertices, k=1, workers=-1)
    return source_colors[indices]


def _has_vertex_colors(vertex_colors: np.ndarray | None) -> bool:
    """
    Return whether a mesh carries a non-empty per-vertex color array.

    Args:
        vertex_colors (np.ndarray | None): Per-vertex colors loaded from mesh I/O, or ``None``.

    Returns:
        bool: ``True`` when ``vertex_colors`` has at least one vertex color.
    """
    return vertex_colors is not None and vertex_colors.size > 0 and vertex_colors.shape[0] > 0


def _as_float_vertex_colors(colors: np.ndarray) -> np.ndarray:
    """
    Convert vertex colors to float RGB(A) in ``[0, 1]`` for ``point_cloud_utils`` I/O.

    Args:
        colors (np.ndarray): Input colors as ``uint8`` or float; values above ``1.0`` are treated as 8-bit.

    Returns:
        np.ndarray: Float32 colors clipped to ``[0, 1]``.
    """
    colors = np.asarray(colors)
    if colors.dtype == np.uint8:
        colors = colors.astype(np.float64) / 255.0
    else:
        colors = colors.astype(np.float64)
        if colors.size > 0 and colors.max() > 1.0:
            colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0).astype(np.float32)


def extract_segment_mesh(
    *,
    full_mesh_path: Path,
    segment_ply_path: Path,
    output_path: Path,
    distance_threshold: float,
    no_gap_fill: bool,
    resolution: int,
    device: str,
    verbose: bool,
) -> None:
    """
    Extract a mesh patch from a full-scene mesh using a Gaussian splat segment as a spatial mask.

    Mesh vertices within ``distance_threshold`` of any segment Gaussian are kept. Faces whose
    three vertices all lie in that set are exported. Unless ``no_gap_fill`` is set, boundary holes
    are closed with harmonic Laplacian fill and the patch is made watertight for simulation.

    Args:
        full_mesh_path (Path): Path to the full-scene triangle mesh (``.ply`` or other ``point_cloud_utils`` format).
        segment_ply_path (Path): Path to the segment Gaussian splat scene (``.ply``).
        output_path (Path): Output path for the extracted segment mesh.
        distance_threshold (float): Maximum distance in world units from segment Gaussians to include mesh vertices.
        no_gap_fill (bool): When ``True``, skip harmonic hole fill and watertight remeshing.
        resolution (int): Octree resolution passed to ``pcu.make_mesh_watertight`` (capped to segment size).
        device (str): Torch device for loading the segment splat PLY (for example ``"cuda"``).
        verbose (bool): Enable debug logging.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")
    logger = logging.getLogger(__name__)

    # Load the full mesh
    logger.info(f"Loading full scene mesh from {full_mesh_path}")
    vertices, faces, vertex_colors = pcu.load_mesh_vfc(str(full_mesh_path))
    logger.info(f"Loaded mesh with {len(vertices):,} vertices and {len(faces):,} faces")

    # Load the segment Gaussians
    logger.info(f"Loading segment Gaussians from {segment_ply_path}")
    segment_splat, _ = GaussianSplat3d.from_ply(segment_ply_path, device=device)
    segment_means = segment_splat.means.cpu().numpy()  # [N, 3]
    logger.info(f"Loaded {len(segment_means):,} Gaussians in segment")

    # Build KD-tree for fast nearest neighbor queries
    logger.info("Building KD-tree for segment Gaussians...")
    tree = cKDTree(segment_means)

    # Find mesh vertices within distance threshold of any segment Gaussian
    logger.info(f"Finding mesh vertices within {distance_threshold:.3f} units of segment...")
    distances, _ = tree.query(vertices, k=1, distance_upper_bound=distance_threshold)
    close_vertex_mask = distances < distance_threshold
    num_close_vertices = close_vertex_mask.sum()

    logger.info(
        f"Found {num_close_vertices:,} vertices ({100 * num_close_vertices / len(vertices):.1f}%) " f"within threshold"
    )

    if num_close_vertices == 0:
        raise ValueError(
            f"No mesh vertices found within {distance_threshold} units of segment Gaussians. "
            f"Try increasing --distance-threshold."
        )

    # Create a mapping from old vertex indices to new vertex indices
    old_to_new_vertex_idx = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new_vertex_idx[close_vertex_mask] = np.arange(num_close_vertices)

    # Extract only faces where all 3 vertices are close to the segment
    face_mask = np.all(close_vertex_mask[faces], axis=1)
    extracted_faces = faces[face_mask]
    num_extracted_faces = len(extracted_faces)

    logger.info(
        f"Extracted {num_extracted_faces:,} faces ({100 * num_extracted_faces / len(faces):.1f}%) "
        f"where all vertices are close to segment"
    )

    if num_extracted_faces == 0:
        raise ValueError(
            "No complete faces found within the distance threshold. " "Try increasing --distance-threshold."
        )

    # Reindex faces to use new vertex indices
    extracted_faces_reindexed = old_to_new_vertex_idx[extracted_faces]

    # Extract the corresponding vertices and colors
    extracted_vertices = vertices[close_vertex_mask]
    extracted_colors = (
        _as_float_vertex_colors(vertex_colors[close_vertex_mask]) if _has_vertex_colors(vertex_colors) else None
    )

    if not no_gap_fill:
        # Use the raw extracted patch for color lookup after watertight remeshes vertices.
        color_source_vertices = extracted_vertices.copy()
        color_source_colors = extracted_colors

        logger.info("Harmonic Laplacian gap fill on extracted patch...")
        num_faces_before = len(extracted_faces_reindexed)
        num_vertices_before = len(extracted_vertices)
        extracted_vertices, extracted_faces_reindexed = fill_mesh_gaps(extracted_vertices, extracted_faces_reindexed)
        if len(extracted_faces_reindexed) != num_faces_before:
            logger.info(
                "Harmonic fill added %d faces and %d vertices (%d -> %d faces)",
                len(extracted_faces_reindexed) - num_faces_before,
                len(extracted_vertices) - num_vertices_before,
                num_faces_before,
                len(extracted_faces_reindexed),
            )
        effective_resolution = int(min(resolution, max(2_000, len(extracted_faces_reindexed) // 2)))
        if effective_resolution != resolution:
            logger.info(
                "Capped watertight resolution %d -> %d for segment size",
                resolution,
                effective_resolution,
            )
        logger.info("Making mesh watertight (resolution=%d)...", effective_resolution)
        num_faces_before = len(extracted_faces_reindexed)
        num_vertices_before = len(extracted_vertices)
        extracted_vertices, extracted_faces_reindexed = pcu.make_mesh_watertight(
            extracted_vertices,
            extracted_faces_reindexed,
            resolution=effective_resolution,
        )
        logger.info(
            "Watertight mesh: %d vertices, %d faces (was %d vertices, %d faces)",
            len(extracted_vertices),
            len(extracted_faces_reindexed),
            num_vertices_before,
            num_faces_before,
        )
        # Reproject colors onto the new mesh vertices if colors are available
        if color_source_colors is not None:
            extracted_colors = _reproject_vertex_colors(extracted_vertices, color_source_vertices, color_source_colors)

    # Save the extracted mesh
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving extracted mesh to {output_path}")

    if extracted_colors is not None:
        pcu.save_mesh_vfc(
            str(output_path),
            extracted_vertices,
            extracted_faces_reindexed,
            extracted_colors,
        )
    else:
        pcu.save_mesh_vf(str(output_path), extracted_vertices, extracted_faces_reindexed)

    logger.info(
        f"Successfully saved mesh with {len(extracted_vertices):,} vertices "
        f"and {len(extracted_faces_reindexed):,} faces"
    )


def fill_mesh_gaps(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    k: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Close boundary holes with fan caps and harmonic Laplacian fairing.

    Each open boundary loop is triangulated with a Steiner vertex at the loop centroid. Cap vertex
    positions are relaxed by solving a k-harmonic PDE (``k=1``: Laplacian, ``k=2``: biharmonic)
    with the original mesh vertices fixed.

    Args:
        vertices (np.ndarray): Mesh vertex positions, shape ``(#V, 3)``.
        faces (np.ndarray): Triangle indices, shape ``(#F, 3)``.
        k (int): Order of the harmonic operator (``1`` = harmonic/Laplacian, ``2`` = biharmonic).

    Returns:
        tuple[np.ndarray, np.ndarray]: Updated ``(vertices, faces)`` with holes capped.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be an (#F, 3) triangle index array")

    orig_vertex_count = vertices.shape[0]

    while True:
        loop = igl.boundary_loop(faces)
        if loop.size < 3:
            break

        centroid = vertices[loop].mean(axis=0, keepdims=True)
        cap_idx = vertices.shape[0]
        vertices = np.vstack([vertices, centroid])

        cap_faces = np.array(
            [[cap_idx, int(loop[i]), int(loop[(i + 1) % loop.size])] for i in range(loop.size)],
            dtype=np.int32,
        )
        faces = np.vstack([faces, cap_faces])

    if vertices.shape[0] == orig_vertex_count:
        return vertices.astype(np.float32), faces

    b = np.arange(orig_vertex_count, dtype=np.int32)
    bc = vertices[:orig_vertex_count]
    smoothed = igl.harmonic(vertices, faces, b, bc, k)
    if not np.isfinite(smoothed).all():
        bad = ~np.isfinite(smoothed).all(axis=1)
        smoothed = np.asarray(smoothed, dtype=np.float64)
        smoothed[bad] = vertices[bad]
    vertices = smoothed

    return vertices.astype(np.float32), faces


def main() -> None:
    """Parse CLI arguments and run :func:`extract_segment_mesh`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Extract a submesh from a full-scene mesh using a Gaussian splat segment "
            "as a spatial mask. Segment PLY can come from any segmentation pipeline."
        ),
    )
    parser.add_argument("--input-splat", type=Path, help="Input splat segment file (PLY format)")
    parser.add_argument("--input-mesh", type=Path, help="Input full scene mesh file (PLY/OBJ format)")
    parser.add_argument("--output-path", type=Path, required=True, help="Output path")
    parser.add_argument(
        "--no-gap-fill",
        action="store_true",
        help="Skip watertight + harmonic gap fill; may cause collision issues in Isaac Sim",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=20_000,
        help="Manifold octree resolution for make_mesh_watertight (default: 20000)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.5,
        help="Maximum distance from segment Gaussians to include mesh vertices (default: 0.5)",
    )
    args = parser.parse_args()
    if args.input_splat is None or args.input_mesh is None:
        parser.error("Both --input-splat and --input-mesh are required")

    extract_segment_mesh(
        full_mesh_path=args.input_mesh,
        segment_ply_path=args.input_splat,
        output_path=args.output_path,
        distance_threshold=args.distance_threshold,
        device=args.device,
        verbose=args.verbose,
        no_gap_fill=args.no_gap_fill,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
