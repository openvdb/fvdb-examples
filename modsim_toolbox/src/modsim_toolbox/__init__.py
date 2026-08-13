"""Reusable components for simulating imaging systems on voxelized scenes.

The toolbox generalizes over three things and nothing else: **geometry**,
**voxels**, and **ray tracing**, with a viewer for looking at the result. It
holds no physics -- no spectra, no materials, no units of radiance -- and reads
no files: every function takes tensors and returns tensors, so what a modality
means is decided entirely by its caller. See ``examples/car`` for one worked
modality end to end.

Import the modules, not their contents, so that every call site says which of
the four concepts it is reaching for::

    from modsim_toolbox import geometry, trace, visualization, voxelize

    topology, normals = voxelize.voxelize_mesh(vertices, faces, normals, size)
    origins, directions = trace.rays_from_pinhole(...)
    hit = trace.first_hit(topology, normals, origins, directions)
"""

from modsim_toolbox import geometry, trace, visualization, voxelize

__all__ = ["geometry", "trace", "visualization", "voxelize"]
