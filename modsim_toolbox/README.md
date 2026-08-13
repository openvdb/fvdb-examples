# ModSim Toolbox

ModSim Toolbox is a minimal example of building reusable, accelerated, and
differentiable infrastructure for multimodal simulation on top of
[fVDB](https://github.com/AcademySoftwareFoundation/openvdb/tree/master/fvdb).
It is not a simulator for any particular sensor. Instead, it factors out the
geometric work that most sensor simulations share, so that the physics
specific to a given modality can be written separately, on top of a common
tensor-based interface.

## Components

The toolbox covers three areas:

- **geometry** — rotations and the coordinate frames a sensor is mounted in.
- **voxels** — converting group-jagged meshes into an `fvdb.GridBatch`, and
  carrying per-corner attributes onto the resulting voxels.
- **ray tracing** — casting a bundle of rays against a voxel grid batch using
  fVDB's voxel ray tracing, finding the nearest surface each ray hits, and
  computing visibility between a surface and a direction (for example,
  toward a light source).

There is also a viewer (`visualization.Visualizer`) for displaying the
result. Because ray tracing runs against voxel grids rather than triangle
meshes, and because fVDB's operations are differentiable, the same trace can
be used both for fast forward simulation and for gradient-based tasks such as
calibration or inverse rendering.

The toolbox does not model physics: it has no notion of spectra, materials,
or units of radiance, and it does not read files. Every function operates on
tensors and returns tensors, and the modules are stateless, with the
exception of `visualization.Visualizer`, which owns a live viser server. What
a modality's outputs represent — radiance, a range value, a label — is
decided by the caller. This is what allows the same underlying trace to
support different modalities, such as a panchromatic camera or a
spectrometer, without changes to the toolbox itself.

## Example

[`examples/car`](examples/car) shows one modality built on the toolbox: a
DIRSIG vehicle-glint scene, voxelized per OBJ group, imaged by the
panchromatic frame camera described in the scene's own platform file, and
shaded using solar irradiance and Beard-Maxwell BRDFs. The physics specific
to that example — spectral reflectance, shading, band integration — is
implemented in the example itself, not in the toolbox.
