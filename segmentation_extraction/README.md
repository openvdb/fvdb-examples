# Segment Mesh Extraction

Extract a submesh from a full-scene mesh using a Gaussian splat segment as a spatial mask.

This tool is **segmentation-method agnostic**: the segment splat PLY can come from
any segmentation method provided in fvdb-examples, manual cropping, or any other pipeline that
produces a Gaussian splat `.ply` for the region of interest.

## Typical workflow

1. Create a 3D Gaussian splat of a real-world scene using `frgs reconstruct` (fvdb-reality-capture) or a similar Gaussian splatting method.
2. Create a mesh from the splat using `frgs mesh-dlnr` or similar.
3. Run a segmentation method like GarfVDB or LangSplatV2 to get a segment of the splat (in Gaussian splat PLY format) for the object or region you care about.
4. Use `extract_mesh_segments.py` to rip out the part of the larger mesh that corresponds with your chosen segment.
5. Optionally, use the segmented splat (in Gaussian splat PLY format) and mesh pair to make a USDZ with `frgs convert`. The USDZ can then be used for downstream simulation in Isaac Sim or other similar tools.

Ripping objects from large scenes can result in holes where the object was lying on a surface. Additionally, if the mesh isn't watertight, robots and other meshes can get stuck or fall through the mesh. These issues are fixed by using a harmonic fill method to fill the gap and a watertight remeshing step to fill in smaller holes and solidify the mesh. Use `--no-gap-fill` to skip this step.

## Installation

```bash
conda env create -f segmentation_extraction_environment.yml
conda activate fvdb_segment_mesh
pip install -e .
```

## Usage

```bash
python extract_mesh_segments.py \
    --input-mesh /path/to/full_scene_mesh.ply \
    --input-splat /path/to/segment_splats.ply \
    --output-path /path/to/segment_mesh.ply
```

### Optional Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--distance-threshold` | `0.5` | Max distance (world units) from segment Gaussians to include mesh vertices |
| `--no-gap-fill` | off | Skip harmonic hole fill and watertight remeshing |
| `--resolution` | `20000` | Octree resolution for `make_mesh_watertight` (auto-capped for small segments) |
| `--device` | `cuda` | Device for loading the segment splat PLY |

## Related examples

- [GarfVDB](../instance_segmentation/garfvdb/) — instance segmentation
- [LangSplatV2](../open_vocabulary_segmentation/langsplatv2/) — semantic segmentation
