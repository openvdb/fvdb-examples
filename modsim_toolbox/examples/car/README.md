# Car: vehicle-glint example

This example uses the toolbox to simulate a panchromatic frame camera imaging
a car, based on RIT's DIRSIG vehicle-glint demo scene. The toolbox handles
voxelization and ray tracing; this example adds the optical physics —
solar irradiance, Beard-Maxwell BRDFs, spectral reflectance, and band
integration — in `radiometry.py` and `shade`.

## Layout

| File | Contents |
|---|---|
| `main.py` | `build_scene`, `build_sensor`, `simulate`, `serve` |
| `radiometry.py` | Solar spectrum, Beard-Maxwell, Fresnel, spectral resampling |
| `loader.py` | Wavefront OBJ and DIRSIG `.platform` / `.ppd` / `.mat` / `.ems` / `.fit` parsing |

## Scene data

The scene data is not checked in. Download the DIRSIG vehicle-glint demo
scene from the [DIRSIG demo scenes page](https://dirsig.cis.rit.edu/) and
unpack it into `scene/`, subject to RIT's license terms for that data. This
example reads five files from it:

```
scene/geometry/infiniti_g35_vn.obj   car geometry, per group, with vertex normals
scene/demo.platform                  320x240 at 16 um, f = 100 mm, 0.400-0.800 um "Pan"
scene/demo.ppd                       platform pose: 150 m up, yawed 135 deg
scene/materials/demo.mat             material table, referencing:
scene/materials/*.ems, *.fit         emissivity curves and Beard-Maxwell fits
```

The 6 km ground plane declared in the scene is replaced with a 40 m patch,
and capture time and location are constants in `main.py` rather than parsed
from the scene files. Everything else the demo ships is unused.

## Running it

```bash
uv run python main.py
```

This opens a viser page showing the voxelized scene alongside the simulated
focal plane, with controls for solar azimuth, elevation, and shadows.
