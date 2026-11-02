"""Simulate the DIRSIG vehicle glint demo with fVDB, and serve it in viser.

The demo scene ships everything a sensor simulation needs, and this example
reads all of it rather than hard-coding any of it:

``geometry/infiniti_g35_vn.obj``
    One Infiniti G35 with vertex normals, voxelized per OBJ group via
    :func:`modsim_toolbox.voxelize.voxelize_mesh`.
``demo.platform`` / ``demo.ppd``
    A 320 x 240, 16 um focal plane behind a 100 mm lens, 150 m up and yawed
    135 degrees. Parsed by :mod:`loader`, which casts one ray per element.
``materials/demo.mat``
    Spectral emissivities and Beard-Maxwell BRDF fits, loaded by
    :mod:`loader` and evaluated by :mod:`radiometry`. The glossy paint and
    glass are what glint.
``demo.scene`` / ``demo.tasks``
    Where and when: Rochester NY, 2010-08-03 11:06 local, which is what the
    sun sliders default to.

The four stages below run in the order they are written:
:meth:`Scene.from_obj` turns the geometry into voxel grids,
:meth:`Sensor.from_dirsig` reads the optics and the materials, :func:`simulate`
casts one ray per detector element and shades what it hits, and :func:`serve`
puts both in a browser. Every knob either stage has -- voxel sizing, the ground
patch, the spectral grid, the capture time and place -- is a defaulted argument
on the one that uses it, so the only module constants are the paths into the
scene and the capture the scene was authored around.

The radiometry is deliberately single-bounce: direct solar plus uniform diffuse
sky, no interreflection and no path radiance between the scene and the
platform. That is enough to reproduce what this demo exists to show -- glints
off the car's specular paint and glass moving as the sun moves -- but it is not
a substitute for DIRSIG's own radiometry solver.

Wavelengths are microns and radiances are ``W / (m^2 sr)`` integrated over the
channel's passband, unless stated otherwise.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import NamedTuple

import fvdb
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingImports]
import pvlib.solarposition  # pyright: ignore[reportMissingImports]
import torch

from modsim_toolbox import geometry, trace, visualization, voxelize

import loader  # pyright: ignore[reportImplicitRelativeImport]
import radiometry  # pyright: ignore[reportImplicitRelativeImport]

DEVICE = "cuda"

SCENE_DIR = Path(__file__).parent / "scene"
OBJ_FILE = SCENE_DIR / "geometry" / "infiniti_g35_vn.obj"
PLATFORM_FILE = SCENE_DIR / "demo.platform"
MOTION_FILE = SCENE_DIR / "demo.ppd"
MATERIAL_FILE = SCENE_DIR / "materials" / "demo.mat"

# Rochester NY, 2010-08-03 11:06 local, which is what demo.scene and demo.tasks
# declare and what the sun sliders default to.
CAPTURE_LATITUDE = 43.12
CAPTURE_LONGITUDE = -77.67
CAPTURE_TIME = dt.datetime(
    2010, 8, 3, 10, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5))
) + dt.timedelta(seconds=4000)


class Scene(NamedTuple):
    """The voxelized scene a ray is traced against.

    Attributes:
        topology: Voxel grids, one per OBJ group plus the ground.
        normals: Unit surface normal per voxel, jagged shape: (G, M_g, 3).
            G: number of groups; M_g: voxels in group g.
        materials: ``usemtl`` id of the group each voxel belongs to, jagged
            shape: (G, M_g).
        names: Name of each group, length G.
        voxel_sizes: Voxel edge length per group, in scene meters, shape: (G,).
        ground_z: Height of the ground plane in scene meters.
    """

    topology: fvdb.GridBatch
    normals: fvdb.JaggedTensor
    materials: fvdb.JaggedTensor
    names: list[str]
    voxel_sizes: torch.Tensor
    ground_z: float

    @classmethod
    def from_obj(
        cls,
        path: Path,
        detail_fraction: float = 0.35,
        voxel_size_bounds: tuple[float, float] = (0.001, 0.1),
        ground_half_extent: float = 20.0,
        ground_voxel_size: float = 0.5,
        ground_z: float = 0.0,
        ground_material_id: int = 8000,
        device: str = DEVICE,
    ) -> Scene:
        """Voxelize an OBJ and the ground it sits on.

        Each OBJ group is voxelized at its own tessellation, so the wing mirrors
        resolve without the underbody costing a fortune, and the ground patch is
        added as one more group at a size that only has to catch shadows.

        The defaults describe this demo: the scene's own 6 km ground box, whose
        top is at z = 0 and which ``geometry.glist`` gives material 8000, stood
        in for by a patch just big enough to hold the car's shadow.

        Args:
            path: Path to the ``.obj`` file.
            detail_fraction: Voxel size per group as a fraction of that group's
                own tessellation. Smaller oversamples; larger blurs.
            voxel_size_bounds: Floor and ceiling on that size, in scene meters.
            ground_half_extent: Half-width of the square ground patch, in scene
                meters.
            ground_voxel_size: Voxel edge length of the ground, in scene meters.
                It only has to catch shadows, so it is coarse.
            ground_z: Height of the ground plane in scene meters.
            ground_material_id: ``usemtl`` id to give the ground.
            device: Torch device to build everything on.

        Returns:
            The voxelized scene.
        """
        vertices, normals, faces, names, material_ids = loader.load_obj(path, device)
        sizes = voxelize.estimate_voxel_size(
            fvdb.JaggedTensor(vertices),
            fvdb.JaggedTensor(faces),
            detail_scale=detail_fraction,
            smallest=voxel_size_bounds[0],
            largest=voxel_size_bounds[1],
        )

        ground_vertices, ground_normals, ground_faces = loader.ground_plane_mesh(
            ground_half_extent, ground_z, device
        )
        names = names + ["ground"]
        material_ids = material_ids + [ground_material_id]
        voxel_sizes = torch.cat([sizes, sizes.new_tensor([ground_voxel_size])])

        topology, raw_normals = voxelize.voxelize_mesh(
            fvdb.JaggedTensor(vertices + [ground_vertices]),
            fvdb.JaggedTensor(faces + [ground_faces]),
            fvdb.JaggedTensor(normals + [ground_normals]),
            voxel_sizes,
        )
        return cls(
            topology=topology,
            normals=raw_normals.jagged_like(
                torch.nn.functional.normalize(raw_normals.jdata, dim=-1)
            ),
            # The trace reports which voxel it hit, so materials are resolved
            # per voxel: every voxel of a group carries that group's ``usemtl``.
            materials=voxelize.broadcast_to_voxels(
                topology, torch.tensor(material_ids, device=topology.device)
            ),
            names=names,
            voxel_sizes=voxel_sizes,
            ground_z=ground_z,
        )


class Sensor(NamedTuple):
    """The instrument, where it is, and what it is looking at things made of.

    Attributes:
        instrument: Optics and focal plane, from the platform file.
        position: Platform location in scene ENU coordinates, in meters,
            shape: (3,).
        rotation: Rotation taking instrument-frame directions into scene ENU,
            shape: (3, 3).
        materials: Every material the scene declares, matched to a voxel by
            :attr:`loader.Material.id`.
        specular_scales: First-surface normalization for each entry of
            ``materials``.
        wavelength: Wavelengths the channel response is sampled at, in microns,
            shape: (W,). W: number of spectral samples.
        weights: Band-integration weight per wavelength, shape: (W,).
    """

    instrument: loader.Instrument
    position: np.ndarray
    rotation: torch.Tensor
    materials: list[loader.Material]
    specular_scales: list[float]
    wavelength: torch.Tensor
    weights: torch.Tensor

    @classmethod
    def from_dirsig(
        cls,
        platform_path: Path,
        motion_path: Path,
        material_path: Path,
        spectral_step: float = 0.01,
        device: str = DEVICE,
    ) -> Sensor:
        """Read the optics, the pose and the materials from DIRSIG scene files.

        ``demo.platform`` declares a 0.400-0.800 um bandpass and a single
        rectangular "Pan" channel spanning all of it, so the channel response is
        sampled as a boxcar over a grid of that step.

        Args:
            platform_path: Path to the ``.platform`` file, giving the optics and
                the focal plane.
            motion_path: Path to the ``.ppd`` file, giving the platform's pose.
            material_path: Path to the ``.mat`` file, and through it the
                emissivity curves and BRDF fits it references.
            spectral_step: Spacing of the wavelength grid the channel response
                and every spectrum are sampled on, in microns.
            device: Torch device to build everything on.

        Returns:
            The sensor.
        """
        instrument = loader.load_platform(platform_path)
        position, orientation = loader.load_platform_motion(motion_path)

        # Instrument frame -> platform frame -> scene ENU, composed as matrices.
        rotation = geometry.euler_to_mat(
            torch.as_tensor(orientation, dtype=torch.float32, device=device)
        ) @ geometry.euler_to_mat(
            torch.as_tensor(
                instrument.mount_rotation, dtype=torch.float32, device=device
            )
        )

        # Sampled and windowed in float64: at the ends of the bandpass the
        # boxcar edge falls exactly on a sample, so which side of it that sample
        # lands on is decided by the last bit of the comparison.
        low, high = instrument.bandpass
        grid = np.arange(low, high + 1e-9, spectral_step)
        window = np.zeros_like(grid)
        for channel in instrument.channels:
            window[np.abs(grid - channel.center) <= 0.5 * channel.width] = 1.0
        wavelength = torch.as_tensor(grid, dtype=torch.float32, device=device)
        response = torch.as_tensor(window, dtype=torch.float32, device=device)

        materials = list(
            loader.load_materials(
                material_path, float(np.mean(instrument.bandpass))
            ).values()
        )
        return cls(
            instrument=instrument,
            position=position,
            rotation=rotation,
            materials=materials,
            specular_scales=[specular_scale(m) for m in materials],
            wavelength=wavelength,
            weights=radiometry.band_weights(wavelength, response),
        )


def specular_scale(material: loader.Material) -> float:
    """Normalization for a material's first-surface lobe.

    Args:
        material: The material to scale, specular or not.

    Returns:
        The multiplier that reproduces the material's declared DHR, or 1.0 for a
        Lambertian material, which has no first-surface term to scale.
    """
    if not material.is_specular:
        return 1.0
    assert material.dhr is not None
    return radiometry.dhr_specular_scale(material.dhr, **material.brdf)


def shade(
    hit: trace.Hit,
    material: torch.Tensor,
    sunlit: torch.Tensor,
    view: torch.Tensor,
    toward_sun: torch.Tensor,
    elevation: float,
    sensor: Sensor,
    shadows: bool = True,
) -> torch.Tensor:
    """Compute band-integrated radiance toward the sensor for every ray.

    Direct sunlight is scattered by the material's BRDF, the shadow mask gates
    it, and a uniform sky adds a Lambertian term weighted by how much of the
    hemisphere the surface can see.

    Args:
        hit: Where each ray met the scene. N: number of rays.
        material: ``usemtl`` id of the surface at each hit, -1 where the ray
            escaped, shape: (N,).
        sunlit: True where the sun reaches the surface, shape: (N,), dtype bool.
        view: Unit direction from each surface point back toward the sensor,
            shape: (N, 3).
        toward_sun: Unit direction from the scene toward the sun, shape: (3,).
        elevation: Solar elevation in degrees above the horizon.
        sensor: Supplies the material table and the channel weighting.
        shadows: Whether the shadow mask gates the direct beam.

    Returns:
        Radiance in ``W / (m^2 sr)`` integrated over the channel, shape: (N,).
    """
    direct, diffuse = radiometry.solar_spectrum(sensor.wavelength, elevation)

    cos_incident = (hit.normal * toward_sun).sum(-1)
    cos_reflected = (hit.normal * view).sum(-1)
    half = torch.nn.functional.normalize(toward_sun + view, dim=-1)
    cos_half = (hit.normal * half).sum(-1)
    cos_bistatic = (half * toward_sun).sum(-1)
    lit = (cos_incident > 0.0) & (cos_reflected > 0.0) & hit.hit
    if shadows:
        lit &= sunlit

    # How much of the sky hemisphere the surface sees, for the ambient term.
    sky_view = (0.5 * (1.0 + hit.normal[:, 2])).clamp(0.0, 1.0)

    radiance = torch.zeros(len(hit.point), device=hit.point.device)
    for index, entry in enumerate(sensor.materials):
        selected = material == entry.id
        if not bool(selected.any()):
            continue
        wl, value = entry.reflectance
        reflectance = radiometry.resample(
            wl.to(DEVICE), value.to(DEVICE), sensor.wavelength
        ).to(torch.float32)

        if not entry.is_specular:
            # Lambertian: SPECULAR_FRACTION is 0 for both ClassicEmissivity
            # materials in this scene.
            brdf = (reflectance / torch.pi)[None, :].expand(int(selected.sum()), -1)
        else:
            # The material's DHR is the first-surface share of the total
            # reflectance; whatever is left scatters diffusely and carries the
            # material's color.
            assert entry.dhr is not None
            lobe = radiometry.beard_maxwell(
                cos_incident[selected],
                cos_reflected[selected],
                cos_half[selected],
                cos_bistatic[selected],
                specular_scale=sensor.specular_scales[index],
                **entry.brdf,
            )
            body = ((reflectance - entry.dhr).clamp(min=0.0) / torch.pi)[None, :]
            brdf = lobe[:, None] + body

        beam = (
            brdf
            * direct[None, :]
            * (cos_incident[selected] * lit[selected]).clamp(min=0.0)[:, None]
        )
        sky = (
            (reflectance / torch.pi)[None, :]
            * diffuse[None, :]
            * sky_view[selected][:, None]
        )
        radiance[selected] = ((beam + sky) * sensor.weights[None, :]).sum(-1)

    return torch.where(hit.hit, radiance, 0.0)


def simulate(
    scene: Scene,
    sensor: Sensor,
    azimuth: float,
    elevation: float,
    shadows: bool = True,
) -> np.ndarray:
    """Render one full-resolution frame of the focal plane.

    Args:
        scene: The voxelized scene to trace against.
        sensor: The instrument to trace from.
        azimuth: Solar azimuth in degrees, clockwise from north.
        elevation: Solar elevation in degrees above the horizon.
        shadows: Whether to trace shadow rays.

    Returns:
        Band-integrated radiance in ``W / (m^2 sr)``, shape: (H, W). H:
        detector rows, y element count. W: columns, x element count.
    """
    instrument = sensor.instrument
    origins, directions = trace.rays_from_pinhole(
        instrument.focal_length,
        instrument.x_count,
        instrument.y_count,
        instrument.x_spacing * (-1.0 if instrument.x_flip else 1.0),
        instrument.y_spacing * (-1.0 if instrument.y_flip else 1.0),
        torch.as_tensor(sensor.position, dtype=torch.float32, device=DEVICE),
        sensor.rotation,
    )

    hit = trace.first_hit(scene.topology, scene.normals, origins, directions)
    toward_sun = radiometry.sun_direction(azimuth, elevation).to(DEVICE)
    sunlit = ~trace.shadowed(scene.topology, hit.point, hit.normal, toward_sun)

    radiance = shade(
        hit,
        material=torch.where(hit.hit, scene.materials.jdata[hit.voxel], -1),
        sunlit=sunlit,
        view=-directions,
        toward_sun=toward_sun,
        elevation=elevation,
        sensor=sensor,
        shadows=shadows,
    )
    return radiance.reshape(instrument.y_count, instrument.x_count).cpu().numpy()


def two_sigma_scale(image: np.ndarray, sigmas: float = 2.0) -> np.ndarray:
    """Scale a high dynamic range image the way DIRSIG's viewer does.

    The demo's own README points out that min/max scaling hides everything once
    a glint saturates the range, and that its "two sigma" scaling is what makes
    the scene legible. This is that: clip to the mean plus or minus a couple of
    standard deviations, then stretch.

    Args:
        image: Radiance image, shape: (H, W). H: detector rows. W: columns.
        sigmas: Half-width of the retained range, in standard deviations.

    Returns:
        Display image in ``[0, 255]``, shape: (H, W, 3), dtype uint8.
    """
    mean = float(image.mean())
    deviation = float(image.std())
    low = max(image.min(), mean - sigmas * deviation)
    high = min(image.max(), mean + sigmas * deviation)
    scaled = np.clip((image - low) / max(high - low, 1e-12), 0.0, 1.0)
    return np.repeat((255.0 * scaled).astype(np.uint8)[:, :, None], 3, axis=2)


def serve(
    scene: Scene,
    sensor: Sensor,
    capture_time: dt.datetime = CAPTURE_TIME,
    latitude: float = CAPTURE_LATITUDE,
    longitude: float = CAPTURE_LONGITUDE,
    port: int = 8080,
) -> None:
    """Show the voxels and the simulated focal plane in a viser page.

    Blocks until the process is interrupted.

    Args:
        scene: The voxelized scene, drawn as colored point clouds.
        sensor: The instrument to simulate with.
        capture_time: When the scene is being imaged, which with the
            geolocation sets where the sun sliders start.
        latitude: Scene latitude in degrees north.
        longitude: Scene longitude in degrees east.
        port: TCP port for the viser web interface.
    """
    vis = visualization.Visualizer(host="0.0.0.0", port=port)
    server = vis.server
    vis.add_grid_batch(
        "/voxels",
        scene.topology,
        scene.normals,
        visualization.normal_colormap(),
        point_size=scene.voxel_sizes * 0.5,
        names=scene.names,
    )

    position = pvlib.solarposition.get_solarposition(
        pd.DatetimeIndex([capture_time]), latitude, longitude
    )
    sun = (float(position["azimuth"].iloc[0]), float(position["elevation"].iloc[0]))
    print(f"sun at capture time: azimuth {sun[0]:.1f} deg, elevation {sun[1]:.1f} deg")

    with server.gui.add_folder("Sun"):
        azimuth_slider = server.gui.add_slider(
            "Azimuth (deg)", min=0.0, max=360.0, step=1.0, initial_value=sun[0]
        )
        elevation_slider = server.gui.add_slider(
            "Elevation (deg)", min=0.0, max=90.0, step=1.0, initial_value=sun[1]
        )
        shadow_toggle = server.gui.add_checkbox("Shadows", initial_value=True)
        reset_button = server.gui.add_button("Reset to capture time")
        resimulate_button = server.gui.add_button("Resimulate")

    status = server.gui.add_markdown("")
    image_handle = server.gui.add_image(
        np.zeros((sensor.instrument.y_count, sensor.instrument.x_count, 3), np.uint8),
        label="Simulated focal plane",
        format="png",
    )

    def resimulate() -> None:
        """Re-render the focal plane for the sliders' current sun angle."""
        azimuth, elevation = float(azimuth_slider.value), float(elevation_slider.value)
        status.content = (
            f"Simulating at azimuth {azimuth:.0f}, elevation {elevation:.0f}..."
        )
        start = time.perf_counter()
        image = simulate(
            scene, sensor, azimuth, elevation, shadows=bool(shadow_toggle.value)
        )
        image_handle.image = two_sigma_scale(image)
        server.scene.add_point_cloud(
            "/sun",
            points=(200.0 * radiometry.sun_direction(azimuth, elevation))
            .unsqueeze(0)
            .numpy(),
            colors=np.array([[255, 240, 120]], np.uint8),
            point_size=6.0,
            point_shape="circle",
        )
        status.content = (
            f"azimuth {azimuth:.0f} deg, elevation {elevation:.0f} deg  \n"
            f"radiance {image.min():.2f} to {image.max():.2f} W/(m^2 sr)  \n"
            f"{time.perf_counter() - start:.2f} s"
        )

    @resimulate_button.on_click
    def _(_event) -> None:
        resimulate()

    @reset_button.on_click
    def _(_event) -> None:
        azimuth_slider.value, elevation_slider.value = sun
        resimulate()

    resimulate()
    print(f"viser running at http://localhost:{server.get_port()} (ctrl-c to exit)")
    while True:
        time.sleep(1.0)


def main() -> None:
    """Voxelize the car, read the sensor, and serve the simulation in viser."""
    scene = Scene.from_obj(OBJ_FILE)
    print(
        f"{OBJ_FILE.stem}: {len(scene.names)} groups, "
        f"{len(scene.normals.jdata)} voxels, "
        f"{1000 * float(scene.voxel_sizes.min()):.1f}-"
        f"{1000 * float(scene.voxel_sizes.max()):.1f} mm"
    )

    sensor = Sensor.from_dirsig(PLATFORM_FILE, MOTION_FILE, MATERIAL_FILE)
    instrument = sensor.instrument
    ifov = instrument.x_spacing / instrument.focal_length
    altitude = sensor.position[2] - scene.ground_z
    print(
        f"sensor: {instrument.x_count}x{instrument.y_count} elements, "
        f"{1000 * instrument.x_spacing:.1f} um pitch, f = {instrument.focal_length:.1f} mm, "
        f"bandpass {instrument.bandpass[0]:.3f}-{instrument.bandpass[1]:.3f} um\n"
        f"  IFOV {1e6 * ifov:.1f} urad, GSD {100 * ifov * altitude:.2f} cm at "
        f"{altitude:.0f} m, materials "
        + ", ".join(m.name[:24] for m in sensor.materials)
    )

    serve(scene, sensor)


if __name__ == "__main__":
    main()
