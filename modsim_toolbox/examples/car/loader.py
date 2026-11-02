"""Load Wavefront OBJ files and DIRSIG scene files into GPU tensors.

:class:`Material`, :class:`Channel` and :class:`Instrument` are the data
objects those files are loaded into.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch


@dataclass(frozen=True)
class Material:
    """One ``MATERIAL_ENTRY`` of a DIRSIG ``.mat`` file.

    Lambertian ``ClassicEmissivity`` materials have all Beard-Maxwell fields
    set to ``None``; ``ShellTarget`` materials carry the full set. The fit
    nearest the sensor's band is used, since the two visible-band entries in
    these files are identical.

    Attributes:
        id: The ``usemtl`` id the geometry refers to this material by.
        name: Human readable material name.
        reflectance: Spectral reflectance as ``(wavelength, value)`` tensors,
            derived from the emissivity curve as ``1 - emissivity``.
            Wavelengths in microns, shape: (S,) each. S: spectral samples.
        n: Real part of the complex index of refraction, ``None`` for
            Lambertian materials.
        k: Imaginary part (extinction coefficient) of the index.
        dhr: Directional hemispherical reflectance of the first-surface term,
            which sets its absolute scale.
        bias: Amplitude of the Gaussian microfacet orientation distribution.
        sigma: Width of that Gaussian, in radians of half-vector tilt.
        tau: Beard-Maxwell shadowing decay constant, in degrees of bistatic
            angle.
        omega: Beard-Maxwell obscuration constant, in degrees of half-vector
            tilt.
        rho_d: Lambertian volumetric reflectance.
        rho_v: Directional-diffuse volumetric reflectance.
    """

    id: int
    name: str
    reflectance: tuple[torch.Tensor, torch.Tensor]
    n: float | None = None
    k: float | None = None
    dhr: float | None = None
    bias: float | None = None
    sigma: float | None = None
    tau: float | None = None
    omega: float | None = None
    rho_d: float | None = None
    rho_v: float | None = None

    @property
    def is_specular(self) -> bool:
        """Whether this material has a first-surface (glinting) lobe."""
        return self.n is not None

    @property
    def brdf(self) -> dict[str, float]:
        """The Beard-Maxwell parameters, ready to splat into the BRDF calls.

        Returns:
            The eight lobe and volumetric parameters keyed by the argument names
            :func:`radiometry.beard_maxwell` and
            :func:`radiometry.dhr_specular_scale` use. Empty for a Lambertian
            material.
        """
        keys = ("n", "k", "bias", "sigma", "tau", "omega", "rho_d", "rho_v")
        return {key: value for key in keys if (value := getattr(self, key)) is not None}


@dataclass(frozen=True)
class Channel:
    """One spectral channel of the focal plane's response.

    Attributes:
        name: Channel name as it appears in the platform file, e.g. ``"Pan"``.
        center: Center wavelength, in microns.
        width: Full width of the channel's passband, in microns.
    """

    name: str
    center: float
    width: float


@dataclass(frozen=True)
class Instrument:
    """A generic DIRSIG instrument: its optics, focal plane and mounting.

    Attributes:
        focal_length: Distance from the entrance pupil to the focal plane, in
            millimeters. DIRSIG declares no unit for it; millimeters is the
            format's convention.
        x_count: Number of detector elements along the focal plane's x axis.
        y_count: Number of elements along its y axis.
        x_spacing: Center-to-center element pitch along x, in millimeters.
        y_spacing: Pitch along y, in millimeters.
        x_flip: Whether the x element index runs opposite to focal plane x.
        y_flip: Whether the y element index runs opposite to focal plane y.
        mount_rotation: Euler xyz rotation of the mount relative to the
            platform, in radians, shape: (3,).
        bandpass: Wavelength ``(minimum, maximum)`` the focal plane responds
            over, in microns.
        channels: The focal plane's spectral channels, in file order.
    """

    focal_length: float
    x_count: int
    y_count: int
    x_spacing: float
    y_spacing: float
    x_flip: bool
    y_flip: bool
    mount_rotation: np.ndarray
    bandpass: tuple[float, float]
    channels: tuple[Channel, ...]


def load_emissivity(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the first curve of a DIRSIG ``.ems`` emissivity file.

    The format is a count of curves, then 91 further header lines, then one or
    more ``CURVE_BEGIN`` blocks of ``wavelength emissivity`` pairs. Files may
    hold many curves -- ``grass_mixed.ems`` has 300 realizations of a grass
    mixture -- and only the first is used here, which is the simplification
    this renderer makes in place of DIRSIG's per-facet material sampling.

    Args:
        path: Path to the ``.ems`` file.

    Returns:
        A pair ``(wavelength, emissivity)`` with wavelengths in microns,
        shape: (S,) each. S: number of samples in the curve.
    """
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "CURVE_BEGIN")

    wavelength: list[float] = []
    value: list[float] = []
    for line in lines[start + 1 :]:
        if line.strip() == "CURVE_BEGIN":
            break
        fields = line.split()
        if len(fields) == 2:
            wavelength.append(float(fields[0]))
            value.append(float(fields[1]))

    return (
        torch.tensor(wavelength, dtype=torch.float64),
        torch.tensor(value, dtype=torch.float64),
    )


def _payloads(text: str, keyword: str) -> list[str]:
    """Collect the payload of every line opening with ``keyword``.

    Args:
        text: Block of OBJ text to scan.
        keyword: Line keyword to match, such as ``v`` or ``f``.

    Returns:
        The remainder of each matching line, keyword and its space stripped.
    """
    prefix = keyword + " "
    return [
        line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)
    ]


def _float_rows(lines: list[str], device: str) -> torch.Tensor:
    """Parse whitespace-separated ``xyz`` payloads into one tensor.

    Trailing columns (vertex colors, the ``w`` of a rational position) are
    dropped. Parsing is batched through numpy so a whole table is converted at
    C speed and reaches the device in a single transfer.

    Args:
        lines: Payload of each ``v`` or ``vn`` line.
        device: Torch device to place the result on.

    Returns:
        Parsed rows, shape: (N, 3). N: number of lines.
    """
    if not lines:
        return torch.empty(0, 3, device=device)
    return torch.from_numpy(
        np.array([line.split()[:3] for line in lines], dtype=np.float32)
    ).to(device)


def load_obj(
    path: Path, device: str = "cuda"
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[str],
    list[int],
]:
    """Load a Wavefront OBJ, jagged over its ``g`` groups.

    A group is the unit of authoring in a modelling package and the unit of
    simulation here, so it is the jagged dimension: groups hold wildly
    different vertex counts. A ``g`` line opens a group and everything up to
    the next one belongs to it, so the file is split on those and read one
    block at a time, each block yielding the group's own tensors.

    OBJ indices are global and cumulative, and a file may declare its vertices
    all at once up front or block by block, so the ``v`` and ``vn`` tables are
    accumulated as the blocks are read and a group is resolved against
    everything declared so far. A group is assumed to reference only its own
    vertices, so its slice of the table is the span between the lowest and
    highest position index its faces touch, and rebasing the faces onto that
    slice is all the reconciliation needed.

    Normals are not group-local: a block may reference a ``vn`` declared in an
    earlier one, so they stay indices into the accumulated table. They are also
    stored per face corner rather than per vertex, because the OBJ picks a
    ``vn`` per corner independently of the position — a hard crease reuses a
    position with a different normal on each side, which the corner represents
    directly instead of splitting the vertex to encode it. A corner with no
    ``vn`` takes the flat geometric normal of its own triangle.

    Args:
        path: Path to the ``.obj`` file.
        device: Torch device to place all tensors on.

    Returns:
        A five-tuple describing one group per ``g`` block that declares faces,
        in file order. The first three are lists of length G, one tensor per
        group, ready to hand to :class:`fvdb.JaggedTensor` once any further
        groups have been appended:

        - ``vertices``: Group-local world-space vertex positions, shape:
          (V_g, 3) each. G: number of groups; V_g: vertices in group g.
        - ``normals``: Per-corner normals, three independent vectors per face,
          shape: (F_g, 3, 3) each. F_g: triangles in group g. Not unit length;
          the OBJ's ``vn`` values are passed through as written.
        - ``faces``: Triangle corner indices into the group's own vertices,
          shape: (F_g, 3) each.
        - ``group_names``: Name of each group, length G.
        - ``material_ids``: ``usemtl`` id of each group, length G (``-1`` if
          absent).
    """
    # Anything before the first ``g`` is header and any tables the file
    # declares up front; each split that follows is one group's block, opening
    # with its name.
    preamble, *blocks = re.split(r"^g[ \t]+", path.read_text(), flags=re.MULTILINE)

    group_names: list[str] = []
    material_ids: list[int] = []
    group_vertices: list[torch.Tensor] = []
    group_normals: list[torch.Tensor] = []
    group_faces: list[torch.Tensor] = []

    # Positions and normals contributed by the preamble and every block read so
    # far, which is all the current group may reference.
    v_blocks = [_float_rows(_payloads(preamble, "v"), device)]
    vn_blocks = [_float_rows(_payloads(preamble, "vn"), device)]

    for block in blocks:
        name, _, body = block.partition("\n")
        v_blocks.append(_float_rows(_payloads(body, "v"), device))
        vn_blocks.append(_float_rows(_payloads(body, "vn"), device))

        v = torch.cat(v_blocks)  # (V_so_far, 3)
        # Zero row appended so the flat-normal branch below always has a row to
        # read, including for a file that declares no ``vn`` at all.
        vn = torch.cat(vn_blocks + [torch.zeros(1, 3, device=device)])

        # Corner indices into the file's tables, three corners per triangle.
        # Polygons are fan-triangulated about their first corner, and a corner
        # written without a ``vn`` gets index -1.
        positions: list[int] = []
        normal_refs: list[int] = []
        for face in _payloads(body, "f"):
            corner_v: list[int] = []
            corner_vn: list[int] = []
            for token in face.split():
                parts = token.split("/")
                corner_v.append(int(parts[0]) - 1)
                corner_vn.append(
                    int(parts[2]) - 1 if len(parts) > 2 and parts[2] else -1
                )
            for i in range(1, len(corner_v) - 1):
                for c in (0, i, i + 1):
                    positions.append(corner_v[c])
                    normal_refs.append(corner_vn[c])

        if not positions:
            continue

        v_index = torch.tensor(positions, dtype=torch.int64, device=device).reshape(
            -1, 3
        )  # (F_g, 3)
        vn_index = torch.tensor(normal_refs, dtype=torch.int64, device=device).reshape(
            -1, 3
        )  # (F_g, 3)

        # The group's own span of the table, with its faces rebased onto it.
        base = int(v_index.min())
        vertices = v[base : int(v_index.max()) + 1]  # (V_g, 3)
        faces = (v_index - base).int()  # (F_g, 3)

        # Flat geometric normal of each triangle, the fallback for corners the
        # file leaves without a ``vn``.
        corners = vertices[faces.long()]  # (F_g, 3, 3)
        face_normal = torch.nn.functional.normalize(
            torch.linalg.cross(
                corners[:, 1] - corners[:, 0],
                corners[:, 2] - corners[:, 0],
                dim=-1,
            ),
            dim=-1,
        )  # (F_g, 3)

        materials = _payloads(body, "usemtl")
        group_names.append(name.strip())
        material_ids.append(int(materials[-1]) if materials else -1)
        group_vertices.append(vertices)
        group_normals.append(
            torch.where(
                (vn_index >= 0)[..., None],
                vn[vn_index.clamp(min=0)],
                face_normal[:, None, :],
            )
        )  # (F_g, 3, 3)
        group_faces.append(faces)

    return group_vertices, group_normals, group_faces, group_names, material_ids


def ground_plane_mesh(
    half_extent: float = 20.0,
    z: float = 0.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a flat rectangular mesh at height ``z`` for voxelization.

    The returned tensors are a single geometry group shaped like one entry of
    what :func:`load_obj` returns, so they can be appended to its lists before
    constructing a :class:`fvdb.JaggedTensor`.

    Args:
        half_extent: Half-width of the square patch along each axis, in scene
            meters.
        z: Height of the plane in scene meters.
        device: Torch device to place all tensors on.

    Returns:
        A 3-tuple of:

        - ``vertices``: Corner positions, shape: (4, 3).
        - ``normals``: Per-corner normals aligned with ``faces``, all pointing
          up (+z), shape: (2, 3, 3). F: triangles (2).
        - ``faces``: Triangle corner indices into ``vertices``, shape: (2, 3).
    """
    e = half_extent
    vertices = torch.tensor(
        [[-e, -e, z], [e, -e, z], [e, e, z], [-e, e, z]],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device=device)
    up = torch.zeros(2, 3, 3, device=device)
    up[..., 2] = 1.0
    return vertices, up, faces


def _text(element: ElementTree.Element, tag: str) -> str:
    """Return the text of a required child element, asserting it is present."""
    value = element.findtext(tag)
    assert value is not None, f"Required element {tag!r} missing"
    return value


# DIRSIG declares spatial and angular units as attributes on the element that
# carries the values, so the scale factor is looked up rather than assumed.
_LENGTH_TO_MM = {"microns": 1e-3, "millimeters": 1.0, "meters": 1e3}
_LENGTH_TO_METERS = {"millimeters": 1e-3, "meters": 1.0, "kilometers": 1e3}
_ANGLE_TO_RADIANS = {"radians": 1.0, "degrees": float(np.pi / 180.0)}


def load_platform(path: Path) -> Instrument:
    """Read a DIRSIG ``.platform`` file's instrument description.

    Only the first enabled focal plane of the first instrument is read, which is
    all this demo's platform declares.

    Args:
        path: Path to the ``.platform`` file.

    Returns:
        The instrument, with every length converted to millimeters and every
        angle to radians.
    """
    root = ElementTree.parse(path).getroot()

    mount = root.find(".//mount/data")
    assert mount is not None
    mount_scale = _ANGLE_TO_RADIANS[mount.get("angularunits", "radians")]
    mount_rotation = mount_scale * np.array(
        [float(_text(mount, f"{axis}rotation")) for axis in "xyz"], dtype=np.float64
    )

    array = root.find(".//detectorarray")
    assert array is not None
    to_mm = _LENGTH_TO_MM[array.get("spatialunits", "microns")]

    bandpass = root.find(".//spectralresponse/bandpass")
    assert bandpass is not None
    return Instrument(
        focal_length=float(_text(root, ".//instrument/properties/focallength")),
        x_count=int(_text(array, "xelementcount")),
        y_count=int(_text(array, "yelementcount")),
        x_spacing=to_mm * float(_text(array, "xelementspacing")),
        y_spacing=to_mm * float(_text(array, "yelementspacing")),
        x_flip=bool(int(_text(array, "xflipaxis"))),
        y_flip=bool(int(_text(array, "yflipaxis"))),
        mount_rotation=mount_rotation,
        bandpass=(
            float(_text(bandpass, "minimum")),
            float(_text(bandpass, "maximum")),
        ),
        channels=tuple(
            Channel(
                name=channel.get("name", ""),
                center=float(_text(channel, "center")),
                width=float(_text(channel, "width")),
            )
            for channel in root.findall(".//channellist/channel")
        ),
    )


def load_platform_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the platform's pose from a DIRSIG ``.ppd`` file.

    Only the first entry is read: this demo's platform is static, so its single
    entry holds for every capture time.

    Args:
        path: Path to the ``.ppd`` file.

    Returns:
        A pair of ``(position, orientation)`` in scene ENU coordinates.
        ``position``: Platform location in meters, shape: (3,).
        ``orientation``: Euler xyz rotation in radians, shape: (3,).
    """
    data = ElementTree.parse(path).getroot().find("data")
    assert data is not None
    # The camera model below rotates instrument-frame directions straight into
    # scene ENU, which only holds for these two conventions.
    assert data.get("rotationframe") == "sceneenu"
    assert data.get("rotationorder") == "xyz"

    entry = data.find("entry")
    assert entry is not None
    point = entry.find("position/location/point")
    assert point is not None
    triple = entry.find("orientation/eulerangles/cartesiantriple")
    assert triple is not None
    position = _LENGTH_TO_METERS[data.get("spatialunits", "meters")] * np.array(
        [float(_text(point, axis)) for axis in "xyz"], dtype=np.float64
    )
    orientation = _ANGLE_TO_RADIANS[data.get("angularunits", "radians")] * np.array(
        [float(_text(triple, axis)) for axis in "xyz"], dtype=np.float64
    )
    return position, orientation


def load_materials(path: Path, wavelength: float) -> dict[int, Material]:
    """Read a DIRSIG ``.mat`` file and everything its entries reference.

    For ``ShellTarget`` entries the Beard-Maxwell parameters are read from the
    ``.fit`` file referenced by ``BRDF_FIT_FILE`` and inlined directly onto the
    :class:`Material`. The fit block whose ``LAMBDA`` is nearest
    ``wavelength`` is used; the two visible-band blocks in these files carry
    identical parameters so the spectral variation is left entirely to the
    reflectance curve.

    Args:
        path: Path to the ``.mat`` file. Emissivity and fit files are resolved
            relative to its directory, as ``demo.scene`` declares.
        wavelength: Band center in microns, used to select the nearest BRDF fit
            block.

    Returns:
        Every material in the file, keyed by its ``usemtl`` id.
    """

    def _number(block: str, key: str) -> float:
        m = re.search(rf"\b{key}\s*=\s*(\S+)", block)
        assert m is not None
        return float(m.group(1))

    result: dict[int, Material] = {}
    for block in re.findall(
        r"MATERIAL_ENTRY\s*\{(.*?)\n\}", path.read_text(), re.DOTALL
    ):
        emissivity_match = re.search(r"FILENAME\s*=\s*(\S+)", block) or re.search(
            r"EMISSIVITY_FILE\s*=\s*(\S+)", block
        )
        fit_match = re.search(r"BRDF_FIT_FILE\s*=\s*(\S+)", block)

        assert emissivity_match is not None
        emissivity_wl, emissivity_val = load_emissivity(
            path.parent / emissivity_match.group(1)
        )

        bm: dict[str, float] = {}
        if fit_match:
            fit_blocks: list[str] = re.findall(
                r"FIT_PARAMS\s*\{(.*?)\n\}",
                (path.parent / fit_match.group(1)).read_text(),
                re.DOTALL,
            )
            parsed = [
                {
                    "n": _number(b, "N"),
                    "k": _number(b, "K"),
                    "dhr": _number(b, "DHR"),
                    "bias": _number(b, "BIAS"),
                    "sigma": _number(b, "SIGMA"),
                    "tau": _number(b, "TAU"),
                    "omega": _number(b, "OMEGA"),
                    "rho_d": _number(b, "RHO_D"),
                    "rho_v": _number(b, "RHO_V"),
                    "_lambda": _number(b, "LAMBDA"),
                }
                for b in fit_blocks
            ]
            best = min(parsed, key=lambda f: abs(f["_lambda"] - wavelength))
            bm = {k: v for k, v in best.items() if k != "_lambda"}

        id_match = re.search(r"\bID\s*=\s*(\d+)", block)
        name_match = re.search(r"\bNAME\s*=\s*(.+)", block)
        assert id_match is not None and name_match is not None
        material = Material(
            id=int(id_match.group(1)),
            name=name_match.group(1).strip(),
            # Kirchhoff: an opaque surface reflects what it does not absorb.
            reflectance=(emissivity_wl, 1.0 - emissivity_val),
            **bm,
        )
        result[material.id] = material
    return result
