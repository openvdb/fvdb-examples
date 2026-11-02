"""The optical physics this example simulates.

Sunlight reaching the ground, how a surface scatters it, and how a spectral
curve is resampled and integrated against a channel. None of this belongs to
the toolbox, which is deliberately modality-free; it is what makes *this*
example a reflective-band optical simulation rather than some other kind.
"""

import math

import torch

# Solar constant, band-integrated irradiance at the top of the atmosphere, W/m^2.
SOLAR_CONSTANT = 1361.0
# Effective blackbody temperature of the sun's photosphere, in kelvin.
SOLAR_TEMPERATURE = 5778.0
# Angstrom turbidity parameters for the aerosol optical depth.
AEROSOL_BETA = 0.05
AEROSOL_ALPHA = 1.3


def sun_direction(azimuth: float, elevation: float) -> torch.Tensor:
    """Unit vector pointing at the sun, in scene ENU coordinates.

    Args:
        azimuth: Solar azimuth in degrees, clockwise from north.
        elevation: Solar elevation in degrees above the horizon.

    Returns:
        Unit direction from the scene toward the sun, shape: (3,).
    """
    az = math.radians(azimuth)
    el = math.radians(elevation)
    return torch.tensor(
        [
            math.sin(az) * math.cos(el),
            math.cos(az) * math.cos(el),
            math.sin(el),
        ],
        dtype=torch.float32,
    )


def solar_spectrum(
    wavelength: torch.Tensor, elevation: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct and diffuse solar spectral irradiance at the ground.

    The extraterrestrial spectrum is a 5778 K blackbody normalized to the solar
    constant. It is attenuated by Rayleigh scattering and an Angstrom aerosol
    term along the Kasten-Young air mass path; the diffuse sky term takes half
    of what Rayleigh scattering removes, the standard first-order approximation
    that also gives the sky its blue.

    Args:
        wavelength: Wavelengths to evaluate, in microns, shape: (W,). W: number
            of wavelengths.
        elevation: Solar elevation in degrees above the horizon.

    Returns:
        A pair of spectral irradiances in ``W / (m^2 um)``: the direct beam
        measured normal to itself, shape: (W,), and the diffuse sky irradiance
        on a horizontal surface, shape: (W,).
    """
    if elevation <= 0.0:
        return torch.zeros_like(wavelength), torch.zeros_like(wavelength)

    micron = wavelength * 1e-6
    radiance = 1.0 / (
        micron**5 * (torch.exp(0.01438777 / (micron * SOLAR_TEMPERATURE)) - 1.0)
    )

    # Normalize the full Planck integral to the solar constant.
    full = torch.linspace(0.15, 5.0, 4000, device=wavelength.device) * 1e-6
    total = torch.trapezoid(
        1.0 / (full**5 * (torch.exp(0.01438777 / (full * SOLAR_TEMPERATURE)) - 1.0)),
        full * 1e6,
    )
    top_of_atmosphere = SOLAR_CONSTANT * radiance / total

    # Kasten-Young (1989) relative air mass.
    sine = math.sin(math.radians(elevation))
    air_mass: float = 1.0 / (sine + 0.50572 * (elevation + 6.07995) ** (-1.6364))

    rayleigh = torch.exp(-air_mass * 0.008735 * wavelength**-4.08)
    aerosol = torch.exp(-air_mass * AEROSOL_BETA * wavelength**-AEROSOL_ALPHA)

    direct = top_of_atmosphere * rayleigh * aerosol
    diffuse = 0.5 * top_of_atmosphere * sine * (1.0 - rayleigh) * aerosol
    return direct, diffuse


def beard_maxwell(
    cos_incident: torch.Tensor,
    cos_reflected: torch.Tensor,
    cos_half: torch.Tensor,
    cos_bistatic: torch.Tensor,
    n: float,
    k: float,
    bias: float,
    sigma: float,
    tau: float,
    omega: float,
    rho_d: float,
    rho_v: float,
    specular_scale: float = 1.0,
) -> torch.Tensor:
    """Evaluate the NEF Beard-Maxwell BRDF.

    Implements the Westlund-Meyer (Graphics Interface 2002) equation 10:

        rho = R(beta)/R(0) * rho_fs(theta_h) * cos^2(theta_h)
              / (cos(theta_i) cos(theta_r)) * SO
            + rho_d + 2 rho_v / (cos(theta_i) + cos(theta_r))

    where ``rho_fs`` is a Gaussian in half-vector tilt and SO is a
    shadowing/obscuration factor.

    Args:
        cos_incident: Cosine between the surface normal and the direction to
            the source, shape: (N,). N: number of samples.
        cos_reflected: Cosine between the surface normal and the direction to
            the sensor, shape: (N,).
        cos_half: Cosine between the surface normal and the half vector
            (microfacet tilt), shape: (N,).
        cos_bistatic: Cosine between the incident direction and the half
            vector (angle of incidence onto the microfacet), shape: (N,).
        n: Real part of the complex index of refraction.
        k: Imaginary part (extinction coefficient) of the index.
        bias: Amplitude of the Gaussian microfacet orientation distribution.
        sigma: Width of that Gaussian, in radians of half-vector tilt.
        tau: Shadowing decay constant, in degrees of bistatic angle.
        omega: Obscuration constant, in degrees of half-vector tilt.
        rho_d: Lambertian volumetric reflectance.
        rho_v: Directional-diffuse volumetric reflectance.
        specular_scale: Multiplier on the first-surface term, used to match
            the material's declared DHR.

    Returns:
        BRDF value in inverse steradians, shape: (N,).
    """
    theta_half = torch.arccos(cos_half.clamp(-1.0, 1.0))
    beta = torch.arccos(cos_bistatic.clamp(-1.0, 1.0))

    orientation = bias * torch.exp(-((theta_half / sigma) ** 2))

    tilt = torch.rad2deg(theta_half) / omega
    shadow = (1.0 + tilt * torch.exp(-2.0 * torch.rad2deg(beta) / tau)) / (1.0 + tilt)

    fresnel_ratio = fresnel(cos_bistatic, n, k) / fresnel(
        torch.ones_like(cos_bistatic), n, k
    )
    denominator = (cos_incident * cos_reflected).clamp(min=1e-4)
    first_surface = (
        specular_scale
        * fresnel_ratio
        * orientation
        * cos_half.clamp(min=0.0) ** 2
        / denominator
        * shadow
    )

    volumetric = rho_d + 2.0 * rho_v / (cos_incident + cos_reflected).clamp(min=1e-4)
    return first_surface + volumetric


def fresnel(cos_angle: torch.Tensor, n: float, k: float) -> torch.Tensor:
    """Unpolarized Fresnel reflectance of a surface with complex index n - ik.

    Averages the two polarization states, which is what an unpolarized source
    and a sensor with ``<polarizer type="none"/>`` between them see.

    Args:
        cos_angle: Cosine of the angle of incidence onto the microfacet,
            shape: (N,). N: number of samples.
        n: Real part of the index of refraction.
        k: Extinction coefficient.

    Returns:
        Reflectance in ``[0, 1]``, shape: (N,).
    """
    # Floored away from zero: the parallel branch below divides by cos^2, which
    # at exactly grazing incidence would produce inf / inf.
    cos_sq = cos_angle.clamp(0.0, 1.0) ** 2 + 1e-12
    sin_sq = (1.0 - cos_sq).clamp(min=0.0)

    # Solve for the real and imaginary parts of the transmitted cosine.
    common = n * n - k * k - sin_sq
    radical = torch.sqrt((common**2 + (2.0 * n * k) ** 2).clamp(min=0.0))
    a_sq = (0.5 * (radical + common)).clamp(min=0.0)
    b_sq = (0.5 * (radical - common)).clamp(min=0.0)
    a = torch.sqrt(a_sq)
    cos = torch.sqrt(cos_sq)

    perpendicular = (a_sq + b_sq - 2.0 * a * cos + cos_sq) / (
        a_sq + b_sq + 2.0 * a * cos + cos_sq
    )
    parallel = perpendicular * (
        (a_sq + b_sq - 2.0 * a * cos * torch.sqrt(sin_sq) + sin_sq * sin_sq / cos_sq)
        / (a_sq + b_sq + 2.0 * a * cos * torch.sqrt(sin_sq) + sin_sq * sin_sq / cos_sq)
    )
    return (0.5 * (perpendicular + parallel)).clamp(0.0, 1.0)


# Zenith and azimuth samples used to integrate a BRDF over the hemisphere when
# solving for the DHR normalization. The specular lobes here are a few degrees
# wide, so the zenith grid has to be fine to resolve them.
DEVICE = "cuda"
DHR_ZENITH_SAMPLES = 2048
DHR_AZIMUTH_SAMPLES = 256


def dhr_specular_scale(
    dhr: float,
    n: float,
    k: float,
    bias: float,
    sigma: float,
    tau: float,
    omega: float,
    rho_d: float = 0.0,
    rho_v: float = 0.0,
) -> float:
    """Solve for the first-surface scale that reproduces a material's DHR.

    A fit file declares a directional hemispherical reflectance but not how the
    orientation Gaussian is normalized, so the lobe's absolute height is
    undetermined. Integrating the unscaled first-surface term over the reflected
    hemisphere at normal incidence and dividing the declared DHR by the result
    recovers it, which is what NEF equation 11 does spectrally.

    The integral is taken at normal incidence, where the lobe sits around the
    normal and the quadrature resolves it.

    Args:
        dhr: Declared directional hemispherical reflectance of the first-surface
            term.
        n: Real part of the complex index of refraction.
        k: Imaginary part (extinction coefficient) of the index.
        bias: Amplitude of the Gaussian microfacet orientation distribution.
        sigma: Width of that Gaussian, in radians of half-vector tilt.
        tau: Shadowing decay constant, in degrees of bistatic angle.
        omega: Obscuration constant, in degrees of half-vector tilt.
        rho_d: Ignored. The DHR quantifies the first-surface lobe alone, so the
            volumetric terms are held at zero for the integral; accepted only so
            a material's full parameter set can be splatted in.
        rho_v: Ignored, as ``rho_d``.

    Returns:
        Multiplier for the first-surface term of :func:`beard_maxwell`.
    """
    del rho_d, rho_v

    zenith = (
        torch.arange(DHR_ZENITH_SAMPLES, device=DEVICE, dtype=torch.float64) + 0.5
    ) * (0.5 * math.pi / DHR_ZENITH_SAMPLES)
    azimuth = (
        torch.arange(DHR_AZIMUTH_SAMPLES, device=DEVICE, dtype=torch.float64) + 0.5
    ) * (2.0 * math.pi / DHR_AZIMUTH_SAMPLES)
    grid_zenith, grid_azimuth = torch.meshgrid(zenith, azimuth, indexing="ij")

    # Normal incidence: the source is straight up, so theta_i = 0.
    incident = torch.tensor([0.0, 0.0, 1.0], device=DEVICE, dtype=torch.float64)
    reflected = torch.stack(
        [
            torch.sin(grid_zenith) * torch.cos(grid_azimuth),
            torch.sin(grid_zenith) * torch.sin(grid_azimuth),
            torch.cos(grid_zenith),
        ],
        dim=-1,
    ).reshape(-1, 3)
    half = torch.nn.functional.normalize(reflected + incident, dim=-1)

    unscaled = beard_maxwell(
        cos_incident=torch.ones(len(reflected), device=DEVICE, dtype=torch.float64),
        cos_reflected=reflected[:, 2],
        cos_half=half[:, 2],
        cos_bistatic=(half * incident).sum(-1),
        n=n,
        k=k,
        bias=bias,
        sigma=sigma,
        tau=tau,
        omega=omega,
        rho_d=0.0,
        rho_v=0.0,
    )

    # Integrate f * cos(theta_r) over the hemisphere.
    solid_angle = (
        torch.sin(grid_zenith).reshape(-1)
        * (0.5 * math.pi / DHR_ZENITH_SAMPLES)
        * (2.0 * math.pi / DHR_AZIMUTH_SAMPLES)
    )
    return dhr / float((unscaled * reflected[:, 2] * solid_angle).sum())


def resample(x: torch.Tensor, y: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
    """Linearly resample a tabulated curve, holding its end values beyond it.

    Args:
        x: Sample positions of the curve, ascending, shape: (S,). S: number of
            tabulated samples.
        y: Curve value at each, shape: (S,).
        at: Positions to evaluate at, shape: (W,). W: number of query points.

    Returns:
        Curve value at each query position, shape: (W,).
    """
    i = torch.searchsorted(x.contiguous(), at.contiguous()).clamp(1, len(x) - 1)
    t = ((at - x[i - 1]) / (x[i] - x[i - 1])).clamp(0.0, 1.0)
    return y[i - 1] + t * (y[i] - y[i - 1])


def band_weights(wavelength: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    """Trapezoidal quadrature weights for integrating a spectrum over a channel.

    Args:
        wavelength: Wavelengths of the spectral samples, ascending, in microns,
            shape: (W,). W: number of spectral samples.
        response: Channel response at each wavelength, shape: (W,).

    Returns:
        Weight per sample, so that ``(spectrum * weights).sum(-1)`` is the
        band-integrated quantity, shape: (W,).
    """
    delta = torch.diff(wavelength)
    width = torch.cat([delta[:1], (wavelength[2:] - wavelength[:-2]) / 2, delta[-1:]])
    return response * width
