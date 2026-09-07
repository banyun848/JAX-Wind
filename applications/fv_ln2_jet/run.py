"""Direct finite-volume HITSZ liquid-nitrogen jet simulation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import NamedTuple


SMAGORINSKY_COEFFICIENT = 0.16

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True, slots=True)
class JetCase:
    cells: tuple[int, int, int]
    lengths: tuple[float, float, float]
    mapping_types: tuple[str, str, str]
    mapping_focus: tuple[float, float, float]
    mapping_strength: tuple[float, float, float]
    dt: float
    steps: int
    chunk_steps: int
    checkpoint_every: int
    ambient_temperature: float
    ambient_relative_humidity: float
    ambient_water_vapor: float
    pressure: float
    ambient_density: float
    dry_air_density: float
    ambient_heat_capacity: float
    ambient_gas_constant: float
    kinematic_viscosity: float
    scalar_diffusivity: float
    scalar_advection_scheme: str
    nitrogen_buoyancy_coefficient: float
    gravity: tuple[float, float, float]
    roughness: float
    source_mode: str
    fully_vaporized_within_source_cell: bool
    gas_inlet_radius: float
    gas_inlet_temperature: float
    subgrid_jet_enabled: bool
    subgrid_support_radius_cells: float
    subgrid_transition_width_cells: float
    subgrid_momentum_length_cells: float
    subgrid_turbulence_intensity: float
    subgrid_integral_scale_cells: float
    subgrid_correlation_time: float
    subgrid_transverse_ratio: float
    nozzle: tuple[float, float, float]
    radius: float
    speed: float
    mass_flow_rate: float
    vapor_quality: float
    jet_temperature: float
    liquid_density: float
    liquid_latent_heat: float
    ramp_time: float
    initial_diameter: float
    minimum_diameter: float
    maximum_diameter: float
    rosin_rammler_spread: float
    parcels_per_step: int
    maximum_parcels: int
    parcel_substeps: int
    cone_half_angle_degrees: float
    edge_speed_ratio: float
    edge_diameter_ratio: float
    profile_radius_m: tuple[float, ...]
    profile_axial_velocity_m_s: tuple[float, ...]
    profile_radial_velocity_m_s: tuple[float, ...]
    profile_d10_m: tuple[float, ...]
    output: Path
    gmg_tolerance: float
    gmg_presweeps: int
    gmg_postsweeps: int
    flow_formulation: str
    momentum_closure: str
    time_integration: str


class DifferentiableInletControl(NamedTuple):
    """Continuous physical controls carried as a JAX pytree."""

    gas_radius: object
    gas_temperature: object
    initial_diameter: object
    speed_scale: object
    edge_speed_ratio: object
    edge_diameter_ratio: object


class CryogenicState(NamedTuple):
    velocity: object
    pressure: object
    density: object
    momentum_tendency: object
    temperature: object
    temperature_tendency: object
    water_vapor: object
    water_vapor_tendency: object
    nitrogen: object
    nitrogen_density: object
    nitrogen_density_tendency: object
    liquid_water: object
    liquid_water_tendency: object
    ice_water: object
    ice_water_tendency: object
    parcels: object
    continuity_error: object
    time: object
    step: object


def load_case(path: str | Path) -> JetCase:
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    domain = document["domain"]
    time_table = document["time"]
    ambient = document["ambient"]
    walls = document["walls"]
    jet = document["jet"]
    source = document.get("source", {})
    numerics = document["numerics"]
    output = document["output"]
    cells = tuple(int(value) for value in domain["cells"])
    lengths = tuple(float(value) for value in domain["lengths_m"])
    mapping_table = domain.get("mapping", {})
    mapping_types = tuple(
        str(value).replace("_", "-")
        for value in mapping_table.get("types", ("uniform",) * 3)
    )
    mapping_focus = tuple(
        float(value)
        for value in mapping_table.get(
            "focus_m", tuple(0.5 * length for length in lengths)
        )
    )
    mapping_strength = tuple(
        float(value)
        for value in mapping_table.get("strength", (0.0,) * 3)
    )
    if len(cells) != 3 or len(lengths) != 3 or min(cells) <= 0:
        raise ValueError("domain cells and lengths must contain three positive values")
    if not all(
        len(values) == 3
        for values in (mapping_types, mapping_focus, mapping_strength)
    ):
        raise ValueError("domain mapping entries must contain three values")
    case = JetCase(
        cells=cells,
        lengths=lengths,
        mapping_types=mapping_types,
        mapping_focus=mapping_focus,
        mapping_strength=mapping_strength,
        dt=float(time_table["dt_seconds"]),
        steps=int(time_table["steps"]),
        chunk_steps=int(time_table["chunk_steps"]),
        checkpoint_every=int(time_table["checkpoint_every_steps"]),
        ambient_temperature=float(ambient["temperature_k"]),
        ambient_relative_humidity=float(ambient["relative_humidity"]),
        ambient_water_vapor=float(ambient["water_vapor_mixing_ratio"]),
        pressure=float(ambient["pressure_pa"]),
        ambient_density=float(ambient.get("density_kg_m3", 1.225)),
        dry_air_density=float(
            ambient.get(
                "dry_air_density_kg_m3",
                ambient.get("density_kg_m3", 1.225),
            )
        ),
        ambient_heat_capacity=float(
            ambient.get("heat_capacity_j_kg_k", 1005.0)
        ),
        ambient_gas_constant=float(
            ambient.get("gas_constant_j_kg_k", 287.05)
        ),
        kinematic_viscosity=float(
            ambient.get("kinematic_viscosity_m2_s", 1.5e-5)
        ),
        scalar_diffusivity=float(
            ambient.get("scalar_diffusivity_m2_s", 2.2e-5)
        ),
        scalar_advection_scheme=str(
            numerics.get("scalar_advection_scheme", "central")
        ).replace("_", "-"),
        nitrogen_buoyancy_coefficient=float(
            ambient.get("nitrogen_buoyancy_coefficient", 0.03398)
        ),
        gravity=tuple(
            float(value)
            for value in ambient.get("gravity_m_s2", (0.0, 0.0, -9.81))
        ),
        roughness=float(walls["roughness_length_m"]),
        source_mode=str(source.get("mode", "volume")),
        fully_vaporized_within_source_cell=bool(
            source.get("fully_vaporized_within_source_cell", False)
        ),
        gas_inlet_radius=float(
            source.get("gas_inlet_radius_m", jet["radius_m"])
        ),
        gas_inlet_temperature=float(
            source.get(
                "gas_inlet_temperature_k",
                jet.get("temperature_k", 77.34),
            )
        ),
        subgrid_jet_enabled=bool(source.get("subgrid_jet_enabled", False)),
        subgrid_support_radius_cells=float(
            source.get("subgrid_support_radius_cells", 2.5)
        ),
        subgrid_transition_width_cells=float(
            source.get("subgrid_transition_width_cells", 0.5)
        ),
        subgrid_momentum_length_cells=float(
            source.get("subgrid_momentum_length_cells", 4.0)
        ),
        subgrid_turbulence_intensity=float(
            source.get("subgrid_turbulence_intensity", 0.0)
        ),
        subgrid_integral_scale_cells=float(
            source.get("subgrid_integral_scale_cells", 4.0)
        ),
        subgrid_correlation_time=float(
            source.get("subgrid_correlation_time_seconds", 3.0e-4)
        ),
        subgrid_transverse_ratio=float(
            source.get("subgrid_transverse_ratio", 1.0)
        ),
        nozzle=tuple(float(value) for value in jet["position_m"]),
        radius=float(jet["radius_m"]),
        speed=float(jet["speed_m_s"]),
        mass_flow_rate=float(jet["mass_flow_rate_kg_s"]),
        vapor_quality=float(jet["vapor_quality"]),
        jet_temperature=float(jet.get("temperature_k", 77.34)),
        liquid_density=float(jet.get("liquid_density_kg_m3", 806.11)),
        liquid_latent_heat=float(
            jet.get("liquid_latent_heat_j_kg", 199_180.0)
        ),
        ramp_time=float(jet.get("ramp_time_seconds", 0.05)),
        initial_diameter=float(jet.get("initial_diameter_m", 150.0e-6)),
        minimum_diameter=float(jet.get("minimum_diameter_m", 50.0e-6)),
        maximum_diameter=float(jet.get("maximum_diameter_m", 300.0e-6)),
        rosin_rammler_spread=float(jet.get("rosin_rammler_spread", 3.0)),
        parcels_per_step=int(jet.get("parcels_per_step", 8)),
        maximum_parcels=int(jet.get("maximum_parcels", 16_384)),
        parcel_substeps=int(jet.get("parcel_substeps", 4)),
        cone_half_angle_degrees=float(jet.get("cone_half_angle_degrees", 0.0)),
        edge_speed_ratio=float(jet.get("edge_speed_ratio", 1.0)),
        edge_diameter_ratio=float(jet.get("edge_diameter_ratio", 1.0)),
        profile_radius_m=tuple(
            float(value) for value in jet.get("profile_radius_m", ())
        ),
        profile_axial_velocity_m_s=tuple(
            float(value)
            for value in jet.get("profile_axial_velocity_m_s", ())
        ),
        profile_radial_velocity_m_s=tuple(
            float(value)
            for value in jet.get("profile_radial_velocity_m_s", ())
        ),
        profile_d10_m=tuple(
            float(value) for value in jet.get("profile_d10_m", ())
        ),
        output=Path(output["directory"]),
        gmg_tolerance=float(numerics["gmg_tolerance"]),
        gmg_presweeps=int(numerics["gmg_presweeps"]),
        gmg_postsweeps=int(numerics["gmg_postsweeps"]),
        flow_formulation=str(numerics.get("flow_formulation", "low-mach")),
        momentum_closure=str(
            numerics.get("momentum_closure", "amd")
        ).replace("_", "-"),
        time_integration=str(numerics["time_integration"]),
    )
    finite_positive = (
        *case.lengths,
        case.dt,
        case.ambient_temperature,
        case.pressure,
        case.ambient_density,
        case.dry_air_density,
        case.ambient_heat_capacity,
        case.ambient_gas_constant,
        case.kinematic_viscosity,
        case.scalar_diffusivity,
        case.roughness,
        case.gas_inlet_radius,
        case.gas_inlet_temperature,
        case.subgrid_support_radius_cells,
        case.subgrid_transition_width_cells,
        case.subgrid_momentum_length_cells,
        case.subgrid_integral_scale_cells,
        case.subgrid_correlation_time,
        case.radius,
        case.speed,
        case.mass_flow_rate,
        case.jet_temperature,
        case.liquid_density,
        case.liquid_latent_heat,
        case.initial_diameter,
        case.minimum_diameter,
        case.maximum_diameter,
        case.rosin_rammler_spread,
        case.edge_speed_ratio,
        case.edge_diameter_ratio,
        case.gmg_tolerance,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in finite_positive):
        raise ValueError("physical dimensions, timestep, and material values must be positive")
    if any(kind not in {"uniform", "tanh", "sinh"} for kind in case.mapping_types):
        raise ValueError("mapping types must be uniform, tanh, or sinh")
    if not all(
        math.isfinite(focus) and 0.0 <= focus <= length
        for focus, length in zip(case.mapping_focus, case.lengths)
    ):
        raise ValueError("mapping focus must lie inside each physical axis")
    if not all(
        math.isfinite(strength) and strength >= 0.0
        for strength in case.mapping_strength
    ):
        raise ValueError("mapping strength must be finite and nonnegative")
    if len(case.gravity) != 3 or not all(math.isfinite(value) for value in case.gravity):
        raise ValueError("gravity_m_s2 must contain three finite values")
    if not math.isfinite(case.ramp_time) or case.ramp_time < 0.0:
        raise ValueError("jet ramp time must be finite and nonnegative")
    if case.steps <= 0 or case.chunk_steps <= 0 or case.checkpoint_every <= 0:
        raise ValueError("steps, chunk size, and checkpoint interval must be positive")
    if case.parcels_per_step <= 0 or case.maximum_parcels < case.parcels_per_step:
        raise ValueError("parcel capacity must accommodate one injection")
    if case.parcel_substeps <= 0:
        raise ValueError("parcel substeps must be positive")
    if case.minimum_diameter >= case.maximum_diameter:
        raise ValueError("minimum droplet diameter must be below maximum")
    if not 0.0 <= case.cone_half_angle_degrees < 90.0:
        raise ValueError("cone half-angle must lie in [0, 90) degrees")
    if case.time_integration not in {"rk3", "fast-rk3"}:
        raise ValueError("time_integration must be .rk3. or .fast-rk3.")
    if case.scalar_advection_scheme not in {"central", "upwind"}:
        raise ValueError(
            "scalar_advection_scheme must be .central. or .upwind."
        )
    if case.flow_formulation not in {"low-mach", "incompressible"}:
        raise ValueError(
            "flow_formulation must be .low-mach. or .incompressible."
        )
    if case.momentum_closure not in {"amd", "classical-static-smagorinsky"}:
        raise ValueError(
            "momentum_closure must be .amd. or .classical-static-smagorinsky."
        )
    if case.source_mode not in {"volume", "inflow"}:
        raise ValueError("source mode must be .volume. or .inflow.")
    if case.subgrid_jet_enabled and case.source_mode != "inflow":
        raise ValueError("the subgrid boundary jet requires source mode .inflow.")
    if case.fully_vaporized_within_source_cell and case.source_mode != "volume":
        raise ValueError(
            "fully vaporized within-source-cell closure requires volume source mode"
        )
    if not math.isfinite(case.subgrid_turbulence_intensity) or case.subgrid_turbulence_intensity < 0.0:
        raise ValueError("subgrid turbulence intensity must be finite and nonnegative")
    if not math.isfinite(case.subgrid_transverse_ratio) or case.subgrid_transverse_ratio < 0.0:
        raise ValueError("subgrid transverse ratio must be finite and nonnegative")
    if not 0.0 <= case.ambient_relative_humidity <= 1.0:
        raise ValueError("ambient relative humidity must lie in [0, 1]")
    if not 0.0 <= case.vapor_quality < 1.0:
        raise ValueError("vapor quality must lie in [0, 1)")
    profiles = (
        case.profile_radius_m,
        case.profile_axial_velocity_m_s,
        case.profile_radial_velocity_m_s,
        case.profile_d10_m,
    )
    if any(profiles) and not all(
        len(values) == len(case.profile_radius_m) for values in profiles
    ):
        raise ValueError("empirical LN2 source profiles must have equal lengths")
    inside = all(
        0.0 < coordinate < length
        for coordinate, length in zip(case.nozzle, case.lengths)
    )
    at_inflow = (
        case.source_mode == "inflow"
        and case.nozzle[0] == 0.0
        and 0.0 < case.nozzle[1] < case.lengths[1]
        and 0.0 < case.nozzle[2] < case.lengths[2]
    )
    if not (inside or at_inflow):
        raise ValueError(
            "the nozzle must lie inside the domain or on its inflow"
        )
    return case


def _add_velocity(left, right):
    from jaxwind import StaggeredVelocity

    return StaggeredVelocity(
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
    )


def _cell_vector_to_faces(x, y, z, grid):
    from jaxwind import StaggeredVelocity
    from jaxwind.discretization import _cells_to_faces

    return StaggeredVelocity(
        _cells_to_faces(x, grid, 2, periodic=False, boundary="copy"),
        _cells_to_faces(y, grid, 1, periodic=False, boundary="zero"),
        _cells_to_faces(z, grid, 0, periodic=False, boundary="zero"),
    )


def _enforce_scalar(field, ambient):
    """Zero-gradient outlet with a quiescent ambient inlet."""

    field = field.at[..., 0].set(ambient)
    outlet = (4.0 * field[..., -2] - field[..., -3]) / 3.0
    return field.at[..., -1].set(outlet)


def _conservative_nonnegative(field, cell_volumes):
    """Remove undershoots without changing the volume-weighted inventory."""

    import jax.numpy as jnp

    volumes = jnp.asarray(cell_volumes, field.dtype)
    target = jnp.maximum(jnp.sum(field * volumes), 0.0)
    positive = jnp.maximum(field, 0.0)
    positive_inventory = jnp.sum(positive * volumes)
    tiny = jnp.finfo(field.dtype).tiny
    scale = jnp.where(
        positive_inventory > tiny,
        target / jnp.maximum(positive_inventory, tiny),
        0.0,
    )
    return positive * scale


def build_simulation(case: JetCase, *, differentiable_inlet: bool = False):
    import jax
    import jax.numpy as jnp

    from jaxwind.domain import (
        AnalyticalGrid,
        SinhMapping,
        TanhMapping,
        UniformGrid,
    )
    from jaxwind import (
        AnisotropicMinimumDissipation,
        StaticSmagorinsky,
        FREE_SLIP,
        IdealGasMixture,
        OPEN,
        Boundaries,
        FlowModel,
        InflowPlane,
        MoninObukhovWall,
        PassiveScalar,
        StaggeredVelocity,
        Wall,
        build_pressure_poisson,
        build_tendency,
        cell_velocity,
        conservative_specific_tendency,
        courant_number,
        eddy_viscosity,
        enforce_open_velocity,
        face_density,
        continuity_residual,
        dilatation_correction,
        divergence,
        pressure_gradient,
        project,
        project_low_mach,
        scalar_tendency,
        zeros,
    )
    from jaxwind.cryogenic import (
        LN2InletControl,
        LN2Jet,
        ParcelExchange,
        advance_ln2_parcels,
        initial_ln2_parcels,
        vapor_nozzle_source,
    )
    from jaxwind.physics.cryogenic import (
        CryogenicMicrophysicsConfig,
        advance_fog_microphysics,
    )

    def axis_mapping(kind, focus, strength, length):
        normalized_focus = focus / length
        if kind == "sinh":
            return SinhMapping(normalized_focus, strength)
        # TanhMapping(0) is also the identity used for an unstretched axis.
        return TanhMapping(strength, normalized_focus)

    mappings = tuple(
        axis_mapping(kind, focus, strength, length)
        for kind, focus, strength, length in zip(
            case.mapping_types,
            case.mapping_focus,
            case.mapping_strength,
            case.lengths,
        )
    )
    grid = (
        UniformGrid(*case.cells, *case.lengths)
        if all(
            kind == "uniform" or strength == 0.0
            for kind, strength in zip(
                case.mapping_types, case.mapping_strength
            )
        )
        else AnalyticalGrid(*case.cells, *case.lengths, *mappings)
    )
    boundaries = Boundaries(
        Wall(FREE_SLIP),
        Wall(FREE_SLIP),
        streamwise=OPEN,
        spanwise=FREE_SLIP,
    )
    wall = MoninObukhovWall(case.roughness)
    closure = (
        AnisotropicMinimumDissipation()
        if case.momentum_closure == "amd"
        else StaticSmagorinsky(SMAGORINSKY_COEFFICIENT)
    )
    flow_model = FlowModel(
        viscosity=case.kinematic_viscosity,
        subfilter=closure,
        surface=wall,
        sidewalls=wall,
    )
    momentum_rhs = build_tendency(grid, boundaries, flow_model)
    scalar_model = PassiveScalar(
        diffusivity=case.scalar_diffusivity,
        turbulent_prandtl=0.74,
        advection_scheme=case.scalar_advection_scheme,
    )
    poisson = build_pressure_poisson(
        grid,
        backend="gmg",
        periodic_x=False,
        periodic_y=False,
        dtype="float32",
        config={
            "tolerance": case.gmg_tolerance,
            "presweeps": case.gmg_presweeps,
            "postsweeps": case.gmg_postsweeps,
            "max_iterations": 200,
        },
    )
    jet = LN2Jet(
        *case.nozzle,
        case.radius,
        case.speed,
        case.mass_flow_rate,
        case.vapor_quality,
        initial_temperature=case.jet_temperature,
        liquid_density=case.liquid_density,
        ramp_time=case.ramp_time,
        initial_diameter=case.initial_diameter,
        minimum_diameter=case.minimum_diameter,
        maximum_diameter=case.maximum_diameter,
        rosin_rammler_spread=case.rosin_rammler_spread,
        parcels_per_step=case.parcels_per_step,
        maximum_parcels=case.maximum_parcels,
        substeps=case.parcel_substeps,
        cone_half_angle_degrees=case.cone_half_angle_degrees,
        edge_speed_ratio=case.edge_speed_ratio,
        edge_diameter_ratio=case.edge_diameter_ratio,
        profile_radius_m=case.profile_radius_m,
        profile_axial_velocity_m_s=case.profile_axial_velocity_m_s,
        profile_radial_velocity_m_s=case.profile_radial_velocity_m_s,
        profile_d10_m=case.profile_d10_m,
        gravity_x=case.gravity[0],
        gravity_y=case.gravity[1],
        gravity_z=case.gravity[2],
    )
    microphysics = CryogenicMicrophysicsConfig(
        pressure=case.pressure,
        dry_air_density=case.dry_air_density,
        dry_air_heat_capacity=case.ambient_heat_capacity,
        dry_air_gas_constant=case.ambient_gas_constant,
        liquid_nitrogen_density=case.liquid_density,
        liquid_nitrogen_latent_heat=case.liquid_latent_heat,
        nitrogen_boiling_temperature=case.jet_temperature,
        outlet_start_x=0.875 * grid.lx,
        outlet_end_x=grid.lx,
    )
    equation_of_state = IdealGasMixture(
        pressure=case.pressure,
        background_gas_constant=case.ambient_gas_constant,
        species_gas_constants=(
            microphysics.nitrogen_gas_constant,
            microphysics.water_vapor_gas_constant,
        ),
    )
    dry_water = case.ambient_water_vapor == 0.0
    volume_source = case.source_mode == "volume"
    incompressible = case.flow_formulation == "incompressible"
    direct_vapor_mass_flow_rate = (
        jet.mass_flow_rate
        if case.fully_vaporized_within_source_cell
        else jet.vapor_mass_flow_rate
    )
    flash_mass_flow_rate = (
        jet.liquid_mass_flow_rate
        if case.fully_vaporized_within_source_cell
        else 0.0
    )

    def thermodynamic_density(temperature, water_vapor, nitrogen_density):
        if dry_water:
            remainder_gas_constant = jnp.asarray(
                case.ambient_gas_constant, temperature.dtype
            )
        else:
            water_ratio = jnp.maximum(water_vapor, 0.0)
            remainder_gas_constant = (
                case.ambient_gas_constant
                + water_ratio * microphysics.water_vapor_gas_constant
            ) / (1.0 + water_ratio)
        pressure_over_temperature = equation_of_state.pressure_field(
            temperature
        ) / jnp.maximum(temperature, equation_of_state.temperature_floor)
        density = (
            pressure_over_temperature
            - nitrogen_density
            * (
                microphysics.nitrogen_gas_constant
                - remainder_gas_constant
            )
        ) / remainder_gas_constant
        return jnp.maximum(
            density,
            nitrogen_density + equation_of_state.density_floor,
        )

    def local_width(faces, coordinate):
        index = int(
            np.clip(np.searchsorted(faces, coordinate) - 1, 0, len(faces) - 2)
        )
        return float(faces[index + 1] - faces[index])

    local_dx = local_width(grid.x_faces, jet.x)
    local_dy = local_width(grid.y_faces, jet.y)
    local_dz = local_width(grid.z_faces, jet.z)
    cell_volume = jnp.asarray(grid.cell_volumes, jnp.float32)
    inlet_cell_area = jnp.asarray(
        grid.z_widths[:, None] * grid.y_widths[None, :], jnp.float32
    )
    y_face_widths = np.concatenate(
        (
            (0.5 * grid.y_widths[0],),
            0.5 * (grid.y_widths[:-1] + grid.y_widths[1:]),
            (0.5 * grid.y_widths[-1],),
        )
    )
    z_face_widths = np.concatenate(
        (
            (0.5 * grid.z_widths[0],),
            0.5 * (grid.z_widths[:-1] + grid.z_widths[1:]),
            (0.5 * grid.z_widths[-1],),
        )
    )
    inlet_y_face_area = jnp.asarray(
        grid.z_widths[:, None] * y_face_widths[None, :], jnp.float32
    )
    inlet_z_face_area = jnp.asarray(
        z_face_widths[:, None] * grid.y_widths[None, :], jnp.float32
    )
    shape = (grid.nz, grid.ny, grid.nx)
    inlet_shape = (grid.nz, grid.ny)

    base_control = DifferentiableInletControl(
        jnp.asarray(case.gas_inlet_radius, jnp.float32),
        jnp.asarray(case.gas_inlet_temperature, jnp.float32),
        jnp.asarray(case.initial_diameter, jnp.float32),
        jnp.asarray(1.0, jnp.float32),
        jnp.asarray(case.edge_speed_ratio, jnp.float32),
        jnp.asarray(case.edge_diameter_ratio, jnp.float32),
    )
    inlet_water_vapor = jnp.full(
        inlet_shape, case.ambient_water_vapor, jnp.float32
    )
    x_cells = jnp.asarray(grid.x_centers, jnp.float32)
    y_cells = jnp.asarray(grid.y_centers, jnp.float32)
    z_cells = jnp.asarray(grid.z_centers, jnp.float32)
    y_faces = jnp.asarray(grid.y_faces, jnp.float32)
    z_faces = jnp.asarray(grid.z_faces, jnp.float32)
    if volume_source:
        nozzle_kernel = vapor_nozzle_source(grid, jet)
        inlet_radius = jnp.zeros(inlet_shape, jnp.float32)
    else:
        inlet_radius = jnp.sqrt(
            (y_cells[None, :] - jet.y) ** 2
            + (z_cells[:, None] - jet.z) ** 2
        )
        nozzle_kernel = jnp.zeros(shape, jnp.float32)
    grid_spacing = max(local_dy, local_dz)
    default_transition_width = 1.5 * grid_spacing
    zero_cells = jnp.zeros(shape, jnp.float32)
    zero_acceleration = _cell_vector_to_faces(
        zero_cells, zero_cells, zero_cells,
        grid,
    )

    def normalise_mode(mode, weight):
        tiny = jnp.finfo(mode.dtype).tiny
        total = jnp.maximum(jnp.sum(weight), tiny)
        centered = mode - jnp.sum(weight * mode) / total
        rms = jnp.sqrt(jnp.sum(weight * centered**2) / total)
        return centered / jnp.maximum(rms, tiny)

    def inlet_fields(control, time):
        momentum_correction = zero_acceleration
        midpoint = time + 0.5 * case.dt
        inlet_ramp = (
            jnp.asarray(1.0, jnp.float32)
            if case.ramp_time == 0.0
            else 0.5
            * (
                1.0
                - jnp.cos(
                    jnp.pi
                    * jnp.clip(midpoint / case.ramp_time, 0.0, 1.0)
                )
            )
        )
        if volume_source:
            inlet_temperature = jnp.full(
                inlet_shape, case.ambient_temperature, jnp.float32
            )
            inlet_nitrogen_density = jnp.zeros(
                inlet_shape, jnp.float32
            )
            inlet_x_velocity = jnp.zeros(inlet_shape, jnp.float32)
            inlet_y_velocity = jnp.zeros(
                (grid.nz, grid.ny + 1), jnp.float32
            )
            inlet_z_velocity = jnp.zeros(
                (grid.nz + 1, grid.ny), jnp.float32
            )
        else:
            support_radius = (
                jnp.asarray(
                    case.subgrid_support_radius_cells * grid_spacing,
                    jnp.float32,
                )
                if case.subgrid_jet_enabled
                else control.gas_radius
            )
            transition_width = (
                case.subgrid_transition_width_cells * grid_spacing
                if case.subgrid_jet_enabled
                else default_transition_width
            )
            inlet_weight = 0.5 * (
                1.0
                - jnp.tanh(
                    (inlet_radius - support_radius) / transition_width
                )
            )
            pure_nitrogen_density = (
                case.ambient_density
                if incompressible
                else case.pressure
                / (
                    microphysics.nitrogen_gas_constant
                    * control.gas_temperature
                )
            )
            inlet_nitrogen_density = (
                pure_nitrogen_density * inlet_weight
            ).astype(jnp.float32)
            inlet_temperature = (
                case.ambient_temperature
                + inlet_weight
                * (control.gas_temperature - case.ambient_temperature)
            ).astype(jnp.float32)
            inlet_total_density = (
                jnp.full_like(inlet_temperature, case.ambient_density)
                if incompressible
                else thermodynamic_density(
                    inlet_temperature,
                    inlet_water_vapor,
                    inlet_nitrogen_density,
                )
            )
            inlet_speed = jet.vapor_mass_flow_rate / (
                jnp.sum(
                    inlet_nitrogen_density
                    * inlet_weight
                    * inlet_cell_area
                )
            )
            inlet_x_velocity = inlet_ramp * inlet_speed * inlet_weight
            inlet_y_velocity = jnp.zeros(
                (grid.nz, grid.ny + 1), jnp.float32
            )
            inlet_z_velocity = jnp.zeros(
                (grid.nz + 1, grid.ny), jnp.float32
            )

            if (
                case.subgrid_jet_enabled
                and case.subgrid_turbulence_intensity > 0.0
            ):
                length_scale = (
                    case.subgrid_integral_scale_cells * grid_spacing
                )
                wavenumber = 2.0 * jnp.pi / length_scale
                phase = time / case.subgrid_correlation_time
                relative_y = y_cells[None, :] - jet.y
                relative_z = z_cells[:, None] - jet.z
                axial_mode = (
                    jnp.sin(wavenumber * relative_y + phase)
                    * jnp.cos(wavenumber * relative_z - 0.7 * phase)
                    + 0.5
                    * jnp.cos(2.0 * wavenumber * relative_y - 1.3 * phase)
                    * jnp.sin(wavenumber * relative_z + 0.4 * phase)
                )
                axial_mode = normalise_mode(
                    axial_mode,
                    inlet_total_density * inlet_weight * inlet_cell_area
                )
                amplitude = (
                    inlet_ramp
                    * case.subgrid_turbulence_intensity
                    * inlet_speed
                )
                inlet_x_velocity = inlet_x_velocity + (
                    amplitude * inlet_weight * axial_mode
                )

                radius_y = jnp.sqrt(
                    (y_faces[None, :] - jet.y) ** 2
                    + (z_cells[:, None] - jet.z) ** 2
                )
                weight_y = 0.5 * (
                    1.0
                    - jnp.tanh(
                        (radius_y - support_radius) / transition_width
                    )
                )
                transverse_y = (
                    jnp.sin(
                        wavenumber * (z_cells[:, None] - jet.z) + phase
                    )
                    * jnp.cos(
                        wavenumber * (y_faces[None, :] - jet.y)
                        - 0.6 * phase
                    )
                )
                transverse_y = normalise_mode(
                    transverse_y, weight_y * inlet_y_face_area
                )
                inlet_y_velocity = (
                    case.subgrid_transverse_ratio
                    * amplitude
                    * weight_y
                    * transverse_y
                )

                radius_z = jnp.sqrt(
                    (y_cells[None, :] - jet.y) ** 2
                    + (z_faces[:, None] - jet.z) ** 2
                )
                weight_z = 0.5 * (
                    1.0
                    - jnp.tanh(
                        (radius_z - support_radius) / transition_width
                    )
                )
                transverse_z = (
                    -jnp.cos(
                        wavenumber * (z_faces[:, None] - jet.z) + phase
                    )
                    * jnp.sin(
                        wavenumber * (y_cells[None, :] - jet.y)
                        - 0.6 * phase
                    )
                )
                transverse_z = normalise_mode(
                    transverse_z, weight_z * inlet_z_face_area
                )
                inlet_z_velocity = (
                    case.subgrid_transverse_ratio
                    * amplitude
                    * weight_z
                    * transverse_z
                )

            if case.subgrid_jet_enabled:
                resolved_momentum = (
                    jnp.sum(
                        inlet_total_density
                        * inlet_x_velocity**2
                        * inlet_cell_area
                    )
                )
                target_momentum = (
                    jet.vapor_mass_flow_rate
                    * jet.speed
                    * inlet_ramp**2
                )
                momentum_deficit = jnp.maximum(
                    target_momentum - resolved_momentum, 0.0
                )
                momentum_length = (
                    case.subgrid_momentum_length_cells * local_dx
                )
                streamwise_weight = jnp.exp(
                    -0.5 * (x_cells / momentum_length) ** 2
                )
                momentum_kernel = (
                    inlet_weight[..., None]
                    * streamwise_weight[None, None, :]
                )
                momentum_kernel = momentum_kernel / (
                    jnp.sum(momentum_kernel * cell_volume)
                )
                axial_acceleration = (
                    momentum_deficit
                    / case.ambient_density
                    * momentum_kernel
                )
                momentum_correction = _cell_vector_to_faces(
                    axial_acceleration, zero_cells, zero_cells,
                    grid,
                )

            inlet_x_velocity = inlet_x_velocity.astype(jnp.float32)
            inlet_y_velocity = inlet_y_velocity.astype(jnp.float32)
            inlet_z_velocity = inlet_z_velocity.astype(jnp.float32)
        plane = InflowPlane(
            inlet_x_velocity,
            inlet_y_velocity,
            inlet_z_velocity,
            inlet_temperature,
        )
        return (
            plane,
            inlet_temperature,
            inlet_nitrogen_density,
            momentum_correction,
        )

    def make_initial(control):
        (
            inflow_plane,
            inlet_temperature,
            inlet_nitrogen_density,
            _momentum_correction,
        ) = inlet_fields(control, jnp.asarray(0.0, jnp.float32))
        velocity = enforce_open_velocity(
            zeros(grid, "float32", boundaries), inflow_plane, grid
        )
        zero_velocity = StaggeredVelocity(
            jnp.zeros_like(velocity.x),
            jnp.zeros_like(velocity.y),
            jnp.zeros_like(velocity.z),
        )
        initial_temperature = jnp.full(
            shape, case.ambient_temperature, jnp.float32
        ).at[..., 0].set(inlet_temperature)
        initial_water_vapor = jnp.full(
            shape, case.ambient_water_vapor, jnp.float32
        ).at[..., 0].set(inlet_water_vapor)
        initial_nitrogen_density = jnp.zeros(
            shape, jnp.float32
        ).at[..., 0].set(inlet_nitrogen_density)
        thermodynamic_initial_density = thermodynamic_density(
            initial_temperature,
            initial_water_vapor,
            initial_nitrogen_density,
        )
        initial_density = (
            jnp.full(shape, case.ambient_density, jnp.float32)
            if incompressible
            else thermodynamic_initial_density
        )
        initial_nitrogen = jnp.clip(
            initial_nitrogen_density / initial_density, 0.0, 1.0
        )
        return CryogenicState(
            velocity,
            jnp.zeros(shape, jnp.float32),
            initial_density,
            zero_velocity,
            initial_temperature,
            jnp.zeros(shape, jnp.float32),
            initial_water_vapor,
            jnp.zeros(shape, jnp.float32),
            initial_nitrogen,
            initial_nitrogen_density,
            jnp.zeros(shape, jnp.float32),
            jnp.zeros(shape, jnp.float32),
            jnp.zeros(shape, jnp.float32),
            jnp.zeros(shape, jnp.float32),
            jnp.zeros(shape, jnp.float32),
            initial_ln2_parcels(jet),
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0, jnp.int32),
        )

    initial = make_initial(base_control)

    def rk3_step(state, control):
        """Wray three-stage RK3 with selectable pressure projection cadence."""
        (
            inflow_plane,
            inlet_temperature,
            inlet_nitrogen_density,
            subgrid_momentum,
        ) = inlet_fields(control, state.time)
        reference_temperature = jnp.full(
            shape, case.ambient_temperature, state.temperature.dtype
        ).at[..., 0].set(inlet_temperature)
        reference_nitrogen_density = jnp.zeros_like(
            state.nitrogen_density
        ).at[..., 0].set(inlet_nitrogen_density)
        reference_density = thermodynamic_density(
            reference_temperature,
            jnp.full_like(state.water_vapor, case.ambient_water_vapor),
            reference_nitrogen_density,
        )
        parcel_control = LN2InletControl(
            control.initial_diameter,
            control.speed_scale,
            control.edge_speed_ratio,
            control.edge_diameter_ratio,
        )
        current_velocity = enforce_open_velocity(
            state.velocity, inflow_plane, grid
        )
        parcel_exchange = (
            ParcelExchange(
                state.parcels,
                zero_acceleration.x,
                zero_acceleration.y,
                zero_acceleration.z,
                zero_cells,
                zero_cells,
                zero_cells,
                zero_cells,
                jnp.asarray(0.0, state.temperature.dtype),
            )
            if case.fully_vaporized_within_source_cell
            else advance_ln2_parcels(
                state.parcels,
                current_velocity,
                state.temperature,
                grid,
                state.step,
                case.dt,
                jet,
                microphysics,
                state.density,
                parcel_control,
            )
        )
        dtype = state.temperature.dtype
        midpoint = (state.step.astype(dtype) + 0.5) * case.dt
        ramp = (
            jnp.asarray(1.0, dtype)
            if jet.ramp_time == 0.0
            else 0.5
            * (
                1.0
                - jnp.cos(
                    jnp.pi
                    * jnp.clip(midpoint / jet.ramp_time, 0.0, 1.0)
                )
            )
        )
        vapor_mass_rate = (
            direct_vapor_mass_flow_rate
            * ramp
            * nozzle_kernel
            / cell_volume
            if volume_source
            else jnp.zeros(shape, dtype)
        )
        flash_power_density = (
            flash_mass_flow_rate
            * case.liquid_latent_heat
            * ramp
            * nozzle_kernel
            / cell_volume
            if volume_source
            else jnp.zeros(shape, dtype)
        )
        vapor_mixing_source = vapor_mass_rate / state.density
        gas_mass_source = parcel_exchange.gas_mass_source + vapor_mass_rate
        parcel_momentum = StaggeredVelocity(
            parcel_exchange.acceleration_x,
            parcel_exchange.acceleration_y,
            parcel_exchange.acceleration_z,
        )

        def tendencies(
            velocity,
            density,
            temperature,
            water_vapor,
            nitrogen,
            nitrogen_density,
            liquid,
            ice,
            execution_time,
        ):
            u_cell, _, _ = cell_velocity(velocity)
            vapor_momentum = _cell_vector_to_faces(
                vapor_mixing_source * (jet.speed - u_cell),
                jnp.zeros(shape, dtype),
                jnp.zeros(shape, dtype),
                grid,
            )
            momentum = _add_velocity(
                momentum_rhs(velocity, execution_time), parcel_momentum
            )
            momentum = _add_velocity(momentum, subgrid_momentum)
            if not incompressible:
                momentum = _add_velocity(
                    momentum, dilatation_correction(velocity, grid)
                )
            momentum = _add_velocity(momentum, vapor_momentum)
            buoyancy_density = (
                thermodynamic_density(
                    temperature, water_vapor, nitrogen_density
                )
                if incompressible
                else density
            )
            density_anomaly = (
                buoyancy_density - reference_density
            ) / jnp.maximum(buoyancy_density, 1.0e-6)
            momentum = _add_velocity(
                momentum,
                _cell_vector_to_faces(
                    case.gravity[0] * density_anomaly,
                    case.gravity[1] * density_anomaly,
                    case.gravity[2] * density_anomaly,
                    grid,
                ),
            )
            viscosity = eddy_viscosity(
                velocity, grid, boundaries, closure
            )
            flow_dilatation = divergence(velocity, grid)

            def transported(field):
                return (
                    scalar_tendency(
                        field,
                        velocity,
                        grid,
                        scalar_model,
                        eddy_viscosity=viscosity,
                    )
                    + field * flow_dilatation
                )

            temperature_rhs = (
                transported(temperature)
                + parcel_exchange.temperature_source
                + vapor_mixing_source
                * (jet.initial_temperature - temperature)
                - flash_power_density
                / (density * case.ambient_heat_capacity)
            )
            nitrogen_density_rhs = conservative_specific_tendency(
                nitrogen,
                density,
                velocity,
                grid,
                scalar_model,
                eddy_diffusivity=(
                    viscosity / scalar_model.turbulent_prandtl
                ),
            ) + gas_mass_source
            if dry_water:
                water_rhs = jnp.zeros_like(water_vapor)
                liquid_rhs = jnp.zeros_like(liquid)
                ice_rhs = jnp.zeros_like(ice)
            else:
                water_rhs = transported(water_vapor)
                liquid_rhs = transported(liquid)
                ice_rhs = transported(ice)
            return (
                momentum,
                temperature_rhs,
                water_rhs,
                nitrogen_density_rhs,
                liquid_rhs,
                ice_rhs,
            )

        weights = (
            (8.0 / 15.0, 0.0),
            (5.0 / 12.0, -17.0 / 60.0),
            (3.0 / 4.0, -5.0 / 12.0),
        )
        velocity = current_velocity
        temperature = state.temperature
        water_vapor = state.water_vapor
        nitrogen = state.nitrogen
        nitrogen_density = state.nitrogen_density
        liquid = state.liquid_water
        ice = state.ice_water
        execution_time = state.time
        pressure = state.pressure
        density = state.density
        previous = (
            state.momentum_tendency,
            state.temperature_tendency,
            state.water_vapor_tendency,
            state.nitrogen_density_tendency,
            state.liquid_water_tendency,
            state.ice_water_tendency,
        )
        current = previous
        lagged_pressure_gradient = (
            pressure_gradient(
                pressure,
                grid,
                periodic_x=poisson.periodic_x,
                periodic_y=poisson.periodic_y,
            )
            if case.time_integration == "fast-rk3"
            else None
        )
        last_stage = len(weights) - 1
        continuity_error = state.continuity_error
        for stage, (current_weight, previous_weight) in enumerate(weights):
            previous_stage_density = density
            current = tendencies(
                velocity,
                density,
                temperature,
                water_vapor,
                nitrogen,
                nitrogen_density,
                liquid,
                ice,
                execution_time,
            )
            current_momentum = current[0]
            previous_momentum = previous[0]
            candidate = StaggeredVelocity(
                velocity.x
                + case.dt
                * (
                    current_weight * current_momentum.x
                    + previous_weight * previous_momentum.x
                ),
                velocity.y
                + case.dt
                * (
                    current_weight * current_momentum.y
                    + previous_weight * previous_momentum.y
                ),
                velocity.z
                + case.dt
                * (
                    current_weight * current_momentum.z
                    + previous_weight * previous_momentum.z
                ),
            )
            substep = case.dt * (current_weight + previous_weight)
            if lagged_pressure_gradient is not None:
                if incompressible:
                    candidate = StaggeredVelocity(
                        candidate.x - substep * lagged_pressure_gradient.x,
                        candidate.y - substep * lagged_pressure_gradient.y,
                        candidate.z - substep * lagged_pressure_gradient.z,
                    )
                else:
                    pressure_density = face_density(density, candidate, grid)
                    candidate = StaggeredVelocity(
                        candidate.x
                        - substep
                        * lagged_pressure_gradient.x
                        / pressure_density.x,
                        candidate.y
                        - substep
                        * lagged_pressure_gradient.y
                        / pressure_density.y,
                        candidate.z
                        - substep
                        * lagged_pressure_gradient.z
                        / pressure_density.z,
                    )
            candidate = enforce_open_velocity(candidate, inflow_plane, grid)

            def stage_scalar(
                field, tendency, old_tendency, ambient, floor=None, ceiling=None
            ):
                updated = field + case.dt * (
                    current_weight * tendency
                    + previous_weight * old_tendency
                )
                if floor is not None:
                    updated = jnp.maximum(updated, floor)
                if ceiling is not None:
                    updated = jnp.minimum(updated, ceiling)
                return _enforce_scalar(updated, ambient)

            temperature = stage_scalar(
                temperature,
                current[1],
                previous[1],
                inlet_temperature,
                jet.initial_temperature,
            )
            nitrogen_density = stage_scalar(
                nitrogen_density,
                current[3],
                previous[3],
                inlet_nitrogen_density,
            )
            nitrogen_density = _conservative_nonnegative(
                nitrogen_density, cell_volume
            )
            if not dry_water:
                water_vapor = stage_scalar(
                    water_vapor,
                    current[2],
                    previous[2],
                    inlet_water_vapor,
                    0.0,
                )
                liquid = stage_scalar(
                    liquid, current[4], previous[4], 0.0, 0.0
                )
                ice = stage_scalar(
                    ice, current[5], previous[5], 0.0, 0.0
                )
                fog = advance_fog_microphysics(
                    temperature,
                    water_vapor,
                    liquid,
                    ice,
                    substep,
                    microphysics,
                )
                temperature = jnp.maximum(
                    fog.temperature, jet.initial_temperature
                )
                water_vapor, liquid, ice = fog.qv, fog.ql, fog.qi
            updated_thermodynamic_density = thermodynamic_density(
                temperature, water_vapor, nitrogen_density
            )
            density = (
                jnp.full_like(updated_thermodynamic_density, case.ambient_density)
                if incompressible
                else updated_thermodynamic_density
            )
            nitrogen = jnp.clip(nitrogen_density / density, 0.0, 1.0)
            if lagged_pressure_gradient is None:
                if incompressible:
                    velocity, pressure = project(
                        candidate,
                        poisson,
                        substep,
                        initial_pressure=pressure,
                    )
                else:
                    velocity, pressure = project_low_mach(
                        candidate,
                        previous_stage_density,
                        density,
                        poisson,
                        substep,
                        mass_source=gas_mass_source,
                        initial_pressure=pressure,
                    )
            elif stage == last_stage:
                if incompressible:
                    velocity, correction = project(candidate, poisson, substep)
                else:
                    velocity, correction = project_low_mach(
                        candidate,
                        state.density,
                        density,
                        poisson,
                        substep,
                        mass_source=gas_mass_source,
                        continuity_dt=case.dt,
                    )
                pressure = pressure + correction * (substep / case.dt)
            else:
                velocity = candidate
            if stage == last_stage:
                if incompressible:
                    continuity_error = case.ambient_density * jnp.max(
                        jnp.abs(divergence(velocity, grid))
                    )
                else:
                    mass_previous = (
                        state.density
                        if lagged_pressure_gradient is not None
                        else previous_stage_density
                    )
                    mass_interval = (
                        case.dt
                        if lagged_pressure_gradient is not None
                        else substep
                    )
                    continuity_error = jnp.max(
                        jnp.abs(
                            continuity_residual(
                                velocity,
                                mass_previous,
                                density,
                                grid,
                                mass_interval,
                                gas_mass_source,
                            )
                        )
                    )
            previous = current
            execution_time = execution_time + substep

        nitrogen = jnp.clip(nitrogen_density / density, 0.0, 1.0)

        return CryogenicState(
            velocity,
            pressure,
            density,
            current[0],
            temperature,
            current[1],
            water_vapor,
            current[2],
            nitrogen,
            nitrogen_density,
            current[3],
            liquid,
            current[4],
            ice,
            current[5],
            parcel_exchange.parcels,
            continuity_error,
            state.time + case.dt,
            state.step + 1,
        )

    def advance_controlled(state, control, count):
        return jax.lax.fori_loop(
            0, count, lambda _, carry: rk3_step(carry, control), state
        )

    controlled = jax.jit(advance_controlled, static_argnums=2)
    if differentiable_inlet:
        return (
            grid,
            jet,
            microphysics,
            jax.jit(make_initial),
            controlled,
            courant_number,
            base_control,
        )

    def advance_block(state, count):
        return controlled(state, base_control, count)

    return (
        grid,
        jet,
        microphysics,
        initial,
        jax.jit(advance_block, static_argnums=1),
        courant_number,
    )


def _save_checkpoint(path: Path, state, grid, jet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lengths_m=np.asarray((grid.lx, grid.ly, grid.lz)),
        x_faces_m=np.asarray(grid.x_faces),
        y_faces_m=np.asarray(grid.y_faces),
        z_faces_m=np.asarray(grid.z_faces),
        x_centers_m=np.asarray(grid.x_centers),
        y_centers_m=np.asarray(grid.y_centers),
        z_centers_m=np.asarray(grid.z_centers),
        nozzle_m=np.asarray((jet.x, jet.y, jet.z)),
        velocity_x=np.asarray(state.velocity.x),
        velocity_y=np.asarray(state.velocity.y),
        velocity_z=np.asarray(state.velocity.z),
        pressure=np.asarray(state.pressure),
        density=np.asarray(state.density),
        temperature=np.asarray(state.temperature),
        water_vapor=np.asarray(state.water_vapor),
        nitrogen=np.asarray(state.nitrogen),
        nitrogen_density=np.asarray(state.nitrogen_density),
        liquid_water=np.asarray(state.liquid_water),
        ice_water=np.asarray(state.ice_water),
        parcel_x=np.asarray(state.parcels.x),
        parcel_y=np.asarray(state.parcels.y),
        parcel_z=np.asarray(state.parcels.z),
        parcel_u=np.asarray(state.parcels.u),
        parcel_v=np.asarray(state.parcels.v),
        parcel_w=np.asarray(state.parcels.w),
        parcel_mass=np.asarray(state.parcels.mass),
        parcel_diameter=np.asarray(state.parcels.diameter),
        parcel_temperature=np.asarray(state.parcels.temperature),
        parcel_multiplicity=np.asarray(state.parcels.multiplicity),
        parcel_active=np.asarray(state.parcels.active),
        continuity_error_kg_m3_s=np.asarray(state.continuity_error),
        time=np.asarray(state.time),
        step=np.asarray(state.step),
    )


def run(case: JetCase, *, steps: int | None = None) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", False)
    grid, jet, microphysics, state, advance, courant_number = build_simulation(case)
    total_steps = case.steps if steps is None else steps
    case.output.mkdir(parents=True, exist_ok=True)
    y_index = grid.ny // 2

    @jax.jit
    def capture_frame(current):
        u = 0.5 * (current.velocity.x[..., :-1] + current.velocity.x[..., 1:])
        v = 0.5 * (current.velocity.y[:, :-1] + current.velocity.y[:, 1:])
        w = 0.5 * (current.velocity.z[:-1] + current.velocity.z[1:])
        speed = jnp.sqrt(u * u + v * v + w * w)
        return (
            speed[:, y_index],
            current.temperature[:, y_index],
            current.nitrogen[:, y_index],
            (current.liquid_water + current.ice_water)[:, y_index],
        )

    velocity_frames = []
    temperature_frames = []
    nitrogen_frames = []
    fog_frames = []
    frame_times = []

    def append_frame(current):
        captured = capture_frame(current)
        jax.block_until_ready(captured)
        velocity_frames.append(np.asarray(captured[0]))
        temperature_frames.append(np.asarray(captured[1]))
        nitrogen_frames.append(np.asarray(captured[2]))
        fog_frames.append(np.asarray(captured[3]))
        frame_times.append(float(current.time))

    append_frame(state)
    started = time.perf_counter()
    last_checkpoint = 0
    while int(state.step) < total_steps:
        count = min(case.chunk_steps, total_steps - int(state.step))
        state = advance(state, count)
        jax.block_until_ready(state.velocity.x)
        append_frame(state)
        completed = int(state.step)
        active = int(jnp.sum(state.parcels.active))
        physical_liquid = float(
            jnp.sum(
                state.parcels.mass
                * state.parcels.multiplicity
                * state.parcels.active
            )
        )
        cfl = float(courant_number(state.velocity, grid, case.dt))
        print(
            f"jet {completed:6d}/{total_steps} time={float(state.time):8.4f}s "
            f"CFL={cfl:.3f} Tmin={float(jnp.min(state.temperature)):.2f}K "
            f"mass_res={float(state.continuity_error):.3e}kg/m3/s "
            f"parcels={active} liquid={physical_liquid:.6e}kg",
            flush=True,
        )
        if (
            completed == total_steps
            or completed - last_checkpoint >= case.checkpoint_every
        ):
            _save_checkpoint(
                case.output / f"checkpoint_{completed:08d}.npz",
                state,
                grid,
                jet,
            )
            last_checkpoint = completed
    elapsed = time.perf_counter() - started
    frames_path = case.output / "animation_frames_4panel.npz"
    np.savez_compressed(
        frames_path,
        velocity_magnitude_y=np.stack(velocity_frames),
        temperature_y=np.stack(temperature_frames),
        nitrogen_y=np.stack(nitrogen_frames),
        total_fog_y=np.stack(fog_frames),
        time_seconds=np.asarray(frame_times),
        lengths_m=np.asarray((grid.lx, grid.ly, grid.lz)),
        x_faces_m=np.asarray(grid.x_faces),
        y_faces_m=np.asarray(grid.y_faces),
        z_faces_m=np.asarray(grid.z_faces),
        x_centers_m=np.asarray(grid.x_centers),
        y_centers_m=np.asarray(grid.y_centers),
        z_centers_m=np.asarray(grid.z_centers),
        nozzle_m=np.asarray((jet.x, jet.y, jet.z)),
    )
    y_index = grid.ny // 2
    z_index = int(
        np.clip(np.searchsorted(grid.z_faces, jet.z) - 1, 0, grid.nz - 1)
    )
    np.savez_compressed(
        case.output / "slices.npz",
        lengths_m=np.asarray((grid.lx, grid.ly, grid.lz)),
        x_faces_m=np.asarray(grid.x_faces),
        y_faces_m=np.asarray(grid.y_faces),
        z_faces_m=np.asarray(grid.z_faces),
        x_centers_m=np.asarray(grid.x_centers),
        y_centers_m=np.asarray(grid.y_centers),
        z_centers_m=np.asarray(grid.z_centers),
        nozzle_m=np.asarray((jet.x, jet.y, jet.z)),
        temperature_y=np.asarray(state.temperature[:, y_index]),
        nitrogen_y=np.asarray(state.nitrogen[:, y_index]),
        liquid_water_y=np.asarray(state.liquid_water[:, y_index]),
        ice_water_y=np.asarray(state.ice_water[:, y_index]),
        temperature_z=np.asarray(state.temperature[z_index]),
        nitrogen_z=np.asarray(state.nitrogen[z_index]),
    )
    result = {
        "steps": total_steps,
        "simulated_seconds": float(state.time),
        "elapsed_seconds": elapsed,
        "steps_per_second": total_steps / elapsed,
        "simulation_to_wall_ratio": float(state.time) / elapsed,
        "minimum_temperature_k": float(jnp.min(state.temperature)),
        "maximum_nitrogen_mass_fraction": float(jnp.max(state.nitrogen)),
        "active_parcels": int(jnp.sum(state.parcels.active)),
        "final_cfl": float(courant_number(state.velocity, grid, case.dt)),
        "maximum_continuity_residual_kg_m3_s": float(
            state.continuity_error
        ),
        "output": str(case.output),
        "vapor_quality": case.vapor_quality,
        "nozzle_exit_vapor_quality": case.vapor_quality,
        "post_source_vapor_quality": (
            1.0 if case.fully_vaporized_within_source_cell else case.vapor_quality
        ),
        "fully_vaporized_within_source_cell": (
            case.fully_vaporized_within_source_cell
        ),
        "vapor_mass_flow_rate_kg_s": (
            jet.mass_flow_rate
            if case.fully_vaporized_within_source_cell
            else jet.vapor_mass_flow_rate
        ),
        "liquid_mass_flow_rate_kg_s": (
            0.0
            if case.fully_vaporized_within_source_cell
            else jet.liquid_mass_flow_rate
        ),
        "flash_latent_cooling_power_w": (
            (
                jet.liquid_mass_flow_rate
                if case.fully_vaporized_within_source_cell
                else 0.0
            )
            * case.liquid_latent_heat
        ),
        "axial_momentum_flux_n": case.mass_flow_rate * case.speed,
        "pressure_pa": microphysics.pressure,
        "ambient_relative_humidity": case.ambient_relative_humidity,
        "ambient_water_vapor_mixing_ratio": case.ambient_water_vapor,
        "flow_formulation": case.flow_formulation,
        "subgrid_boundary_jet": {
            "enabled": case.subgrid_jet_enabled,
            "support_radius_cells": case.subgrid_support_radius_cells,
            "transition_width_cells": case.subgrid_transition_width_cells,
            "momentum_length_cells": case.subgrid_momentum_length_cells,
            "turbulence_intensity": case.subgrid_turbulence_intensity,
            "integral_scale_cells": case.subgrid_integral_scale_cells,
            "correlation_time_seconds": case.subgrid_correlation_time,
            "transverse_ratio": case.subgrid_transverse_ratio,
        },
        "time_integration": case.time_integration,
        "sgs_model": case.momentum_closure,
        "scalar_advection_scheme": case.scalar_advection_scheme,
        **(
            {"amd_poincare_constant": 1.0 / 3.0}
            if case.momentum_closure == "amd"
            else {"smagorinsky_coefficient": SMAGORINSKY_COEFFICIENT}
        ),
        "animation_frames": str(frames_path),
        "animation_frame_count": len(frame_times),
    }
    (case.output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", type=int)
    arguments = parser.parse_args(argv)
    result = run(load_case(arguments.config), steps=arguments.steps)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
