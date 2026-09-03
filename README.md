# JAX-Wind

JAX-Wind is a functional large-eddy simulation solver for atmospheric boundary
layers and wind-energy flows. The package owns numerical meaning and state
transitions; directories under `cases/` contain data only, while
`applications/` owns configuration interpretation, diagnostics, and effects.

The active solver is the staggered finite-volume implementation. It supports
atmospheric boundary layers, wind-energy flows, mapped meshes, and cryogenic
low-Mach cases with FFT, geometric multigrid, or optional AMG pressure
projection.

## Install

JAX-Wind requires Python 3.11 or newer:

```bash
python -m pip install -e .
```

The optional GPU AMG backend is available from the `external/jax-amg`
submodule.

Install the JAX build appropriate for the CPU or accelerator on the target
machine.

The default `pytest` collection is a curated core suite of fewer than 50
solver and active LN₂/wind-farm contracts. Publication reproductions,
multi-process checks, backend-specific oracles, and visualization tests are
kept as an opt-in extended suite:

```bash
pytest
pytest -o 'python_files=test_*.py'
```

## Run a finite-volume ABL case

The atmospheric cases use the staggered finite-volume solver, AB2,
and its direct FFT pressure backend. The FV path currently uses AMD momentum
closure and an eddy-diffusivity passive scalar, so its resolved output records
that closure distinction from other closure formulations. Its extended diagnostics
supply every currently registered Andrén overlay (Figures 2--8, 11, 14, and
15):

```bash
python -m applications.fv_abl cases/Andren1994/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl cases/Andren1994/config.toml \
  --max-steps 10 --overwrite
```

The FV core also provides an opt-in variable-density low-Mach formulation.
`IdealGasMixture` evaluates thermodynamic density from temperature, a scalar
or hydrostatic base-state pressure field, and
transported gas mass fractions, `conservative_specific_tendency` advances a
density-weighted scalar inventory, and `project_low_mach` corrects face
momentum so that
`(rho_new-rho_old)/dt + div(rho*u) = mass_source` holds discretely. Because
the correction is applied to momentum, this conservative formulation reuses
the scalable FFT/GMG pressure operators. The HITSZ LN2 runner uses this path;
the ABL runners remain backward-compatible constant-density/Boussinesq cases.

The FV mesh can also be a separable analytical mapping. `AnalyticalGrid` samples
user callables in normalized computational space before JAX tracing; built-in
`TanhMapping` and `SinhMapping` cover smooth clustering. For example, the
HITSZ inlet mesh uses `TanhMapping(2.2, focus=0.0)` in x and central tanh
branches on the nozzle y-z axis. Fluxes, SGS widths, wall sources, parcels,
scalars, and pressure projection use local cell widths/volumes. Mapped meshes
use GMG or AMG because an FFT direction must remain uniform.

For open-streamwise calculations, the FV application also provides a
configuration-driven warmup/precursor/main workflow. It develops and records
the periodic precursor with FFT, writes only one inflow layer per time step,
then enforces those layers in an open-x main run with GMG and a second-order
outflow condition:

```bash
python -m applications.fv_abl.workflow \
  cases/Andren1994/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl.workflow \
  cases/Andren1994/config.toml --max-steps 2 --overwrite
```

Nieuwstadt uses that same ABL command and schema. It also has an FV comparison
runner with Boussinesq coupling and FFT pressure projection:

```bash
python -m applications.fv_abl cases/Nieuwstadt1993/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl cases/Nieuwstadt1993/config.toml --overwrite
```

GABLS1 also has an FV runner with evolving surface temperature and coupled
Monin–Obukhov momentum and heat fluxes. Its complete overlay covers 27 of the
30 official diagnostics:

```bash
python -m applications.fv_abl cases/GABLS1/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl cases/GABLS1/config.toml \
  --output gabls1_fv_fft_overlays
python tools/overlay_gabls1.py gabls1_fv_fft_overlays \
  --output-dir gabls1_fv_fft_overlays
```

## Solver boundary

`jaxwind` is the only numerical solver API. Applications construct FV
state, operators, pressure projection, and integration directly from their
case data. FFT and multigrid are pressure backends inside this solver, not
separate flow solvers.

## Wind-farm turbine models

`jaxwind.windfarm` owns physical turbine parameterizations and OpenFAST input
adapters. The smallest turbine model is a uniform, non-rotating actuator disk
specified in SI units and lowered explicitly to the solver scales:

```python
from jaxwind.physics import WindTunnelModel
from jaxwind.windfarm import SimpleActuatorDisk

turbine = SimpleActuatorDisk(
    x_m=400.0,
    y_m=250.0,
    hub_height_m=90.0,
    rotor_diameter_m=120.0,
    thrust_coefficient_prime=4.0 / 3.0,
    smoothing_width_m=10.0,
)
wind_tunnel = WindTunnelModel(
    actuator_disk=turbine.to_actuator_disk(scales=scales),
)
```

The disk reuses the force-conserving, filtered pure-thrust kernel. The upstream
OpenFAST source is pinned under `src/jaxwind/windfarm/reference/openfast` only
for implementation and input-format reference; it is excluded from package
discovery and is never imported or linked by JAX-Wind.

## Package structure

| Path | Responsibility |
| --- | --- |
| `src/jaxwind` | Finite-volume state, operators, pressure projection, closures, and integration |
| `src/jaxwind/domain` | Uniform and analytically mapped grids plus physical scales |
| `src/jaxwind/physics` | Shared physical configuration values |
| `src/jaxwind/windfarm` | Turbine parameterizations and OpenFAST input adapters |
| `applications/fv_abl` | Atmospheric FV case execution and precursor workflows |
| `applications/fv_ln2_jet` | Cryogenic low-Mach jet execution |
| `cases` | Data-only case configurations and reference evidence |

## Verify

```bash
python -m pytest -q
```

The default suite covers FV operators, pressure projection, conservative
physics, mapped grids, cryogenic low-Mach flow, turbine forcing, and FV case
composition.

JAX-Wind is released under the [MIT License](LICENSE).
