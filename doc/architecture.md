# Solver architecture

The direct staggered finite-volume numerical implementation remains the solver.
The migration separates its equations from construction, execution, and user
interfaces. Numerical verification is pending compute-node execution; see
[verification instructions](verification.md).

## Ownership and dependencies

`config` resolves SI-valued case data. `simulation` constructs physical models,
initial state and compiled block advancement. `runtime.engine` drives the
simulation, schedules observations, and writes checkpoints. `workflows.engine`
connects simulation stages by named artifacts and delegates advancement to that
same runtime. `io` owns file formats and initialization adapters.

`numerics` owns discretization, pressure solvers, and integration primitives.
The existing physical tendency modules and `physics`/`windfarm` parameter types
remain reusable numerical building blocks. `formulations` owns coupled state
and numerical helpers. It never imports configuration or execution layers.
The larger cryogenic assembly factory is in `simulation.jet`; its numerical
stage ordering is preserved rather than rewritten during the ownership move.

Installed modules do not import `applications`. Experiment-specific analysis
lives under `tools`. Former application entry points have been removed.

## Developer interface

```python
from jaxwind import load_case, build_simulation, RunControls, run

case = load_case("case.toml")
simulation = build_simulation(case)
state = simulation.initialize()
state = simulation.advance(state, RunControls(count=10, target_time=1.0))
result = run(case, output="runs/example")
```

Run this example only on a compute node. `target_time` bounds existing adaptive
controllers; fixed-step kernels retain their configured timestep. Workflow
recorders may return `AdvanceResult(state, outputs)` for runtime persistence.
State types remain formulation-specific immutable JAX pytrees.

Add a closure by implementing its physical tendency and configuration validation,
then composing it in the builder. Add a pressure backend through the existing
pressure-solver contract and explicitly validate its mesh/boundary capabilities.
Add a formulation by defining state and initialization/advancement construction;
reuse runtime execution and persistence. Do not copy an application loop.

## Case schema

Version 1 uses `schema_version`, an explicit `formulation`, and tables for
`case`, `mesh`, `physics`, `numerics`, `time`, `diagnostics`, `initial_conditions`,
and `output`. Formulations are `boussinesq`, `low-mach-abl`,
`cryogenic-incompressible`, and `cryogenic-low-mach`.

Physical model parameters remain grouped by their meaning, for example
`physics.flow`, `physics.scalar`, `physics.thermodynamics`, and `physics.jet`.
Typed internal adapters preserve each existing formulation's numerical choices;
the schema does not make unsupported combinations available.

`extends` names one parent TOML. Tables merge recursively; arrays and scalar
values replace. Input paths resolve relative to the declaring file. Output
paths are relative to invocation. A run saves resolved TOML. Case derivation
writes a small inherited case with independent identity/output settings.
Resolution derivation explicitly selects `case.profile_resampling = "linear"`
for tabulated ABL inputs. It resamples cell-centered means/RMS and vertical-face
velocity on their respective coordinates, clamps endpoints, and preserves wall
impermeability. Existing cases default to strict matching. This is an explicit
initial-data transformation, not a change to the evolution algorithm or restart
history. Low-Mach mesh overrides apply to its declared physical source case.

For Boussinesq adaptive flow, `time.cfl` controls the existing controller and
`time.dt_seconds` is its cap. `time.steps * time.dt_seconds` remains the physical
duration convention. Other formulations retain fixed steps. A CFL option for
a fixed-step formulation is rejected. `--max-steps` limits one invocation and
does not change its saved target.
Cryogenic builders currently require GMG and float32; periodic low-Mach ABL
requires FFT/fast-RK3 and float32. Unsupported backend/precision selections
are rejected instead of being silently ignored.

## Workflows and artifacts

A workflow can declare `[stages.NAME]` with `case`, `operation`, `inputs`,
`overrides`, and `options`. Operations are `simulation`, `periodic`,
`record-inflow`, and `open-inflow`. Inputs bind to paths or
`@STAGE/checkpoint` / `@STAGE/inflow`. Names are labels, not physics selectors.

Atmospheric cases retain a `[workflow]` convenience recipe. The workflow loader
expands it to the same stage nodes before execution. Stages write below their
own directories; `--stage main` includes its declared dependencies.
`jaxwind check WORKFLOW.toml` resolves an explicit graph without requiring
artifacts that its upstream stages have not produced yet. Artifact compatibility
is checked when each stage is constructed. Recipe periodic/recording/open stages
write runtime histories and configured frames; benchmark profile accumulation
belongs to the `simulation` operation, not the inflow adapters.

Checkpoints contain complete state, integrator history, observer accumulators,
event progress, resolved case, mesh, units, and JAX/device metadata. Writes use
an atomic replacement and a versioned schema without pickle. Exact resume
requires a matching configuration fingerprint. Transformed checkpoints are
initialization inputs, not exact-resume runs.

Inflow artifacts contain independently written NPZ chunks plus a versioned
manifest with coordinates and sample counts. Blocks include timestamps and
timestep values. Open-flow consumption validates mesh and fixed cadence;
variable-cadence recordings cannot drive the existing fixed-step open integrator.

Historical checkpoint and inflow files are deliberately unsupported. Continuation
examples declare inputs that must be regenerated in the new format. Do not
rename an old NPZ file to make it appear to be a new checkpoint.

## Migration map

| Previous interface | New interface |
| --- | --- |
| `python -m applications.fv_abl CASE` | `jaxwind run CASE` |
| `python -m applications.fv_abl.workflow CASE` | `jaxwind workflow CASE` |
| Low-Mach / LN2 application commands | `jaxwind run CASE` |
| `--dry-run` | `jaxwind check CASE` |
| `--overwrite` | New output directory, or explicit resume |
| `domain.cells` | `mesh.cells` |
| `finite_volume.cfl_ceiling` | `time.cfl` |
| `finite_volume.*` numerical controls | `numerics.*` |
| Application-private helpers | Public owning package modules |

ABI/schema evolution follows ADR-0018. The historical semantic/interpreter
design records remain historical and are not new implementation requirements.
