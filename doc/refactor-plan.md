# Solver refactor plan and handoff

The implementation is on `feat/refactor`. The numerical acceptance gate is
still open: no Python, installation, build, tests, or simulations were run on
the login node. See [compute-node verification](verification.md).

## Scope

Move application ownership into the installed package without introducing an
inheritance-heavy solver framework. Keep direct JAX arrays, functional state
transitions, pressure backend contracts, numerical stage ordering, and existing
fixed/adaptive integration policies. Break old configuration/API/artifact
contracts deliberately; do not migrate historical output directories.

## Implementation sequence

1. Record boundaries in ADR-0018. Separate configuration, numerical kernels,
   formulation state, simulation construction, runtime, artifact I/O, workflows,
   and CLI ownership.
2. Add schema version 1 and declaration-relative inheritance. Lower shared
   documents into typed formulation inputs. Derive resolution/CFL cases with
   independent identity/output and explicit profile-resampling policy.
3. Move all active ABL, periodic low-Mach, and cryogenic builders into the
   package. Move discretization, projection, and integration into `numerics`.
4. Introduce one host runtime and `Simulation`/`RunControls` contract. Persist
   full state, histories, observer accumulators, and targets atomically. Separate
   exact resume from explicit initialization/conversion/prolongation.
5. Compose named workflow stages using checkpoint/inflow artifacts. Validate
   dependencies, geometry, cadence, and configuration identity; reuse runtime.
6. Migrate tracked cases, analysis consumers, launchers, and documentation to
   `jaxwind check/run/resume/workflow/case derive`. Remove old application code.
7. Provide bounded baseline/candidate trajectory scripts and regression tests.
   Execute them on compute, review failures and repeated performance measurements,
   and only then accept the numerical migration.

Steps 1–6 and the scripts/tests for step 7 are implemented. Execution and
acceptance of step 7 are pending, not inferred from source inspection.

## Acceptance checklist — compute node

- Installed CLI works outside the source checkout.
- Configuration inheritance, derived meshes, CFL rejection, and path resolution
  behave as documented; missing inputs fail clearly.
- Baseline/candidate fixed, adaptive, open-inflow, low-Mach, and cryogenic
  trajectories agree within declared tolerances; state remains finite.
- Projection, conservation, and differentiable inlet tests pass.
- Split/resumed runs preserve every state/history array and accumulated profiles.
- Workflow dependencies, inflow coverage, and incompatible artifacts are rejected
  correctly. Interrupted runs retain a complete recovery checkpoint.
- Repeat accelerator timings at production-relevant size before judging speed
  or memory regressions; the six-step timing matrix is only a smoke measurement.

## Intentional limits

This refactor does not add new numerical algorithms or historical artifact
readers. Fixed-step thermodynamic formulations do not acquire CFL adaptation.
Variable-cadence recordings cannot drive the existing fixed-step open integrator.
The cryogenic factory remains large so its coupling order stays explicit;
further decomposition should follow the verified numerical baseline.

Full-state NPZ checkpoints and checkpointed frame histories can add host memory
and I/O costs. Measure these on compute before large production runs. Configuration
fingerprints identify resolved settings, not the contents of arbitrary external
assets; preserve source profiles, model decks, and initialization artifacts for
reproducibility. Benchmark reference data and external OpenFAST/AMG dependencies
remain separately supplied inputs.
