# Andrén et al. (1994) neutral ABL case

> Run all commands below on a compute node, including configuration checks.
> This case uses schema version 1. Historical outputs cannot be resumed;
> regenerate inputs in the new format. See [verification](../../doc/verification.md).

This data-only case is configured by the fixed-schema
[`config.toml`](config.toml) and composed by the finite-volume
[`fv_abl`](../../src/jaxwind/config/abl.py) application. The TOML
contains canonical SI inputs: the grid, Coriolis and
geostrophic values, wall roughness, passive-scalar flux, initial profile,
physical times, and numerical controls. The composition owns SI-to-execution
scaling; it performs no execution and never dispatches on the case name.

There is no neutral/stable/convective selector. This case's scalar is
explicitly passive, so it has no buoyancy feedback and the resolved stability
is the neutral limit.

The schema has no solver registry or case-specific solver. The `[numerics]`
table selects pressure, integration, and closure settings; `[time]`,
`[diagnostics]`, and `[output]` declare their own controls;
unknown tables and keys are rejected. Run the configured case with:

```bash
jaxwind check cases/Andren1994/config.toml
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  jaxwind run cases/Andren1994/config.toml \
  --max-steps 10 --output /tmp/andren1994-fv-smoke
```

This FV realization uses AMD for momentum and its eddy viscosity for passive-
scalar diffusion. `resolved_case.toml` and `summary.json` record that choice. During the configured
statistics window it writes total momentum and scalar fluxes, signed AMD TKE
transfer, momentum and scalar diffusivities, streamwise spectra, total resolved
TKE history, and momentum-stationarity metrics. AMD has no prognostic SGS
TKE, so the reported modeled SGS-TKE contribution is explicitly zero rather than an inferred prognostic quantity.

## FV warmup, precursor, and enforced-main workflow

The `[workflow]` table in the same case TOML supplies only stage
lengths, the recorded x-plane, chunking, and output location. The workflow
fixes the pressure and boundary choices required by each stage: warmup and
precursor are periodic and use FFT, while the enforced main run is open in x
and uses GMG. Display the resolved contract without constructing simulation fields:

```bash
jaxwind check \
  cases/Andren1994/config.toml
```

Run the complete chain, or run each restartable stage separately:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  jaxwind workflow \
  cases/Andren1994/config.toml

jaxwind workflow \
  cases/Andren1994/config.toml --stage warmup
jaxwind workflow \
  cases/Andren1994/config.toml --stage precursor --resume
jaxwind workflow \
  cases/Andren1994/config.toml --stage main --resume
```

The precursor stores exactly one `yz` layer per time step in versioned NPZ chunks under `precursor/inflow/`: the three staggered velocity
components and scalar. The main domain directly enforces the matching layer at
its inlet. At the outlet, tangential velocity and scalar use the three-point
second-order zero-gradient extrapolation; pressure projection selects the
normal outflow velocity using inlet-Neumann/outlet-Dirichlet pressure
conditions. Because x is not periodic in this stage, attempting to construct
its pressure solve with FFT is rejected.

For a short end-to-end smoke run, `--max-steps 2` pauses the current stage after two steps while
retaining the same boundary and backend choices.

The reference profile and published comparison envelope live under
[`reference`](reference/).

Clean crops of all 19 published figures live under
[`reference/figure_panels`](reference/figure_panels/). Their source pages,
crop boxes, and active plot-axis registrations are recorded in
[`manifest.json`](reference/figure_panels/manifest.json).

Overlay a completed active run on the directly comparable profile panels:

```bash
python tools/overlay_andren1994.py \
  outputs/andren1994_fv_gmg_40x40x40
```

This writes individual overlays for Figures 2 through 8, 11, 14, and 15, a
compact diagnostic sheet, and a complete 19-figure sheet under the
result directory's `paper_overlays/`. Figures 2, 3, 6, 8, 11, 14, and 15 use
the extended restartable diagnostics: total TKE, component momentum
stationarity, resolved-plus-SGS fluxes, signed SGS TKE transfer, SGS
diffusivities, and streamwise spectra. Reference-only panels are labeled
explicitly when an older run did not record their required observable. The
tool reads `history.csv`, `profiles.csv`, `spectra.csv`, and `summary.json`; it
does not run or reconfigure the solver. To reproduce the checked-in crops from
the DLR article scan:

```bash
python tools/overlay_andren1994.py \
  --extract-from-pdf /path/to/qj-1457-1994.pdf
```
