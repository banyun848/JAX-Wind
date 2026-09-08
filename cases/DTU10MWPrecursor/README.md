# DTU 10-MW finite-volume AD-BEM benchmark

> Run all commands below on a compute node, including configuration checks.
> This case uses schema version 1. Historical outputs cannot be resumed;
> regenerate inputs in the new format. See [verification](../../doc/verification.md).

## Finite-volume open-domain workflow

[`fv_workflow.toml`](fv_workflow.toml) runs the same `128 x 64 x 256` domain
and fixed DTU operating point through the FV warmup/precursor/main workflow.
Its warmup and one-hour precursor use the periodic FFT projection. The
precursor records one `yz` layer every `0.1 s`; the main domain enforces those
layers at its inlet, disables the background pressure force, uses the
second-order open outlet, and projects with GMG. AD-BEM, nacelle, and tower
loads are active only in the main stage.

The turbine declaration remains data. Set its configured environment variable
to an AeroDyn15-compatible DTU 10 MW OpenFAST deck, then inspect or run it:

```bash
export JAXWIND_DTU10MW_FAST=/path/to/DTU_10MW_AeroDyn15.fst
jaxwind check \
  cases/DTU10MWPrecursor/fv_workflow.toml
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  jaxwind workflow \
  cases/DTU10MWPrecursor/fv_workflow.toml
```

The complete configuration advances 360,000 warmup steps, records 36,000
precursor layers, and advances the turbine domain for 36,000 steps. The four
recorded inflow fields contain about 9.5 GB of uncompressed float32 data, substantially less
than the former 11-plane HDF5 recording. Use `--max-steps 2` for a
full-resolution paused stage; continue with `--resume`. Artifacts use NPZ chunks.

The workflow preserves the physical domain, pressure driving, roughness, turbine
geometry, fixed rotor speed, and stage durations while using the FV AMD closure
and open-boundary discretization.
