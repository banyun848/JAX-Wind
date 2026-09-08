# Nieuwstadt et al. (1993)

> Run all commands below on a compute node, including configuration checks.
> This case uses schema version 1. Historical outputs cannot be resumed;
> regenerate inputs in the new format. See [verification](../../doc/verification.md).

This data-only case reproduces the dry, shear-free boundary-layer comparison
on the paper 40 x 40 x 48 grid. Scalar initialization, surface flux, and
buoyancy coupling determine the dynamics without a case-specific solver mode.

Run the finite-volume case with hydrostatic-free Boussinesq coupling, AMD,
AB2, and the configured pressure backend:

```bash
jaxwind check cases/Nieuwstadt1993/config.toml
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  jaxwind run cases/Nieuwstadt1993/config.toml
```

Use `--max-steps 10` for a short smoke run. The application writes
`profiles.csv`, `radial_spectra.csv`, and `summary.json`; the fixed turbulent
Prandtl number and spectrum policy are declared in `[numerics]`.

Compare a completed result with the included reference figures:

```bash
python tools/overlay_nieuwstadt1993.py \
  --results outputs/nieuwstadt1993_fv_gmg_40x40x48 \
  --legend-label "JAX-Wind FV AMD GMG 40×40×48 GPU"
```
