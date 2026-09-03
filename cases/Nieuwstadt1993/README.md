# Nieuwstadt et al. (1993)

This data-only case reproduces the dry, shear-free boundary-layer comparison
on the paper 40 x 40 x 48 grid. Scalar initialization, surface flux, and
buoyancy coupling determine the dynamics without a case-specific solver mode.

Run the finite-volume case with hydrostatic-free Boussinesq coupling, AMD,
AB2, and the configured pressure backend:

```bash
python -m applications.fv_abl cases/Nieuwstadt1993/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl cases/Nieuwstadt1993/config.toml --overwrite
```

Use `--max-steps 10 --overwrite` for a short smoke run. The application writes
`profiles.csv`, `radial_spectra.csv`, and `summary.json`; the fixed turbulent
Prandtl number and spectrum policy are declared in `[finite_volume]`.

Compare a completed result with the included reference figures:

```bash
python tools/overlay_nieuwstadt1993.py \
  --results outputs/nieuwstadt1993_fv_gmg_40x40x48 \
  --legend-label "JAX-Wind FV AMD GMG 40×40×48 GPU"
```
