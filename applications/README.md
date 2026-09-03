# Applications

Applications translate data-only case configurations into finite-volume solver
components and own initialization, diagnostics, checkpoints, and output effects.
There is no solver selector or case-name registry.

`fv_abl` is the shared atmospheric application. Neutral, stable, and convective
behavior follows from the configured scalar coupling, initial stratification,
and surface exchange:

```bash
python -m applications.fv_abl cases/Andren1994/config.toml --dry-run
python -m applications.fv_abl cases/Nieuwstadt1993/config.toml --dry-run
python -m applications.fv_abl cases/GABLS1/config.toml --dry-run
```

The application also owns a restartable warmup, precursor, and open-domain
workflow. Warmup and precursor stages use the FV FFT pressure backend; the
open-x main stage uses GMG and a second-order outlet:

```bash
python -m applications.fv_abl.workflow \
  cases/Andren1994/config.toml --dry-run
```

Additional finite-volume applications cover mapped low-Mach ABL runs,
cryogenic jets, and experiment-specific postprocessing under `applications/`.
All pressure backend choices belong to `jaxwind`; they are not separate
flow solvers.
