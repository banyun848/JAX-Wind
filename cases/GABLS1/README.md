# GABLS1

> Run all commands below on a compute node, including configuration checks.
> This case uses schema version 1. Historical outputs cannot be resumed;
> regenerate inputs in the new format. See [verification](../../doc/verification.md).

This is the canonical 32³, 400 m GABLS1 case integrated for nine hours, with
statistics collected over hours 8–9. The physical data includes an evolving
surface potential temperature of −0.25 K h⁻¹. Coupled Monin–Obukhov exchange
sets momentum and scalar surface fluxes, while an advection frame removes the
uniform 8 m s⁻¹ translation from evolved fields.

Run or inspect the finite-volume configuration:

```bash
PYTHONPATH=src:. jaxwind check \
  cases/GABLS1/config.toml
PYTHONPATH=src:. JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  jaxwind run cases/GABLS1/config.toml
```

The solver uses AMD closure, AB2 integration, and the pressure backend selected
in `[numerics]`. After the statistics window, compare the profiles with
the included official participant ensemble:

```bash
PYTHONPATH=src:. python tools/overlay_gabls1.py \
  outputs/gabls1_fv_gmg_32x32x32 \
  --output-dir outputs/gabls1_fv_gmg_32x32x32
```

The FV overlay supplies 27 of the 30 official quantities. Boundary-layer
height, TKE storage, and total TKE transport remain reference-only because the
statistics accumulator does not expose equivalent fields. The refined 64³
configuration is available as [`config_64.toml`](config_64.toml).
