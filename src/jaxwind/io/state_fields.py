"""Explicit field adapters for new-format initialization and analysis."""
from __future__ import annotations
import numpy as np
from .checkpoint import checkpoint_metadata


def initialize_state(path, template, grid, formulation):
    from .checkpoint import _restore
    header = checkpoint_metadata(path)
    if header.get("formulation") != formulation:
        raise ValueError("initialization formulation differs; use an explicit conversion")
    for axis in ("x", "y", "z"):
        if not np.array_equal(header["mesh"][axis + "_faces"], np.asarray(getattr(grid, axis + "_faces"))):
            raise ValueError("initialization mesh differs; use an explicit transformation")
    with np.load(path, allow_pickle=False) as arrays:
        return _restore(header["state"], arrays, template)


def state_fields(path):
    header = checkpoint_metadata(path)
    with np.load(path, allow_pickle=False) as source:
        fields = {name.removeprefix("state/").replace("/", "_"): np.array(source[name]) for name in source.files if name.startswith("state/")}
    return fields, header


def atmospheric_state(path, grid):
    import jax.numpy as jnp
    from jaxwind import AtmosphericSolution, StaggeredVelocity, validate
    fields, header = state_fields(path)
    if header.get("formulation") != "boussinesq":
        raise ValueError("initialization requires a Boussinesq checkpoint")
    for axis in ("x", "y", "z"):
        expected = np.asarray(getattr(grid, axis + "_faces"))
        actual = np.asarray(header["mesh"][axis + "_faces"])
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError("checkpoint mesh differs; use an explicit mesh transformation")
    vector = lambda name: StaggeredVelocity(*(jnp.asarray(fields[name + "_" + axis]) for axis in ("x", "y", "z")))
    state = AtmosphericSolution(vector("velocity"), jnp.asarray(fields["pressure"]), vector("momentum_tendency"),
                                jnp.asarray(fields["scalar"]), jnp.asarray(fields["scalar_tendency"]),
                                jnp.asarray(fields["time"]), jnp.asarray(fields["step"]))
    validate(state.velocity, grid)
    return state
