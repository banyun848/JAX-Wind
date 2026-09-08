"""Atmospheric field adapter for analysis of versioned checkpoints."""
from pathlib import Path
from .field_archive import open_fields


def load_solution(path: Path, jnp):
    from jaxwind import AtmosphericSolution, StaggeredVelocity
    with open_fields(path) as data:
        vector = lambda name: StaggeredVelocity(*(jnp.asarray(data[name + "_" + axis]) for axis in ("x", "y", "z")))
        return AtmosphericSolution(vector("velocity"), jnp.asarray(data["pressure"]), vector("momentum_tendency"),
                                   jnp.asarray(data["scalar"]), jnp.asarray(data["scalar_tendency"]),
                                   jnp.asarray(data["time"]), jnp.asarray(data["step"]))
