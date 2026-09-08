"""Recorded inflow analysis adapters."""
from pathlib import Path
from .recording import InflowReader


def load_inflow_block(directory: Path, start: int, stop: int, jnp=None):
    return InflowReader(directory).read(start, stop)
