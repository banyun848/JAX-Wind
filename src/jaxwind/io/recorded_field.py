"""Sliceable chunked inflow fields for streaming postprocessing."""
from pathlib import Path
import json
import numpy as np
from .recording import SCHEMA


class RecordedField:
    def __init__(self, path):
        path = Path(path)
        self.directory = path.parent
        self.name = path.stem
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if self.metadata.get("schema") != SCHEMA:
            raise ValueError("unsupported inflow schema")
        if not self.metadata["chunks"]:
            raise ValueError("inflow recording is empty")
        with np.load(self.directory / self.metadata["chunks"][0]["file"], allow_pickle=False) as archive:
            self.shape = (self.metadata["samples"], *archive[self.name].shape[1:])
            self.dtype = archive[self.name].dtype

    def __getitem__(self, index):
        if isinstance(index, int):
            return self[index:index+1][0]
        start, stop, stride = index.indices(self.shape[0])
        if stride != 1:
            raise ValueError("recorded fields support contiguous sample slices")
        parts, offset = [], 0
        for chunk in self.metadata["chunks"]:
            end = offset + chunk["samples"]
            if start < end and stop > offset:
                with np.load(self.directory / chunk["file"], allow_pickle=False) as archive:
                    parts.append(np.asarray(archive[self.name][max(0, start-offset):min(chunk["samples"], stop-offset)]))
            offset = end
        return np.concatenate(parts) if parts else np.empty((0, *self.shape[1:]), dtype=self.dtype)
