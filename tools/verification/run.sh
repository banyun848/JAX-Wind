#!/usr/bin/env bash
# Run ONLY inside a compute allocation. This script never installs packages.
set -euo pipefail

confirmed=0
forward=()
for argument in "$@"; do
    if [[ "$argument" == "--confirm-compute-node" ]]; then
        confirmed=1
    else
        forward+=("$argument")
    fi
done
if [[ "$confirmed" != 1 && ( -z "${SLURM_JOB_ID:-}" || -z "${SLURMD_NODENAME:-}" ) ]]; then
    echo "Refusing execution outside a compute step. Enter with srun, or explicitly pass --confirm-compute-node on a non-Slurm compute node." >&2
    exit 2
fi
export JAXWIND_COMPUTE_CONFIRMED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
verification_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${JAXWIND_PYTHON:-python3}" "$verification_dir/verify.py" "${forward[@]}"
