#!/usr/bin/env bash
# Supply account, partition, CPUs, memory, time and GPU resources to sbatch.
#SBATCH --job-name=jaxwind-verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=jaxwind-verify-%j.out
#SBATCH --error=jaxwind-verify-%j.err
set -euo pipefail

: "${JAXWIND_REPO:?Set JAXWIND_REPO to the candidate checkout}"
# Activate a preconfigured environment before submission or supply the absolute
# interpreter path in JAXWIND_PYTHON. No modules or packages are installed here.
srun --ntasks=1 bash "$JAXWIND_REPO/tools/verification/run.sh" "$@"
