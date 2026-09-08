#!/usr/bin/env bash
#SBATCH --job-name=jaxwind-bench
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --time=00:30:00
#SBATCH --output=jaxwind-bench-%j.out
#SBATCH --error=jaxwind-bench-%j.err

set -euo pipefail
cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"

module purge
module load compiler/dtk/26.04
module load mpi/openmpi/openmpi-4.1.5-gcc9.3.0
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-jax060}"

export LD_LIBRARY_PATH="/public/software/compiler/dtk-26.04/dcc/gcvm/lib:${LD_LIBRARY_PATH:-}"
MPI_LIBDIR="$(mpicc --showme:libdirs | awk '{print $1}')"
export LD_PRELOAD="${MPI_LIBDIR}/libmpi.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --tres-per-task=gres/dcu:1 --kill-on-bad-exit=1 \
  python tools/benchmark_hitsz_step.py \
    --block-steps 100 --samples 7 --component-repeats 20 \
    --json "benchmark_results/refactor-rocm-${SLURM_JOB_ID}.json" "$@"
