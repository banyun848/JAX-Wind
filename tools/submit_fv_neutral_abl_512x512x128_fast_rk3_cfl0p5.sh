#!/bin/bash -l
#SBATCH --job-name=fv-512-fast-c05
#SBATCH --account=rzg_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=fv-neutral-512x512x128-fast-cfl0p5-%j.out
#SBATCH --error=fv-neutral-512x512x128-fast-cfl0p5-%j.err

set -euo pipefail

source ~/venvs/numba_cuda_waterboa/bin/activate_cuda_env.sh

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

steps="${STEPS:-20000}"
cfl="${CFL:-0.5}"
chunk="${CHUNK:-100}"
run_directory="outputs/fv_neutral_abl_512x512x128_fast_rk3_cfl0p5_${SLURM_JOB_ID}"
mkdir -p "$run_directory"

echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "repository=$repo_root"
echo "output=$run_directory"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun python -u tools/run_fv_neutral_abl.py \
  --cells 512 512 128 \
  --lengths 10240 10240 1280 \
  --friction 0.4 \
  --roughness 0.005 \
  --backend gmg \
  --scheme fast-rk3 \
  --precision float32 \
  --courant "$cfl" \
  --steps "$steps" \
  --spinup 0.5 \
  --chunk "$chunk" \
  --profile-every 2000 \
  --plot "$run_directory/loglaw_velocity_profile.png"
