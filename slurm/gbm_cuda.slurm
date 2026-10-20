#!/bin/bash
#SBATCH -J gbm_cuda
#SBATCH -o gbm_cuda-%j.out
#SBATCH -e gbm_cuda-%j.err
#SBATCH -c 32
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 00:15:00

set -euo pipefail
module load cuda/system
cd "$SLURM_SUBMIT_DIR"
mkdir -p bin
nvcc -O3 -o bin/gbm_cuda src/gbm_cuda.cu

# Sweep: 10M, 100M, 500M paths (same parameters as gbm_engine.c)
for N in 10000000 100000000 500000000; do
  srun ./bin/gbm_cuda 100 0.08 0.25 252 $N 256
  echo "---"
done

# Block size sweep at 100M
for B in 128 256 512 1024; do
  srun ./bin/gbm_cuda 100 0.08 0.25 252 100000000 $B
  echo "---"
done
