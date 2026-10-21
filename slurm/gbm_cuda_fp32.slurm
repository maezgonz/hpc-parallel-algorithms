#!/bin/bash
#SBATCH -J gbm_fp32
#SBATCH -o gbm_fp32-%j.out
#SBATCH -e gbm_fp32-%j.err
#SBATCH -c 32
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 00:10:00

set -euo pipefail
module load cuda/system
cd "$SLURM_SUBMIT_DIR"
mkdir -p bin
nvcc -O3 --use_fast_math -o bin/gbm_cuda_fp32 src/gbm_cuda_fp32.cu

# FP32 sweep at block=1024 (proven optimal from FP64 sweep)
for N in 10000000 100000000 500000000; do
  srun ./bin/gbm_cuda_fp32 100 0.08 0.25 252 $N 1024
  echo "---"
done
