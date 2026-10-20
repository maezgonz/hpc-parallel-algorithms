#!/bin/bash
#SBATCH -J cuda_vec_add
#SBATCH -o cuda_vec_add-%j.out
#SBATCH -e cuda_vec_add-%j.err
#SBATCH -c 32
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 00:10:00

set -euo pipefail
module load cuda/11.2
cd "$SLURM_SUBMIT_DIR"
nvcc -O3 -o bin/vector_add_cuda src/vector_add_cuda.cu
# Print device info, then benchmark at 3 sizes
srun ./bin/vector_add_cuda 0
srun ./bin/vector_add_cuda 1048576 1024
srun ./bin/vector_add_cuda 16777216 1024
srun ./bin/vector_add_cuda 134217728 1024
