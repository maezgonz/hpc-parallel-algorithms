#!/bin/bash
#SBATCH --job-name=jacobi-2d
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load gcc openmpi
mkdir -p logs
srun ./bin/jacobi_2d 1024 5000 1e-4 field_rank.csv
