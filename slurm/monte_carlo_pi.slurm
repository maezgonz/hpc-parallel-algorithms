#!/bin/bash
#SBATCH --job-name=mcpi-openmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load gcc
mkdir -p logs
srun ./bin/monte_carlo_pi 1000000000
