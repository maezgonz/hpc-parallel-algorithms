#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error
#SBATCH --time=0-00:15:00
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem-per-cpu 4G

set -euo pipefail
module load intel impi
mkdir -p benchmarks
srun ./bin/gbm_engine 100 0.08 0.25 252 1000000000 benchmarks/gbm_bands.csv
