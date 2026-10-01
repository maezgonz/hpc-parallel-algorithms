#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error
#SBATCH --time=0-00:10:00
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem-per-cpu 1G

set -euo pipefail
module load intel impi
mkdir -p logs
srun ./bin/monte_carlo_pi 1000000000
