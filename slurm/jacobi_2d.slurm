#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error
#SBATCH --time=0-00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH -c 32
#SBATCH --mem-per-cpu 1G

set -euo pipefail
module load intel impi
mkdir -p logs
mpirun ./bin/jacobi_2d 1024 5000 1e-4 field_rank.csv
