#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error
#SBATCH --time=0-00:15:00
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem-per-cpu 1G

set -euo pipefail
module load intel impi
# Weak scaling: constant work per thread (25M samples/thread)
OUT=benchmarks/mcpi_weak_scaling_ft3.csv
PER_THREAD=25000000
echo 'threads,total_samples,wall_time' > "$OUT"
for T in 1 2 4 8 16 32 64; do
  N=$((PER_THREAD * T))
  LINE=$(OMP_NUM_THREADS=$T ./bin/monte_carlo_pi "$N")
  W=$(echo "$LINE" | grep -oP 'wall_s=\K[0-9.]+')
  echo "$T,$N,$W" >> "$OUT"
  echo "threads=$T total=$N wall_s=$W"
done
echo "--- result ---"
cat "$OUT"
