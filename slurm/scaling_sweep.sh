#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error
#SBATCH --time=0-00:15:00
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem-per-cpu 1G

set -euo pipefail
module load intel impi
OUT=benchmarks/mcpi_scaling_ft3.csv
echo 'threads,wall_time' > "$OUT"
for T in 1 2 4 8 16 32 64; do
  LINE=$(OMP_NUM_THREADS=$T ./bin/monte_carlo_pi 1000000000)
  W=$(echo "$LINE" | grep -oP 'wall_s=\K[0-9.]+')
  NT=$(echo "$LINE" | grep -oP 'threads=\K[0-9]+')
  echo "$T,$W" >> "$OUT"
  echo "sweep threads_set=$T threads_reported=$NT wall_s=$W"
done
echo "--- result ---"
cat "$OUT"
