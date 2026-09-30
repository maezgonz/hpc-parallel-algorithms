# hpc-parallel-algorithms

![License](https://img.shields.io/badge/License-MIT-blue)
![Language](https://img.shields.io/badge/C17-OpenMP%20%7C%20MPI-4A90D9)
![CI](https://github.com/maezgonz/hpc-parallel-algorithms/actions/workflows/ci.yml/badge.svg)

## Objective

A curated collection of parallel algorithms in C, developed and benchmarked as part of an MSc in High-Performance Computing. Workloads run on **CESGA FinisTerrae-3** (Galician Supercomputing Center) through Slurm. The repository demonstrates production-grade shared-memory (OpenMP) and distributed-memory (MPI) patterns with a focus on **low latency, cache-friendly access and measurable speedups**.

## Architecture

```text
┌────────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────────┐
│  src/*.c   │───▶│ Makefile │───▶│  Slurm jobs │───▶│  benchmarks/ │
│ algorithms │    │ -O3/march│    │  sbatch/srun│    │  timings CSV │
└────────────┘    └──────────┘    └─────────────┘    └──────────────┘
```

- **`src/`** — self-contained algorithm implementations, each compiled independently.
- **`slurm/`** — job scripts capturing resource requests (CPUs per task, time, partition) for reproducible FinisTerrae-3 runs.
- **`benchmarks/`** — scaling results (strong/weak) exported as CSV and plotted.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | C17 |
| Shared memory | OpenMP (reductions, static scheduling, per-thread RNG) |
| Distributed memory | MPI (roadmap) |
| Scheduler | Slurm (CESGA FinisTerrae-3) |
| Build | GNU Make, GCC (`-O3 -march=native`) |
| CI | GitHub Actions (build + smoke test) |

## Repository Layout

```text
hpc-parallel-algorithms/
├── src/
│   └── monte_carlo_pi.c      # OpenMP Monte Carlo pi, per-thread erand48 seeding
├── slurm/
│   └── monte_carlo_pi.slurm  # SBATCH wrapper: 1 node, 64 CPUs, 10 min
├── benchmarks/               # scaling results (CSV + plots)
├── Makefile
└── .github/workflows/ci.yml
```

## Execution

### Local build and run

```bash
make all
./bin/monte_carlo_pi 100000000
# samples=100000000 threads=8 pi=3.14167618 abs_error=8.35e-05 wall_s=0.412
```

### CESGA FinisTerrae-3 (Slurm)

```bash
# adapt partition and CPUs to the node topology (check with `sinfo`, `lscpu`)
sbatch slurm/monte_carlo_pi.slurm
squeue -u $USER
sacct -j <jobid> --format=JobID,Elapsed,NTasks,State
```

## Results

Strong-scaling study of the Monte Carlo kernel on FinisTerrae-3 (1× AMD EPYC node, 10⁹ samples):

| Threads | Wall time (s) | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | TBD | 1.00× | 100% |
| 8 | TBD | TBD | TBD |
| 32 | TBD | TBD | TBD |
| 64 | TBD | TBD | TBD |

![Strong scaling — Monte Carlo pi](benchmarks/strong_scaling.png)

> Figures are generated from `benchmarks/*.csv`; populated as runs complete.

### Benchmark harness

Scaling results are analyzed automatically: feed a CSV of `(threads, wall_time)` pairs and the harness derives speedup, parallel efficiency and a markdown table ready for this README — rendered in dark mode.

```bash
python benchmarks/plot_scaling.py --csv benchmarks/sample_scaling.csv \
  --title "Strong scaling - Monte Carlo pi" --output benchmarks/scaling.png
```

**Sample output** (illustrative data until FinisTerrae-3 runs land — see `benchmarks/sample_scaling.csv`):

| Threads | Wall time (s) | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 40.500 | 1.00x | 100% |
| 8 | 5.500 | 7.36x | 92% |
| 32 | 1.900 | 21.32x | 67% |
| 64 | 1.600 | 25.31x | 40% |

![Sample strong scaling](benchmarks/sample_scaling.png)

## Roadmap

- [x] Strong-scaling harness with automated dark-mode plotting
- [ ] 2D stencil (Jacobi) solver with OpenMP + MPI hybrid decomposition
- [ ] MPI point-to-point and collective communication benchmarks
- [ ] Weak-scaling study and NUMA-aware memory placement

## License

[MIT](LICENSE) — Matias Gonzalez
