# MPI Labs — MSc in HPC coursework

Hands-on MPI exercises developed during the MSc in High-Performance Computing and run on **CESGA FinisTerrae-3**. They progress from rank basics to point-to-point communication and the collective operations used by the solvers in `src/`.

## Exercises

| File | Topic |
|---|---|
| `example1_hello_rank_size.c` | MPI init, rank and world size |
| `example2_send_receive.c` | Blocking point-to-point basics |
| `example3_send_receive.c` | Point-to-point with explicit tags |
| `example4_send_receive_vector.c` | Sending vectors (`MPI_Send`/`MPI_Recv` with counts) |
| `example5_send_receive_Deadlock.c` | Deadlock demonstration and how to avoid it |
| `example6_receive.c` | Receiving with `MPI_ANY_SOURCE`/`MPI_STATUS_IGNORE` |
| `example7_Broadcast.c` | `MPI_Bcast` |
| `example8_Scatter.c` | `MPI_Scatter` |
| `example9_reduce.c` | `MPI_Reduce` (dot product + timing statistics) |
| `example10_Scan.c` | `MPI_Scan` (prefix sums) |
| `example11_Reduce_scatter.c` | `MPI_Reduce_scatter` |
| `example12_Pi_integral.c` | Domain decomposition: numerical integration of pi |

## Build and run (local)

```bash
make all
mpirun -np 4 ./example9_reduce
```

## Run on CESGA FinisTerrae-3

`job_ft3.sh` captures the real job configuration used during the master's labs: Intel toolchain via `module load intel impi`, 64 cores per node, `--mem-per-cpu 1G`.

```bash
module load intel impi
make all
sbatch job_ft3.sh
squeue -u $USER
```
