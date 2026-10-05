/**
 * @file  mpi_collectives.c
 * @brief Collective communication benchmark: Bcast, Scatter, Gather, Reduce, Allreduce.
 *
 * Times each collective over increasing message sizes (warmup + repeated
 * trials); every rank times locally and the maximum elapsed time is
 * reported via MPI_Allreduce, which measures the true duration of the
 * collective across the slowest participant. Standard pattern for
 * characterizing the InfiniBand fabric of CESGA FinisTerrae-3.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_COUNT (1 << 17)
#define WARMUP 5
#define REPEATS 50

/**
 * Synchronize all ranks and return the slowest rank's elapsed time for one
 * timed block, measured around the caller's collective loop. Must be
 * invoked by every rank: it contains an MPI_Allreduce.
 *
 * @param t0 Start timestamp captured with MPI_Wtime before the loop.
 * @return Maximum elapsed time across ranks.
 */
static double collect_max_elapsed(const double t0) {
    const double local = MPI_Wtime() - t0;
    double max_elapsed = 0.0;
    MPI_Allreduce(&local, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return max_elapsed;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *buffer = malloc(MAX_COUNT * sizeof(double));
    double *scratch = malloc(MAX_COUNT * sizeof(double));
    if (buffer == NULL || scratch == NULL) {
        fprintf(stderr, "error: allocation failed\n");
        free(buffer);
        free(scratch);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    for (int i = 0; i < MAX_COUNT; ++i) {
        buffer[i] = (double)(i % 7);
    }

    const int counts[] = {1, 16, 256, 4096, 65536, MAX_COUNT};
    const int n_counts = (int)(sizeof(counts) / sizeof(counts[0]));

    for (int c = 0; c < n_counts; ++c) {
        const int count = counts[c];
        const int chunk = count / size > 0 ? count / size : 1;
        double elapsed = 0.0;

        if (rank == 0) {
            printf("=== count=%d doubles (%d B/rank, %d ranks) ===\n",
                   count, chunk * 8, size);
        }

        for (int i = 0; i < WARMUP; ++i) {
            MPI_Bcast(buffer, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int i = 0; i < REPEATS; ++i) {
            MPI_Bcast(buffer, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        elapsed = collect_max_elapsed(t0);
        if (rank == 0) {
            printf("%-10s avg_us=%.2f\n", "Bcast", elapsed / REPEATS * 1e6);
        }

        for (int i = 0; i < WARMUP; ++i) {
            MPI_Scatter(buffer, chunk, MPI_DOUBLE, scratch, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < REPEATS; ++i) {
            MPI_Scatter(buffer, chunk, MPI_DOUBLE, scratch, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        elapsed = collect_max_elapsed(t0);
        if (rank == 0) {
            printf("%-10s avg_us=%.2f\n", "Scatter", elapsed / REPEATS * 1e6);
        }

        for (int i = 0; i < WARMUP; ++i) {
            MPI_Gather(buffer, chunk, MPI_DOUBLE, scratch, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < REPEATS; ++i) {
            MPI_Gather(buffer, chunk, MPI_DOUBLE, scratch, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        elapsed = collect_max_elapsed(t0);
        if (rank == 0) {
            printf("%-10s avg_us=%.2f\n", "Gather", elapsed / REPEATS * 1e6);
        }

        for (int i = 0; i < WARMUP; ++i) {
            MPI_Reduce(buffer, scratch, count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < REPEATS; ++i) {
            MPI_Reduce(buffer, scratch, count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        elapsed = collect_max_elapsed(t0);
        if (rank == 0) {
            printf("%-10s avg_us=%.2f\n", "Reduce", elapsed / REPEATS * 1e6);
        }

        for (int i = 0; i < WARMUP; ++i) {
            MPI_Allreduce(buffer, scratch, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < REPEATS; ++i) {
            MPI_Allreduce(buffer, scratch, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        elapsed = collect_max_elapsed(t0);
        if (rank == 0) {
            printf("%-10s avg_us=%.2f\n", "Allreduce", elapsed / REPEATS * 1e6);
        }
    }

    free(buffer);
    free(scratch);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
