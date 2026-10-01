/**
 * @file  mpi_pingpong.c
 * @brief MPI point-to-point latency and bandwidth benchmark (ping-pong).
 *
 * Ranks 0 and 1 exchange messages of increasing size; round-trip time is
 * measured with MPI_Wtime and reported as half-RTT latency and one-way
 * bandwidth. Standard pattern for characterizing interconnects such as
 * the InfiniBand fabric of CESGA FinisTerrae-3.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BYTES (8 * 1024 * 1024)
#define WARMUP 10
#define REPEATS 100

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "error: need at least 2 ranks\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    char *buffer = malloc(MAX_BYTES);
    if (buffer == NULL) {
        fprintf(stderr, "error: allocation failed for %d bytes\n", MAX_BYTES);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    memset(buffer, 1, MAX_BYTES);

    if (rank == 0) {
        printf("%10s %14s %18s\n", "bytes", "latency_us", "bandwidth_GBps");
    }

    for (int bytes = 1; bytes <= MAX_BYTES; bytes *= 2) {
        if (rank == 0) {
            for (int i = 0; i < WARMUP; ++i) {
                MPI_Send(buffer, bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer, bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else if (rank == 1) {
            for (int i = 0; i < WARMUP; ++i) {
                MPI_Recv(buffer, bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer, bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            const double t0 = MPI_Wtime();
            for (int i = 0; i < REPEATS; ++i) {
                MPI_Send(buffer, bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer, bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            const double total = MPI_Wtime() - t0;
            const double rtt_seconds = total / REPEATS;
            const double rtt_us = rtt_seconds * 1e6;
            const double bandwidth_GBps = (double)bytes / (rtt_seconds / 2.0) / 1e9;
            printf("%10d %14.2f %18.3f\n", bytes, rtt_us, bandwidth_GBps);
        } else if (rank == 1) {
            for (int i = 0; i < REPEATS; ++i) {
                MPI_Recv(buffer, bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer, bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(buffer);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
