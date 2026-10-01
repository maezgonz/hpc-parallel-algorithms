/**
 * @file  jacobi_2d.c
 * @brief 2D Laplace solver via Jacobi iteration, hybrid MPI + OpenMP.
 *
 * Global N x N grid decomposed by rows across MPI ranks; each rank
 * parallelizes its row block with OpenMP and exchanges halo rows with
 * MPI_Sendrecv. Fixed boundary conditions: top edge at 100, bottom edge
 * at 0, side columns linearly ramped between them. Designed for runs on
 * CESGA FinisTerrae-3 through the Slurm wrapper in slurm/.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define TAG_TO_UPPER 100
#define TAG_TO_LOWER 101
#define TOP_TEMP 100.0
#define BOTTOM_TEMP 0.0

/**
 * Number of global rows owned by a rank under balanced 1D decomposition.
 *
 * @param n Global grid size.
 * @param size Number of MPI ranks.
 * @param rank Rank index.
 * @return Rows owned by @p rank.
 */
static size_t block_rows(size_t n, int size, int rank) {
    return n / (size_t)size + ((int)rank < (int)(n % (size_t)size) ? 1 : 0);
}

/**
 * First global row owned by a rank under balanced 1D decomposition.
 *
 * @param n Global grid size.
 * @param size Number of MPI ranks.
 * @param rank Rank index.
 * @return First global row of @p rank.
 */
static size_t row_start_of(size_t n, int size, int rank) {
    const size_t base = n / (size_t)size;
    const size_t rem = n % (size_t)size;
    return rank < (int)rem ? (size_t)rank * (base + 1) : rem + (size_t)rank * base;
}

/**
 * Apply the fixed boundary conditions to the real rows owned by this rank.
 * Halo rows are left untouched: they are either filled by the halo
 * exchange or never read (ranks owning the physical top/bottom edges).
 *
 * @param grid Local array of (local_rows + 2) x n doubles.
 * @param local_rows Real rows owned by this rank.
 * @param row_start First global row owned by this rank.
 * @param n Global grid size.
 */
static void apply_boundary(double *grid, size_t local_rows, size_t row_start, size_t n) {
    for (size_t i = 1; i <= local_rows; ++i) {
        const size_t g = row_start + i - 1;
        double *row = &grid[i * n];
        if (g == 0) {
            for (size_t j = 0; j < n; ++j) {
                row[j] = TOP_TEMP;
            }
        } else if (g == n - 1) {
            for (size_t j = 0; j < n; ++j) {
                row[j] = BOTTOM_TEMP;
            }
        } else {
            const double edge =
                TOP_TEMP - (TOP_TEMP - BOTTOM_TEMP) * (double)g / (double)(n - 1);
            for (size_t j = 0; j < n; ++j) {
                row[j] = 0.0;
            }
            row[0] = edge;
            row[n - 1] = edge;
        }
    }
}

/**
 * Exchange halo rows with the up and down neighbors. Ranks owning the
 * physical top/bottom edges skip the corresponding exchange.
 *
 * @param curr Local array of (local_rows + 2) x n doubles.
 * @param local_rows Real rows owned by this rank.
 * @param n Global grid size.
 * @param rank Rank index.
 * @param size Number of MPI ranks.
 */
static void exchange_halos(double *curr, size_t local_rows, size_t n, int rank, int size) {
    if (rank > 0) {
        MPI_Sendrecv(&curr[1 * n], (int)n, MPI_DOUBLE, rank - 1, TAG_TO_LOWER,
                     &curr[0], (int)n, MPI_DOUBLE, rank - 1, TAG_TO_UPPER,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size - 1) {
        MPI_Sendrecv(&curr[local_rows * n], (int)n, MPI_DOUBLE, rank + 1, TAG_TO_UPPER,
                     &curr[(local_rows + 1) * n], (int)n, MPI_DOUBLE, rank + 1, TAG_TO_LOWER,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

/**
 * Perform one Jacobi sweep over the interior cells owned by this rank,
 * writing into @p next and tracking the maximum change per thread with an
 * OpenMP max reduction. Fixed boundary rows are never updated.
 *
 * @param curr Previous iterate, (local_rows + 2) x n.
 * @param next Next iterate, (local_rows + 2) x n.
 * @param local_rows Real rows owned by this rank.
 * @param row_start First global row owned by this rank.
 * @param n Global grid size.
 * @return Maximum change over this rank's interior cells.
 */
static double jacobi_sweep(const double *curr, double *next, size_t local_rows,
                           size_t row_start, size_t n) {
    double local_max = 0.0;

#pragma omp parallel for reduction(max : local_max) schedule(static)
    for (size_t i = 1; i <= local_rows; ++i) {
        const size_t g = row_start + i - 1;
        if (g == 0 || g == n - 1) {
            continue;
        }
        for (size_t j = 1; j < n - 1; ++j) {
            const double updated = 0.25 * (curr[(i - 1) * n + j] + curr[(i + 1) * n + j] +
                                           curr[i * n + j - 1] + curr[i * n + j + 1]);
            next[i * n + j] = updated;
            const double diff = fabs(updated - curr[i * n + j]);
            if (diff > local_max) {
                local_max = diff;
            }
        }
    }
    return local_max;
}

/**
 * Write the global field to a CSV file (rank 0 only).
 *
 * @param path Destination CSV path.
 * @param field Global field of n x n doubles.
 * @param n Global grid size.
 */
static void write_csv(const char *path, const double *field, size_t n) {
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", path);
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            fprintf(file, "%.6f%c", field[i * n + j], j + 1 < n ? ',' : '\n');
        }
    }
    fclose(file);
}

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 256;
    unsigned long long max_iters = 10000;
    double tol = 1e-4;
    const char *output = NULL;

    if (argc > 1) {
        char *end = NULL;
        n = strtoull(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || n == 0) {
            if (rank == 0) {
                fprintf(stderr, "error: invalid grid size '%s'\n", argv[1]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }
    if (argc > 2) {
        char *end = NULL;
        max_iters = strtoull(argv[2], &end, 10);
        if (end == argv[2] || *end != '\0' || max_iters == 0) {
            if (rank == 0) {
                fprintf(stderr, "error: invalid max iterations '%s'\n", argv[2]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }
    if (argc > 3) {
        char *end = NULL;
        tol = strtod(argv[3], &end);
        if (end == argv[3] || *end != '\0' || tol <= 0.0) {
            if (rank == 0) {
                fprintf(stderr, "error: invalid tolerance '%s'\n", argv[3]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }
    if (argc > 4) {
        output = argv[4];
    }

    if (n < 3 || (size_t)size > n) {
        if (rank == 0) {
            fprintf(stderr, "error: need n >= 3 and n >= ranks (n=%zu ranks=%d)\n", n, size);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const size_t local_rows = block_rows(n, size, rank);
    const size_t row_start = row_start_of(n, size, rank);
    const size_t padded = (local_rows + 2) * n;

    double *curr = calloc(padded, sizeof(double));
    double *next = calloc(padded, sizeof(double));
    if (curr == NULL || next == NULL) {
        fprintf(stderr, "error: allocation failed for %zu doubles\n", padded);
        free(curr);
        free(next);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    apply_boundary(curr, local_rows, row_start, n);
    apply_boundary(next, local_rows, row_start, n);

    unsigned long long iterations = 0;
    double residual = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (; iterations < max_iters; ++iterations) {
        exchange_halos(curr, local_rows, n, rank, size);
        const double local_max = jacobi_sweep(curr, next, local_rows, row_start, n);
        MPI_Allreduce(&local_max, &residual, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        double *swap = curr;
        curr = next;
        next = swap;
        if (residual < tol) {
            break;
        }
    }
    iterations += 1;

    const double elapsed = MPI_Wtime() - t0;

    if (output != NULL) {
        double *packed = malloc(local_rows * n * sizeof(double));
        if (packed == NULL) {
            fprintf(stderr, "error: allocation failed for output buffer\n");
        } else {
            for (size_t i = 0; i < local_rows; ++i) {
                memcpy(packed + i * n, &curr[(i + 1) * n], n * sizeof(double));
            }
            int *rcounts = NULL;
            int *displs = NULL;
            double *full = NULL;
            if (rank == 0) {
                rcounts = malloc((size_t)size * sizeof(int));
                displs = malloc((size_t)size * sizeof(int));
                full = malloc(n * n * sizeof(double));
                if (rcounts != NULL && displs != NULL && full != NULL) {
                    for (int r = 0; r < size; ++r) {
                        rcounts[r] = (int)(block_rows(n, size, r) * n);
                        displs[r] = (int)(row_start_of(n, size, r) * n);
                    }
                }
            }
            MPI_Gatherv(packed, (int)(local_rows * n), MPI_DOUBLE, full, rcounts, displs,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank == 0 && full != NULL) {
                write_csv(output, full, n);
            }
            free(rcounts);
            free(displs);
            free(full);
            free(packed);
        }
    }

    if (rank == 0) {
        printf("n=%zu ranks=%d threads_per_rank=%d iters=%llu residual=%.3e "
               "converged=%s wall_s=%.3f\n",
               n, size, omp_get_max_threads(), iterations, residual,
               residual < tol ? "yes" : "no", elapsed);
    }

    free(curr);
    free(next);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
