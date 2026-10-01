/**
 * @file  gbm_engine.c
 * @brief High-throughput Monte Carlo GBM engine (OpenMP, antithetic variates).
 *
 * Streams n_paths GBM trajectories accumulating exact terminal statistics
 * (Welford, merged per thread) while keeping a stride sample of complete
 * paths for per-day percentile bands. Antithetic variates halve the
 * effective sample count with variance reduction. The emitted CSV carries
 * a '# key=value' metadata header consumed by the Python bridge
 * (quant-trading-models, src/hpc_bridge.py). Designed for runs on CESGA
 * FinisTerrae-3 through the Slurm wrapper in slurm/.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define TWO_PI 6.28318530717958647692
#define LCG_MULT 6364136223846793005ULL
#define LCG_INC 1442695040888963407ULL
#define RNG_SEED 0x853C49E6748FEA9BULL
#define GOLDEN_RATIO 0x9E3779B97F4A7C15ULL
#define TARGET_SAMPLE_PATHS 2000

/**
 * Streaming accumulator for exact mean and variance (Welford's algorithm).
 */
typedef struct {
    double mean;
    double m2;
    unsigned long long count;
} RunningStats;

/**
 * Fold one observation into the accumulator.
 *
 * @param stats Accumulator to update.
 * @param x New observation.
 */
static void stats_update(RunningStats *stats, double x) {
    stats->count += 1;
    const double delta = x - stats->mean;
    stats->mean += delta / (double)stats->count;
    stats->m2 += delta * (x - stats->mean);
}

/**
 * Merge a second accumulator into the first (Chan et al. parallel formula).
 *
 * @param target Accumulator to merge into.
 * @param other Accumulator to merge from.
 */
static void stats_merge(RunningStats *target, const RunningStats *other) {
    if (other->count == 0) {
        return;
    }
    if (target->count == 0) {
        *target = *other;
        return;
    }
    const double delta = other->mean - target->mean;
    const double total = (double)(target->count + other->count);
    target->m2 += other->m2 + delta * delta *
                  (double)target->count * (double)other->count / total;
    target->mean += delta * (double)other->count / total;
    target->count += other->count;
}

/**
 * Draw one uniform sample in [0, 1) from a 64-bit LCG state.
 *
 * @param state Pointer to the caller-owned 64-bit LCG state.
 * @return Uniform double in [0, 1).
 */
static inline double next_uniform(unsigned long long *state) {
    *state = *state * LCG_MULT + LCG_INC;
    return (double)(*state >> 11) / 9007199254740992.0;
}

/**
 * Draw one standard normal sample via Box-Muller from two uniforms.
 *
 * @param state Pointer to the caller-owned 64-bit LCG state.
 * @return Standard normal double.
 */
static inline double next_gaussian(unsigned long long *state) {
    double u1 = next_uniform(state);
    if (u1 <= 0.0) {
        u1 = 1e-12;
    }
    const double u2 = next_uniform(state);
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

/**
 * Comparison function for qsort (ascending doubles).
 */
static int compare_doubles(const void *a, const void *b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
}

/**
 * Interpolated percentile from a pre-sorted array.
 *
 * @param sorted Ascending-sorted values.
 * @param n Number of values.
 * @param q Quantile in [0, 1].
 * @return Interpolated percentile value.
 */
static double percentile_sorted(const double *sorted, size_t n, double q) {
    const double pos = q * (double)(n - 1);
    const size_t lo = (size_t)pos;
    const size_t hi = lo + 1 < n ? lo + 1 : lo;
    const double frac = pos - (double)lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

int main(int argc, char **argv) {
    double s0 = 100.0;
    double mu = 0.08;
    double sigma = 0.25;
    long long horizon = 252;
    long long n_paths = 100000000;
    unsigned long long seed = 42;
    const char *output = NULL;

    if (argc > 1) {
        char *end = NULL;
        s0 = strtod(argv[1], &end);
        if (end == argv[1] || *end != '\0' || s0 <= 0.0) {
            fprintf(stderr, "error: invalid spot price '%s'\n", argv[1]);
            return EXIT_FAILURE;
        }
    }
    if (argc > 2) {
        char *end = NULL;
        mu = strtod(argv[2], &end);
        if (end == argv[2] || *end != '\0') {
            fprintf(stderr, "error: invalid drift '%s'\n", argv[2]);
            return EXIT_FAILURE;
        }
    }
    if (argc > 3) {
        char *end = NULL;
        sigma = strtod(argv[3], &end);
        if (end == argv[3] || *end != '\0' || sigma < 0.0) {
            fprintf(stderr, "error: invalid volatility '%s'\n", argv[3]);
            return EXIT_FAILURE;
        }
    }
    if (argc > 4) {
        char *end = NULL;
        horizon = strtoll(argv[4], &end, 10);
        if (end == argv[4] || *end != '\0' || horizon <= 0) {
            fprintf(stderr, "error: invalid horizon '%s'\n", argv[4]);
            return EXIT_FAILURE;
        }
    }
    if (argc > 5) {
        char *end = NULL;
        n_paths = strtoll(argv[5], &end, 10);
        if (end == argv[5] || *end != '\0' || n_paths <= 0) {
            fprintf(stderr, "error: invalid path count '%s'\n", argv[5]);
            return EXIT_FAILURE;
        }
    }
    if (argc > 6) {
        output = argv[6];
    }
    if (argc > 7) {
        char *end = NULL;
        seed = strtoull(argv[7], &end, 10);
        if (end == argv[7] || *end != '\0') {
            fprintf(stderr, "error: invalid seed '%s'\n", argv[7]);
            return EXIT_FAILURE;
        }
    }

    if (n_paths % 2 == 1) {
        n_paths += 1;
    }
    const long long n_pairs = n_paths / 2;
    long long stride = n_pairs / TARGET_SAMPLE_PATHS;
    if (stride < 1) {
        stride = 1;
    }
    const long long n_sample = n_pairs / stride + (n_pairs % stride ? 1 : 0);

    const int max_threads = omp_get_max_threads();
    RunningStats *thread_stats = calloc((size_t)max_threads, sizeof(RunningStats));
    double *sample = malloc((size_t)n_sample * (size_t)horizon * sizeof(double));
    double *shocks = malloc((size_t)horizon * sizeof(double));
    if (thread_stats == NULL || sample == NULL || shocks == NULL) {
        fprintf(stderr, "error: allocation failed\n");
        free(thread_stats);
        free(sample);
        free(shocks);
        return EXIT_FAILURE;
    }

    const double dt = 1.0 / 252.0;
    const double drift = (mu - 0.5 * sigma * sigma) * dt;
    const double vol_dt = sigma * sqrt(dt);
    RunningStats terminal_exact = {0.0, 0.0, 0};

    const double t0 = omp_get_wtime();

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        unsigned long long state =
            RNG_SEED ^ (GOLDEN_RATIO * (unsigned long long)(tid + 1));
        RunningStats local = {0.0, 0.0, 0};

#pragma omp for schedule(static)
        for (long long p = 0; p < n_pairs; ++p) {
            for (long long t = 0; t < horizon; ++t) {
                shocks[t] = next_gaussian(&state);
            }

            const int keep = (p % stride == 0);
            double *row = keep ? &sample[(p / stride) * horizon] : NULL;

            double log_price = 0.0;
            for (long long t = 0; t < horizon; ++t) {
                log_price += drift + vol_dt * shocks[t];
                if (row != NULL) {
                    row[t] = s0 * exp(log_price);
                }
            }
            stats_update(&local, s0 * exp(log_price));

            log_price = 0.0;
            for (long long t = 0; t < horizon; ++t) {
                log_price += drift - vol_dt * shocks[t];
            }
            stats_update(&local, s0 * exp(log_price));
        }

        thread_stats[tid] = local;
    }

    const double elapsed = omp_get_wtime() - t0;

    for (int t = 0; t < max_threads; ++t) {
        stats_merge(&terminal_exact, &thread_stats[t]);
    }
    const double exact_std = terminal_exact.count > 1
        ? sqrt(terminal_exact.m2 / (double)(terminal_exact.count - 1))
        : 0.0;

    if (output != NULL) {
        FILE *file = fopen(output, "w");
        if (file == NULL) {
            fprintf(stderr, "error: cannot open '%s' for writing\n", output);
        } else {
            fprintf(file, "# gbm_engine v1\n");
            fprintf(file, "# s0=%.6f mu=%.6f sigma=%.6f horizon=%lld n_paths=%lld seed=%llu\n",
                    s0, mu, sigma, horizon, n_paths, seed);
            fprintf(file, "# exact_terminal_mean=%.6f exact_terminal_std=%.6f\n",
                    terminal_exact.mean, exact_std);
            fprintf(file, "# elapsed_s=%.3f throughput_paths_per_s=%.3e threads=%d\n",
                    elapsed, (double)n_paths / elapsed, max_threads);
            fprintf(file, "day,mean,std,p05,p50,p95\n");
            fprintf(file, "0,%.6f,0.000000,%.6f,%.6f,%.6f\n", s0, s0, s0, s0);

            double *col = malloc((size_t)n_sample * sizeof(double));
            if (col != NULL) {
                for (long long t = 0; t < horizon; ++t) {
                    double sum = 0.0;
                    double sum_sq = 0.0;
                    for (long long si = 0; si < n_sample; ++si) {
                        const double value = sample[si * horizon + t];
                        col[si] = value;
                        sum += value;
                        sum_sq += value * value;
                    }
                    const double mean = sum / (double)n_sample;
                    const double variance =
                        sum_sq / (double)n_sample - mean * mean;
                    qsort(col, (size_t)n_sample, sizeof(double), compare_doubles);
                    fprintf(file, "%lld,%.6f,%.6f,%.6f,%.6f,%.6f\n", t + 1, mean,
                            sqrt(variance > 0.0 ? variance : 0.0),
                            percentile_sorted(col, (size_t)n_sample, 0.05),
                            percentile_sorted(col, (size_t)n_sample, 0.50),
                            percentile_sorted(col, (size_t)n_sample, 0.95));
                }
                free(col);
            }
            fclose(file);
        }
    }

    printf("s0=%.2f mu=%.4f sigma=%.4f horizon=%lld paths=%lld threads=%d\n",
           s0, mu, sigma, horizon, n_paths, max_threads);
    printf("exact_terminal_mean=%.6f exact_terminal_std=%.6f\n",
           terminal_exact.mean, exact_std);
    printf("elapsed_s=%.3f throughput_paths_per_s=%.3e\n",
           elapsed, (double)n_paths / elapsed);
    if (output != NULL) {
        printf("bands_csv=%s\n", output);
    }

    free(thread_stats);
    free(sample);
    free(shocks);
    return EXIT_SUCCESS;
}
