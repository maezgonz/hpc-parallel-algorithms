/**
 * @file  monte_carlo_pi.c
 * @brief OpenMP Monte Carlo estimation of pi with per-thread RNG seeding.
 *
 * Demonstrates shared-memory parallelism patterns used across this
 * repository: static scheduling, work sharing with reductions, and
 * wall-clock timing via omp_get_wtime(). Designed for runs on CESGA
 * FinisTerrae-3 through the Slurm wrapper in slurm/.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEFAULT_SAMPLES 100000000ULL
#define PI_REF 3.14159265358979323846
#define LCG_MULT 6364136223846793005ULL
#define LCG_INC 1442695040888963407ULL
#define RNG_SEED 0x853C49E6748FEA9BULL
#define GOLDEN_RATIO 0x9E3779B97F4A7C15ULL

/**
 * Draw one uniform sample in [0, 1) from a 64-bit LCG state, publishing the
 * top 53 bits (PCG-style output) for a full double mantissa.
 *
 * @param state Pointer to the caller-owned 64-bit LCG state.
 * @return Uniform double in [0, 1).
 */
static inline double next_uniform(unsigned long long *state) {
    *state = *state * LCG_MULT + LCG_INC;
    return (double)(*state >> 11) / 9007199254740992.0;
}

/**
 * Estimate pi by counting uniform samples inside the unit quarter circle.
 * Each OpenMP thread derives an independent LCG state so streams never
 * overlap, avoiding correlation bias between threads.
 *
 * @param samples Total number of uniform samples to draw.
 * @return Monte Carlo estimate of pi.
 */
static double estimate_pi(unsigned long long samples) {
    unsigned long long hits = 0;

#pragma omp parallel reduction(+ : hits)
    {
        const unsigned int tid = (unsigned int)omp_get_thread_num();
        unsigned long long state =
            RNG_SEED ^ (GOLDEN_RATIO * (unsigned long long)(tid + 1));

#pragma omp for schedule(static)
        for (unsigned long long i = 0; i < samples; ++i) {
            const double x = next_uniform(&state);
            const double y = next_uniform(&state);
            if (x * x + y * y <= 1.0) {
                hits += 1;
            }
        }
    }

    return 4.0 * (double)hits / (double)samples;
}

int main(int argc, char **argv) {
    unsigned long long samples = DEFAULT_SAMPLES;

    if (argc > 1) {
        char *end = NULL;
        samples = strtoull(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || samples == 0) {
            fprintf(stderr, "error: invalid sample count '%s'\n", argv[1]);
            return EXIT_FAILURE;
        }
    }

    const double t0 = omp_get_wtime();
    const double pi = estimate_pi(samples);
    const double elapsed = omp_get_wtime() - t0;

    printf("samples=%llu threads=%d pi=%.8f abs_error=%.2e wall_s=%.3f\n",
           samples, omp_get_max_threads(), pi, fabs(pi - PI_REF), elapsed);

    return EXIT_SUCCESS;
}
