/**
 * @file  gbm_cuda.cu
 * @brief CUDA Monte Carlo GBM engine — the same financial kernel as gbm_engine.c,
 *        ported to the A100 GPU.
 *
 * Simulates Geometric Brownian Motion paths on the GPU with per-thread
 * RNG states (Box-Muller), accumulating terminal-price statistics via
 * an atomic-free two-pass reduction. Designed to demonstrate when the
 * GPU wins: high compute-to-data ratio (many steps per byte transferred).
 *
 * Architecture:
 *   1. Each thread simulates `n_paths_per_thread` complete GBM paths
 *   2. Terminal prices are stored to global memory
 *   3. A second kernel computes mean and std-dev on the GPU
 *   4. Only 2 scalars are transferred back to the host
 *
 * Usage: gbm_cuda [s0] [mu] [sigma] [horizon] [n_paths] [block_size]
 *   Defaults: 100 0.08 0.25 252 10000000 256
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI_REF 3.14159265358979323846
#define TWO_PI 6.28318530717958647692
#define LCG_MULT 6364136223846793005ULL
#define LCG_INC 1442695040888963407ULL
#define RNG_SEED 0x853C49E6748FEA9BULL
#define GOLDEN_RATIO 0x9E3779B97F4A7C15ULL
#define DEFAULT_BLOCK 256

typedef struct timeval resnfo;

#define timestamp(sample) gettimeofday((sample), 0)

static double elapsed(const resnfo start, const resnfo end) {
    return (end.tv_sec + end.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6);
}

/**
 * CUDA kernel: simulate GBM paths. Each thread handles
 * n_paths_per_thread paths, storing terminal prices to d_terminal.
 *
 * @param d_terminal Output: terminal price per path (device memory).
 * @param s0 Spot price.
 * @param mu Annualized drift.
 * @param sigma Annualized volatility.
 * @param dt Time step (1/252).
 * @param n_total_paths Total number of paths.
 * @param n_paths_per_thread Paths handled by each thread.
 */
__global__ void gbm_kernel(
    double *const d_terminal,
    const double s0, const double mu, const double sigma,
    const double dt, const int horizon,
    const unsigned long long n_total_paths,
    const int n_paths_per_thread
) {
    const unsigned long long gid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long state =
        RNG_SEED ^ (GOLDEN_RATIO * (gid + 1));

    const double drift = (mu - 0.5 * sigma * sigma) * dt;
    const double vol_dt = sigma * sqrt(dt);

    for (int p = 0; p < n_paths_per_thread; ++p) {
        const unsigned long long path_idx = gid * (unsigned long long)n_paths_per_thread + p;
        if (path_idx >= n_total_paths) {
            break;
        }

        double log_price = 0.0;
        for (int t = 0; t < horizon; ++t) {
            // Box-Muller from two LCG uniforms
            state = state * LCG_MULT + LCG_INC;
            double u1 = (double)(state >> 11) / 9007199254740992.0;
            if (u1 <= 0.0) u1 = 1e-12;
            state = state * LCG_MULT + LCG_INC;
            const double u2 = (double)(state >> 11) / 9007199254740992.0;

            const double z = sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
            log_price += drift + vol_dt * z;
        }
        d_terminal[path_idx] = s0 * exp(log_price);
    }
}

/**
 * CUDA kernel: compute mean and std-dev of terminal prices.
 * Single-block reduction (sufficient for up to ~10M paths with atomic).
 */
__global__ void stats_kernel(
    const double *const d_terminal,
    const unsigned long long n,
    double *const d_mean, double *const d_m2
) {
    __shared__ double s_sum[1024];
    __shared__ double s_sq[1024];

    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;

    double local_sum = 0.0;
    double local_sq = 0.0;

    for (unsigned long long i = tid; i < n; i += n_threads) {
        const double x = d_terminal[i];
        local_sum += x;
        local_sq += x * x;
    }

    s_sum[tid] = local_sum;
    s_sq[tid] = local_sq;
    __syncthreads();

    // Tree reduction
    for (int stride = n_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid] += s_sq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d_mean = s_sum[0];
        *d_m2 = s_sq[0];
    }
}

int main(int argc, char **argv) {
    double s0 = (argc > 1) ? atof(argv[1]) : 100.0;
    double mu = (argc > 2) ? atof(argv[2]) : 0.08;
    double sigma = (argc > 3) ? atof(argv[3]) : 0.25;
    int horizon = (argc > 4) ? atoi(argv[4]) : 252;
    unsigned long long n_paths = (argc > 5) ? strtoull(argv[5], NULL, 10) : 10000000ULL;
    int block_size = (argc > 6) ? atoi(argv[6]) : DEFAULT_BLOCK;

    if (s0 <= 0 || sigma < 0 || horizon <= 0 || n_paths == 0 || block_size <= 0) {
        fprintf(stderr, "error: invalid parameters\n");
        return EXIT_FAILURE;
    }

    const double dt = 1.0 / 252.0;
    const int n_threads = block_size;
    const int n_paths_per_thread = (int)((n_paths + n_threads - 1) / n_threads);
    // Round up total to multiple of threads for clean indexing
    const unsigned long long n_alloc = (unsigned long long)n_threads * (unsigned long long)n_paths_per_thread;

    resnfo t_alloc0, t_alloc1, t_kernel0, t_kernel1, t_stats0, t_stats1;

    // Allocate device memory for terminal prices
    timestamp(&t_alloc0);
    double *d_terminal;
    cudaMalloc(&d_terminal, n_alloc * sizeof(double));

    double *d_mean, *d_m2;
    cudaMalloc(&d_mean, sizeof(double));
    cudaMalloc(&d_m2, sizeof(double));
    timestamp(&t_alloc1);

    const int total_threads = (int)((n_alloc + n_threads - 1) / n_threads);
    dim3 block(block_size);
    dim3 grid(total_threads);

    // Launch GBM kernel
    timestamp(&t_kernel0);
    gbm_kernel<<<grid, block>>>(
        d_terminal, s0, mu, sigma, dt, horizon, n_alloc, n_paths_per_thread
    );
    cudaDeviceSynchronize();
    timestamp(&t_kernel1);

    // Launch stats kernel (single block of up to 1024 threads)
    timestamp(&t_stats0);
    stats_kernel<<<1, 1024>>>(d_terminal, n_alloc, d_mean, d_m2);
    cudaDeviceSynchronize();
    timestamp(&t_stats1);

    // Transfer only 2 scalars back to host
    double h_mean = 0.0, h_sq_sum = 0.0;
    cudaMemcpy(&h_mean, d_mean, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sq_sum, d_m2, sizeof(double), cudaMemcpyDeviceToHost);

    // Compute final statistics
    const double mean = h_mean / (double)n_alloc;
    const double variance = h_sq_sum / (double)n_alloc - mean * mean;
    const double std_dev = sqrt(variance > 0 ? variance : 0);

    // Closed-form reference
    const double expected_mean = s0 * exp(mu * (double)horizon * dt);
    const double T = (double)horizon * dt;
    const double expected_std = s0 * exp(mu * T) *
        sqrt(exp(sigma * sigma * T) - 1.0);

    const double kernel_time = elapsed(t_kernel0, t_kernel1);
    const double total_time = elapsed(t_alloc0, t_stats1);

    printf("s0=%.2f mu=%.4f sigma=%.4f horizon=%d paths=%llu block=%d\n",
           s0, mu, sigma, horizon, n_paths, block_size);
    printf("gpu_terminal_mean=%.6f expected=%.6f diff=%.4e\n",
           mean, expected_mean, fabs(mean - expected_mean));
    printf("gpu_terminal_std=%.6f expected=%.6f diff=%.4e\n",
           std_dev, expected_std, fabs(std_dev - expected_std));
    printf("gpu_kernel_s=%.6f gpu_total_s=%.6f\n", kernel_time, total_time);
    printf("throughput_paths_per_s=%.3e\n", (double)n_paths / kernel_time);
    printf("throughput_steps_per_s=%.3e\n",
           (double)n_paths * horizon / kernel_time);

    cudaFree(d_terminal);
    cudaFree(d_mean);
    cudaFree(d_m2);
    return EXIT_SUCCESS;
}
