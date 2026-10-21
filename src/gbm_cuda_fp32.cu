/**
 * @file  gbm_cuda_fp32.cu
 * @brief CUDA Monte Carlo GBM engine in single precision (FP32) with fast math.
 *
 * Same financial kernel as gbm_cuda.cu (FP64), ported to float with CUDA
 * intrinsics (__expf, __logf, __cosf). Designed to demonstrate the 2x
 * FP32 throughput of the A100 — the "how to make the GPU win" companion
 * to the double-precision benchmark.
 *
 * Key optimizations vs FP64:
 *   - All computation in float (FP32)
 *   - __expf, __logf, __cosf: single-precision hardware instructions
 *   - 24-bit uniform conversion (sufficient for float Monte Carlo)
 *   - Same LCG state (64-bit) for direct comparability with FP64 results
 *
 * Usage: gbm_cuda_fp32 [s0] [mu] [sigma] [horizon] [n_paths] [block_size]
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI_REF 3.14159265358979323846
#define TWO_PI 6.28318530717958647692
#define LCG_MULT 6364136223846793005ULL
#define LCG_INC 1442695040888963407ULL
#define RNG_SEED 0x853C49E6748FEA9BULL
#define GOLDEN_RATIO 0x9E3779B97F4A7C15ULL
#define DEFAULT_BLOCK 1024

typedef struct timeval resnfo;

#define timestamp(sample) gettimeofday((sample), 0)

static double elapsed(const resnfo start, const resnfo end) {
    return (end.tv_sec + end.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6);
}

/**
 * CUDA kernel: simulate GBM paths in FP32 with fast math intrinsics.
 */
__global__ void gbm_kernel_fp32(
    float *const d_terminal,
    const float s0, const float mu, const float sigma,
    const float dt, const int horizon,
    const unsigned long long n_total_paths,
    const int n_paths_per_thread
) {
    const unsigned long long gid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long state =
        RNG_SEED ^ (GOLDEN_RATIO * (gid + 1));

    const float drift = (mu - 0.5f * sigma * sigma) * dt;
    const float vol_dt = sigma * sqrtf(dt);

    for (int p = 0; p < n_paths_per_thread; ++p) {
        const unsigned long long path_idx =
            gid * (unsigned long long)n_paths_per_thread + p;
        if (path_idx >= n_total_paths) {
            break;
        }

        float log_price = 0.0f;
        for (int t = 0; t < horizon; ++t) {
            // LCG -> float uniform (24 bits, [0, 1))
            state = state * LCG_MULT + LCG_INC;
            float u1 = (float)(state >> 40) * (1.0f / 16777216.0f);
            if (u1 <= 0.0f) u1 = 1e-12f;
            state = state * LCG_MULT + LCG_INC;
            const float u2 = (float)(state >> 40) * (1.0f / 16777216.0f);

            // Box-Muller with CUDA intrinsics
            const float z = sqrtf(-2.0f * __logf(u1)) * __cosf(TWO_PI * u2);
            log_price += drift + vol_dt * z;
        }
        d_terminal[path_idx] = s0 * __expf(log_price);
    }
}

/**
 * CUDA kernel: FP32 reduction for mean and sum-of-squares.
 */
__global__ void stats_kernel_fp32(
    const float *const d_terminal,
    const unsigned long long n,
    float *const d_sum, float *const d_sq_sum
) {
    __shared__ float s_sum[1024];
    __shared__ float s_sq[1024];

    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;

    float local_sum = 0.0f;
    float local_sq = 0.0f;

    for (unsigned long long i = tid; i < n; i += n_threads) {
        const float x = d_terminal[i];
        local_sum += x;
        local_sq += x * x;
    }

    s_sum[tid] = local_sum;
    s_sq[tid] = local_sq;
    __syncthreads();

    for (int stride = n_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid] += s_sq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d_sum = s_sum[0];
        *d_sq_sum = s_sq[0];
    }
}

int main(int argc, char **argv) {
    double s0 = (argc > 1) ? atof(argv[1]) : 100.0;
    double mu = (argc > 2) ? atof(argv[2]) : 0.08;
    double sigma = (argc > 3) ? atof(argv[3]) : 0.25;
    int horizon = (argc > 4) ? atoi(argv[4]) : 252;
    unsigned long long n_paths = (argc > 5) ? strtoull(argv[5], NULL, 10) : 100000000ULL;
    int block_size = (argc > 6) ? atoi(argv[6]) : DEFAULT_BLOCK;

    if (s0 <= 0 || sigma < 0 || horizon <= 0 || n_paths == 0 || block_size <= 0) {
        fprintf(stderr, "error: invalid parameters\n");
        return EXIT_FAILURE;
    }

    const float dt = 1.0f / 252.0f;
    const int n_paths_per_thread = (int)((n_paths + block_size - 1) / block_size);
    const unsigned long long n_alloc =
        (unsigned long long)block_size * (unsigned long long)n_paths_per_thread;

    resnfo t_alloc0, t_alloc1, t_kernel0, t_kernel1, t_stats0, t_stats1;

    timestamp(&t_alloc0);
    float *d_terminal;
    cudaMalloc(&d_terminal, n_alloc * sizeof(float));

    float *d_sum, *d_sq_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_sq_sum, sizeof(float));
    timestamp(&t_alloc1);

    const int total_threads = (int)((n_alloc + block_size - 1) / block_size);
    dim3 block(block_size);
    dim3 grid(total_threads);

    timestamp(&t_kernel0);
    gbm_kernel_fp32<<<grid, block>>>(
        d_terminal, (float)s0, (float)mu, (float)sigma, dt, horizon,
        n_alloc, n_paths_per_thread
    );
    cudaDeviceSynchronize();
    timestamp(&t_kernel1);

    timestamp(&t_stats0);
    stats_kernel_fp32<<<1, 1024>>>(d_terminal, n_alloc, d_sum, d_sq_sum);
    cudaDeviceSynchronize();
    timestamp(&t_stats1);

    float h_sum = 0.0f, h_sq_sum = 0.0f;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sq_sum, d_sq_sum, sizeof(float), cudaMemcpyDeviceToHost);

    const double mean = (double)h_sum / (double)n_alloc;
    const double variance = (double)h_sq_sum / (double)n_alloc - mean * mean;
    const double std_dev = sqrt(variance > 0 ? variance : 0);

    const double expected_mean = s0 * exp(mu * (double)horizon / 252.0);
    const double T = (double)horizon / 252.0;
    const double expected_std = s0 * exp(mu * T) * sqrt(exp(sigma * sigma * T) - 1.0);

    const double kernel_time = elapsed(t_kernel0, t_kernel1);
    const double total_time = elapsed(t_alloc0, t_stats1);

    printf("s0=%.2f mu=%.4f sigma=%.4f horizon=%d paths=%llu block=%d precision=fp32\n",
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
    cudaFree(d_sum);
    cudaFree(d_sq_sum);
    return EXIT_SUCCESS;
}
