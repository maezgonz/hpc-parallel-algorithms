/**
 * @file  vector_add_cuda.cu
 * @brief CUDA kernel: vector addition on CPU vs GPU, with timing breakdown.
 *
 * Adapted from MSc HPC coursework (Module 3: Heterogeneous Programming,
 * USC / CESGA). Compares serial CPU execution against a CUDA kernel,
 * separating GPU compute time from host<->device transfer overhead.
 * Designed for NVIDIA A100 nodes on CESGA FinisTerrae-3.
 *
 * Usage: vector_add_cuda [n_elements] [threads_per_block]
 *   n_elements=0 prints device capabilities only
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DEFAULT_N 1048576
#define DEFAULT_BLOCK 1024

typedef struct timeval resnfo;
typedef double timenfo;

#define timestamp(sample) gettimeofday((sample), 0)

static double elapsed(const resnfo start, const resnfo end) {
    return (end.tv_sec + end.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6);
}

/**
 * Initialize vectors: A[i]=i, B[i]=n-i, R[i]=0.
 */
static void populate(float *a, float *b, float *r, const int n) {
    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)(n - i);
        r[i] = 0.0f;
    }
}

/**
 * Sum all elements (also zeroes the array for reuse).
 */
static float checksum_and_reset(float *arr, const int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
        arr[i] = 0.0f;
    }
    return sum;
}

/**
 * Serial vector addition on CPU.
 */
static void add_cpu(const float *a, const float *b, float *r, const int n) {
    for (int i = 0; i < n; i++) {
        r[i] += a[i] + b[i];
    }
}

/**
 * CUDA kernel: element-wise vector addition.
 */
__global__ void add_kernel(const float *const a, const float *const b,
                           float *const r, const int n) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        r[gid] += a[gid] + b[gid];
    }
}

/**
 * Launch the CUDA kernel and return GPU compute time (excluding transfers).
 */
static double add_gpu(const float *a, const float *b, float *r,
                      const int n, const int block_size) {
    const size_t bytes = (size_t)n * sizeof(float);
    float *da, *db, *dr;
    resnfo t0, t1;

    cudaMalloc(&da, bytes);
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&db, bytes);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dr, bytes);
    cudaMemset(dr, 0, bytes);

    dim3 block(block_size);
    dim3 grid((n + block.x - 1) / block.x);

    timestamp(&t0);
    add_kernel<<<grid, block>>>(da, db, dr, n);
    timestamp(&t1);

    cudaMemcpy(r, dr, bytes, cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dr);

    return elapsed(t0, t1);
}

/**
 * Print device capabilities.
 */
static void device_info(void) {
    struct cudaDeviceProp cap;
    cudaGetDeviceProperties(&cap, 0);
    printf("CUDA Device: %s\n", cap.name);
    printf("  Global mem: %.0f MB\n", cap.totalGlobalMem / 1048576.0);
    printf("  Shared mem/block: %.0f KB\n", cap.sharedMemPerBlock / 1024.0);
    printf("  Max threads/block: %d\n", cap.maxThreadsPerBlock);
    printf("  Warp size: %d\n", cap.warpSize);
    printf("  SMs: %d\n", cap.multiProcessorCount);
    printf("  Clock: %.0f MHz\n", cap.clockRate / 1000.0);
    printf("  Compute capability: %d.%d\n", cap.major, cap.minor);
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    int block = (argc > 2) ? atoi(argv[2]) : DEFAULT_BLOCK;

    if (n == 0) {
        device_info();
        return EXIT_SUCCESS;
    }

    const size_t bytes = (size_t)n * sizeof(float);
    float *a = malloc(bytes);
    float *b = malloc(bytes);
    float *r = malloc(bytes);
    if (!a || !b || !r) {
        fprintf(stderr, "error: allocation failed\n");
        return EXIT_FAILURE;
    }

    populate(a, b, r, n);

    // CPU baseline
    resnfo t0, t1;
    timestamp(&t0);
    add_cpu(a, b, r, n);
    timestamp(&t1);
    double cpu_time = elapsed(t0, t1);
    float cpu_result = checksum_and_reset(r, n);

    // GPU
    timestamp(&t0);
    double gpu_compute = add_gpu(a, b, r, n, block);
    timestamp(&t1);
    double gpu_total = elapsed(t0, t1);
    float gpu_result = checksum_and_reset(r, n);

    printf("n=%d block=%d\n", n, block);
    printf("cpu_wall_s=%.6f\n", cpu_time);
    printf("gpu_total_s=%.6f\n", gpu_total);
    printf("gpu_compute_s=%.6f\n", gpu_compute);
    printf("gpu_transfer_s=%.6f\n", gpu_total - gpu_compute);
    printf("gpu_speedup_vs_cpu=%.2fx\n", cpu_time / gpu_total);
    printf("result_ok=%s\n", (cpu_result == gpu_result) ? "yes" : "NO");
    printf("throughput_gflops=%.2f\n", (3.0 * n) / (gpu_compute * 1e9));

    free(a);
    free(b);
    free(r);
    return EXIT_SUCCESS;
}
