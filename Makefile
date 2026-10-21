CC      := gcc
MPICC   ?= mpicc
CFLAGS  := -O3 -march=native -fopenmp -Wall -Wextra -std=c17
LDFLAGS := -fopenmp -lm

# CUDA targets build only where nvcc exists (FT3 nodes, not CI runners)
NVCC := $(shell which nvcc 2>/dev/null)
NVCFLAGS := -O3 -std=c++17

.PHONY: all run-mcpi run-jacobi run-gbm run-pingpong clean

CUDA_BINS :=
ifneq ($(NVCC),)
CUDA_BINS := bin/vector_add_cuda bin/gbm_cuda bin/gbm_cuda_fp32
endif

all: bin/monte_carlo_pi bin/jacobi_2d bin/gbm_engine bin/mpi_pingpong bin/mpi_collectives $(CUDA_BINS)

bin/monte_carlo_pi: src/monte_carlo_pi.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

bin/jacobi_2d: src/jacobi_2d.c
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

bin/gbm_engine: src/gbm_engine.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

bin/mpi_pingpong: src/mpi_pingpong.c
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -o $@ $< -lm

bin/mpi_collectives: src/mpi_collectives.c
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -o $@ $< -lm

bin/vector_add_cuda: src/vector_add_cuda.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCFLAGS) -o $@ $<

bin/gbm_cuda: src/gbm_cuda.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCFLAGS) -o $@ $<

bin/gbm_cuda_fp32: src/gbm_cuda_fp32.cu
	@mkdir -p $(dir $@)
	$(NVCC) -O3 --use_fast_math -o $@ $<

run-mcpi: bin/monte_carlo_pi
	./bin/monte_carlo_pi 100000000

run-jacobi: bin/jacobi_2d
	mpirun -np 2 ./bin/jacobi_2d 256 10000 1e-4

run-gbm: bin/gbm_engine
	./bin/gbm_engine 100 0.08 0.25 252 100000000 benchmarks/gbm_bands.csv

run-pingpong: bin/mpi_pingpong
	mpirun -np 2 ./bin/mpi_pingpong

clean:
	rm -rf bin
