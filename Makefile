CC      := gcc
MPICC   ?= mpicc
CFLAGS  := -O3 -march=native -fopenmp -Wall -Wextra -std=c17
LDFLAGS := -fopenmp -lm

.PHONY: all run-mcpi run-jacobi clean

all: bin/monte_carlo_pi bin/jacobi_2d

bin/monte_carlo_pi: src/monte_carlo_pi.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

bin/jacobi_2d: src/jacobi_2d.c
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run-mcpi: bin/monte_carlo_pi
	./bin/monte_carlo_pi 100000000

run-jacobi: bin/jacobi_2d
	mpirun -np 2 ./bin/jacobi_2d 256 10000 1e-4

clean:
	rm -rf bin
