CC      := gcc
CFLAGS  := -O3 -march=native -fopenmp -Wall -Wextra -std=c17
LDFLAGS := -fopenmp -lm
BIN     := bin/monte_carlo_pi
SRC     := src/monte_carlo_pi.c

.PHONY: all run clean

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: $(BIN)
	./$(BIN) 100000000

clean:
	rm -rf bin
