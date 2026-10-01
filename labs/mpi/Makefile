CC     := mpicc
CFLAGS := -O2 -Wall -std=c17
EXECS  := $(patsubst %.c,%,$(wildcard example*.c))

.PHONY: all clean

all: $(EXECS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f $(EXECS)
