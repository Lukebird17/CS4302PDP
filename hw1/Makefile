# macOS 自带的 clang 不支持 OpenMP，故使用 Homebrew 安装的 GCC
GCC15 := $(shell which gcc-15 2>/dev/null)
GCC14 := $(shell which gcc-14 2>/dev/null)
GCC13 := $(shell which gcc-13 2>/dev/null)
GCC12 := $(shell which gcc-12 2>/dev/null)

ifneq ($(GCC15),)
    CC = gcc-15
else ifneq ($(GCC14),)
    CC = gcc-14
else ifneq ($(GCC13),)
    CC = gcc-13
else ifneq ($(GCC12),)
    CC = gcc-12
else
    CC = gcc
endif

CFLAGS = -O2 -fopenmp -std=c11 -Wall -Wextra
LDFLAGS = -fopenmp -lm

TARGETS = prob1 prob2 prob3

all: $(TARGETS)

prob1: prob1.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

prob2: prob2.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

prob3: prob3.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: all clean run1 run2 run3 test

RUN1_ARGS ?= 1024 8
RUN2_ARGS ?= 10000000 111 222 333 8
RUN3_ARGS ?= 256 256 3 42 4

run1: prob1
	./prob1 $(RUN1_ARGS)

run2: prob2
	./prob2 $(RUN2_ARGS)

run3: prob3
	./prob3 $(RUN3_ARGS)

clean:
	rm -f $(TARGETS) p1 p2 p3 *.o prob1_test
