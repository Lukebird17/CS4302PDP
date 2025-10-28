# Makefile for CUDA Programming Assignment

# Compiler and flags
CC = nvcc
# -O3 for optimization
# -arch=native to build for the GPU available on the system
# -Wno-deprecated-gpu-targets to suppress the warning about old architectures
CFLAGS = -O3 -arch=native -Wno-deprecated-gpu-targets --std=c++17

# Executable names
EXECS = prob1.out prob2.out

# Default target: build all executables
all: $(EXECS)

# Rule to build each executable from its corresponding .cu file
prob1.out: prob1.cu
	$(CC) $(CFLAGS) -o $@ $<

prob2.out: prob2.cu
	$(CC) $(CFLAGS) -o $@ $<

# Rule to clean up all generated files
clean:
	rm -f $(EXECS) prob1.o prob2.o
