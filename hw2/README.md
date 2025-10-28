# CUDA Programming Assignment: Optimizations

This project implements and optimizes two common computational problems using CUDA:
1.  **Matrix Multiplication**
2.  **Array Summation**

The goal is to write standalone, reusable functions that leverage the CUDA memory hierarchy and parallel processing capabilities for maximum performance.

## How to Compile and Run

A `Makefile` is provided. To compile the code, simply run:

```bash
make
```

This will generate an executable file named `program.out`.

To run the program, execute:

```bash
./program.out
```

To clean up all generated files, run:

```bash
make clean
```

## Optimization Techniques

### 1. Matrix Multiplication (`prob1.cu`)

The matrix multiplication kernel is optimized using the following techniques:

#### a. Tiled (Block) Matrix Multiplication using Shared Memory

-   **Concept**: Instead of having each thread fetch data directly from global memory for every single multiplication, we first load small sub-matrices (tiles) into the fast on-chip shared memory. Each thread in a thread block then cooperates to compute the product of these sub-matrices.
-   **Benefit**: This dramatically reduces the number of high-latency reads from global memory. Global memory access is slow, whereas shared memory access is orders of magnitude faster. By loading a tile into shared memory once, we can reuse its data for multiple calculations (32 times in this implementation), maximizing data reuse and hiding memory latency.
-   **Implementation**:
    -   The kernel loads a `32x32` tile of matrix `A` and a `32x32` tile of matrix `B` into shared memory arrays (`shared_A`, `shared_B`).
    -   `__syncthreads()` is used to ensure all threads in the block have finished loading data into shared memory before the computation begins, and to ensure all computations on the current tiles are finished before loading the next tiles.

#### b. Loop Unrolling (Manual)

-   **Concept**: Loop unrolling is a technique to reduce the number of loop control instructions (like incrementing and comparing the loop counter) and to increase the instruction-level parallelism. By processing more data in a single loop iteration, we reduce the loop overhead.
-   **Benefit**: Fewer instruction cycles are wasted on loop management, and the GPU's instruction scheduler has more independent instructions to work with, which helps hide instruction and memory latencies.
-   **Implementation**: The inner loop for the dot product calculation within a tile is manually unrolled by a factor of 4. Instead of one multiplication-addition per iteration, we perform four.

### 2. Array Summation (`prob2.cu`)

The array summation is implemented using a parallel reduction algorithm, which is optimized as follows:

#### a. Parallel Reduction using Shared Memory

-   **Concept**: A reduction algorithm combines a list of elements into a single value using a binary operator (in this case, addition). A naive approach where a single variable is updated by all threads (`atomicAdd`) would be slow due to massive contention. Instead, we perform the reduction in stages.
-   **Benefit**: This approach is highly parallel and scalable.
-   **Implementation**:
    1.  **Local Reduction**: Each thread block first sums a portion of the array. The partial sum for each thread is stored in shared memory.
    2.  **Intra-Block Reduction**: Within each block, threads perform a parallel reduction on the data in shared memory. The work is halved in each step (`stride /= 2`). For example, 256 threads add their values to produce 128 sums, then 64, 32, 16, 8, 4, 2, and finally 1 partial sum for the entire block. `__syncthreads()` is critical here to synchronize threads between each step.
    3.  **Global Reduction**: The first thread of each block (`tid == 0`) atomically adds its block's final partial sum to a global result variable in global memory. Using `atomicAdd` here is efficient because only one thread per block writes to the global result, minimizing contention.

This multi-level reduction strategy effectively utilizes the fast shared memory and the massive parallelism of the GPU.
