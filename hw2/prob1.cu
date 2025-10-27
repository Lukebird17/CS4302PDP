#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(float *A, float *B, float *C, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    
    __shared__ float shared_A[32][32];  // Shared memory for sub-matrix A
    __shared__ float shared_B[32][32];  // Shared memory for sub-matrix B

    float C_value = 0.0f;
    
    // Loop over sub-matrices to compute the product
    for (int k = 0; k < (N / 32); ++k) {
        shared_A[ty][tx] = A[row * N + (k * 32 + tx)];
        shared_B[ty][tx] = B[(k * 32 + ty) * N + col];
        __syncthreads();

        for (int n = 0; n < 32; ++n) {
            C_value += shared_A[ty][n] * shared_B[n][tx];
        }
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = C_value;
    }
}

void matrixMultiplyCUDA(float *A, float *B, float *C, int N) {
    // Allocate memory on the GPU
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Set up the grid and block sizes
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(N / 32, N / 32);

    // Launch the kernel
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
