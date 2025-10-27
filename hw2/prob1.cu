#include <iostream>
#include <cuda_runtime.h>

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

int main() {
    const int N = 32;  // Matrix size (N x N)
    float A[N][N], B[N][N], C[N][N];

    // Initialize matrices A and B with example values
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = 1.0f;
            B[i][j] = 1.0f;
        }
    }

    // Allocate memory on the GPU
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices to the GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(N / 32, N / 32);

    // Launch kernel
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy the result back to the host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Display result (for demonstration)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
