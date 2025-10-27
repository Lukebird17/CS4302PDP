#include <iostream>
#include <cuda_runtime.h>

__global__ void sumArray(float *A, float *result, int N) {
    extern __shared__ float shared_data[];

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    
    float sum = 0;
    if (index < N) {
        sum = A[index];
    }

    shared_data[tid] = sum;
    __syncthreads();

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Store result from block 0
    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

int main() {
    const int N = 1024;  // Array size
    float A[N], result = 0.0f;

    // Initialize array A with example values
    for (int i = 0; i < N; ++i) {
        A[i] = 1.0f;
    }

    // Allocate memory on the GPU
    float *d_A, *d_result;
    size_t size = N * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_result, sizeof(float));

    // Copy array A to the GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    // Set up grid and block dimensions
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    sumArray<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_A, d_result, N);

    // Copy the result back to the host
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Display result
    std::cout << "Sum of array: " << result << std::endl;

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_result);

    return 0;
}
