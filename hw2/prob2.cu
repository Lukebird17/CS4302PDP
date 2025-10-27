#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for array summation using reduction
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

void sumArrayCUDA(float *A, float *result, int N) {
    float *d_A, *d_result;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    // Launch the kernel with block size of 256 threads
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    sumArray<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_A, d_result, N);

    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_result);
}
