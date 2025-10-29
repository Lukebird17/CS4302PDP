#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>


// First Kernel 
__global__ void sumArrayKernel(const float *A, float *partials, int N) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float my_sum = 0.0f;
    while (i < N) {
        my_sum += A[i];
        i += gridDim.x * blockDim.x;
    }

    shared_data[tid] = my_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = shared_data[0];
    }
}

// Second kernel 
__global__ void reduceFinal(float *partials, float *result, int num_partials) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int i = tid;

    float my_sum = 0.0f;
    while (i < num_partials) {
        my_sum += partials[i];
        i += blockDim.x;
    }

    shared_data[tid] = my_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = shared_data[0];
    }
}

void sumArrayCUDA(const std::vector<float>& A, float& result, int N) {
    float *d_A, *d_partials, *d_result;
    int threadsPerBlock = 256;
    int blocks = std::min((N + threadsPerBlock - 1) / threadsPerBlock, 1024);
    
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_partials, blocks * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    size_t shared_mem_size = threadsPerBlock * sizeof(float);
    sumArrayKernel<<<blocks, threadsPerBlock, shared_mem_size>>>(d_A, d_partials, N);

    int finalThreads = std::min(256, blocks);
    size_t final_shared_mem = finalThreads * sizeof(float);
    reduceFinal<<<1, finalThreads, final_shared_mem>>>(d_partials, d_result, blocks);

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_partials);
    cudaFree(d_result);
}

// CPU 
void sumArrayCPU(const std::vector<float>& A, float& result, int N) {
    result = 0.0f;
    for (int i = 0; i < N; ++i) {
        result += A[i];
    }
}

int main() {
    const int N = 100000000; 

    std::cout << "--- Array Summation Performance ---" << std::endl;
    std::cout << "Array size: " << N << " elements" << std::endl;

    std::vector<float> h_A(N);
    std::generate(h_A.begin(), h_A.end(), [](){ return 1.0f; }); 

    float result_cpu = 0.0f;
    float result_gpu = 0.0f;

    std::cout << "\nRunning CPU version..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    sumArrayCPU(h_A, result_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "CPU Result: " << result_cpu << std::endl;

    std::cout << "\nRunning GPU version..." << std::endl;
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    sumArrayCUDA(h_A, result_gpu, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;
    std::cout << "GPU Result: " << result_gpu << std::endl;
    std::cout << "\nGPU Speedup: " << cpu_time.count() / gpu_time << "x" << std::endl;

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
