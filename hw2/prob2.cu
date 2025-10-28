#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// --- CUDA Kernel and Host Function ---

// Optimized parallel reduction kernel
__global__ void sumArrayKernel(const float *A, float *result, int N) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop to handle large arrays
    float my_sum = 0.0f;
    while (i < N) {
        my_sum += A[i];
        i += gridDim.x * blockDim.x;
    }
    shared_data[tid] = my_sum;
    __syncthreads();

    // Intra-block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // First thread of each block writes its partial sum to global memory
    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

// Wrapper function for the CUDA implementation
void sumArrayCUDA(const std::vector<float>& A, float& result, int N) {
    float *d_A, *d_result;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    int threadsPerBlock = 256;
    int blocks = std::min((N + threadsPerBlock - 1) / threadsPerBlock, 1024);
    size_t shared_mem_size = threadsPerBlock * sizeof(float);

    sumArrayKernel<<<blocks, threadsPerBlock, shared_mem_size>>>(d_A, d_result, N);

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_result);
}

// --- C++ CPU Implementation ---
void sumArrayCPU(const std::vector<float>& A, float& result, int N) {
    result = 0.0f;
    for (int i = 0; i < N; ++i) {
        result += A[i];
    }
}

// --- Main Function for Comparison ---
int main() {
    const int N = 100000000; // Array size (e.g., 100 million elements)

    std::cout << "--- Array Summation Performance ---" << std::endl;
    std::cout << "Array size: " << N << " elements" << std::endl;

    // Initialize array
    std::vector<float> h_A(N);
    std::generate(h_A.begin(), h_A.end(), [](){ return 1.0f; }); // Fill with 1.0 for easy verification

    float result_cpu = 0.0f;
    float result_gpu = 0.0f;

    // --- CPU Execution and Timing ---
    std::cout << "\nRunning CPU version..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    sumArrayCPU(h_A, result_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "CPU Result: " << result_cpu << std::endl;

    // --- GPU Execution and Timing ---
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

    // --- Speedup ---
    std::cout << "\nGPU Speedup: " << cpu_time.count() / gpu_time << "x" << std::endl;

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
