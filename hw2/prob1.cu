#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

// --- CUDA Kernel and Host Function ---

// Tile size for shared memory optimization
constexpr int TILE_SIZE = 32;

// Optimized matrix multiplication kernel using shared memory tiling
__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int N) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * blockDim.y + ty;
    const int col = blockIdx.x * blockDim.x + tx;

    // Shared memory tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float value = 0.0f;
    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Iterate over tiles
    for (int t = 0; t < numTiles; ++t) {
        const int tileRow = row;
        const int tileCol = t * TILE_SIZE + tx;
        if (tileRow < N && tileCol < N) {
            tileA[ty][tx] = A[tileRow * N + tileCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        const int tileRowB = t * TILE_SIZE + ty;
        const int tileColB = col;
        if (tileRowB < N && tileColB < N) {
            tileB[ty][tx] = B[tileRowB * N + tileColB];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Multiply tiles
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += tileA[ty][k] * tileB[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// Wrapper function for the CUDA implementation
void matrixMultiplyCUDA(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    matrixMultiplyKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// --- C++ CPU Implementation ---
void matrixMultiplyCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float value = 0.0f;
            for (int k = 0; k < N; ++k) {
                value += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = value;
        }
    }
}

// --- Main Function for Comparison ---
int main() {
    const int N = 1024; // Matrix size (e.g., 1024x1024)

    std::cout << "--- Matrix Multiplication Performance ---" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;

    // Initialize matrices
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C_cpu(N * N);
    std::vector<float> h_C_gpu(N * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::generate(h_A.begin(), h_A.end(), [&](){ return dist(gen); });
    std::generate(h_B.begin(), h_B.end(), [&](){ return dist(gen); });

    // --- CPU Execution and Timing ---
    std::cout << "\nRunning CPU version..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " ms" << std::endl;

    // --- GPU Execution and Timing ---
    std::cout << "\nRunning GPU version..." << std::endl;
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    matrixMultiplyCUDA(h_A, h_B, h_C_gpu, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

    // --- Verification and Speedup ---
    float max_error = 0.0f;
    for (size_t i = 0; i < h_C_cpu.size(); ++i) {
        max_error = std::max(max_error, std::abs(h_C_cpu[i] - h_C_gpu[i]));
    }
    std::cout << "\nVerification Max Error: " << max_error << std::endl;
    std::cout << "GPU Speedup: " << cpu_time.count() / gpu_time << "x" << std::endl;

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
