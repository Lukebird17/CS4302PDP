#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>


#define TILE_WIDTH 16  

__global__ void matrixMultiplySharedMemory(float *A, float *B, float *C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // 全局内位置
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0;
    for (int i = 0; i < (A_cols + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
        if (i * TILE_WIDTH + threadIdx.x < A_cols && row < A_rows) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * A_cols + i * TILE_WIDTH + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (i * TILE_WIDTH + threadIdx.y < A_cols && col < B_cols) {
            shared_B[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * B_cols + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads(); 

        for (int j = 0; j < TILE_WIDTH; j++) {
            value += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
        }
        __syncthreads();  
    }

    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = value;
    }
}

void matrixMultiplyCUDA(const std::vector<float>& A, const std::vector<float>& B,
                        std::vector<float>& C,
                        int A_rows, int A_cols, int B_cols) {
    int B_rows = A_cols;
    float *d_A, *d_B, *d_C;
    size_t A_bytes = (size_t)A_rows * A_cols * sizeof(float);
    size_t B_bytes = (size_t)B_rows * B_cols * sizeof(float);
    size_t C_bytes = (size_t)A_rows * B_cols * sizeof(float);
    
    cudaMalloc((void**)&d_A, A_bytes);
    cudaMalloc((void**)&d_B, B_bytes);
    cudaMalloc((void**)&d_C, C_bytes);
    
    cudaMemcpy(d_A, A.data(), A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B_bytes, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((B_cols + TILE_WIDTH - 1) / TILE_WIDTH, (A_rows + TILE_WIDTH - 1) / TILE_WIDTH);
    
    matrixMultiplySharedMemory<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    
    cudaMemcpy(C.data(), d_C, C_bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// CPU 
void matrixMultiplyCPU(const std::vector<float>& A, const std::vector<float>& B, 
                       std::vector<float>& C, int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; k++) {
                sum += A[i * A_cols + k] * B[k * B_cols + j];
            }
            C[i * B_cols + j] = sum;
        }
    }
}


int main(int argc, char** argv) {
    int A_rows = 512;
    int A_cols = 1024; // 也是 B_rows
    int B_cols = 1024;
    if (argc == 4) {
        A_rows = std::max(1, std::atoi(argv[1]));
        A_cols = std::max(1, std::atoi(argv[2]));
        B_cols = std::max(1, std::atoi(argv[3]));
    } else {
        std::cout << "Usage: " << argv[0] << " [A_rows A_cols B_cols]\n";
        std::cout << "No args provided — using default sizes A=" << A_rows << "x" << A_cols
                  << ", B=" << A_cols << "x" << B_cols << "\n";
    }

    std::cout << "Matrix Multiplication: " << A_rows << "×" << A_cols << " * "
              << A_cols << "×" << B_cols << std::endl;

    std::vector<float> A((size_t)A_rows * A_cols);
    std::vector<float> B((size_t)A_cols * B_cols);
    std::vector<float> C_cpu((size_t)A_rows * B_cols);
    std::vector<float> C_gpu((size_t)A_rows * B_cols);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < A.size(); i++) A[i] = dist(gen);
    for (size_t i = 0; i < B.size(); i++) B[i] = dist(gen);

    std::cout << "\nRunning CPU version..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(A, B, C_cpu, A_rows, A_cols, B_cols);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;

    std::cout << "\nRunning GPU version..." << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMultiplyCUDA(A, B, C_gpu, A_rows, A_cols, B_cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

    float max_error = 0.0f;
    size_t C_size = (size_t)A_rows * B_cols;
    for (size_t i = 0; i < C_size; i++) {
        float error = std::abs(C_cpu[i] - C_gpu[i]);
        max_error = std::max(max_error, error);
    }

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Max Error: " << max_error << std::endl;
    if (gpu_time > 0.0f)
        std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;

    if (max_error < 1e-3f) {
        std::cout << "✓ Results match!" << std::endl;
    } else {
        std::cout << "✗ Results don't match!" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
