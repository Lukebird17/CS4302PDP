#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>


#define TILE_WIDTH 16  // 矩阵分块的大小，可以根据硬件情况进行调整

// CUDA核函数，计算矩阵乘法
__global__ void matrixMultiplySharedMemory(float *A, float *B, float *C, int A_rows, int A_cols, int B_cols) {
    // 计算线程的行和列
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // 全局内位置
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // 创建共享内存
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0;

    // 遍历A的列和B的行，进行矩阵乘法
    for (int i = 0; i < (A_cols + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
        // 将A和B的子块加载到共享内存
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

        __syncthreads();  // 确保所有线程在进行计算之前共享内存已加载

        // 进行矩阵乘法计算
        for (int j = 0; j < TILE_WIDTH; j++) {
            value += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
        }

        __syncthreads();  // 确保线程同步
    }

    // 将计算结果写入C矩阵
    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = value;
    }
}

// CUDA 包装函数（与 main 函数对齐）
void matrixMultiplyCUDA(const std::vector<float>& A, const std::vector<float>& B,
                        std::vector<float>& C, int N) {
    float *d_A, *d_B, *d_C;
    
    size_t bytes = N * N * sizeof(float);
    
    // 分配GPU内存
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);
    
    // 设置CUDA的线程和块的配置
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    
    // 调用核函数进行矩阵乘法
    matrixMultiplySharedMemory<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, N, N);
    
    // 将计算结果从设备复制回主机
    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// CPU 版本（用于验证正确性）
void matrixMultiplyCPU(const std::vector<float>& A, const std::vector<float>& B, 
                       std::vector<float>& C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


int main() {
    const int N = 1024;
    
    std::cout << "Matrix Multiplication: " << N << "×" << N << std::endl;
    
    // 初始化矩阵
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C_cpu(N * N);
    std::vector<float> C_gpu(N * N);
    
    // 随机填充
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * N; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }
    
    // CPU 计算
    std::cout << "\nRunning CPU version..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(A, B, C_cpu, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    
    // GPU 计算
    std::cout << "\nRunning GPU version..." << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrixMultiplyCUDA(A, B, C_gpu, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;
    
    // 验证正确性
    float max_error = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float error = std::abs(C_cpu[i] - C_gpu[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Max Error: " << max_error << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
    
    if (max_error < 1e-3) {
        std::cout << "✓ Results match!" << std::endl;
    } else {
        std::cout << "✗ Results don't match!" << std::endl;
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
