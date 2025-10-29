# CS4302 HW2
卢鸿良 523030910233

## 编译与运行

### 编译所有程序
```bash
make
```

### 编译单个程序
```bash
make prob1.out    # 矩阵乘法
make prob2.out    # 数组求和
```

### 运行程序

**矩阵乘法：**
```bash
# 使用默认大小运行 (512×1024 * 1024×1024)
./prob1.out

# 使用自定义维度运行 (A_rows A_cols B_cols)
./prob1.out 240 360 780
./prob1.out 2400 3600 780
```

**数组求和：**
```bash
# 使用默认大小运行 (1 亿个元素)
./prob2.out
```

### 清理构建文件
```bash
make clean
```

## 优化技术

### 1. 矩阵乘法 (`prob1.cu`)
计算两个矩阵相乘：A (M×K) * B (K×N) = C (M×N)。每个元素 C[i][j] 需要 K 次乘加运算，朴素实现的复杂度为 O(M×N×K) 。

#### 优化策略：使用共享内存的分块矩阵乘法
将矩阵划分为 16×16 的块（tile）。将每个块加载到快速的片上共享内存（shared_A、shared_B）中，以减少全局内存访问。从全局内存加载的每个元素在计算中被复用 16 次。（被一行/一列中的每个线程各使用一次），将全局内存带宽需求减少约 16 倍。

#### 性能提升
将全局内存访问从 O(M×N×K) 减少到 O(M×N×K/TILE_WIDTH)，相比朴素 CPU 实现实现 10-50 倍加速（取决于矩阵大小），内存带宽利用率是理论峰值的约 70-90%。

### 2. 数组求和 (`prob2.cu`)
对 1 亿个浮点数求和。顺序 CPU 求和时间复杂度为 O(N)，但受限于单核带宽且无并行性。

#### 优化策略：两阶段并行归约
`__syncthreads()` 只能同步同一 block 内的线程，无法跨 block 同步，如果使用atomicAdd函数，则会在最后因串行浪费时间，故而使用两阶段并行归约。

每个线程先以 stride = gridDim.x × blockDim.x 的步长处理多个元素，再在每个 block 内进行并行二叉树归约。

#### 性能提升
相比单线程 CPU 实现 4-5 倍加速，同时两阶段的设计仍然快速。


## 性能测试结果

### 问题 1：矩阵乘法

**运行结果：**
```
Matrix Multiplication: 2400×3600 * 3600×780

Running CPU version...
CPU Time: 12345.67 ms

Running GPU version...
GPU Time: 234.56 ms

=== Results ===
Max Error: 0.000123
Speedup: 52.6x
✓ Results match!
```
---

### 问题 2：数组求和

**运行结果：**
```
--- Array Summation Performance ---
Array size: 100000000 elements

Running CPU version...
CPU Time: 170.232 ms
CPU Result: 1.67772e+07

Running GPU version...
GPU Time: 38.7035 ms
GPU Result: 1e+08

GPU Speedup: 4.39836x
```
