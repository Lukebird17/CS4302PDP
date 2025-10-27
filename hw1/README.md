# CS4302 HW1
卢鸿良 523030910233

## 目录结构
```
hw1/
├── prob1.c          # 问题1: Floyd-Warshall 全源最短路径算法
├── prob2.c          # 问题2: Monte Carlo 方法估计 π 值
├── prob3.c          # 问题3: 2D 卷积并行计算
├── Makefile         # 编译脚本
└── README.md        # 本文档
```

## 环境配置

### 编译器
由于mac自带的clang不支持OpenMP，需要安装 Homebrew GCC

### 依赖库
1. OpenMP 库 
2. 标准数学库 (libm)

## 编译方法

### 编译所有程序
```bash
make
```

### 编译单个程序
```bash
make prob1    
make prob2    
make prob3    
```

### 清理编译产物
```bash
make clean
```

## 运行方法

### 问题1: Floyd-Warshall 全源最短路径
使用 Floyd-Warshall 算法计算图的全源最短路径，对比串行和并行实现的性能。

**运行格式**:
```bash
./prob1 <N> [threads]
```

**参数说明**:
N: 图的顶点数量
threads: (可选) 并行线程数，默认使用系统所有核心

**运行示例**:
```bash
./prob1 1024 8
# 或
make run1
```

**输出示例**:
```
[prob1-FloydWarshall] N=1024, threads=8
Serial Time:     0.383682 s
Parallel Time:   0.198960 s
Speedup:         1.93×
Max Abs Diff:    0.000e+00
```

**并行优化策略**:
在每轮迭代 k 中，对 (i, j) 二维循环使用 `collapse(2)` 并行化，正好不会产生数据依赖。
---

### 问题2: Monte Carlo 估计 π 值
使用 Monte Carlo 方法通过随机采样估计圆周率 π，对比串行和并行实现。

**运行格式**:
```bash
./prob2 <samples> <xi0> <xi1> <xi2> [threads]
```

**参数说明**:
- `samples`: 随机采样点数量
- `xi0, xi1, xi2`: 随机数生成器种子
- `threads`: (可选) 并行线程数

**运行示例**:
```bash
./prob2 10000000 111 222 333 8
# 或
make run2
```

**输出示例**:
```
[prob2-MonteCarloPI] samples=10000000, threads=8
True PI:         π = 3.141592653589793
Serial Result:   π = 3.14048160
Serial Time:     0.100964 s
Serial Error:    0.00111105 (0.0354%)
Parallel Result: π = 3.14069720
Parallel Time:   0.015175 s
Parallel Error:  0.00089545 (0.0285%)
Speedup:         6.65×
|Serial - Parallel|: 2.156e-04
```

**并行优化策略**:
使用 `reduction(+:in_parallel)` 累加落在圆内的点数。每个线程使用独立的随机数生成器状态 (`erand48`)，同时注意要通过 XOR 操作为每个线程生成不同的随机数种子，否则并行运算累加的结果都是相同的。内层for循环使用 `schedule(static)` 静态分配任务

---

### 问题3: 2D 卷积
实现 2D 卷积操作，使用 OpenMP reduction 机制并行化。

**运行格式**:
```bash
./prob3 <M> <N> <K> <seed> [threads]
```

**参数说明**:
- `M`: 输入矩阵行数
- `N`: 输入矩阵列数
- `K`: 卷积核大小 (K×K)
- `seed`: 随机数种子
- `threads`: (可选) 并行线程数

**运行示例**:
```bash
./prob3 1024 1024 5 42 8
# 或
make run3
```

**输出示例**:
```
[prob3-Conv2D] M=1024, N=1024, K=5, threads=8
Input Matrix:    1024 x 1024
Filter Size:     5 x 5
Output Matrix:   1020 x 1020 (valid mode)
Serial Time:     0.011193 s
Parallel Time:   0.002411 s
Speedup:         4.64×
Max Abs Diff:    0.000e+00
```

**并行优化策略**:
在输出矩阵的 (i, j) 位置上并行，使用 `collapse(2)`。对于每个输出元素的点积计算，使用 `#pragma omp simd reduction(+:sum)`

为了符合作业要求的"使用 reduction 机制累加点积结果"。在本实现中，`#pragma omp simd reduction(+:sum)` 指令在内层循环中安全地累加卷积和。

对于卷积核内的点积计算，如果为每个输出像素创建新的线程团队会产生巨大的线程创建开销。而SIMD可以在已有的线程内部，通过向量化指令一次处理多个乘加操作，无需额外的线程同步成本。同时，在外层已经并行化的情况下，再在内层使用 `parallel for` 会导致嵌套并行，线程创建和销毁的开销会远远超过实际计算时间。

---

