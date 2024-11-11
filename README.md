# optimize-gemm

GEMM is crucial in AI inference, since it is the core of computations in many layers such as convolution and Transformer (attention). This repo shows how I optimized GEMM by methods like SIMD registers, blocking/tiling, packing, parallel computing to achieve over **100x** speed up (0.2837 -> 28.8882 GFLOPS) on Raspberry Pi 4 Model B (4GB RAM) using CPU only, and reaches the performance level of high-performance linear algebra libraries.

All matrices are row-major and initialized with values from text files, and thus the matrix C should be constant. Once the computation is complete, the sum and some elements will print for correctness checks.

## Optimization Steps

### 1. Naive

Basic triple-nested loop to compute each element of C. [mm_naive.cpp](https://github.com/Avafly/optimize-gemm/blob/main/mm_naive.cpp) achieved 0.2837 GFLOPS with elapsed time of 11976.29 ms.

### 2. Cache optimization

Matrix A and C are accessed sequentially in above code, but matrix B is accessed with a stride of N, which results in low cache hit rate for large matrices. Swapping the k and n loops can ensure sequential access to matrix B.

[mm_naive_cache.cpp](https://github.com/Avafly/optimize-gemm/blob/main/mm_naive_cache.cpp) improved performance to 2.3665 GFLOPS with elapsed time of 1435.60 ms.

### 3. 4x4 Kernel

This step used neon registers (ARM's SIMD equivalent to x64's AVX) to process multiple data with single instructions. A 4x4 kernel computes 16 elements of matrix C simultaneously, and elements in A and B only accessed from memory once per k loop instead of four times.

[mm_kernel4x4.cpp](https://github.com/Avafly/optimize-gemm/blob/main/mm_kernel4x4.cpp) improved performance to 2.4971 GFLOPS with elapsed time of 1360.52 ms.

### 4. 12x8 Kernel

The next step is selecting a suitable kernel size. [Cortex-A72](https://developer.arm.com/documentation/102467/0201/Check-your-knowledge) has 32 neon registers, and 20-30 registers is optimal for use. 12x8 was chosen to maximize register utilization, which requires 29 registers total (24 for C elements, 3 for A, and 2 for B).

[mm_kernel12x8.cpp](https://github.com/Avafly/optimize-gemm/blob/main/mm_kernel12x8.cpp) improved performance to 6.2343 GFLOPS with elapsed time of 544.95 ms.

### 5. Further optimizations

The kernel efficiently utilizes registers, but cache usage remained suboptimal. For large matrices, FLOPS [drop significantly](https://en.algorithmica.org/hpc/algorithms/matmul/) due to frequent memory access. This can be addressed with blocking/tiling, which improves cache hit rates by enhancing data locality. Additionally, blocking naturally enables parallel computation. Parallelization also improves cache and neon register utilization since each core has its own cache and registers. Additionally, packing the blocks before computation also enhances data locality.

[mm_optimize.cpp](https://github.com/Avafly/optimize-gemm/blob/main/mm_optimize.cpp) improved performance to 28.8882 GFLOPS with elapsed time of 117.60 ms - a **100x** speedup compared to the naive implementation! @_@

## Some Linear Algebra Libraries

Most linear algebra libraries provide gemm, and I tested against some well-known libraries: [eigen](https://eigen.tuxfamily.org/), [openblas](https://www.openblas.net/), and [gsl](https://www.gnu.org/software/gsl/).

|          | Elapsed Time (x4) | GFLOPS (x4) | Version |
| :------: | :---------------: | :---------: | :-----: |
| optimize |     117.60 ms     |   28.8882   |    -    |
|  eigen   |     190.91 ms     |   17.7960   |  3.4.0  |
| openblas |     107.62 ms     |   31.5694   | 0.3.28  |
|   gsl    |    4365.52 ms     |   0.7782    |   2.8   |

My gemm outperforms eigen by over 10 GFLOPS and is about 2 GFLOPS slower than openblas. Gsl doesn not provide an efficient implementation by default, resulting in poor performance (but it can utilize other libs as backends for gemm computation if specified during build).