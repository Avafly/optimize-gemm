#include <iostream>
#include <cstdio>
#include <omp.h>
#include "utils.hpp"
#include <arm_neon.h>

constexpr int ALIGN_SIZE = 64;
constexpr int KERNEL_ROW = 4;
constexpr int KERNEL_COL = 4;

#if !defined(M_SIZE) || !defined(N_SIZE) || !defined(K_SIZE)
    #define M_SIZE      1536
    #define N_SIZE      1152
    #define K_SIZE      960
#endif

float A[M_SIZE * K_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float B[K_SIZE * N_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float C[M_SIZE * N_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float A_pack[KERNEL_ROW * K_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };

void Kernel4x4(const float *A, const float *B, float *C,
    const int K, const int lda, const int ldb, const int ldc)
{
    float32x4_t c0 = vdupq_n_f32(0.0f);
    float32x4_t c1 = vdupq_n_f32(0.0f);
    float32x4_t c2 = vdupq_n_f32(0.0f);
    float32x4_t c3 = vdupq_n_f32(0.0f);

    const float *a_ptr = A;
    const float *b_ptr = B;

    for (int k = 0; k < K; ++k)
    {
        float32x4_t a0 = vld1q_f32(a_ptr);
        float32x4_t b0 = vld1q_f32(b_ptr);

        c0 = vfmaq_laneq_f32(c0, b0, a0, 0);
        c1 = vfmaq_laneq_f32(c1, b0, a0, 1);
        c2 = vfmaq_laneq_f32(c2, b0, a0, 2);
        c3 = vfmaq_laneq_f32(c3, b0, a0, 3);

        a_ptr += lda;
        b_ptr += ldb;
    }

    float *c_ptr = C;
    vst1q_f32(c_ptr, c0);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c1);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c2);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c3);
}

void matmul(const float *A, const float *B, float *C,
    const int M, const int N, const int K)
{
    for (int m = 0; m < M; m += KERNEL_ROW)
    {
        PackTrans(&A[m * K], A_pack, KERNEL_ROW, K, K);
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n += KERNEL_COL)
        {
            Kernel4x4(A_pack, &B[n], &C[m * N + n], K, KERNEL_ROW, N, N);
        }
    }
}

int main(int argc, char *argv[])
{
    constexpr size_t FLOPs = 2 * static_cast<size_t>(M_SIZE) * static_cast<size_t>(N_SIZE) * static_cast<size_t>(K_SIZE);
    std::printf("M = %d, N = %d, K = %d, FLOPs = %zu\n", M_SIZE, N_SIZE, K_SIZE, FLOPs);
    std::printf("Kernel row = %d, col = %d\n", KERNEL_ROW, KERNEL_COL);
    // dim check
    if (M_SIZE % KERNEL_ROW != 0 || N_SIZE % KERNEL_COL != 0)
    {
        std::printf("Error: M and N must be divisible by kernel dims\n");
        return 1;
    }

    // load array
    if (LoadArray("../data/arr1.txt", A, M_SIZE * K_SIZE) == false ||
        LoadArray("../data/arr2.txt", B, K_SIZE * N_SIZE) == false)
    {
        std::printf("Failed to load data\n");
        return 1;
    }

    // timer
    double start_time = omp_get_wtime();
    // gemm
    matmul(A, B, C, M_SIZE, N_SIZE, K_SIZE);
    // show speed
    double elapsed_time = (omp_get_wtime() - start_time) * 1000.0;
    std::printf("Elapsed time: %.2f ms | ", elapsed_time);
    std::printf("GFLOPS: %.4f\n", static_cast<double>(FLOPs) / (elapsed_time * 1000000.0));

    // check
    std::printf("Checksum: %.4f\n", Checksum(C, static_cast<size_t>(M_SIZE * N_SIZE)));
    ShowResult(C, 100, 5);
    
    return 0;
}