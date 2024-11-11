#include <cstdio>
#include <omp.h>
#include "utils.hpp"
#include <arm_neon.h>

constexpr int ALIGN_SIZE = 64;
constexpr int KERNEL_ROW = 12;
constexpr int KERNEL_COL = 8;

#if !defined(M_SIZE) || !defined(N_SIZE) || !defined(K_SIZE)
    #define M_SIZE      1536
    #define N_SIZE      1152
    #define K_SIZE      960
#endif

float A[M_SIZE * K_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float B[K_SIZE * N_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float C[M_SIZE * N_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float A_pack[KERNEL_ROW * K_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };

void Kernel12x8(const float *A, const float *B, float *C,
    const int K, const int lda, const int ldb, const int ldc)
{
    float32x4_t c00 = vdupq_n_f32(0.0f);
    float32x4_t c01 = vdupq_n_f32(0.0f);
    float32x4_t c10 = vdupq_n_f32(0.0f);
    float32x4_t c11 = vdupq_n_f32(0.0f);
    float32x4_t c20 = vdupq_n_f32(0.0f);
    float32x4_t c21 = vdupq_n_f32(0.0f);
    float32x4_t c30 = vdupq_n_f32(0.0f);
    float32x4_t c31 = vdupq_n_f32(0.0f);
    float32x4_t c40 = vdupq_n_f32(0.0f);
    float32x4_t c41 = vdupq_n_f32(0.0f);
    float32x4_t c50 = vdupq_n_f32(0.0f);
    float32x4_t c51 = vdupq_n_f32(0.0f);
    float32x4_t c60 = vdupq_n_f32(0.0f);
    float32x4_t c61 = vdupq_n_f32(0.0f);
    float32x4_t c70 = vdupq_n_f32(0.0f);
    float32x4_t c71 = vdupq_n_f32(0.0f);
    float32x4_t c80 = vdupq_n_f32(0.0f);
    float32x4_t c81 = vdupq_n_f32(0.0f);
    float32x4_t c90 = vdupq_n_f32(0.0f);
    float32x4_t c91 = vdupq_n_f32(0.0f);
    float32x4_t c100 = vdupq_n_f32(0.0f);
    float32x4_t c101 = vdupq_n_f32(0.0f);
    float32x4_t c110 = vdupq_n_f32(0.0f);
    float32x4_t c111 = vdupq_n_f32(0.0f);

    const float *a_ptr = A;
    const float *b_ptr = B;

    for (int k = 0; k < K; ++k)
    {
        float32x4_t a0 = vld1q_f32(a_ptr);
        float32x4_t a1 = vld1q_f32(a_ptr + 4);
        float32x4_t a2 = vld1q_f32(a_ptr + 8);

        float32x4_t b0 = vld1q_f32(b_ptr);
        float32x4_t b1 = vld1q_f32(b_ptr + 4);

        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c40 = vfmaq_laneq_f32(c40, b0, a1, 0);
        c41 = vfmaq_laneq_f32(c41, b1, a1, 0);
        c50 = vfmaq_laneq_f32(c50, b0, a1, 1);
        c51 = vfmaq_laneq_f32(c51, b1, a1, 1);
        c60 = vfmaq_laneq_f32(c60, b0, a1, 2);
        c61 = vfmaq_laneq_f32(c61, b1, a1, 2);
        c70 = vfmaq_laneq_f32(c70, b0, a1, 3);
        c71 = vfmaq_laneq_f32(c71, b1, a1, 3);
        c80 = vfmaq_laneq_f32(c80, b0, a2, 0);
        c81 = vfmaq_laneq_f32(c81, b1, a2, 0);
        c90 = vfmaq_laneq_f32(c90, b0, a2, 1);
        c91 = vfmaq_laneq_f32(c91, b1, a2, 1);
        c100 = vfmaq_laneq_f32(c100, b0, a2, 2);
        c101 = vfmaq_laneq_f32(c101, b1, a2, 2);
        c110 = vfmaq_laneq_f32(c110, b0, a2, 3);
        c111 = vfmaq_laneq_f32(c111, b1, a2, 3);

        a_ptr += lda;
        b_ptr += ldb;
    }

    float *c_ptr = C;
    vst1q_f32(c_ptr, c00);
    vst1q_f32(&c_ptr[4], c01);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c10);
    vst1q_f32(&c_ptr[4], c11);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c20);
    vst1q_f32(&c_ptr[4], c21);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c30);
    vst1q_f32(&c_ptr[4], c31);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c40);
    vst1q_f32(&c_ptr[4], c41);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c50);
    vst1q_f32(&c_ptr[4], c51);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c60);
    vst1q_f32(&c_ptr[4], c61);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c70);
    vst1q_f32(&c_ptr[4], c71);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c80);
    vst1q_f32(&c_ptr[4], c81);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c90);
    vst1q_f32(&c_ptr[4], c91);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c100);
    vst1q_f32(&c_ptr[4], c101);
    c_ptr += ldc;
    vst1q_f32(c_ptr, c110);
    vst1q_f32(&c_ptr[4], c111);
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
            Kernel12x8(A_pack, &B[n], &C[m * N + n], K, KERNEL_ROW, N, N);
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