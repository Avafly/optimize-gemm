#include <cstdio>
#include <omp.h>
#include "utils.hpp"
#include <arm_neon.h>

constexpr int ALIGN_SIZE = 64;
constexpr int KERNEL_ROW = 12;
constexpr int KERNEL_COL = 8;
constexpr int BLOCK_M = 384;
constexpr int BLOCK_N = 96;
constexpr int BLOCK_K = 64;

#if !defined(M_SIZE) || !defined(N_SIZE) || !defined(K_SIZE)
    #define M_SIZE      1536
    #define N_SIZE      1152
    #define K_SIZE      960
#endif

float A[M_SIZE * K_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float B[K_SIZE * N_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float C[M_SIZE * N_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float A_pack[BLOCK_M * K_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float B_pack[BLOCK_K * BLOCK_N] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };

void Kernel12x8(const float *A, const float *B, float *C,
    const int K, const int lda, const int ldb, const int ldc)
{
    float32x4_t c00 = vld1q_f32(C);
    float32x4_t c01 = vld1q_f32(C + 4);
    float32x4_t c10 = vld1q_f32(C + ldc);
    float32x4_t c11 = vld1q_f32(C + ldc + 4);
    float32x4_t c20 = vld1q_f32(C + 2 * ldc);
    float32x4_t c21 = vld1q_f32(C + 2 * ldc + 4);
    float32x4_t c30 = vld1q_f32(C + 3 * ldc);
    float32x4_t c31 = vld1q_f32(C + 3 * ldc + 4);
    float32x4_t c40 = vld1q_f32(C + 4 * ldc);
    float32x4_t c41 = vld1q_f32(C + 4 * ldc + 4);
    float32x4_t c50 = vld1q_f32(C + 5 * ldc);
    float32x4_t c51 = vld1q_f32(C + 5 * ldc + 4);
    float32x4_t c60 = vld1q_f32(C + 6 * ldc);
    float32x4_t c61 = vld1q_f32(C + 6 * ldc + 4);
    float32x4_t c70 = vld1q_f32(C + 7 * ldc);
    float32x4_t c71 = vld1q_f32(C + 7 * ldc + 4);
    float32x4_t c80 = vld1q_f32(C + 8 * ldc);
    float32x4_t c81 = vld1q_f32(C + 8 * ldc + 4);
    float32x4_t c90 = vld1q_f32(C + 9 * ldc);
    float32x4_t c91 = vld1q_f32(C + 9 * ldc + 4);
    float32x4_t c100 = vld1q_f32(C + 10 * ldc);
    float32x4_t c101 = vld1q_f32(C + 10 * ldc + 4);
    float32x4_t c110 = vld1q_f32(C + 11 * ldc);
    float32x4_t c111 = vld1q_f32(C + 11 * ldc + 4);

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
    for (int m = 0; m < M; m += BLOCK_M)
    {
        for (int n = 0; n < N; n += BLOCK_N)
        {
            for (int k = 0; k < K; k += BLOCK_K)
            {
                PackCopy(&B[k * N + n], B_pack, BLOCK_K, BLOCK_N, N);
                #pragma omp parallel for schedule(static)
                for (int bm = 0; bm < BLOCK_M; bm += KERNEL_ROW)
                {
                    PackTrans(&A[m * K + k + bm * K], &A_pack[bm * BLOCK_K], KERNEL_ROW, BLOCK_K, K);
                    for (int bn = 0; bn < BLOCK_N; bn += KERNEL_COL)
                    {
                        Kernel12x8(
                            &A_pack[bm * BLOCK_K],
                            &B_pack[bn],
                            &C[m * N + n + bm * N + bn],
                            BLOCK_K, KERNEL_ROW, BLOCK_N, N
                        );
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    constexpr size_t FLOPs = 2 * static_cast<size_t>(M_SIZE) * static_cast<size_t>(N_SIZE) * static_cast<size_t>(K_SIZE);
    std::printf("M = %d, N = %d, K = %d, FLOPs = %zu\n", M_SIZE, N_SIZE, K_SIZE, FLOPs);
    std::printf("Kernel row = %d, col = %d\n", KERNEL_ROW, KERNEL_COL);
    std::printf("BLOCK M = %d, N = %d, K = %d\n", BLOCK_M, BLOCK_N, BLOCK_K);
    // dim check
    if (M_SIZE % KERNEL_ROW != 0 || N_SIZE % KERNEL_COL != 0)
    {
        std::printf("Error: M and N must be divisible by kernel dims\n");
        return 1;
    }
    if (M_SIZE % BLOCK_M != 0 || N_SIZE % BLOCK_N != 0 || K_SIZE % BLOCK_K != 0)
    {
        std::printf("Error: MNK must be divisible by block dims\n");
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