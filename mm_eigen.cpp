#include <cstdio>
#include <omp.h>
#include "utils.hpp"
#include <Eigen/Dense>

constexpr int ALIGN_SIZE = 64;

#if !defined(M_SIZE) || !defined(N_SIZE) || !defined(K_SIZE)
    #define M_SIZE      1536
    #define N_SIZE      1152
    #define K_SIZE      960
#endif

float A[M_SIZE * K_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float B[K_SIZE * N_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };
float C[M_SIZE * N_SIZE] __attribute__((aligned(ALIGN_SIZE))) = { 0.0f, };

void matmul(const float *A, const float *B, float *C,
    const int M, const int N, const int K)
{
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a(A, M, K);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b(B, K, N);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> c(C, M, N);
    
    c = a * b;
}

int main(int argc, char *argv[])
{
    constexpr size_t FLOPs = 2 * static_cast<size_t>(M_SIZE) * static_cast<size_t>(N_SIZE) * static_cast<size_t>(K_SIZE);
    std::printf("M = %d, N = %d, K = %d, FLOPs = %zu\n", M_SIZE, N_SIZE, K_SIZE, FLOPs);
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