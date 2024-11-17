cmake_minimum_required(VERSION 3.10)
project(Optimize-GEMM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-DM_SIZE=1536 -DN_SIZE=1152 -DK_SIZE=1344 -O3")

# openmp
find_package(OpenMP REQUIRED)

# naive
add_executable(mm_naive mm_naive.cpp)
target_link_libraries(mm_naive OpenMP::OpenMP_CXX)

# kernel 4x4
add_executable(mm_kernel4x4 mm_kernel4x4.cpp)
target_link_libraries(mm_kernel4x4 OpenMP::OpenMP_CXX)

# kernel 12x8
add_executable(mm_kernel12x8 mm_kernel12x8.cpp)
target_link_libraries(mm_kernel12x8 OpenMP::OpenMP_CXX)

# optimize
add_executable(mm_optimize mm_optimize.cpp)
target_link_libraries(mm_optimize OpenMP::OpenMP_CXX)

set(ENV_ROOT "/your/envs")
# openblas
set(OPENBLAS_INCLUDE_DIR "${ENV_ROOT}/openblas/include")
set(OPENBLAS_LIB_DIR "${ENV_ROOT}/openblas/lib")
# gsl
set(GSL_INCLUDE_DIR "${ENV_ROOT}/gsl/include")
set(GSL_LIB_DIR "${ENV_ROOT}/gsl/lib")
# eigen
set(EIGEN_INCLUDE_DIR "${ENV_ROOT}/eigen/include/eigen3")

# openblas
add_executable(mm_blas mm_blas.cpp)
target_include_directories(mm_blas PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_link_directories(mm_blas PRIVATE ${OPENBLAS_LIB_DIR})
target_link_libraries(mm_blas -lopenblas OpenMP::OpenMP_CXX -lpthread -lm)

# eigen
add_executable(mm_eigen mm_eigen.cpp)
target_include_directories(mm_eigen PRIVATE ${EIGEN_INCLUDE_DIR})
target_link_libraries(mm_eigen OpenMP::OpenMP_CXX)

# gsl
add_executable(mm_gsl mm_gsl.cpp)
target_include_directories(mm_gsl PRIVATE ${GSL_INCLUDE_DIR})
target_link_directories(mm_gsl PRIVATE ${GSL_LIB_DIR})
target_link_libraries(mm_gsl -lgsl -lgslcblas OpenMP::OpenMP_CXX -lm)