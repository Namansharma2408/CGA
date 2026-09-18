cmake_minimum_required(VERSION 3.20)
project(CGA_BFS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── CUDA Architecture ─────────────────────────────────────────────────────────
# Set to match your GPU: 86=RTX 30xx, 89=RTX 40xx, 80=A100, 75=RTX 20xx
# Use: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── OpenMP ────────────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

# ── CUDA libraries ────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Compile flags ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# ═══════════════════════════════════════════════════════════════════════════════
# Source collections
# ═══════════════════════════════════════════════════════════════════════════════

# Matrix / Vector
set(MATRIX_SRCS
    src/matrix/csr.cpp
    src/matrix/csc.cpp
    src/matrix/loader.cpp
)
set(VECTOR_SRCS
    src/vector/dense_vector.cpp
    src/vector/sparse_vector.cpp
    src/vector/conversion.cpp
)

# CPU kernels + utilities
set(CPU_UTIL_SRCS
    src/cpu/utils/hash_table.cpp
    src/cpu/utils/bucket.cpp
    src/cpu/utils/load_balance.cpp
)
set(CPU_KERNEL_SRCS
    src/cpu/spmspv/pm_bhash.cpp
    src/cpu/spmspv/lb_pm_bhash.cpp
    src/cpu/spmspv/pb_mspa.cpp
    src/cpu/spmspv/lb_pb_mspa.cpp
    src/cpu/spmspv/gustavson.cpp
)

# GPU CUDA kernels
set(GPU_SRCS
    src/gpu/utils/memory.cu
    src/gpu/utils/conversion.cu
    src/gpu/spmspv/sort_based.cu
    src/gpu/spmv/csr_vector.cu
    src/gpu/spmv/merge_based.cu
)

# ML Models
set(MODEL_SRCS
    src/models/features.cpp
    src/models/model1_platform.cpp
    src/models/model2_cpu.cpp
    src/models/model3_gpu.cpp
    src/models/model4_universal.cpp
    src/models/inference.cpp
)

# Core BFS
set(CORE_SRCS
    src/core/bfs.cpp
    src/core/execution_flow.cpp
    src/core/scheduler.cpp
)

# Utilities
set(UTIL_SRCS
    src/utils/timer.cpp
    src/utils/logger.cpp
)

set(ALL_SRCS
    ${MATRIX_SRCS}
    ${VECTOR_SRCS}
    ${CPU_UTIL_SRCS}
    ${CPU_KERNEL_SRCS}
    ${GPU_SRCS}
    ${MODEL_SRCS}
    ${CORE_SRCS}
    ${UTIL_SRCS}
)

include_directories(src)

# ═══════════════════════════════════════════════════════════════════════════════
# Main executable: cga_bfs
# ═══════════════════════════════════════════════════════════════════════════════
add_executable(cga_bfs src/main.cpp ${ALL_SRCS})
target_link_libraries(cga_bfs
    PRIVATE
        OpenMP::OpenMP_CXX
        CUDA::cudart
        CUDA::cusparse
        CUDA::cublas
)
target_compile_options(cga_bfs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

# ═══════════════════════════════════════════════════════════════════════════════
# Test target: test_cpu
# ═══════════════════════════════════════════════════════════════════════════════
add_executable(test_cpu
    tests/test_cpu.cpp
    ${MATRIX_SRCS}
    ${VECTOR_SRCS}
    ${CPU_UTIL_SRCS}
    ${CPU_KERNEL_SRCS}
    ${UTIL_SRCS}
)
target_link_libraries(test_cpu PRIVATE OpenMP::OpenMP_CXX CUDA::cudart)
target_include_directories(test_cpu PRIVATE src)

# ═══════════════════════════════════════════════════════════════════════════════
# Test target: test_gpu
# ═══════════════════════════════════════════════════════════════════════════════
add_executable(test_gpu
    tests/test_gpu.cu
    ${MATRIX_SRCS}
    ${VECTOR_SRCS}
    ${GPU_SRCS}
    ${UTIL_SRCS}
)
target_link_libraries(test_gpu PRIVATE CUDA::cudart CUDA::cusparse)
target_include_directories(test_gpu PRIVATE src)

# ═══════════════════════════════════════════════════════════════════════════════
# Test target: test_models
# ═══════════════════════════════════════════════════════════════════════════════
add_executable(test_models
    tests/test_models.cpp
    ${MODEL_SRCS}
    ${MATRIX_SRCS}
    ${VECTOR_SRCS}
    ${UTIL_SRCS}
)
target_link_libraries(test_models PRIVATE OpenMP::OpenMP_CXX CUDA::cudart)
target_include_directories(test_models PRIVATE src)

# ═══════════════════════════════════════════════════════════════════════════════
# CTest
# ═══════════════════════════════════════════════════════════════════════════════
enable_testing()
add_test(NAME CPU_Kernels  COMMAND test_cpu)
add_test(NAME GPU_Kernels  COMMAND test_gpu)
add_test(NAME ML_Models    COMMAND test_models WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

message(STATUS "CMAKE_BUILD_TYPE  = ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Architectures= ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CXX Compiler      = ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
