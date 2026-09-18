cmake_minimum_required(VERSION 3.20)
project(CGA_BFS_Improved LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# ── Base sources (from parent src/ — unchanged files) ─────────────────────────
set(PARENT_SRC "${CMAKE_SOURCE_DIR}/../src")

set(MATRIX_SRCS  ${PARENT_SRC}/matrix/csr.cpp
                  ${PARENT_SRC}/matrix/csc.cpp
                  ${PARENT_SRC}/matrix/loader.cpp)
set(VECTOR_SRCS  ${PARENT_SRC}/vector/dense_vector.cpp
                  ${PARENT_SRC}/vector/sparse_vector.cpp
                  ${PARENT_SRC}/vector/conversion.cpp)
set(CPU_UTIL_SRCS ${PARENT_SRC}/cpu/utils/hash_table.cpp
                   ${PARENT_SRC}/cpu/utils/bucket.cpp
                   ${PARENT_SRC}/cpu/utils/load_balance.cpp)
set(CPU_KRN_SRCS  ${PARENT_SRC}/cpu/spmspv/pm_bhash.cpp
                   ${PARENT_SRC}/cpu/spmspv/lb_pm_bhash.cpp
                   ${PARENT_SRC}/cpu/spmspv/pb_mspa.cpp
                   ${PARENT_SRC}/cpu/spmspv/lb_pb_mspa.cpp
                   ${PARENT_SRC}/cpu/spmspv/gustavson.cpp)
set(GPU_SRCS      ${PARENT_SRC}/gpu/utils/memory.cu
                   ${PARENT_SRC}/gpu/utils/conversion.cu
                   ${PARENT_SRC}/gpu/spmspv/sort_based.cu
                   ${PARENT_SRC}/gpu/spmv/csr_vector.cu
                   ${PARENT_SRC}/gpu/spmv/merge_based.cu)
set(UTIL_SRCS     ${PARENT_SRC}/utils/timer.cpp
                   ${PARENT_SRC}/utils/logger.cpp)

# ── Improved sources (override features, inference, bfs) ─────────────────────
set(IMPR_SRC "${CMAKE_SOURCE_DIR}")
set(MODEL_SRCS    ${IMPR_SRC}/models/features.cpp   # IMPROVED (DIM=25, CSC fix)
                  ${IMPR_SRC}/models/inference.cpp  # IMPROVED (all DTA fixes)
                  ${PARENT_SRC}/models/model1_platform.cpp
                  ${PARENT_SRC}/models/model2_cpu.cpp
                  ${PARENT_SRC}/models/model3_gpu.cpp
                  ${PARENT_SRC}/models/model4_universal.cpp)
set(CORE_SRCS     ${IMPR_SRC}/core/bfs.cpp          # IMPROVED (CSR+CSC call-site)
                  ${PARENT_SRC}/core/execution_flow.cpp
                  ${PARENT_SRC}/core/scheduler.cpp)

set(ALL_SRCS ${MATRIX_SRCS} ${VECTOR_SRCS}
             ${CPU_UTIL_SRCS} ${CPU_KRN_SRCS}
             ${GPU_SRCS} ${MODEL_SRCS}
             ${CORE_SRCS} ${UTIL_SRCS})

# ── Include paths: improved headers shadow parent headers ─────────────────────
include_directories(${IMPR_SRC})         # improved/models/features.h takes priority
include_directories(${PARENT_SRC})       # everything else from original src/

# ── Executable (P1-10/P2-3: canonical improved main with --dta-type) ──────────
add_executable(cga_bfs_improved ${IMPR_SRC}/src/main.cpp ${ALL_SRCS})
target_link_libraries(cga_bfs_improved
    PRIVATE OpenMP::OpenMP_CXX CUDA::cudart CUDA::cusparse CUDA::cublas)
target_compile_options(cga_bfs_improved
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

# ── Test target (uses improved models) ────────────────────────────────────────
add_executable(test_models_improved
    ${PARENT_SRC}/../tests/test_models.cpp
    ${MODEL_SRCS}
    ${MATRIX_SRCS}
    ${VECTOR_SRCS}
    ${UTIL_SRCS})
target_link_libraries(test_models_improved
    PRIVATE OpenMP::OpenMP_CXX CUDA::cudart)
target_include_directories(test_models_improved
    PRIVATE ${IMPR_SRC} ${PARENT_SRC})

# P0-10 regression (DIM 25, t_var index, one-copy k4).
add_executable(test_improved_dta
    ${PARENT_SRC}/../tests/test_improved_dta.cpp
    ${MODEL_SRCS}
    ${MATRIX_SRCS}
    ${VECTOR_SRCS}
    ${UTIL_SRCS})
target_link_libraries(test_improved_dta
    PRIVATE OpenMP::OpenMP_CXX CUDA::cudart)
target_include_directories(test_improved_dta
    PRIVATE ${IMPR_SRC} ${PARENT_SRC})

enable_testing()
add_test(NAME Improved_Models COMMAND test_models_improved
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..)
add_test(NAME Improved_DTA COMMAND test_improved_dta
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..)

message(STATUS "Improved build: DIM=25, CSC col_counts fix, all DTA improvements")
message(STATUS "CUDA Architectures = ${CMAKE_CUDA_ARCHITECTURES}")
