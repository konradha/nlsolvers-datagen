# cmake .. -DENABLE_GPU=ON -DCMAKE_CXX_COMPILER=nvcc
cmake_minimum_required(VERSION 3.14)
project(nlsolvers VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(ENABLE_GPU "Enable GPU support" OFF)
option(OPTIMIZE_SIZE "Optimize binaries for size" OFF)
option(OPTIMIZE_SPEED "Optimize for maximum perf" ON)

option(BUILD_NLSE_CUBIC "Build NLSE with cubic nonlinearity" ON)
option(BUILD_KG "Build KGE" ON)

### EASILY EXTENED HERE: JUST COPY HEADER AND DRIVER TO BUILD NEW NONLINEARITY INTO LIBRARY


if(ENABLE_GPU)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures") 
  set(CMAKE_CXX_COMPILER ${CMAKE_CUDA_COMPILER})
  find_package(CUDAToolkit REQUIRED)

  
  if(OPTIMIZE_SPEED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --use_fast_math --expt-relaxed-constexpr --extended-lambda")
  endif()
  include_directories(
    ${CMAKE_SOURCE_DIR}/device/include
  )

  set(CUDA_LIBRARIES 
      CUDA::cusolver
      CUDA::cusparse
      CUDA::cublas
  )

else()
  if(OPTIMIZE_SPEED)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      add_compile_options(-O2 -march=native -ftree-vectorize -ffast-math)
      add_compile_options(-fprefetch-loop-arrays -fopenmp-simd)
      add_compile_options(-flto -fPIC -funroll-loops)
      add_link_options(-flto)
      add_link_options(-fuse-ld=gold)
      add_link_options(-Wl,--gc-sections)
      add_link_options(-Wl,-O3)
    endif()
  elseif(OPTIMIZE_SIZE)
    add_compile_options(-Os)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      add_compile_options(-ffunction-sections -fdata-sections)
      add_link_options(-Wl,--gc-sections)
    endif()
  endif()
  add_definitions(-DEIGEN_NO_DEBUG)
  add_definitions(-DEIGEN_MAX_ALIGN_BYTES=64)
  # these might hurt the cluster
  # add_definitions(-DEIGEN_USE_BLAS)
  # add_definitions(-DEIGEN_USE_LAPACKE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang") 
    add_link_options(-static-libgcc -static-libstdc++)
  endif()
endif()

find_package(Eigen3 REQUIRED)

# these might hurt the cluster
# find_package(BLAS REQUIRED)
# find_package(LAPACK REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
link_libraries(${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

set(LIBNPY_DIR "${CMAKE_SOURCE_DIR}/../libnpy" CACHE PATH "Path to libnpy directory")
if(EXISTS "${LIBNPY_DIR}/include/npy.hpp")
  include_directories(${LIBNPY_DIR}/include)
  add_definitions(-DHAVE_LIBNPY)
else()
  message(FATAL_ERROR "libnpy not found at ${LIBNPY_DIR}. Specify with -DLIBNPY_DIR=path/to/libnpy")
endif()


include_directories(
  ${CMAKE_SOURCE_DIR}/common/include
  ${CMAKE_SOURCE_DIR}/host/include
)

add_subdirectory(common)
add_subdirectory(host)

if(ENABLE_GPU)
  add_subdirectory(device)
endif()
