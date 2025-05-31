#ifndef COMMON_KERNELS__CUH
#define COMMON_KERNELS__CUH

#include <cuda_runtime.h>

__global__ void velocity_kernel(double *v, const double *u,
                                const double *u_past, const double dt,
                                const uint32_t n) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    v[idx] = (u[idx] - u_past[idx]) / dt;
  }
}

#endif
