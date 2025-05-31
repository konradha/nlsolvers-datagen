#ifndef SG_HYPERBOLIC_CUH
#define SG_HYPERBOLIC_CUH

#include "matfunc_real.hpp"
#include "pragmas.hpp"
#include "spmv.hpp"

#include <cuda_runtime.h>

namespace device {

namespace SGEHyperbolicSolver {

__global__ void neg_sinh_kernel(double *out, const double *in, const double *m,
                                const uint32_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = -(m[idx] * sinh(in[idx]));
  }
}

__global__ void gautschi_kernel(double *u_next, const double *u,
                                const double *u_past, const double *costu,
                                const double *filtered_sinc, const double tau,
                                const uint32_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    u_next[idx] =
        2.0 * costu[idx] - u_past[idx] + tau * tau * filtered_sinc[idx];
  }
}

void step(double *d_u, double *d_u_past, double *d_buf, double *d_buf2,
          double *d_buf3, MatrixFunctionApplicatorReal *matfunc,
          const double *d_m, const double tau, const uint32_t n,
          const dim3 grid, const dim3 block) {
  cudaMemcpy(d_buf, d_u, n * sizeof(double), cudaMemcpyDeviceToDevice);
  matfunc->apply(d_buf2, d_u, tau,
                 MatrixFunctionApplicatorReal::FunctionType::ID_SQRT);
  neg_sinh_kernel<<<grid, block>>>(d_buf2, d_buf2, d_m, n);
  matfunc->apply(d_buf3, d_buf2, tau,
                 MatrixFunctionApplicatorReal::FunctionType::SINC2_SQRT);
  matfunc->apply(d_buf2, d_u, tau,
                 MatrixFunctionApplicatorReal::FunctionType::COS_SQRT);
  gautschi_kernel<<<grid, block>>>(d_u, d_u, d_u_past, d_buf2, d_buf3, tau, n);
  cudaMemcpy(d_u_past, d_buf, n * sizeof(double), cudaMemcpyDeviceToDevice);
}

} // namespace SGEHyperbolicSolver

} // namespace device

#endif // SG_SINGLE_CUH
