#ifndef NLSE_CUBIC_QUINTIC_CUH
#define NLSE_CUBIC_QUINTIC_CUH

#include <cuda_runtime.h>
#include <thrust/complex.h>

namespace device {

__global__ void density_cubic_quintic(thrust::complex<double> *rho,
                                      const thrust::complex<double> *u,
                                      const double *m, const double sigma1,
                                      const double sigma2, const uint32_t nx,
                                      const uint32_t ny) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < nx && j < ny) {
    const uint32_t idx = j * nx + i;
    thrust::complex<double> val = u[idx];
    double abs_u_squared = val.real() * val.real() + val.imag() * val.imag();
    rho[idx] = m[idx] * (sigma1 * abs_u_squared +
                         sigma2 * abs_u_squared * abs_u_squared);
  }
}

__global__ void nonlin_part_cubic_quintic(thrust::complex<double> *out,
                                          const thrust::complex<double> *in,
                                          const thrust::complex<double> *rho,
                                          const thrust::complex<double> tau,
                                          const uint32_t nx,
                                          const uint32_t ny) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < nx && j < ny) {
    const uint32_t idx = j * nx + i;
    thrust::complex<double> val = rho[idx];
    out[idx] = in[idx] * exp(tau * val);
  }
}

} // namespace device

#endif
