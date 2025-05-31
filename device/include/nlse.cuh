#ifndef NLSE_CUH
#define NLSE_CUH

#include <cuda_runtime.h>
#include <thrust/complex.h>

namespace device {

__global__ void density(thrust::complex<double> *rho,
                        const thrust::complex<double> *u, const double *m,
                        const uint32_t nx, const uint32_t ny) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < nx && j < ny) {
    const uint32_t idx = j * nx + i;
    thrust::complex<double> val = u[idx];
    rho[idx] = m[idx] * (val.real() * val.real() + val.imag() * val.imag());
  }
}

__global__ void nonlin_part(thrust::complex<double> *out,
                            const thrust::complex<double> *in,
                            const thrust::complex<double> *rho,
                            const thrust::complex<double> tau,
                            const uint32_t nx, const uint32_t ny) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < nx && j < ny) {
    const uint32_t idx = j * nx + i;
    thrust::complex<double> val = rho[idx];
    out[idx] = in[idx] * exp(tau * val);
  }
}

__global__ void density_3d(thrust::complex<double> *rho,
                         const thrust::complex<double> *u, const double *m,
                         const uint32_t nx, const uint32_t ny, const uint32_t nz) {
  // TODO verify this indexing
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < nx && j < ny && k < nz) {
    const uint32_t idx = k * nx * ny + j * nx + i;
    thrust::complex<double> val = u[idx];
    rho[idx] = m[idx] * (val.real() * val.real() + val.imag() * val.imag());
  }
}

__global__ void nonlin_part_3d(thrust::complex<double> *out,
                             const thrust::complex<double> *in,
                             const thrust::complex<double> *rho,
                             const thrust::complex<double> tau,
                             const uint32_t nx, const uint32_t ny,
			     const uint32_t nz) {
  // TODO verify this indexing
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < nx && j < ny && k < nz) {
    const uint32_t idx = k * nx * ny + j * nx + i;
    thrust::complex<double> val = rho[idx];
    out[idx] = in[idx] * exp(tau * val);
  }
}


__global__ void B(thrust::complex<double> *out,
		const double * m,
		const thrust::complex<double> *u,
		const uint32_t nx, const uint32_t ny) {
  
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < nx && j < ny) {
    const uint32_t idx = j * nx + i;
    thrust::complex<double> val = u[idx];
    out[idx] = -m[idx] * (val.real() * val.real() + val.imag() * val.imag()) * val;
  }
}

__global__ void B_3d(thrust::complex<double> *out,
		const double * m,
		const thrust::complex<double> *u,
		const uint32_t nx, const uint32_t ny, const uint32_t nz) { 
  // TODO verify this indexing
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i < nx && j < ny && k < nz) {
    const uint32_t idx = k * nx * ny + j * nx + i;
    thrust::complex<double> val = u[idx];
    out[idx] = -m[idx] * (val.real() * val.real() + val.imag() * val.imag()) * val;
  }
}

__global__ void sewi(thrust::complex<double> * out, const thrust::complex<double> * exp_prev,
		     const thrust::complex<double> * exp_psi_B, const thrust::complex<double> tau,
		     const uint32_t nx, const uint32_t ny) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < nx && j < ny) {
    const uint32_t idx = j * nx + i;
    out[idx] = exp_prev[idx] - 2. * tau * exp_psi_B[idx];
  }
}

__global__ void sewi_3d(thrust::complex<double> * out, const thrust::complex<double> * exp_prev,
		     const thrust::complex<double> * exp_psi_B, const thrust::complex<double> tau,
		     const uint32_t nx, const uint32_t ny, const uint32_t nz) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i < nx && j < ny && k < nz) {
    const uint32_t idx = k * nx * ny + j * nx + i;
    out[idx] = exp_prev[idx] - 2. * tau * exp_psi_B[idx];
  }
}

} // namespace device

#endif
