#ifndef KG_SINGLE_CUH
#define KG_SINGLE_CUH

#include "common_kernels.cuh"
#include "matfunc_real.hpp"
#include "pragmas.hpp"
#include "spmv.hpp"

#include <cuda_runtime.h>
// let's try to finally get rid of hand-rolled kernels
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

namespace device {

namespace KGESolver {

__global__ void neg_kernel(double *out, const double *in, const double *m,
                           const uint32_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = -m[idx] * in[idx] * in[idx] * in[idx];
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

__global__ void sv_kernel(double *u, double *u_past, double *l,
                          double *nl, // linear operator, nonlinear operator
                          const double tau, const uint32_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    u[idx] = 2 * u[idx] - u_past[idx] + tau * tau * (l[idx] + nl[idx]);
  }
}

void m_u_cubed(double * out, double* m, double* u, uint32_t n) {
    thrust::device_ptr<double> m_ptr(m);
    thrust::device_ptr<double> u_ptr(u);
    thrust::device_ptr<double> result_ptr(out);

    thrust::transform(u_ptr, u_ptr + n, u_ptr, result_ptr, thrust::multiplies<double>());
    thrust::transform(result_ptr, result_ptr + n, u_ptr, result_ptr, thrust::multiplies<double>());
    thrust::transform(m_ptr, m_ptr + n, result_ptr, result_ptr, thrust::multiplies<double>());
}

void step(double *d_v, double *d_u, double *d_u_past, double *d_buf,
          double *d_buf2, double *d_buf3, MatrixFunctionApplicatorReal *matfunc,
          const double *d_m, const double tau, const uint32_t n,
          const dim3 grid, const dim3 block) {
  // cudaMemcpy(d_buf, d_u, n * sizeof(double), cudaMemcpyDeviceToDevice);
  // matfunc->apply(d_buf2, d_u, tau,
  //                MatrixFunctionApplicatorReal::FunctionType::ID_SQRT);
  // neg_kernel<<<grid, block>>>(d_buf2, d_buf2, d_m, n);
  // matfunc->apply(d_buf3, d_buf2, tau,
  //                MatrixFunctionApplicatorReal::FunctionType::SINC2_SQRT);
  // matfunc->apply(d_buf2, d_u, tau,
  //                MatrixFunctionApplicatorReal::FunctionType::COS_SQRT);
  // gautschi_kernel<<<grid, block>>>(d_u, d_u, d_u_past, d_buf2, d_buf3, tau, n);
  // cudaMemcpy(d_u_past, d_buf, n * sizeof(double), cudaMemcpyDeviceToDevice);
  // velocity_kernel<<<grid, block>>>(d_v, d_u, d_u_past, tau, n);

  // thrust::device_ptr<double> d_u_ptr(d_u);
  // thrust::device_ptr<double> d_buf_ptr(d_buf);
  // thrust::copy(d_u_ptr, d_u_ptr + n, d_buf_ptr);
  // matfunc->apply(d_buf2, d_u, tau,
  //                MatrixFunctionApplicatorReal::FunctionType::ID_SQRT);
  // thrust::device_ptr<const double> d_m_ptr(d_m);
  // thrust::device_ptr<double> d_buf2_ptr(d_buf2);
  // thrust::transform(d_u_ptr, d_u_ptr + n, d_m_ptr, d_buf2_ptr,
  //                    [] __device__ (double u_val, double m_val) {
  //                        return -m_val * u_val * u_val * u_val;
  //                    });
  // matfunc->apply(d_buf3, d_buf2, tau,
  //                MatrixFunctionApplicatorReal::FunctionType::SINC2_SQRT);
  // matfunc->apply(d_buf2, d_u, tau,
  //                MatrixFunctionApplicatorReal::FunctionType::COS_SQRT);

  // thrust::transform(d_buf2_ptr, d_buf2_ptr + n, d_buf2_ptr,
  //                 thrust::placeholders::_1 * 2.0);



  thrust::device_ptr<double> d_u_ptr(d_u);
  thrust::device_ptr<double> d_u_past_ptr(d_u_past);
  thrust::device_ptr<double> d_buf_ptr(d_buf);
  thrust::device_ptr<double> d_buf2_ptr(d_buf2);
  thrust::device_ptr<double> d_buf3_ptr(d_buf3);
  thrust::device_ptr<const double> d_m_ptr(d_m);

  thrust::copy(d_u_ptr, d_u_ptr + n, d_buf_ptr);
  
  matfunc->apply(d_buf2, d_u, tau, MatrixFunctionApplicatorReal::FunctionType::COS_SQRT);
  thrust::transform(d_buf2_ptr, d_buf2_ptr + n, d_buf2_ptr,
                    thrust::placeholders::_1 * 2.0);
  thrust::transform(d_u_ptr, d_u_ptr + n, d_m_ptr, d_buf3_ptr,
                    [] __device__ (double u_val, double m_val) {
                        return -m_val * u_val * u_val * u_val;
                    });
 
  matfunc->apply(d_buf3, d_buf3, tau, MatrixFunctionApplicatorReal::FunctionType::SINC2_SQRT);
  double tau_squared = tau * tau;
  thrust::transform(d_buf3_ptr, d_buf3_ptr + n, d_buf3_ptr,
                    thrust::placeholders::_1 * tau_squared);
 
  thrust::transform(d_buf2_ptr, d_buf2_ptr + n, d_u_past_ptr, d_u_ptr,
                    thrust::minus<double>());
  thrust::transform(d_u_ptr, d_u_ptr + n, d_buf3_ptr, d_u_ptr,
                    thrust::plus<double>());
  thrust::copy(d_buf_ptr, d_buf_ptr + n, d_u_past_ptr);
 
  thrust::device_ptr<double> d_v_ptr(d_v);
  thrust::transform(d_u_ptr, d_u_ptr + n, d_u_past_ptr, d_v_ptr,
                    [tau] __device__ (double u, double u_past) {
                        return (u - u_past) / tau;
                    });
}

void step_sv(double *d_v, double *d_u, double *d_u_past, double *d_buf,
             double *d_buf2, double *d_buf3,
             MatrixFunctionApplicatorReal *matfunc, const double *d_m,
             const double tau, const uint32_t n, const dim3 grid,
             const dim3 block) {
  /*
  // u_n-1 bookkeeping
  cudaMemcpy(d_buf3, d_u, n * sizeof(double), cudaMemcpyDeviceToDevice);
  // 2 u_n - u_{n-1} + dt² (Lu_n - m f(u))
  matfunc->expose_spmv()->multiply(d_u, d_buf);
  neg_kernel<<<grid, block>>>(d_buf2, d_u, d_m, n);
  sv_kernel<<<grid, block>>>(d_u, d_u_past, d_buf, d_buf2, tau, n);
  cudaMemcpy(d_u_past, d_buf3, n * sizeof(double), cudaMemcpyDeviceToDevice);
  velocity_kernel<<<grid, block>>>(d_v, d_u, d_u_past, tau, n);
  */

    thrust::device_ptr<double> d_u_ptr(d_u);
    thrust::device_ptr<double> d_buf3_ptr(d_buf3);
    thrust::copy(d_u_ptr, d_u_ptr + n, d_buf3_ptr);
   
    matfunc->expose_spmv()->multiply(d_u, d_buf);
    thrust::device_ptr<const double> d_m_ptr(d_m);
    thrust::device_ptr<double> d_buf2_ptr(d_buf2);
    thrust::transform(d_u_ptr, d_u_ptr + n, d_m_ptr, d_buf2_ptr,
                     [] __device__ (double u_val, double m_val) {
                         return -m_val * u_val * u_val * u_val;
                     });
    
    double dt_squared = tau * tau;
    thrust::device_ptr<double> d_buf_ptr(d_buf);
    thrust::transform(d_buf_ptr, d_buf_ptr + n, d_buf_ptr,
                    [dt_squared] __device__ (double val) { return dt_squared * val; });
    thrust::transform(d_buf2_ptr, d_buf2_ptr + n, d_buf2_ptr,
                    [dt_squared] __device__ (double val) { return dt_squared * val; });
    
    thrust::device_ptr<double> d_u_past_ptr(d_u_past);
    thrust::transform(d_u_ptr, d_u_ptr + n, d_u_past_ptr, d_u_ptr,
                    [] __device__ (double u, double u_past) { return 2.0 * u - u_past; });
    
    thrust::transform(d_u_ptr, d_u_ptr + n, d_buf_ptr, d_u_ptr,
                    [] __device__ (double u, double dt2_Lu) { return u + dt2_Lu; });
    thrust::transform(d_u_ptr, d_u_ptr + n, d_buf2_ptr, d_u_ptr,
                    [] __device__ (double u, double dt2_nmu3) { return u + dt2_nmu3; });
    
    thrust::copy(d_buf3_ptr, d_buf3_ptr + n, d_u_past_ptr);
    thrust::device_ptr<double> d_v_ptr(d_v);
    thrust::transform(d_u_ptr, d_u_ptr + n, d_u_past_ptr, d_v_ptr,
                    [tau] __device__ (double u, double u_past) {
                        return (u - u_past) / tau;
                    });
}

} // namespace KGESolver

} // namespace device

#endif // KG_SINGLE_CUH
