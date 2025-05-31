#ifndef LANCZOS_HPP
#define LANCZOS_HPP

#include "pragmas.hpp"
#include "spmv.hpp"

#include <iostream>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>



#define NUM_BLOCKS 256
#define BLOCK_SIZE 256

struct KrylovInfo {
  double *__restrict__ T;
  double *__restrict__ V;
  double *__restrict__ buf1;
  double *__restrict__ buf2;
  double *__restrict__ d_beta;
  uint32_t n;
  uint32_t m;
};

/*
__global__ void norm(const double *__restrict__ vec,
                     double *__restrict__ result, const uint32_t n) {
  __shared__ double shared_mem[BLOCK_SIZE];
  const uint32_t tid = threadIdx.x;
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0.0;
  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    const double val = vec[i];
    sum += val * val;
  }
  shared_mem[tid] = sum;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_mem[tid] += shared_mem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, shared_mem[0]);
  }
}


__global__ void norm(const double *__restrict__ vec,
                     double *__restrict__ result, const uint32_t n) {
  __shared__ double shared_mem[BLOCK_SIZE];
  const uint32_t tid = threadIdx.x;
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  double sum = 0.0;
  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    double val = vec[i];
    sum += val * val;
  }

  shared_mem[tid] = sum;
  __syncthreads();

  for (uint32_t s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_mem[tid] += shared_mem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, shared_mem[0]);
  }
}
*/

// __global__ void inv_scale(double *__restrict__ vec, const double scalar,
//                           const uint32_t n) {
//   const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
// 
//   for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
//     vec[i] /= scalar;
//   }
// }

// __global__ void dot(const double *__restrict__ v1,
//                     const double *__restrict__ v2, double *__restrict__ result,
//                     const uint32_t n) {
//   __shared__ double shared_mem[BLOCK_SIZE];
//   const uint32_t tid = threadIdx.x;
//   const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
// 
//   double sum = 0.0;
//   for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
//     sum += v1[i] * v2[i];
//   }
// 
//   shared_mem[tid] = sum;
//   __syncthreads();
// 
//   for (uint32_t s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
//     if (tid < s) {
//       shared_mem[tid] += shared_mem[tid + s];
//     }
//     __syncthreads();
//   }
// 
//   if (tid == 0) {
//     atomicAdd(result, shared_mem[0]);
//   }
// }
// 
// __global__ void axpby(double *__restrict__ v1, const double *__restrict__ v2,
//                       const double alpha, const double beta, const uint32_t n) {
//   const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
// 
//   for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
//     v1[i] = alpha * v1[i] + beta * v2[i];
//   }
// }
// 
// __global__ void scalar_sqrt(double *__restrict__ x) {
//   const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid == 0)
//     x[0] = sqrt(x[0]);
// }

// void lanczos_iteration(DeviceSpMV<double> *spmv, KrylovInfo *krylov,
//                        const double *__restrict__ u) {
//   const uint32_t n = krylov->n;
//   const uint32_t m = krylov->m;
//   cudaMemset(krylov->V, 0, n * m * sizeof(double));
//   dim3 grid(NUM_BLOCKS);
//   dim3 block(BLOCK_SIZE);
//   cudaMemcpy(krylov->V, (void *)u, n * sizeof(double),
//              cudaMemcpyDeviceToDevice);
// 
//   // \beta = ||u||
//   cudaMemset(krylov->d_beta, 0, sizeof(double));
//   norm<<<grid, block>>>(krylov->V, krylov->d_beta, n);
// 
//   double beta;
//   cudaMemcpy(&beta, krylov->d_beta, sizeof(double), cudaMemcpyDeviceToHost);
//   beta = std::sqrt(beta);
//   cudaMemcpy(krylov->d_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);
//   // std::cout << "Device beta (in Lanczos): " << beta << "\n";
//   inv_scale<<<grid, block>>>(krylov->V, beta, n);
// 
//   // this small matrix will disappear after some optimization such that this
//   // entire function runs on device only
//   Eigen::MatrixX<double> T = Eigen::MatrixX<double>::Zero(m, m);
//   cudaMemset(krylov->T, 0, m * m * sizeof(double));
//   for (uint32_t j = 0; j < m - 1; j++) {
//     spmv->multiply(&krylov->V[j * n], krylov->buf1);
//     if (j > 0) {
//       axpby<<<grid, block>>>(krylov->buf1, &krylov->V[(j - 1) * n], 1.0,
//                              -T(j - 1, j), n);
//     }
// 
//     cudaMemset(krylov->T + m * j + j, 0, sizeof(double));
//     dot<<<grid, block>>>(krylov->buf1, &(krylov->V[j * n]),
//                          krylov->T + m * j + j, n);
//     cudaMemcpy(&T(j, j), krylov->T + m * j + j, sizeof(double),
//                cudaMemcpyDeviceToHost);
//     axpby<<<grid, block>>>(krylov->buf1, &krylov->V[j * n], 1.0, -T(j, j), n);
// 
//     for (uint32_t i = 0; i <= j; i++) {
//       cudaMemset(krylov->buf2, 0, sizeof(double));
//       dot<<<grid, block>>>(krylov->buf1, &(krylov->V[i * n]), krylov->buf2, n);
//       double coeff;
//       cudaMemcpy(&coeff, krylov->buf2, sizeof(double), cudaMemcpyDeviceToHost);
//       // MGS
//       axpby<<<grid, block>>>(krylov->buf1, &krylov->V[i * n], 1.0, -coeff, n);
//     }
// 
//     cudaMemset(krylov->T + m * j + j + 1, 0, sizeof(double));
//     norm<<<grid, block>>>(krylov->buf1, krylov->T + m * j + j + 1, n);
//     scalar_sqrt<<<1, 1>>>(krylov->T + m * j + j + 1);
//     cudaDeviceSynchronize();
//     cudaMemcpy(krylov->T + m * (j + 1) + j, krylov->T + m * j + j + 1,
//                sizeof(double), cudaMemcpyDeviceToDevice);
//     cudaMemcpy(&T(j + 1, j), krylov->T + m * j + j + 1, sizeof(double),
//                cudaMemcpyDeviceToHost);
//     cudaMemcpy(&T(j, j + 1), krylov->T + m * j + j + 1, sizeof(double),
//                cudaMemcpyDeviceToHost);
// 
//     // V.col(j + 1) = w / T(j + 1, j);
//     inv_scale<<<grid, block>>>(krylov->buf1, T(j + 1, j), n);
//     cudaMemcpy(&krylov->V[(j + 1) * n], krylov->buf1, n * sizeof(double),
//                cudaMemcpyDeviceToDevice);
//   }
//   // Eigen::MatrixX<double> V = Eigen::MatrixX<double>::Zero(n, m);
//   // cudaMemcpy(V.data(), krylov->V, n * m * sizeof(double),
//   // cudaMemcpyDeviceToHost); std::cout << "Device V.row(0) after Lanczos
//   // done:\n"; std::cout << V.row(0) << "\n";
// }

void lanczos_iteration(DeviceSpMV<double> *spmv, KrylovInfo *krylov,
                       const double *__restrict__ u) {
  const uint32_t n = krylov->n;
  const uint32_t m = krylov->m;
  cublasHandle_t handle; 
  cublasCreate(&handle);

  cudaMemcpy(krylov->buf1, u, n * sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpy(krylov->V, u, n * sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemset(krylov->T, 0, m * m * sizeof(double));

  double beta;
  cublasDnrm2(handle, n, krylov->V, 1, &beta);
  cudaMemcpy(krylov->d_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);

  double scale = 1.0 / beta;
  cublasDscal(handle, n, &scale, krylov->V, 1);

  double dot_result;
  double *h;
  cudaMalloc(&h, m * sizeof(double));
  double one = 1.0;
  double zero = 0.0;
  double neg_one = -1.0;

  cudaDeviceSynchronize();
  for (uint32_t j = 0; j < m - 1; j++) {
    spmv->multiply(&krylov->V[j * n], krylov->buf1);

    if (j > 0) {
      cublasDdot(handle, n, krylov->buf1, 1, &krylov->V[(j - 1) * n], 1, &dot_result);
      cudaMemcpy(&krylov->T[m * (j - 1) + j], &dot_result, sizeof(double), cudaMemcpyHostToDevice);
      scale = -dot_result;
      cublasDaxpy(handle, n, &scale, &krylov->V[(j - 1) * n], 1, krylov->buf1, 1);
    }

    cublasDdot(handle, n, krylov->buf1, 1, &krylov->V[j * n], 1, &dot_result);
    cudaMemcpy(&krylov->T[m * j + j], &dot_result, sizeof(double), cudaMemcpyHostToDevice);
    scale = -dot_result;
    cublasDaxpy(handle, n, &scale, &krylov->V[j * n], 1, krylov->buf1, 1);

    cublasDgemv(handle, CUBLAS_OP_T, n, j + 1, &one, krylov->V, n, krylov->buf1, 1, &zero, h, 1);
    cublasDgemv(handle, CUBLAS_OP_N, n, j + 1, &neg_one, krylov->V, n, h, 1, &one, krylov->buf1, 1);

    cublasDnrm2(handle, n, krylov->buf1, 1, &beta);

    cudaMemcpy(&krylov->T[m * j + j + 1], &beta, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&krylov->T[m * (j + 1) + j], &beta, sizeof(double), cudaMemcpyHostToDevice);

    if (beta > 0.0) {
      scale = 1.0 / beta;
      cublasDscal(handle, n, &scale, krylov->buf1, 1);
    }

    cudaMemcpy(&krylov->V[(j + 1) * n], krylov->buf1, n * sizeof(double), cudaMemcpyDeviceToDevice);
  }

  cudaFree(h);
  cublasDestroy(handle);
}

#endif
