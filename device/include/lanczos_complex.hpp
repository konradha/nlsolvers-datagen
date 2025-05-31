#ifndef LANCZOS_COMPLEX_HPP
#define LANCZOS_COMPLEX_HPP

/*
 *
 * nvcc -O3 -use_fast_math --ptxas-options=-v --fmad=true --restrict -std=c++17
 * --expt-relaxed-constexpr -diag-suppress 20012 -diag-suppress 20011 -lcusparse
 * -lcublas -lcusolver -x cu --extended-lambda -arch=sm_70
 * test_lanczos_complex.cpp -o to_lanczos_complex_test
 */

#define COUNTER 0

#include "pragmas.hpp"
#include "spmv.hpp"
#include <thrust/complex.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>

#define NUM_BLOCKS 256
#define BLOCK_SIZE 1024
#define MAX_BLOCKS_PER_DIM 65535

struct KrylovInfoComplex {
  thrust::complex<double> *T;
  thrust::complex<double> *V;
  thrust::complex<double> *buf1;
  thrust::complex<double> *buf2;
  double *d_beta;
  double *reconstruct_beta;
  uint32_t n;
  uint32_t m;
  cublasHandle_t handle;
};

__global__ void norm_complex(const thrust::complex<double> *__restrict__ vec,
                             double *__restrict__ result, const uint32_t n) {
  __shared__ double shared_mem[BLOCK_SIZE];
  const uint32_t tid = threadIdx.x;
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  double sum = 0.0;
  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    thrust::complex<double> val = vec[i];
    sum += val.real() * val.real() + val.imag() * val.imag();
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

  __syncthreads();
  if (tid == 0 && blockIdx.x == 0) {
    __threadfence();
    *result = sqrt(*result);
  }
}

__global__ void inv_scale_complex(thrust::complex<double> *__restrict__ vec,
                                  const double *scalar, const uint32_t n) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  double local_scalar = *scalar;
  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    vec[i] = vec[i] / local_scalar;
  }
}

__global__ void dot_complex(const thrust::complex<double> *__restrict__ v1,
                            const thrust::complex<double> *__restrict__ v2,
                            thrust::complex<double> *__restrict__ result,
                            const uint32_t n) {
  __shared__ thrust::complex<double> shared_mem[BLOCK_SIZE];
  const uint32_t tid = threadIdx.x;
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  thrust::complex<double> sum(0.0, 0.0);
  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    sum += thrust::conj(v2[i]) * v1[i];
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
    double *result_ptr = reinterpret_cast<double *>(result);
    atomicAdd(&result_ptr[0], shared_mem[0].real());
    atomicAdd(&result_ptr[1], shared_mem[0].imag());
  }
}

__global__ void axpby_complex(thrust::complex<double> *__restrict__ v1,
                              const thrust::complex<double> *__restrict__ v2,
                              thrust::complex<double> *alpha,
                              thrust::complex<double> *beta, const uint32_t n,
                              bool neg_beta = true) {
  thrust::complex<double> local_alpha = alpha[0];
  thrust::complex<double> local_beta = neg_beta ? -beta[0] : beta[0];

  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    v1[i] = local_alpha * v1[i] + local_beta * v2[i];
  }
}

__global__ void copyDoubleToComplexReal(thrust::complex<double> *dest,
                                        const double *src) {
  *dest = thrust::complex<double>(*src, 0.0);
}

/*
void lanczos_iteration_complex(
    DeviceSpMV<thrust::complex<double>> *spmv, KrylovInfoComplex *krylov,
    const thrust::complex<double> *__restrict__ u) {
  const uint32_t n = krylov->n;
  const uint32_t m = krylov->m;

  const uint32_t total_threads_needed = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const uint32_t blocks_x = std::min(total_threads_needed,
                                   static_cast<uint32_t>(MAX_BLOCKS_PER_DIM));
  const uint32_t blocks_y = (total_threads_needed + blocks_x - 1) / blocks_x;

  dim3 grid(blocks_x, blocks_y);
  dim3 block(BLOCK_SIZE);

  //printf("Total threads %d\n", total_threads_needed);
  //printf("Total blocks_x %d\n", blocks_x);
  //printf("Total blocks_y %d\n", blocks_y);

  cudaMemcpy(krylov->V, u, n * sizeof(thrust::complex<double>),
             cudaMemcpyDeviceToDevice);

  cudaMemset(krylov->d_beta, 0, sizeof(double));
  norm_complex<<<grid, block>>>(krylov->V, krylov->d_beta, n);
  cudaMemcpy(krylov->reconstruct_beta, krylov->d_beta,
               sizeof(double), cudaMemcpyDeviceToDevice);
  inv_scale_complex<<<grid, block>>>(krylov->V, krylov->d_beta, n);

  cudaMemset(krylov->T, 0, m * m * sizeof(thrust::complex<double>));

  thrust::complex<double> one(1., 0);
  thrust::complex<double>* one_dev;
  cudaMalloc(&one_dev, sizeof(thrust::complex<double>));
  cudaMemcpy(one_dev, &one, sizeof(thrust::complex<double>),
cudaMemcpyHostToDevice);

  for (uint32_t j = 0; j < m - 1; j++) {
    spmv->multiply(&krylov->V[j * n], krylov->buf1);


    if (j > 0) {
      axpby_complex<<<grid, block>>>(
          krylov->buf1, &krylov->V[(j - 1) * n],
          one_dev,
          &krylov->T[m * (j - 1) + j],
          n, true);
    }


    cudaMemset(&krylov->T[m * j + j], 0, sizeof(thrust::complex<double>));
    dot_complex<<<grid, block>>>(krylov->buf1, &krylov->V[j * n],
                                 &krylov->T[m * j + j], n);

    axpby_complex<<<grid, block>>>(
        krylov->buf1, &krylov->V[j * n], one_dev,
        &krylov->T[m * j + j], n, true);

    for (uint32_t i = 0; i <= j; i++) {
      cudaMemset(krylov->buf2, 0, sizeof(thrust::complex<double>));
      dot_complex<<<grid, block>>>(krylov->buf1, &krylov->V[i * n],
                                   krylov->buf2, n);
      axpby_complex<<<grid, block>>>(
          krylov->buf1, &krylov->V[i * n], one_dev,
          krylov->buf2, n, true);
    }

    cudaMemset(krylov->d_beta, 0, sizeof(double));
    norm_complex<<<grid, block>>>(krylov->buf1, krylov->d_beta, n);

    copyDoubleToComplexReal<<<1, 1>>>(&(krylov->T[m * j + j + 1]),
krylov->d_beta); copyDoubleToComplexReal<<<1, 1>>>(&(krylov->T[m * (j+1) + j]),
krylov->d_beta); inv_scale_complex<<<grid, block>>>(krylov->buf1,
krylov->d_beta, n); cudaMemcpy(&krylov->V[(j + 1) * n], krylov->buf1, n *
sizeof(thrust::complex<double>), cudaMemcpyDeviceToDevice);
  }
  cudaFree(one_dev);
}*/

#include <cublas_v2.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>

__global__ void
fused_lanczos_step(const cuDoubleComplex *__restrict__ spmv_result,
                   cuDoubleComplex *__restrict__ V,
                   cuDoubleComplex *__restrict__ T, const uint32_t j,
                   const uint32_t n, const uint32_t m) {
  __shared__ cuDoubleComplex dot_results[32];
  __shared__ double norm_partial[32];
  const uint32_t tid = threadIdx.x;
  const uint32_t bid = blockIdx.x;
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t grid_size = blockDim.x * gridDim.x;

  cuDoubleComplex w_local = make_cuDoubleComplex(0.0, 0.0);
  if (gid < n) {
    w_local = spmv_result[gid];
    if (j > 0) {
      cuDoubleComplex v_prev = V[gid + (j - 1) * n];
      cuDoubleComplex t_prev = T[j - 1 + j * m];
      w_local.x -= (t_prev.x * v_prev.x - t_prev.y * v_prev.y);
      w_local.y -= (t_prev.x * v_prev.y + t_prev.y * v_prev.x);
    }
  }

  for (uint32_t k = 0; k <= j; k++) {
    cuDoubleComplex dot = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex v_curr = make_cuDoubleComplex(0.0, 0.0);
    if (gid < n) {
      v_curr = V[gid + k * n];
      dot.x += w_local.x * v_curr.x + w_local.y * v_curr.y;
      dot.y += w_local.x * v_curr.y - w_local.y * v_curr.x;
    }

    for (uint32_t offset = 16; offset > 0; offset /= 2) {
      dot.x += __shfl_down_sync(0xffffffff, dot.x, offset);
      dot.y += __shfl_down_sync(0xffffffff, dot.y, offset);
    }

    if (tid % 32 == 0) {
      dot_results[tid / 32] = dot;
    }
    __syncthreads();

    if (tid < 32) {
      dot = tid < (blockDim.x / 32) ? dot_results[tid]
                                    : make_cuDoubleComplex(0.0, 0.0);
      for (uint32_t offset = 16; offset > 0; offset /= 2) {
        dot.x += __shfl_down_sync(0xffffffff, dot.x, offset);
        dot.y += __shfl_down_sync(0xffffffff, dot.y, offset);
      }
      if (tid == 0) {
        dot_results[0] = dot;
      }
    }
    __syncthreads();

    cuDoubleComplex coeff = dot_results[0];
    if (k == j) {
      if (gid < n) {
        T[j + j * m] = coeff;
      }
    }

    if (gid < n) {
      w_local.x -= (coeff.x * v_curr.x - coeff.y * v_curr.y);
      w_local.y -= (coeff.x * v_curr.y + coeff.y * v_curr.x);
    }
    __syncthreads();
  }

  double norm_local = 0.0;
  if (gid < n) {
    norm_local = w_local.x * w_local.x + w_local.y * w_local.y;
  }

  for (uint32_t offset = 16; offset > 0; offset /= 2) {
    norm_local += __shfl_down_sync(0xffffffff, norm_local, offset);
  }

  if (tid % 32 == 0) {
    norm_partial[tid / 32] = norm_local;
  }
  __syncthreads();

  if (tid < 32) {
    norm_local = tid < (blockDim.x / 32) ? norm_partial[tid] : 0.0;
    for (uint32_t offset = 16; offset > 0; offset /= 2) {
      norm_local += __shfl_down_sync(0xffffffff, norm_local, offset);
    }
    if (tid == 0) {
      norm_partial[0] = norm_local;
    }
  }
  __syncthreads();

  double beta = sqrt(norm_partial[0]);
  if (gid < n && beta > 0.0) {
    T[j + 1 + j * m] = make_cuDoubleComplex(beta, 0.0);
    T[j + (j + 1) * m] = make_cuDoubleComplex(beta, 0.0);
    double inv_beta = 1.0 / beta;
    V[gid + (j + 1) * n] =
        make_cuDoubleComplex(w_local.x * inv_beta, w_local.y * inv_beta);
  }
}

void lanczos_iteration_complex(DeviceSpMV<thrust::complex<double>> *spmv,
                               KrylovInfoComplex *krylov,
                               const thrust::complex<double> *__restrict__ u) {
  // printf("Device: Lanczos iteration complex\n");
#if COUNTER
  cudaEvent_t start_data_mov, stop_data_mov;
  cudaEventCreate(&start_data_mov);
  cudaEventCreate(&stop_data_mov);

  cudaEvent_t start_spmv, stop_spmv;
  cudaEventCreate(&start_spmv);
  cudaEventCreate(&stop_spmv);

  cudaEvent_t start_beg_loop, stop_beg_loop;
  cudaEventCreate(&start_beg_loop);
  cudaEventCreate(&stop_beg_loop);

  cudaEvent_t start_inner_loop, stop_inner_loop;
  cudaEventCreate(&start_inner_loop);
  cudaEventCreate(&stop_inner_loop);

  cudaEvent_t start_rest_loop, stop_rest_loop;
  cudaEventCreate(&start_rest_loop);
  cudaEventCreate(&stop_rest_loop);
  float t = 0.f, t_data_mov = 0.f, t_spmv = 0.f, t_beg = 0.f, t_inner = 0.f,
        t_rest = 0.f;
#endif

  const uint32_t n = krylov->n;
  const uint32_t m = krylov->m;

  auto handle = krylov->handle;

#if COUNTER
  cudaEventRecord(start_data_mov);
#endif
  cudaMemcpy(krylov->buf1, u, n * sizeof(thrust::complex<double>),
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(krylov->V, u, n * sizeof(thrust::complex<double>),
             cudaMemcpyDeviceToDevice);
  cudaMemset(krylov->T, 0, m * m * sizeof(thrust::complex<double>));
#if COUNTER
  cudaEventRecord(stop_data_mov);
  cudaEventElapsedTime(&t, start_data_mov, stop_data_mov);
  t_data_mov += t;
  t = 0;
#endif
  double beta;
  cublasNrm2Ex(handle, n, krylov->V, CUDA_C_64F, 1, &beta, CUDA_R_64F,
               CUDA_R_64F);
#if COUNTER
  cudaEventRecord(start_data_mov);
#endif
  cudaMemcpy(krylov->d_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(krylov->reconstruct_beta, &beta, sizeof(double),
             cudaMemcpyHostToDevice);
#if COUNTER
  cudaEventRecord(stop_data_mov);
  cudaEventElapsedTime(&t, start_data_mov, stop_data_mov);
  t_data_mov += t;
  t = 0;
#endif
  cuDoubleComplex scale = make_cuDoubleComplex(1.0 / beta, 0.0);
  cublasZscal(handle, n, &scale, reinterpret_cast<cuDoubleComplex *>(krylov->V),
              1);

  cuDoubleComplex dot_result;
  cuDoubleComplex *h;
  cudaMalloc(&h, m * sizeof(cuDoubleComplex));
  cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
  cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
  cuDoubleComplex neg_one = make_cuDoubleComplex(-1.0, 0.0);

  cudaDeviceSynchronize();
  for (uint32_t j = 0; j < m - 1; j++) {
#if COUNTER
    cudaEventRecord(start_spmv);
#endif
    spmv->multiply(&krylov->V[j * n], krylov->buf1);

    // fused_lanczos_step<<<grid, block,
    // SHARED_MEM_SIZE>>>(reinterpret_cast<cuDoubleComplex*>(krylov->buf1),
    //		reinterpret_cast<cuDoubleComplex*>(krylov->V),
    //		reinterpret_cast<cuDoubleComplex*>(krylov->T),
    //		j, n, m);

#if COUNTER
    cudaEventRecord(stop_spmv);
    cudaEventSynchronize(stop_spmv);
    cudaEventElapsedTime(&t, start_spmv, stop_spmv);
    t_spmv += t;
    t = 0;
    cudaEventRecord(start_beg_loop);
#endif
    if (j > 0) {
      cublasZdotc(
          handle, n, reinterpret_cast<const cuDoubleComplex *>(krylov->buf1), 1,
          reinterpret_cast<const cuDoubleComplex *>(&krylov->V[(j - 1) * n]), 1,
          &dot_result);
#if COUNTER
      cudaEventRecord(start_data_mov);
#endif
      cudaMemcpy(&krylov->T[m * (j - 1) + j], &dot_result,
                 sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

#if COUNTER
      cudaEventRecord(stop_data_mov);
      cudaEventElapsedTime(&t, start_data_mov, stop_data_mov);
      t_data_mov += t;
      t = 0;
#endif
      scale = make_cuDoubleComplex(-dot_result.x, -dot_result.y);
      cublasZaxpy(
          handle, n, &scale,
          reinterpret_cast<const cuDoubleComplex *>(&krylov->V[(j - 1) * n]), 1,
          reinterpret_cast<cuDoubleComplex *>(krylov->buf1), 1);
    }

    cublasZdotc(handle, n,
                reinterpret_cast<const cuDoubleComplex *>(krylov->buf1), 1,
                reinterpret_cast<const cuDoubleComplex *>(&krylov->V[j * n]), 1,
                &dot_result);
#if COUNTER
    cudaEventRecord(start_data_mov);
#endif
    cudaMemcpy(&krylov->T[m * j + j], &dot_result, sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);
#if COUNTER
    cudaEventRecord(stop_data_mov);
    cudaEventElapsedTime(&t, start_data_mov, stop_data_mov);
    t_data_mov += t;
    t = 0;
#endif
    scale = make_cuDoubleComplex(-dot_result.x, -dot_result.y);
    cublasZaxpy(handle, n, &scale,
                reinterpret_cast<const cuDoubleComplex *>(&krylov->V[j * n]), 1,
                reinterpret_cast<cuDoubleComplex *>(krylov->buf1), 1);
#if COUNTER
    cudaEventRecord(stop_beg_loop);
    cudaEventSynchronize(stop_beg_loop);
    cudaEventElapsedTime(&t, start_beg_loop, stop_beg_loop);
    t_beg += t;
    t = 0;

    cudaEventRecord(start_inner_loop);
#endif
    cublasZgemv(handle, CUBLAS_OP_C, n, j + 1, // V: n rows, j+1 cols
                &one, reinterpret_cast<const cuDoubleComplex *>(krylov->V), n,
                reinterpret_cast<const cuDoubleComplex *>(krylov->buf1), 1,
                &zero, h, 1);
    cublasZgemv(handle, CUBLAS_OP_N, n, j + 1, &neg_one,
                reinterpret_cast<const cuDoubleComplex *>(krylov->V), n, h, 1,
                &one, reinterpret_cast<cuDoubleComplex *>(krylov->buf1), 1);
#if COUNTER
    cudaEventRecord(stop_inner_loop);
    cudaEventSynchronize(stop_inner_loop);
    cudaEventElapsedTime(&t, start_inner_loop, stop_inner_loop);
    t_inner += t;
    t = 0;
    cudaEventRecord(start_rest_loop);
#endif
    cublasNrm2Ex(handle, n, krylov->buf1, CUDA_C_64F, 1, &beta, CUDA_R_64F,
                 CUDA_R_64F);

    cuDoubleComplex beta_complex = make_cuDoubleComplex(beta, 0.0);
#if COUNTER
    cudaEventRecord(start_data_mov);
#endif
    cudaMemcpy(&krylov->T[m * j + j + 1], &beta_complex,
               sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(&krylov->T[m * (j + 1) + j], &beta_complex,
               sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
#if COUNTER
    cudaEventRecord(stop_data_mov);
    cudaEventElapsedTime(&t, start_data_mov, stop_data_mov);
    t_data_mov += t;
    t = 0;
#endif
    if (beta > 0.0) {
      scale = make_cuDoubleComplex(1.0 / beta, 0.0);
      cublasZscal(handle, n, &scale,
                  reinterpret_cast<cuDoubleComplex *>(krylov->buf1), 1);
    }
#if COUNTER
    cudaEventRecord(start_data_mov);
#endif
    cudaMemcpy(&krylov->V[(j + 1) * n], krylov->buf1,
               n * sizeof(thrust::complex<double>), cudaMemcpyDeviceToDevice);
#if COUNTER
    cudaEventRecord(stop_data_mov);
    cudaEventElapsedTime(&t, start_data_mov, stop_data_mov);
    t_data_mov += t;
    t = 0;
    cudaEventRecord(stop_rest_loop);
    cudaEventSynchronize(stop_rest_loop);
    cudaEventElapsedTime(&t, start_rest_loop, stop_rest_loop);
    t_rest += t;
    t = 0;
#endif
  }
  cudaFree(h);
#if COUNTER
  float s = t_spmv + t_data_mov + t_inner + t_rest + t_beg;
  printf("%-12s | %-8s | %-10s | %-10s | %-10s\n", "data movement", "spmv",
         "begin loop", "inner loop", "rest loop");
  printf("%12.6f | %8.6f | %10.6f | %10.6f | %10.6f\n", t_data_mov / s,
         t_spmv / s, t_beg / s, t_inner / s, t_rest / s);
#endif

  // printf("Device: End of Lanczos iteration\n");
  // thrust::complex<double> * T_host;
  // T_host = (thrust::complex<double> *)malloc(sizeof(thrust::complex<double>)
  // * m * m); cudaMemcpy(T_host, krylov->T, m * m *
  // sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost); std::cout <<
  // "T:\n"; for(uint32_t i=0;i<m;++i)
  //{
  //   for(uint32_t j=0;j<m;++j)
  //          std::cout << T_host[m * i + j] << " ";
  //   std::cout << "\n";
  // }
  // free(T_host);

  // thrust::complex<double> * V_host;
  // V_host = (thrust::complex<double> *)malloc(sizeof(thrust::complex<double>)
  // * n * m); cudaMemcpy(V_host, krylov->V, n * m *
  // sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost); std::cout <<
  // "V:\n"; for(uint32_t i=0;i<n;++i)
  //{
  //   for(uint32_t j=0;j<m;++j)
  //          std::cout << V_host[m * i + j] << " ";
  //   std::cout << "\n";
  // }
  // free(V_host);
}

#endif
