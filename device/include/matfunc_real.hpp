#ifndef MATFUNC_REAL_HPP
#define MATFUNC_REAL_HPP

#include "lanczos.hpp"
#include "pragmas.hpp"
#include "spmv.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>


#include <iostream>

__global__ void set_diagonal(double* matrix,
                                 double* diag,
                                 uint32_t m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < m) {
        matrix[idx * m + idx] = diag[idx];
    }
}

// 
// // K1: f(t sqrt(λ_i))
// 
// // sinc²
// __global__ void transform_eigenvals_sinc2_sqrt(double *out,
//                                                const double *eigvals,
//                                                const double t,
//                                                const uint32_t m) {
//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   if (i < m) {
//     double x = t * sqrt(abs(eigvals[i]));
//     out[i] = (abs(x) < 1e-8) ? 1.0 : pow(sin(x) / x, 2);
//   }
// }
// 
// // cos
// __global__ void transform_eigenvals_cos_sqrt(double *out, const double *eigvals,
//                                              const double t, const uint32_t m) {
//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   if (i < m) {
//     double x = t * sqrt(abs(eigvals[i]));
//     out[i] = cos(x);
//   }
// }
// 
// // Id
// __global__ void transform_eigenvals_id_sqrt(double *out, const double *eigvals,
//                                             const double t, const uint32_t m) {
//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   if (i < m) {
//     double x = t * sqrt(abs(eigvals[i]));
//     out[i] = x;
//   }
// }
// 
// // V x K multiplication (n x m) * (m x m)
// __global__ void matrix_multiply_VK(const double *V, const double *K,
//                                    double *result, const uint32_t n,
//                                    const uint32_t m) {
//   const int row = blockIdx.y * blockDim.y + threadIdx.y;
//   const int col = blockIdx.x * blockDim.x + threadIdx.x;
// 
//   if (row < n && col < m) {
//     double sum = 0.0;
//     for (int k = 0; k < m; k++) {
//       sum += V[k * n + row] * K[col * m + k];
//     }
//     result[col * n + row] = sum;
//   }
// }
// 
// // Q * diag(f(D)) * Q^T
// __global__ void matrix_multiply_QDQ(const double *Q, const double *D,
//                                     double *result, const uint32_t m) {
//   const int row = blockIdx.y * blockDim.y + threadIdx.y;
//   const int col = blockIdx.x * blockDim.x + threadIdx.x;
// 
//   if (row < m && col < m) {
//     double sum = 0.0;
//     for (int k = 0; k < m; k++) {
//       sum += Q[k * m + row] * D[k] * Q[k * m + col];
//     }
//     result[col * m + row] = sum;
//   }
// }
// 
// // beta * X[0, :]
// __global__ void scale_first_col(const double *X, double *result,
//                                 const double beta, const uint32_t n) {
//   const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < n) {
//     result[idx] = beta * X[idx];
//   }
// }

class MatrixFunctionApplicatorReal {
public:
  enum class FunctionType { COS_SQRT, SINC2_SQRT, ID_SQRT };

  void print_type(FunctionType t) const {
    switch (t) {
    case FunctionType::COS_SQRT:
      std::cout << "type is cos sqrt\n";
      break;
    case FunctionType::SINC2_SQRT:
      std::cout << "type is sinc2 sqrt\n";
      break;
    case FunctionType::ID_SQRT:
      std::cout << "type is Id sqrt\n";
      break;
    default:
      throw std::runtime_error("Invalid matrix func call");
    }
  }

  struct Parameters {
    uint32_t block_size_1d;
    uint32_t block_size_2d;
    double sinc_threshold;
    Parameters() : block_size_1d(64), block_size_2d(8), sinc_threshold(1e-8) {}
  };

  MatrixFunctionApplicatorReal(const int *d_row_ptr, const int *d_col_ind,
                               const double *d_values, uint32_t n, uint32_t m,
                               uint32_t nnz,
                               const Parameters &params = Parameters())
      : n_(n), m_(m), params_(params) {

    spmv_ = new DeviceSpMV<double>(d_row_ptr, d_col_ind, d_values, n, nnz);

    cudaMalloc((void **)&krylov_.V, n * m * sizeof(double));
    cudaMalloc((void **)&krylov_.T, m * m * sizeof(double));
    cudaMalloc((void **)&krylov_.buf1, n * sizeof(double));
    cudaMalloc((void **)&krylov_.buf2, n * sizeof(double));
    cudaMalloc((void **)&krylov_.d_beta, sizeof(double));
    krylov_.n = n;
    krylov_.m = m;

    cudaMalloc(&d_diag_, m * sizeof(double));
    cudaMalloc(&d_small_result_, m * m * sizeof(double));

    cudaMalloc((void **)&d_eigenvalues_, m * sizeof(double));
    cudaMalloc((void **)&d_eigenvectors_, m * m * sizeof(double));
    cudaMalloc((void **)&d_transform_, m * sizeof(double));
    cudaMalloc((void **)&d_work_, m * m * sizeof(double));
    cudaMalloc((void **)&d_work_large_, n * m * sizeof(double));
    cudaMalloc((void **)&d_info_, sizeof(int));

    cusolverDnCreate(&solver_handle_);
    int lwork;
    cusolverDnDsyevd_bufferSize(solver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                                CUBLAS_FILL_MODE_LOWER, m, d_eigenvectors_, m,
                                d_eigenvalues_, &lwork);
    cudaMalloc((void **)&d_solver_work_, lwork * sizeof(double));
    solver_work_size_ = lwork;
    cublasCreate(&cublas_handle_);

    // block_dim_2d_ = dim3(params_.block_size_2d, params_.block_size_2d);
    // grid_dim_2d_ = dim3((m + block_dim_2d_.x - 1) / block_dim_2d_.x,
    //                     (m + block_dim_2d_.y - 1) / block_dim_2d_.y);
    // block_dim_1d_ = dim3(params_.block_size_1d);
    // grid_dim_1d_ = dim3(block_dim_1d_.x);
  }

  ~MatrixFunctionApplicatorReal() {
    delete spmv_;
    cudaFree(krylov_.V);
    cudaFree(krylov_.T);
    cudaFree(krylov_.buf1);
    cudaFree(krylov_.buf2);
    cudaFree(krylov_.d_beta);
    cudaFree(d_eigenvalues_);
    cudaFree(d_eigenvectors_);
    cudaFree(d_diag_);
    cudaFree(d_small_result_);
    cudaFree(d_transform_);
    cudaFree(d_work_);
    cudaFree(d_solver_work_);
    cudaFree(d_info_);
    cudaFree(d_work_large_);
    cusolverDnDestroy(solver_handle_);
    cublasDestroy(cublas_handle_);
  }


  void apply(double *result, const double *input, double t, FunctionType type) {
    lanczos_iteration(spmv_, &krylov_, input);
    cudaDeviceSynchronize();
  
    double beta;
    cudaMemcpy(&beta, krylov_.d_beta, sizeof(double), cudaMemcpyDeviceToHost);
    compute_eigen_decomposition();
  
    thrust::device_ptr<double> d_diag_ptr(d_diag_);
    thrust::device_ptr<double> d_eigvals_ptr(d_eigenvalues_);
  
    switch (type) {
    // THIS HERE IS IMPORTANT TO COMMUNICATE TO THE USER:
    // We rely on the user's sanity to only use this when
    // the entire spectrum of the operator L is negative, ie.
    //   u_tt + Au = g(u)
    //   make use of finite-diff of div(c(x) \nabla)
    //   then: The equation u_tt = div(c(x) \nabla u) - m(x) f(u)
    //   becomes: u_tt - div(c(x) \nabla u) = m(x) f(u)
    //   and thus, if \sigma(\bar{div(c(x) \nabla)}) \in R_{-}
    //   Then this operator's eigenvalues will be positive and we can
    //   actually take the sqrt! Else, not so much!
    case FunctionType::SINC2_SQRT:
      thrust::transform(thrust::make_counting_iterator<uint32_t>(0),
                        thrust::make_counting_iterator<uint32_t>(m_),
                        d_diag_ptr,
                        [=] __device__ (const uint32_t idx) {
                          double x = t * sqrt(abs(d_eigvals_ptr[idx]));
                          return (abs(x) < 1e-8) ? 1.0 : pow(sin(x) / x, 2);
                        });
      break;
    case FunctionType::COS_SQRT:
      thrust::transform(thrust::make_counting_iterator<uint32_t>(0),
                        thrust::make_counting_iterator<uint32_t>(m_),
                        d_diag_ptr,
                        [=] __device__ (const uint32_t idx) {
                          double x = t * sqrt(abs(d_eigvals_ptr[idx]));
                          return cos(x);
                        });
      break;
    case FunctionType::ID_SQRT:
      thrust::transform(thrust::make_counting_iterator<uint32_t>(0),
                        thrust::make_counting_iterator<uint32_t>(m_),
                        d_diag_ptr,
                        [=] __device__ (const uint32_t idx) {
                          double x = t * sqrt(abs(d_eigvals_ptr[idx]));
                          return x;
                        });
      break;
    default:
      throw std::runtime_error("Invalid Matfunc call");
    }
  
    dim3 blockDim(256);
    dim3 gridDim((m_ + blockDim.x - 1) / blockDim.x);
  
    double *temp_work = nullptr;
    cudaMalloc(&temp_work, m_ * m_ * sizeof(double));
  
    const double alpha = 1.0;
    const double beta_cublas = 0.0;
  
    cudaMemset(d_work_, 0, m_ * m_ * sizeof(double));
  
    dim3 block_1d(256);
    dim3 grid_1d((m_ + block_1d.x - 1) / block_1d.x);
    set_diagonal<<<grid_1d, block_1d>>>(d_work_, d_diag_, m_);
  
    cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                m_, m_, m_,
                &alpha,
                d_eigenvectors_, m_,
                d_work_, m_,
                &beta_cublas,
                temp_work, m_);
  
    cublasDgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                m_, m_, m_,
                &alpha,
                temp_work, m_,
                d_eigenvectors_, m_,
                &beta_cublas,
                d_work_, m_);
  
    double *e1 = nullptr;
    cudaMalloc(&e1, m_ * sizeof(double));
    cudaMemset(e1, 0, m_ * sizeof(double));
  
    double one_val = 1.0;
    cudaMemcpy(e1, &one_val, sizeof(double), cudaMemcpyHostToDevice);
  
    double *temp_vec = nullptr;
    cudaMalloc(&temp_vec, m_ * sizeof(double));
  
    cudaMemset(temp_vec, 0, m_ * sizeof(double));
  
    cublasDgemv(cublas_handle_, CUBLAS_OP_N,
                m_, m_,
                &alpha,
                d_work_, m_,
                e1, 1,
                &beta_cublas,
                temp_vec, 1);
  
    cublasDgemv(cublas_handle_, CUBLAS_OP_N,
                n_, m_,
                &alpha,
                krylov_.V, n_,
                temp_vec, 1,
                &beta_cublas,
                d_work_large_, 1);
  
    double scale_factor = beta;
  
    cublasDscal(cublas_handle_, n_,
                &scale_factor,
                d_work_large_, 1);
  
    cudaMemcpy(result, d_work_large_, n_ * sizeof(double), cudaMemcpyDeviceToDevice);
  
    cudaFree(temp_work);
    cudaFree(e1);
    cudaFree(temp_vec);
  }



  // void apply(double *result, const double *input, double t, FunctionType type) {
  //   std::vector<double> input_vec(n_);
  //   cudaMemcpy(input_vec.data(), input, n_ * sizeof(double),
  //              cudaMemcpyDeviceToHost);

  //   lanczos_iteration(spmv_, &krylov_, input);
  //   cudaDeviceSynchronize();

  //   double beta;
  //   cudaMemcpy(&beta, krylov_.d_beta, sizeof(double), cudaMemcpyDeviceToHost);
  //   compute_eigen_decomposition();

  //   // std::vector<double> eigenvals(m_);
  //   // cudaMemcpy(eigenvals.data(), d_eigenvalues_, m_ * sizeof(double),
  //   // cudaMemcpyDeviceToHost); std::cout << "Device eigenvalues: ";
  //   // for(uint32_t i = 0; i < m_; ++i)
  //   //     std::cout << eigenvals[i] << " ";
  //   // std::cout << "\n";

  //   cudaMemset(d_diag_, 0.0, m_ * sizeof(double));
  //   block_dim_1d_ = dim3(256);
  //   // TODO this might need better parametrization depending on sqrt(n) or even
  //   // n^(1/3) (if 3d case)
  //   dim3 block_2d(16, 16);
  //   dim3 grid_VK((m_ + block_2d.x - 1) / block_2d.x,
  //                (n_ + block_2d.y - 1) / block_2d.y);
  //   dim3 grid_QDQ((m_ + block_2d.x - 1) / block_2d.x,
  //                 (m_ + block_2d.y - 1) / block_2d.y);
  //   dim3 block_1d(256);
  //   dim3 grid_1d((n_ + block_1d.x - 1) / block_1d.x);

  //   // TODO check if the values _actually_ check out -- ie. if time stepping
  //   // agrees let's just keep it at first and optimize kernels and kernel tuning
  //   // later

  //   // print_type(type);
  //   switch (type) {
  //   case FunctionType::SINC2_SQRT:
  //     transform_eigenvals_sinc2_sqrt<<<grid_1d, block_1d>>>(
  //         d_diag_, d_eigenvalues_, t, m_);
  //     break;
  //   case FunctionType::COS_SQRT:
  //     transform_eigenvals_cos_sqrt<<<grid_1d, block_1d>>>(
  //         d_diag_, d_eigenvalues_, t, m_);
  //     break;
  //   case FunctionType::ID_SQRT:
  //     transform_eigenvals_id_sqrt<<<grid_1d, block_1d>>>(d_diag_,
  //                                                        d_eigenvalues_, t, m_);
  //     break;

  //   default:
  //     throw std::runtime_error("Invalid Matfunc call");
  //   }

  //   // project matrix back from diagonalized space
  //   matrix_multiply_QDQ<<<grid_QDQ, block_2d>>>(d_eigenvectors_, d_diag_,
  //                                               d_work_, m_);
  //   // apply V to Q f(t sqrt(L)) Q.T
  //   matrix_multiply_VK<<<grid_VK, block_2d>>>(krylov_.V, d_work_, d_work_large_,
  //                                             n_, m_);
  //   // scale with beta
  //   scale_first_col<<<grid_1d, block_1d>>>(d_work_large_, result, beta, n_);
  // }

  // unsafe, workaround for now
  DeviceSpMV<double> *expose_spmv() const { return spmv_; }

private:
  void compute_eigen_decomposition() {
    cudaMemcpy(d_eigenvectors_, krylov_.T, m_ * m_ * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cusolverDnDsyevd(solver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                     CUBLAS_FILL_MODE_LOWER, m_, d_eigenvectors_, m_,
                     d_eigenvalues_, d_solver_work_, solver_work_size_,
                     d_info_);
  }

  double *d_diag_;
  double *d_small_result_;
  cublasHandle_t cublas_handle_;

  DeviceSpMV<double> *spmv_;
  KrylovInfo krylov_;
  cusolverDnHandle_t solver_handle_;
  double *d_eigenvalues_;
  double *d_eigenvectors_;
  double *d_transform_;
  double *d_work_;
  double *d_work_large_;
  double *d_solver_work_;
  int *d_info_;
  int solver_work_size_;
  uint32_t n_;
  uint32_t m_;
  // dim3 grid_dim_2d_;
  // dim3 block_dim_2d_;
  // dim3 grid_dim_1d_;
  // dim3 block_dim_1d_;

  // dim3 block_dim_large_;
  // dim3 grid_dim_large_;
  Parameters params_;
};

#endif
