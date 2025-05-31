#ifndef MATFUNC_COMPLEX_HPP
#define MATFUNC_COMPLEX_HPP

#include "lanczos_complex.hpp"
#include "pragmas.hpp"
#include "spmv.hpp"
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <thrust/complex.h>

#include <Eigen/Sparse>

#define DEBUG 0

// __global__ void transform_eigenvals_exp(thrust::complex<double> *out,
//                                         const double *eigvals,
//                                         thrust::complex<double> dt,
//                                         const uint32_t m) {
//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   if (i < m) {
//     out[i] = thrust::exp(dt * eigvals[i]);
//   }
// }
// 
// __global__ void transform_eigenvals_sinc(thrust::complex<double> *out,
//                                          const double *eigvals,
//                                          thrust::complex<double> dt,
//                                          const uint32_t m) {
//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   if (i < m) {
//     const auto val = dt * eigvals[i];
//     out[i] = thrust::abs(val) < 1e-8 ? thrust::complex<double>(1.0, 0.0)
//                                      : thrust::sin(val) / val;
//   }
// }

__global__ void set_diagonal(thrust::complex<double>* matrix,
                                 const thrust::complex<double>* diag,
                                 uint32_t m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < m) {
        matrix[idx * m + idx] = diag[idx];
    }
}

// __global__ void matrix_multiply_QDQ(const thrust::complex<double> *Q,
//                                    const thrust::complex<double> *D,
//                                    thrust::complex<double> *result,
//                                    const uint32_t m,
//                                    const uint32_t max_grid_x,
//                                    const uint32_t max_grid_y) {
//    const uint32_t block_size_x = blockDim.x;
//    const uint32_t block_size_y = blockDim.y;
// 
//    for (uint32_t grid_y = blockIdx.y; grid_y < (m + block_size_y - 1) / block_size_y; grid_y += max_grid_y) {
//        for (uint32_t grid_x = blockIdx.x; grid_x < (m + block_size_x - 1) / block_size_x; grid_x += max_grid_x) {
//            const uint32_t row = grid_y * block_size_y + threadIdx.y;
//            const uint32_t col = grid_x * block_size_x + threadIdx.x;
// 
//            if (row < m && col < m) {
//                thrust::complex<double> sum = 0.0;
// #pragma unroll
//                for (uint32_t k = 0; k < m; k++) {
//                    sum += Q[k * m + row] * D[k] * Q[k * m + col];
//                }
//                result[col * m + row] = sum;
//            }
//        }
//    }
// }
// 
// 
// __global__ void matrix_multiply_VK(const thrust::complex<double> *V,
//                                    const thrust::complex<double> *K,
//                                    thrust::complex<double> *result,
//                                    const uint32_t n, const uint32_t m,
//                                    const uint32_t max_grid_x,
//                                    const uint32_t max_grid_y) {
//     const uint32_t block_size_x = blockDim.x;
//     const uint32_t block_size_y = blockDim.y;
//     
//     for (uint32_t grid_y = blockIdx.y; grid_y < (n + block_size_y - 1) / block_size_y; grid_y += max_grid_y) {
//         for (uint32_t grid_x = blockIdx.x; grid_x < (m + block_size_x - 1) / block_size_x; grid_x += max_grid_x) {
//             const uint32_t row = grid_y * block_size_y + threadIdx.y;
//             const uint32_t col = grid_x * block_size_x + threadIdx.x;
//             
//             if (row < n && col < m) {
//                 thrust::complex<double> sum = 0.0;
// #pragma unroll
//                 for (uint32_t k = 0; k < m; k++) {
//                     sum += V[k * n + row] * K[col * m + k];
//                 }
//                 result[col * n + row] = sum;
//             }
//         }
//     }
// }
// 
// __global__ void scale_first_col(const thrust::complex<double> *X,
//                                 thrust::complex<double> *result,
//                                 const double beta, const uint32_t n) {
//   const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < n) {
//     result[idx] = beta * X[idx];
//   }
// }

class MatrixFunctionApplicatorComplex {
public:
  void reportAlloc(const char* name, size_t bytes) { 
      std::cout << "GPU Malloc: " << name << " - " 
                << bytes / (1024.0 * 1024.0) << " MB" << std::endl;
      total_allocated_ += bytes;
  }
  struct Parameters {
    uint32_t block_size_1d;
    Parameters() : block_size_1d(256) {}
  };

  MatrixFunctionApplicatorComplex(
      const Eigen::SparseMatrix<std::complex<double> > &A, uint32_t n,
      uint32_t m, uint32_t nnz, const Parameters &params = Parameters())
      : n_(n), m_(m), params_(params) {


#if DEBUG
    size_t complex_double_size = sizeof(thrust::complex<double>);
    size_t double_size = sizeof(double);
    size_t int_size = sizeof(int);

    size_t sparse_matrix_size = (n + 1) * int_size + nnz * int_size + nnz * complex_double_size;
    size_t krylov_size = n * m * complex_double_size + m * m * complex_double_size +
                         2 * n * complex_double_size + 2 * double_size;
    size_t eigen_size = m * double_size + m * m * complex_double_size;
    size_t work_size = m * complex_double_size + m * m * complex_double_size +
                      n * m * complex_double_size;

    size_t total_expected = sparse_matrix_size + krylov_size + eigen_size + work_size;

    std::cout << "Expected GPU memory allocation: " << total_expected / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  - Sparse matrix: " << sparse_matrix_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  - Krylov subspace: " << krylov_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  - Eigendecomposition: " << eigen_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  - Work buffers: " << work_size / (1024.0 * 1024.0) << " MB" << std::endl;

    reportAlloc("CSR row pointers", (n + 1) * int_size);
    reportAlloc("CSR column indices", nnz * int_size);
    reportAlloc("CSR values", nnz * complex_double_size);
    reportAlloc("Krylov V", n * m * complex_double_size);
    reportAlloc("Krylov T", m * m * complex_double_size);
    reportAlloc("Krylov buf1", n * complex_double_size);
    reportAlloc("Krylov buf2", n * complex_double_size);
#endif


    const int *row_ptr = A.outerIndexPtr();
    const int *col_ind = A.innerIndexPtr();
    const std::complex<double> *values = A.valuePtr();
    cudaMalloc(&d_row_ptr_, (n + 1) * sizeof(int));
    cudaMalloc(&d_col_ind_, A.nonZeros() * sizeof(int));
    cudaMalloc(&d_values_, A.nonZeros() * sizeof(thrust::complex<double>));

    cudaMemcpy(d_row_ptr_, row_ptr, (n + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind_, col_ind, A.nonZeros() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_, values,
               A.nonZeros() * sizeof(thrust::complex<double>),
               cudaMemcpyHostToDevice);

    spmv_ = new DeviceSpMV<thrust::complex<double>>(d_row_ptr_, d_col_ind_,
                                                    d_values_, n, nnz);

    cudaMalloc(&krylov_.V, n * m * sizeof(thrust::complex<double>));
    cudaMalloc(&krylov_.T, m * m * sizeof(thrust::complex<double>));
    cudaMalloc(&krylov_.buf1, n * sizeof(thrust::complex<double>));
    cudaMalloc(&krylov_.buf2, n * sizeof(thrust::complex<double>));
    cudaMalloc(&krylov_.d_beta, sizeof(double));
    cudaMalloc(&krylov_.reconstruct_beta, sizeof(double));
    cublasCreate(&krylov_.handle);
    krylov_.n = n;
    krylov_.m = m;

    cudaMalloc(&d_eigenvalues_, m * sizeof(double));

    cudaMalloc(&d_diag_, m * sizeof(thrust::complex<double>));
    cudaMalloc(&d_work_, m * m * sizeof(thrust::complex<double>));
    cudaMalloc(&d_work_large_, n * m * sizeof(thrust::complex<double>));
    cudaMalloc(&d_info_, sizeof(int));
    cudaMalloc(&temp_work_, m_ * m_ * sizeof(thrust::complex<double>));

    cusolverDnCreate(&solver_handle_);
    cublasCreate(&cublas_handle_);

    int lwork;
    cusolverDnZheevd_bufferSize(
        solver_handle_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, m,
        reinterpret_cast<cuDoubleComplex *>(d_eigenvectors_), m, d_eigenvalues_,
        &lwork);
    cudaMalloc(&d_eigenvectors_, m * m * sizeof(cuDoubleComplex));
    cudaMalloc(&d_solver_work_, lwork * sizeof(cuDoubleComplex));
    solver_work_size_ = lwork;

    // block_dim_1d_ = dim3(256);
    // uint32_t block_size_2d = (m <= 32) ? 16 : 8;
    // block_2d_ = dim3(block_size_2d, block_size_2d);
    // 
    // const uint32_t max_grid_dim = 65535;
    // grid_1d_ = dim3(std::min((n_ + block_1d_.x - 1) / block_1d_.x, max_grid_dim));
    // uint32_t grid_x = std::min((m_ + block_2d_.x - 1) / block_2d_.x, max_grid_dim);
    // uint32_t grid_y = std::min((n_ + block_2d_.y - 1) / block_2d_.y, max_grid_dim);
    // grid_VK_ = dim3(grid_x, grid_y);
    // grid_x = std::min((m_ + block_2d_.x - 1) / block_2d_.x, max_grid_dim);
    // grid_y = std::min((m_ + block_2d_.y - 1) / block_2d_.y, max_grid_dim);
    // grid_QDQ_ = dim3(grid_x, grid_y);

    cudaMalloc(&e1_, m_ * sizeof(thrust::complex<double>));
    cudaMemset(e1_, 0, m_ * sizeof(thrust::complex<double>));
    thrust::complex<double> one_val(1.0, 0.0);
    cudaMemcpy(e1_, &one_val, sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    cudaMalloc(&temp_vec_, m_ * sizeof(thrust::complex<double>));
  }

  ~MatrixFunctionApplicatorComplex() {
    cudaFree(e1_);
    cudaFree(d_row_ptr_);
    cudaFree(d_col_ind_);
    cudaFree(d_values_);
    cudaFree(krylov_.V);
    cudaFree(krylov_.T);
    cudaFree(krylov_.buf1);
    cudaFree(krylov_.buf2);
    cudaFree(krylov_.d_beta);
    cudaFree(krylov_.reconstruct_beta);
    cublasDestroy(krylov_.handle);
    cudaFree(d_eigenvalues_);
    cudaFree(d_eigenvectors_);
    cudaFree(d_diag_);
    cudaFree(d_work_);
    cudaFree(d_work_large_);
    cudaFree(d_solver_work_);
    cudaFree(temp_work_);
    cudaFree(temp_vec_);
    cudaFree(d_info_);
    cusolverDnDestroy(solver_handle_);
    cublasDestroy(cublas_handle_);
    delete spmv_;
  }

  void apply(thrust::complex<double> *result,
           const thrust::complex<double> *input, std::complex<double> dt,
           const std::string func = "exp") {
    reset_data();
    lanczos_iteration_complex(spmv_, &krylov_, input);
    double beta;
    cudaMemcpy(&beta, krylov_.reconstruct_beta, sizeof(double),
               cudaMemcpyDeviceToHost);

    compute_eigen_decomposition();
    const uint32_t max_grid_dim = 65535;

    thrust::device_ptr<thrust::complex<double>> d_diag_ptr(d_diag_);
    thrust::device_ptr<double> d_eigvals_ptr(d_eigenvalues_);
    thrust::complex<double> thrust_dt(dt.real(), dt.imag());
    if (func == "exp")
	// thrust::device_ptr<thrust::complex<double>> density_ptr(d_density);
  	// thrust::device_ptr<const thrust::complex<double>> u_ptr(d_u);
  	// thrust::device_ptr<const double> m_ptr(d_m);

  	// thrust::transform(u_ptr, u_ptr + n, m_ptr, density_ptr,
  	//                  [] __device__ (const thrust::complex<double>& u, const double m) {
  	//                      return m * (u.real() * u.real() + u.imag() * u.imag());
  	//                  });
	//
	//  out[i] = thrust::exp(dt * eigvals[i]);

	thrust::transform(thrust::make_counting_iterator<uint32_t>(0),
                 thrust::make_counting_iterator<uint32_t>(m_),
                 d_diag_ptr,
                 [=] __device__ (const uint32_t idx) {
		   double eigval = d_eigvals_ptr[idx]; 
                   return exp(thrust_dt * eigval);
                 });

        // transform_eigenvals_exp<<<grid_1d_, block_1d_>>>(d_diag_, d_eigenvalues_, dt, m_);
    else if (func == "sinc")
        // transform_eigenvals_sinc<<<grid_1d_, block_1d_>>>(d_diag_, d_eigenvalues_, dt, m_);
	
	thrust::transform(thrust::make_counting_iterator<uint32_t>(0),
                 	thrust::make_counting_iterator<uint32_t>(m_),
                        d_diag_ptr,
			[=] __device__ (const uint32_t idx) {
			  double eigval = d_eigvals_ptr[idx];
			  const auto val = thrust_dt * eigval;
			  return abs(val) < 1e-8? thrust::complex<double>(1., 0.) : sin(val) / val; 
			});
                 
    else
        throw std::runtime_error("Matrix function application not implemented");
    
    cudaMemset(d_work_, 0, m_ * m_ * sizeof(thrust::complex<double>));
    dim3 blockDim(256);
    dim3 gridDim((m_ + blockDim.x - 1) / blockDim.x);
    set_diagonal<<<gridDim, blockDim>>>(d_work_, d_diag_, m_); 
   
    // thrust::complex<double> *temp_work;
    // cudaMalloc(&temp_work, m_ * m_ * sizeof(thrust::complex<double>));
    
    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta_cublas = make_cuDoubleComplex(0.0, 0.0);
    
    // Q f(D)
    cublasZgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                m_, m_, m_,
                &alpha,
                reinterpret_cast<const cuDoubleComplex*>(d_eigenvectors_), m_,
                reinterpret_cast<const cuDoubleComplex*>(d_work_), m_,
                &beta_cublas,
                reinterpret_cast<cuDoubleComplex*>(temp_work_), m_);
    
    // (Q f(D)) Q^H
    cublasZgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_C,
                m_, m_, m_,
                &alpha,
                reinterpret_cast<const cuDoubleComplex*>(temp_work_), m_,
                reinterpret_cast<const cuDoubleComplex*>(d_eigenvectors_), m_,
                &beta_cublas,
                reinterpret_cast<cuDoubleComplex*>(d_work_), m_);
    
    // thrust::complex<double> *e1_;
    // cudaMalloc(&e1_, m_ * sizeof(thrust::complex<double>));
    // cudaMemset(e1_, 0, m_ * sizeof(thrust::complex<double>));
    // thrust::complex<double> one_val(1.0, 0.0);
    // cudaMemcpy(e1_, &one_val, sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    
    // thrust::complex<double> *temp_vec;
    // cudaMalloc(&temp_vec, m_ * sizeof(thrust::complex<double>));
    

    cudaMemset(temp_vec_, 0, m_ * sizeof(thrust::complex<double>));
    // f(T) e_1 = (Q f(D)) Q^H e_1
    cublasZgemv(cublas_handle_, CUBLAS_OP_N, 
                m_, m_,
                &alpha,
                reinterpret_cast<const cuDoubleComplex*>(d_work_), m_,
                reinterpret_cast<const cuDoubleComplex*>(e1_), 1,
                &beta_cublas,
                reinterpret_cast<cuDoubleComplex*>(temp_vec_), 1);
    
    // V f(T) e_1
    cublasZgemv(cublas_handle_, CUBLAS_OP_N,
                n_, m_,
                &alpha,
                reinterpret_cast<const cuDoubleComplex*>(krylov_.V), n_,
                reinterpret_cast<const cuDoubleComplex*>(temp_vec_), 1,
                &beta_cublas,
                reinterpret_cast<cuDoubleComplex*>(d_work_large_), 1);
    
    const cuDoubleComplex scale_factor = make_cuDoubleComplex(beta, 0.0);
    
    // beta * V f(T) e_1
    cublasZscal(cublas_handle_, n_,
                &scale_factor,
                reinterpret_cast<cuDoubleComplex*>(d_work_large_), 1);
    
    cudaMemcpy(result, d_work_large_, n_ * sizeof(thrust::complex<double>),
               cudaMemcpyDeviceToDevice);
               
    
    // cudaFree(temp_vec);
  }


  // unsafe, but without full refactoring we take this choice now
  DeviceSpMV<thrust::complex<double>> *expose_spmv() { return spmv_; };

private:
  void compute_eigen_decomposition() {
    cudaMemcpy(d_eigenvectors_, krylov_.T, m_ * m_ * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToDevice);
    cusolverDnZheevd(
        solver_handle_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, m_,
        reinterpret_cast<cuDoubleComplex *>(d_eigenvectors_), m_,
        d_eigenvalues_, reinterpret_cast<cuDoubleComplex *>(d_solver_work_),
        solver_work_size_, d_info_);
  }

  void reset_data() {
    // maybe this can become shorter
    cudaMemset(krylov_.T, 0, m_ * m_ * sizeof(thrust::complex<double>));
    cudaMemset(krylov_.V, 0, n_ * m_ * sizeof(thrust::complex<double>));
    cudaMemset(krylov_.buf1, 0, n_ * sizeof(thrust::complex<double>));
    cudaMemset(krylov_.buf2, 0, n_ * sizeof(thrust::complex<double>));

    cudaMemset(krylov_.d_beta, 0, sizeof(double));
    cudaMemset(krylov_.reconstruct_beta, 0, sizeof(double));
    cudaMemset(d_eigenvectors_, 0, m_ * m_ * sizeof(thrust::complex<double>));
    cudaMemset(d_eigenvalues_, 0, m_ * sizeof(double));
  }

  DeviceSpMV<thrust::complex<double>> *spmv_;
  int *d_row_ptr_, *d_col_ind_;
  thrust::complex<double> *d_values_;

  KrylovInfoComplex krylov_;
  cusolverDnHandle_t solver_handle_;
  cublasHandle_t cublas_handle_;
  double *d_eigenvalues_;
  thrust::complex<double> *d_eigenvectors_;
  thrust::complex<double> *d_diag_;
  thrust::complex<double> *d_work_;
  thrust::complex<double> *d_work_large_;
  thrust::complex<double> *d_solver_work_;
  thrust::complex<double> *temp_work_;
  thrust::complex<double> *temp_vec_;
  thrust::complex<double> *e1_;
  int *d_info_;
  int solver_work_size_;
  uint32_t n_;
  uint32_t m_;
  Parameters params_;

  // dim3 block_dim_1d_;
  // dim3 block_2d_;

  // dim3 grid_VK_;
  // dim3 grid_QDQ_;

  // dim3 block_1d_;
  // dim3 grid_1d_;


  uint64_t total_allocated_;
};

#endif
