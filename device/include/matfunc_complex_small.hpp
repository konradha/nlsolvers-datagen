#ifndef MATFUNC_COMPLEX_HPP
#define MATFUNC_COMPLEX_HPP

#include "lanczos_complex.hpp"
#include "pragmas.hpp"
#include "spmv.hpp"
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <thrust/complex.h>

#define DEBUG 0

__global__ void transform_eigenvals_exp(thrust::complex<double> *out,
                                        const double *eigvals,
                                        thrust::complex<double> dt,
                                        const uint32_t m) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < m) {
    out[i] = thrust::exp(dt * eigvals[i]);
  }
}

__global__ void transform_eigenvals_sinc(thrust::complex<double> *out,
                                         const double *eigvals,
                                         thrust::complex<double> dt,
                                         const uint32_t m) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < m) {
    const auto val = dt * eigvals[i];
    out[i] = thrust::abs(val) < 1e-8 ? thrust::complex<double>(1.0, 0.0)
                                     : thrust::sin(val) / val;
  }
}

__global__ void matrix_multiply_QDQ(const thrust::complex<double> *Q,
                                    const thrust::complex<double> *D,
                                    thrust::complex<double> *result,
                                    const uint32_t m) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < m) {
    thrust::complex<double> sum = 0.0;
    for (int k = 0; k < m; k++) {
      sum += Q[k * m + row] * D[k] * Q[k * m + col];
    }
    result[col * m + row] = sum;
  }
}

__global__ void matrix_multiply_VK(const thrust::complex<double> *V,
                                   const thrust::complex<double> *K,
                                   thrust::complex<double> *result,
                                   const uint32_t n, const uint32_t m) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < m) {
    thrust::complex<double> sum = 0.0;
    for (int k = 0; k < m; k++) {
      sum += V[k * n + row] * K[col * m + k];
    }
    result[col * n + row] = sum;
  }
}

__global__ void scale_first_col(const thrust::complex<double> *X,
                                thrust::complex<double> *result,
                                const double beta, const uint32_t n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    result[idx] = beta * X[idx];
  }
}

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
      const Eigen::SparseMatrix<std::complex<double>> &A, uint32_t n,
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

    block_dim_1d_ = dim3(256);
    block_2d_ = dim3(16, 16);
    grid_VK_ = dim3((m_ + block_2d_.x - 1) / block_2d_.x,
                    (n_ + block_2d_.y - 1) / block_2d_.y);
    grid_QDQ_ = dim3((m_ + block_2d_.x - 1) / block_2d_.x,
                     (m_ + block_2d_.y - 1) / block_2d_.y);
    block_1d_ = dim3(256);
    grid_1d_ = dim3((n_ + block_1d_.x - 1) / block_1d_.x);
  }

  ~MatrixFunctionApplicatorComplex() {
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
    // cudaDeviceSynchronize();
    double beta;
    cudaMemcpy(&beta, krylov_.reconstruct_beta, sizeof(double),
               cudaMemcpyDeviceToHost);

    compute_eigen_decomposition();

    // thrust::apply
    if (func == "exp")
    	transform_eigenvals_exp<<<grid_1d_, block_1d_>>>(d_diag_, d_eigenvalues_,
                                                     dt, m_);
    else if (func == "sinc")
	transform_eigenvals_sinc<<<grid_1d_, block_1d_>>>(d_diag_, d_eigenvalues_,
                                                     dt, m_);	
    else 
	throw std::runtime_error("Matrix function application not implemented");

    matrix_multiply_QDQ<<<grid_QDQ_, block_2d_>>>(d_eigenvectors_, d_diag_,
                                                  d_work_, m_);
    // gemmZ
    matrix_multiply_VK<<<grid_VK_, block_2d_>>>(krylov_.V, d_work_,
                                                d_work_large_, n_, m_);
    // dscalZ
    scale_first_col<<<grid_1d_, block_1d_>>>(d_work_large_, result, beta, n_);
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
  int *d_info_;
  int solver_work_size_;
  uint32_t n_;
  uint32_t m_;
  Parameters params_;

  dim3 block_dim_1d_;
  dim3 block_2d_;

  dim3 grid_VK_;
  dim3 grid_QDQ_;

  dim3 block_1d_;
  dim3 grid_1d_;


  uint64_t total_allocated_;
};

#endif
