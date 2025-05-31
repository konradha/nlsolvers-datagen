#ifndef MATFUNC_HPP
#define MATFUNC_HPP

#include "lanczos.hpp"
#include "pragmas.hpp"
#include "spmv.hpp"

// TODO debug
// TODO test
// TODO refactor
// TODO duplicate for complex doubles as well
// TODO test complex double implementation

class MatrixFunctionApplicator {
public:
  enum class FunctionType { COS_SQRT, SINC2_SQRT, ID_SQRT };

  struct Parameters {
    uint32_t block_size_1d = 64;
    uint32_t block_size_2d = 8;
    double sinc_threshold = 1e-14;
  };

  MatrixFunctionApplicator(DeviceSpMV *spmv, KrylovInfo *krylov,
                           cusolverDnHandle_t handle, uint32_t n, uint32_t m,
                           const Parameters &params = Parameters())
      : spmv_(spmv), krylov_(krylov), solver_handle_(handle), n_(n), m_(m),
        params_(params) {

    cudaMalloc(&d_eigenvalues_, m * sizeof(double));
    cudaMalloc(&d_eigenvectors_, m * m * sizeof(double));
    cudaMalloc(&d_transform_, m * sizeof(double));
    cudaMalloc(&d_work_real_, m * m * sizeof(double));
    cudaMalloc(&d_info_, sizeof(int));

    int lwork;
    cusolverDnDsyevd_bufferSize(solver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                                CUBLAS_FILL_MODE_LOWER, m, d_eigenvectors_, m,
                                d_eigenvalues_, &lwork);
    cudaMalloc(&d_solver_work_, lwork * sizeof(double));
    solver_work_size_ = lwork;

    block_dim_2d_ = dim3(params_.block_size_2d, params_.block_size_2d);
    grid_dim_2d_ = dim3((m + block_dim_2d_.x - 1) / block_dim_2d_.x,
                        (m + block_dim_2d_.y - 1) / block_dim_2d_.y);
    block_dim_1d_ = dim3(params_.block_size_1d);
    grid_dim_1d_ = dim3(1);
  }

  ~MatrixFunctionApplicator() {
    cudaFree(d_eigenvalues_);
    cudaFree(d_eigenvectors_);
    cudaFree(d_transform_);
    cudaFree(d_work_real_);
    cudaFree(d_solver_work_);
    cudaFree(d_info_);
  }

  void apply(double *result, const double *input, double t, FunctionType type) {
    lanczos_iteration(spmv_, krylov_);
    compute_eigen_decomposition();

    if (type == FunctionType::COS_SQRT) {
      transform_cos_sqrt<<<grid_dim_1d_, block_dim_1d_>>>(
          d_transform_, d_eigenvalues_, t, m_, params_.sinc_threshold);
    } else if (type == FunctionType::SINC2_SQRT) {
      transform_sinc2_sqrt<<<grid_dim_1d_, block_dim_1d_>>>(
          d_transform_, d_eigenvalues_, t, m_, params_.sinc_threshold);
    } else {
      transform_id_sqrt<<<grid_dim_1d_, block_dim_1d_>>>(d_transform_,
                                                         d_eigenvalues_, t, m_);
    }

    compute_matrix_function<<<grid_dim_2d_, block_dim_2d_>>>(
        d_work_real_, d_eigenvectors_, d_transform_, m_);

    apply_to_first_column<<<grid_dim_1d_, block_dim_1d_>>>(
        result, krylov_->V, d_work_real_, krylov_->beta, n_, m_);
  }

private:
  void compute_eigen_decomposition() {
    cudaMemcpy(d_eigenvectors_, krylov_->T, m_ * m_ * sizeof(double),
               cudaMemcpyDeviceToDevice);

    cusolverDnDsyevd(solver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                     CUBLAS_FILL_MODE_LOWER, m_, d_eigenvectors_, m_,
                     d_eigenvalues_, d_solver_work_, solver_work_size_,
                     d_info_);
  }

  DeviceSpMV *spmv_;
  KrylovInfo *krylov_;
  cusolverDnHandle_t solver_handle_;
  double *d_eigenvalues_;
  double *d_eigenvectors_;
  double *d_transform_;
  double *d_work_real_;
  double *d_solver_work_;
  int *d_info_;
  int solver_work_size_;
  uint32_t n_;
  uint32_t m_;
  dim3 grid_dim_2d_;
  dim3 block_dim_2d_;
  dim3 grid_dim_1d_;
  dim3 block_dim_1d_;
  Parameters params_;
};

__global__ void transform_cos_sqrt(double *transform, const double *eigenvalues,
                                   double t, uint32_t m, double threshold) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    transform[idx] = cos(t * sqrt(abs(eigenvalues[idx])));
  }
}

__global__ void transform_sinc2_sqrt(double *transform,
                                     const double *eigenvalues, double t,
                                     uint32_t m, double threshold) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    double x = t * sqrt(abs(eigenvalues[idx]));
    transform[idx] = abs(x) < threshold ? 1.0 : (sin(x) / x) * (sin(x) / x);
  }
}

__global__ void transform_id_sqrt(double *transform, const double *eigenvalues,
                                  double t, uint32_t m) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    transform[idx] = t * sqrt(abs(eigenvalues[idx]));
  }
}

__global__ void compute_matrix_function(double *result, const double *Q,
                                        const double *transform, uint32_t m) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < m) {
    double sum = 0;
    for (uint32_t k = 0; k < m; k++) {
      sum += Q[row + k * m] * transform[k] * Q[col + k * m];
    }
    result[row + col * m] = sum;
  }
}

__global__ void apply_to_first_column(double *result, const double *V,
                                      const double *matrix, double beta,
                                      uint32_t n, uint32_t m) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    double sum = 0;
    for (uint32_t j = 0; j < m; j++) {
      sum += V[j * n + idx] * matrix[j];
    }
    result[idx] = beta * sum;
  }
}

#endif
