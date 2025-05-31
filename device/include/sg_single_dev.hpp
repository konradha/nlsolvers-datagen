#ifndef SG_SINGLE_DEV_HPP
#define SG_SINGLE_DEV_HPP

#include "boundaries.cuh"
#include "matfunc_real.hpp"
#include "pragmas.hpp"
#include "sg_single.cuh"
#include "spmv.hpp"

#include <cuda_runtime.h>
#include <vector>

namespace device {

class SGESolverDevice {
public:
  struct Parameters {
    uint32_t block_size;
    uint32_t num_snapshots;
    uint32_t snapshot_freq;
    uint32_t krylov_dim;

    Parameters(uint32_t ns = 100, uint32_t freq = 5, uint32_t m = 10)
        : block_size(256), num_snapshots(ns), snapshot_freq(freq),
          krylov_dim(m) {}
  };

  SGESolverDevice(const int *d_row_ptr, const int *d_col_ind,
                  const double *d_values, const double *h_m, uint32_t n,
                  uint32_t nnz, const double *host_u0, const double *host_v0,
                  double dt, const Parameters &params = Parameters())
      : n_(n), current_snapshot_(0), params_(params), dt_(dt) {

    nx_ = ny_ = std::sqrt(n);
    cudaMalloc(&d_u_, n * sizeof(double));
    cudaMalloc(&d_v_, n * sizeof(double));

    cudaMemcpy(d_u_, host_u0, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_, host_v0, n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_u_past_, n * sizeof(double));

    Eigen::VectorXd u_past = Eigen::Map<const Eigen::VectorXd>(host_u0, n) -
                             dt * Eigen::Map<const Eigen::VectorXd>(host_v0, n);
    cudaMemcpy(d_u_past_, u_past.data(), n * sizeof(double),
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_buf_, n * sizeof(double));
    cudaMalloc(&d_buf2_, n * sizeof(double));
    cudaMalloc(&d_buf3_, n * sizeof(double));
    cudaMalloc(&d_m_, n * sizeof(double));

    cudaMemcpy(d_m_, h_m, n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_u_trajectory_, n * params_.num_snapshots * sizeof(double));

    spmv_ = new DeviceSpMV<double>(d_row_ptr, d_col_ind, d_values, n, nnz);
    matfunc_ = new MatrixFunctionApplicatorReal(d_row_ptr, d_col_ind, d_values,
                                                n, params_.krylov_dim, nnz);

    const uint32_t threads_per_block = params_.block_size;
    grid_dim_ = dim3((n_ + threads_per_block - 1) / threads_per_block);
    block_dim_ = dim3(threads_per_block);

    store_snapshot(0);
  }

  ~SGESolverDevice() {
    delete spmv_;
    delete matfunc_;
    cudaFree(d_u_);
    cudaFree(d_v_);
    cudaFree(d_u_past_);
    cudaFree(d_buf_);
    cudaFree(d_buf2_);
    cudaFree(d_buf3_);
    cudaFree(d_m_);
    cudaFree(d_u_trajectory_);
  }

  void step() {
    device::SGESolver::step(d_u_, d_u_past_, d_buf_, d_buf2_, d_buf3_, matfunc_,
                            d_m_, dt_, n_, grid_dim_, block_dim_);
  }

  void transfer_snapshots(double *dst) {
    cudaMemcpy(dst, d_u_trajectory_,
               params_.num_snapshots * n_ * sizeof(double),
               cudaMemcpyDeviceToHost);
  }

  void apply_bc() { neumann_bc_no_velocity_blocking<double>(d_u_, nx_, ny_); }

  void store_snapshot(const uint32_t snapshot_idx) {
    if (snapshot_idx < params_.num_snapshots) {
      cudaMemcpy(d_u_trajectory_ + snapshot_idx * n_, d_u_, n_ * sizeof(double),
                 cudaMemcpyDeviceToDevice);
      current_snapshot_ = snapshot_idx + 1;
    }
  }

private:
  DeviceSpMV<double> *spmv_;
  MatrixFunctionApplicatorReal *matfunc_;
  double *d_u_;
  double *d_v_;
  double *d_u_past_;
  double *d_m_;
  double *d_buf_;
  double *d_buf2_;
  double *d_buf3_;
  double *d_u_trajectory_;
  uint32_t current_snapshot_;
  uint32_t nx_;
  uint32_t ny_;
  uint32_t n_;
  dim3 grid_dim_;
  dim3 block_dim_;
  Parameters params_;
  double dt_;
};

} // namespace device

#endif // SG_SINGLE_DEV_HPP
