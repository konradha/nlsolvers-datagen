#ifndef STOCHASTIC_PHI4_DEV_HPP
#define STOCHASTIC_PHI4_DEV_HPP

#include "boundaries.cuh"
#include "stochastic_phi4.cuh"
#include "matfunc_real.hpp"
#include "pragmas.hpp"
#include "spmv.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <vector>


namespace device {

class SP4SolverDevice {
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
  SP4SolverDevice(const int *d_row_ptr, const int *d_col_ind,
                  const double *d_values, const double *h_m, uint32_t n,
                  uint32_t nnz, const double *host_u0, const double *host_v0,
                  double dt, const bool is_3d, const double L, const Parameters &params = Parameters())
      : is_3d_(is_3d), n_(n), current_snapshot_(0), params_(params), dt_(dt), L_(L) {

    

    if (!is_3d) {
      nx_ = ny_ = std::sqrt(n);
      assert(nx_ * ny_ == n);
    } else {
      is_3d_ = true;
      nx_ = ny_ = nz_ = std::cbrt(n);
      assert(nx_ * ny_ * nz_ == n);
    }

    // part to make this work "nicely"
    const double kBT = 2.; // let's see if this works
    const double alpha = .01;  // coupling strength
    const double dx = 2 * L_ / (nx_ - 1);  
    //  α sqrt(k_B T dx)
    scaling_factor_ = alpha * sqrt(kBT * dx) / dt; // fluctuation-dissipation?



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
    cudaMalloc(&d_v_trajectory_, n * params_.num_snapshots * sizeof(double));

    spmv_ = new DeviceSpMV<double>(d_row_ptr, d_col_ind, d_values, n, nnz);
    matfunc_ = new MatrixFunctionApplicatorReal(d_row_ptr, d_col_ind, d_values,
                                                n, params_.krylov_dim, nnz);

    const uint32_t threads_per_block = params_.block_size;
    grid_dim_ = dim3((n_ + threads_per_block - 1) / threads_per_block);
    block_dim_ = dim3(threads_per_block);

    store_snapshot(0);
    cublasCreate(&cublas_handle_);
  }

  ~SP4SolverDevice() {
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
    cudaFree(d_v_trajectory_);
    cublasDestroy(cublas_handle_);
  }

  void step() {
    device::SP4Solver::step(d_v_, d_u_, d_u_past_, d_buf_,
                                           d_buf2_, d_buf3_, matfunc_, d_m_,
                                           dt_, n_,  scaling_factor_, grid_dim_, block_dim_);
  }
  void transfer_snapshots(double *dst_u, double *dst_v) {
    cudaMemcpy(dst_u, d_u_trajectory_,
               params_.num_snapshots * n_ * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(dst_v, d_v_trajectory_,
               params_.num_snapshots * n_ * sizeof(double),
               cudaMemcpyDeviceToHost);
  }
  void apply_bc() {
    if (!is_3d_)
      neumann_bc_no_velocity_blocking<double>(d_u_, nx_, ny_);
    else
      neumann_bc_no_velocity_blocking_3d<double>(d_u_, nx_, ny_, nz_);
  }

  void store_snapshot(const uint32_t snapshot_idx) {
    if (snapshot_idx < params_.num_snapshots) {
      cudaMemcpy(d_u_trajectory_ + snapshot_idx * n_, d_u_, n_ * sizeof(double),
                 cudaMemcpyDeviceToDevice);
      cudaMemcpy(d_v_trajectory_ + snapshot_idx * n_, d_v_, n_ * sizeof(double),
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
  double *d_v_trajectory_;
  uint32_t current_snapshot_;
  uint32_t nx_;
  uint32_t ny_;
  uint32_t nz_;
  bool is_3d_ = false;
  uint32_t n_;
  dim3 grid_dim_;
  dim3 block_dim_;
  Parameters params_;
  double dt_;
  cublasHandle_t cublas_handle_;

  double L_;
  double scaling_factor_;
};
}


#endif
