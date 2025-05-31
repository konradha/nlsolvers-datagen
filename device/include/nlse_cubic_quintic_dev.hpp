#ifndef NLSE_CUBIC_QUINTIC_DEV_HPP
#define NLSE_CUBIC_QUINTIC_DEV_HPP

#include "boundaries.cuh"
#include "matfunc_complex.hpp"
#include "nlse_cubic_quintic.cuh"
#include "pragmas.hpp"
#include "spmv.hpp"

#include <cuda_runtime.h>
#include <vector>

namespace device {

class NLSECubicQuinticSolverDevice {
public:
  struct Parameters {
    uint32_t block_size;
    uint32_t num_snapshots;
    uint32_t snapshot_freq;
    uint32_t krylov_dim;
    double sigma1;
    double sigma2;
    Parameters(uint32_t ns = 100, uint32_t freq = 5, uint32_t m = 10,
               double s1 = 1.0, double s2 = 1.0)
        : block_size(256), num_snapshots(ns), snapshot_freq(freq),
          krylov_dim(m), sigma1(s1), sigma2(s2) {}
  };

  NLSECubicQuinticSolverDevice(
      const Eigen::SparseMatrix<std::complex<double>> &L,
      const std::complex<double> *host_u0, const double *host_m,
      const Parameters &params = Parameters())
      : n_(L.rows()), current_snapshot_(0), params_(params) {

    cudaMalloc(&d_u_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_buf_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_density_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_m_, n_ * sizeof(double));
    cudaMalloc(&d_u_trajectory_,
               n_ * params_.num_snapshots * sizeof(thrust::complex<double>));

    cudaMemcpy(d_u_, host_u0, n_ * sizeof(thrust::complex<double>),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_, host_m, n_ * sizeof(double), cudaMemcpyHostToDevice);

    matfunc_ = new MatrixFunctionApplicatorComplex(L, n_, params_.krylov_dim,
                                                   L.nonZeros());

    nx_ = sqrt(n_);
    ny_ = nx_;
    uint32_t threads = 256;
    block_dim_ = dim3(threads / 16, threads / 16);
    grid_dim_ = dim3((nx_ + block_dim_.x - 1) / block_dim_.x,
                     (ny_ + block_dim_.y - 1) / block_dim_.y);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }

    store_snapshot(0);
  }

  ~NLSECubicQuinticSolverDevice() {
    delete matfunc_;
    cudaFree(d_u_);
    cudaFree(d_buf_);
    cudaFree(d_density_);
    cudaFree(d_m_);
    cudaFree(d_u_trajectory_);
  }

  void apply_bc() {
    neumann_bc_no_velocity_blocking<thrust::complex<double>>(d_u_, nx_, ny_);
  }

  void step(const std::complex<double> tau, const uint32_t step_number) {
    device::density_cubic_quintic<<<grid_dim_, block_dim_>>>(
        d_density_, d_u_, d_m_, params_.sigma1, params_.sigma2, nx_, ny_);
    device::nonlin_part_cubic_quintic<<<grid_dim_, block_dim_>>>(
        d_buf_, d_u_, d_density_, -.5 * tau, nx_, ny_);
    matfunc_->apply(d_u_, d_buf_, -tau);

    device::density_cubic_quintic<<<grid_dim_, block_dim_>>>(
        d_density_, d_u_, d_m_, params_.sigma1, params_.sigma2, nx_, ny_);
    device::nonlin_part_cubic_quintic<<<grid_dim_, block_dim_>>>(
        d_u_, d_u_, d_density_, -.5 * tau, nx_, ny_);

    if (step_number % params_.snapshot_freq == 0) {
      store_snapshot(step_number);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
  }

  void transfer_snapshots(std::complex<double> *dst) {
    cudaMemcpy(dst, d_u_trajectory_,
               params_.num_snapshots * n_ * sizeof(thrust::complex<double>),
               cudaMemcpyDeviceToHost);
  }

private:
  void store_snapshot(const uint32_t step_number) {
    cudaMemcpy(d_u_trajectory_ + current_snapshot_ * n_, d_u_,
               n_ * sizeof(thrust::complex<double>), cudaMemcpyDeviceToDevice);
    current_snapshot_++;
  }

  MatrixFunctionApplicatorComplex *matfunc_;
  thrust::complex<double> *d_u_;
  thrust::complex<double> *d_buf_;
  thrust::complex<double> *d_density_;
  thrust::complex<double> *d_u_trajectory_;
  double *d_m_;

  uint32_t nx_, ny_;
  uint32_t current_snapshot_;
  uint32_t n_;
  dim3 grid_dim_;
  dim3 block_dim_;
  Parameters params_;
};
} // namespace device

#endif
