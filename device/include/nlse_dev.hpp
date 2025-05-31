#ifndef NLSE_DEV_HPP
#define NLSE_DEV_HPP

#include "boundaries.cuh"
#include "matfunc_complex.hpp"
#include "nlse.cuh"
#include "pragmas.hpp"
#include "spmv.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define DEBUG 0


namespace device {

// Let's remove more hand-rolled kernels to not incure more penalties!
void compute_density(thrust::complex<double>* d_density, const thrust::complex<double>* d_u, const double* d_m, const size_t n) {
  thrust::device_ptr<thrust::complex<double>> density_ptr(d_density);
  thrust::device_ptr<const thrust::complex<double>> u_ptr(d_u);
  thrust::device_ptr<const double> m_ptr(d_m);

  thrust::transform(u_ptr, u_ptr + n, m_ptr, density_ptr,
                   [] __device__ (const thrust::complex<double>& u, const double m) {
                       return m * (u.real() * u.real() + u.imag() * u.imag());
                   });
}

void apply_nonlinear_part(thrust::complex<double>* d_out, const thrust::complex<double>* d_in,
                           const thrust::complex<double>* d_rho, const std::complex<double>& tau, const size_t n) {
  thrust::device_ptr<thrust::complex<double>> out_ptr(d_out);
  thrust::device_ptr<const thrust::complex<double>> in_ptr(d_in);
  thrust::device_ptr<const thrust::complex<double>> rho_ptr(d_rho);
  thrust::complex<double> thrust_tau(tau.real(), tau.imag());
  thrust::transform(in_ptr, in_ptr + n, rho_ptr, out_ptr,
                   [thrust_tau] __device__ (const thrust::complex<double>& in_val, const thrust::complex<double>& rho_val) {
                       return in_val * exp(thrust_tau * rho_val);
                   });
}

void compute_B(thrust::complex<double>* d_out, const double* d_m, const thrust::complex<double>* d_u, const size_t n) {
  thrust::device_ptr<thrust::complex<double>> out_ptr(d_out);
  thrust::device_ptr<const thrust::complex<double>> u_ptr(d_u);
  thrust::device_ptr<const double> m_ptr(d_m);

  thrust::transform(u_ptr, u_ptr + n, m_ptr, out_ptr,
                   [] __device__ (const thrust::complex<double>& u, const double m) {
                       return -m * (u.real() * u.real() + u.imag() * u.imag()) * u;
                   });
}

void apply_sewi(thrust::complex<double>* d_out, const thrust::complex<double>* d_exp_prev,
                const thrust::complex<double>* d_exp_psi_B, const std::complex<double>& tau, const size_t n) {
  thrust::device_ptr<thrust::complex<double>> out_ptr(d_out);
  thrust::device_ptr<const thrust::complex<double>> exp_prev_ptr(d_exp_prev);
  thrust::device_ptr<const thrust::complex<double>> exp_psi_B_ptr(d_exp_psi_B);
  thrust::complex<double> thrust_tau(tau.real(), tau.imag());
  thrust::transform(exp_prev_ptr, exp_prev_ptr + n, exp_psi_B_ptr, out_ptr,
                   [thrust_tau] __device__ (const thrust::complex<double>& exp_prev, const thrust::complex<double>& exp_psi_B) {
                       return exp_prev - 2.0 * thrust_tau * exp_psi_B;
                   });
}

class NLSESolverDevice {
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

  NLSESolverDevice(const Eigen::SparseMatrix<std::complex<double>> &L,
                   const std::complex<double> *host_u0, const double *host_m,
		   const bool & is_3d,
                   const Parameters &params = Parameters())
      : is_3d_(is_3d), n_(L.rows()), current_snapshot_(0), params_(params) {

#if DEBUG
    size_t free_mem_start, total_mem;
    cudaMemGetInfo(&free_mem_start, &total_mem); 
    std::cout << "GPU Memory: " << total_mem / (1024 * 1024) << " MB total, " 
              << free_mem_start / (1024 * 1024) << " MB free before allocation" << std::endl;

    size_t complex_double_size = sizeof(thrust::complex<double>);
    size_t double_size = sizeof(double);

    size_t total_allocation = 0;
    size_t u_size = n_ * complex_double_size;
    size_t m_size = n_ * double_size;
    size_t traj_size = n_ * params_.num_snapshots * complex_double_size;

    total_allocation += u_size * 4; // all buffers
    total_allocation += m_size;     
    total_allocation += traj_size;  

    std::cout << "Expected allocation: " << total_allocation / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  - Field buffers: " << (u_size * 4) / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  - Trajectory storage: " << traj_size / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  - Coefficient field: " << m_size / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  - Matrix operator (approx): " << (L.nonZeros() * complex_double_size) / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  - Krylov subspace (approx): " << (n_ * params_.krylov_dim * complex_double_size) / (1024 * 1024) << " MB" << std::endl;

    if (total_allocation > free_mem_start * 0.9) {  // 90% ??
        std::cerr << "WARNING: Expected allocation (" << total_allocation / (1024 * 1024)
                  << " MB) exceeds 90% of available GPU memory!" << std::endl;
    }
#endif
    

    cudaMalloc(&d_u_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_buf_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_density_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_buf2_, n_ * sizeof(thrust::complex<double>));
    cudaMalloc(&d_buf3_, n_ * sizeof(thrust::complex<double>));

    cudaMalloc(&d_m_, n_ * sizeof(double));
    // cudaMalloc(&d_u_trajectory_,
    //            n_ * params_.num_snapshots * sizeof(thrust::complex<double>));

    cudaMemcpy(d_u_, host_u0, n_ * sizeof(thrust::complex<double>),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_, host_m, n_ * sizeof(double), cudaMemcpyHostToDevice);

    matfunc_ = new MatrixFunctionApplicatorComplex(L, n_, params_.krylov_dim,
                                                   L.nonZeros());

    if (is_3d_) {
      is_3d_ = true;
      nx_ = ny_ = nz_ = std::cbrt(n_);
      assert(nx_ * ny_ * nz_ == n_);
    } else {
      nx_ = ny_ = std::sqrt(n_);
      assert(nx_ * ny_  == n_);
    }

    // TODO check dims here
    uint32_t threads = 256;
    if (!is_3d_) {
      block_dim_ = dim3(16, 16, 1);
      grid_dim_ = dim3((nx_ + block_dim_.x - 1) / block_dim_.x,
                       (ny_ + block_dim_.y - 1) / block_dim_.y);
    } else {
      block_dim_ = dim3(8, 8, 4);
      grid_dim_ = dim3((nx_ + block_dim_.x - 1) / block_dim_.x,
                       (ny_ + block_dim_.y - 1) / block_dim_.y,
                       (nz_ + block_dim_.z - 1) / block_dim_.z);
    }
    // block_dim_ = dim3(threads / 16, threads / 16);
    // grid_dim_ = dim3((nx_ + block_dim_.x - 1) / block_dim_.x,
    //                  (ny_ + block_dim_.y - 1) / block_dim_.y);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
  }

  ~NLSESolverDevice() {
    delete matfunc_;
    cudaFree(d_u_);
    cudaFree(d_buf_);
    cudaFree(d_buf2_);
    cudaFree(d_buf3_);
    cudaFree(d_density_);
    cudaFree(d_m_);
    // cudaFree(d_u_trajectory_);
    if (d_u_prev_ != nullptr)
      cudaFree(d_u_prev_);
  }

  void apply_bc() {
    if (!is_3d_) {
        neumann_bc_no_velocity_blocking<thrust::complex<double>>(d_u_, nx_, ny_);
      } else {
        neumann_bc_no_velocity_blocking_3d<thrust::complex<double>>(d_u_, nx_,
                                                                    ny_, nz_);
      }
    }
  
  void step(const std::complex<double> tau, const uint32_t step_number, std::complex<double>* host_dst) {
    compute_density(d_density_, d_u_, d_m_, n_);
    apply_nonlinear_part(d_buf_, d_u_, d_density_, 0.5 * tau, n_);
    matfunc_->apply(d_u_, d_buf_, tau, "exp");
  
    compute_density(d_density_, d_u_, d_m_, n_);
    apply_nonlinear_part(d_u_, d_u_, d_density_, 0.5 * tau, n_);
  
    if (step_number % params_.snapshot_freq == 0) {
      store_snapshot_online(host_dst);
    }
  
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
  }
  
  void step_sewi(const std::complex<double> tau, const uint32_t step_number, std::complex<double>* host_dst) {
    if (step_number == 1) {
      cudaMalloc(&d_u_prev_, n_ * sizeof(thrust::complex<double>));
      cudaMemcpy(d_u_prev_, d_u_, n_ * sizeof(thrust::complex<double>), cudaMemcpyDeviceToDevice);
      step(tau, 1, host_dst);
    } else { 
      thrust::device_ptr<thrust::complex<double>> u_ptr(d_u_);
      thrust::device_ptr<thrust::complex<double>> buf_ptr(d_buf_);
      thrust::copy(u_ptr, u_ptr + n_, buf_ptr);

      compute_B(d_buf2_, d_m_, d_u_, n_);

      std::complex<double> std_real_tau(tau.imag(), 0.0);
      thrust::complex<double> real_tau(std_real_tau.real(), std_real_tau.imag());
      matfunc_->apply(d_buf3_, d_buf2_, std_real_tau, "sinc");
      matfunc_->apply(d_buf2_, d_buf3_, tau, "exp");
      
      std::complex<double> two_tau = 2.0 * tau;
      matfunc_->apply(d_buf3_, d_u_prev_, two_tau, "exp");

      apply_sewi(d_u_, d_buf2_, d_buf3_, tau, n_);

      thrust::device_ptr<thrust::complex<double>> u_prev_ptr(d_u_prev_);
      thrust::copy(buf_ptr, buf_ptr + n_, u_prev_ptr);
    }

    if (step_number > 1 && step_number % params_.snapshot_freq == 0) {
      store_snapshot_online(host_dst);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
}

  
  // void step(const std::complex<double> tau, const uint32_t step_number, std::complex<double> * host_dst) {
  //   if (!is_3d_) {
  //     device::density<<<grid_dim_, block_dim_>>>(d_density_, d_u_, d_m_, nx_,
  //                                                ny_);
  //     device::nonlin_part<<<grid_dim_, block_dim_>>>(d_buf_, d_u_, d_density_,
  //                                                    .5 * tau, nx_, ny_);
  //     matfunc_->apply(d_u_, d_buf_, tau, "exp");

  //     device::density<<<grid_dim_, block_dim_>>>(d_density_, d_u_, d_m_, nx_,
  //                                                ny_);
  //     device::nonlin_part<<<grid_dim_, block_dim_>>>(d_u_, d_u_, d_density_,
  //                                                    .5 * tau, nx_, ny_);
  //   } else {
  //     device::density_3d<<<grid_dim_, block_dim_>>>(d_density_, d_u_, d_m_, nx_,
  //                                                ny_, nz_);
  //     device::nonlin_part_3d<<<grid_dim_, block_dim_>>>(d_buf_, d_u_, d_density_,
  //                                                    .5 * tau, nx_, ny_, nz_);
  //     matfunc_->apply(d_u_, d_buf_, tau, "exp");

  //     device::density_3d<<<grid_dim_, block_dim_>>>(d_density_, d_u_, d_m_, nx_,
  //                                                ny_, nz_);
  //     device::nonlin_part_3d<<<grid_dim_, block_dim_>>>(d_u_, d_u_, d_density_,
  //                                                    .5 * tau, nx_, ny_, nz_);
  //   }

  //   if (step_number % params_.snapshot_freq == 0) {
  //     store_snapshot_online(host_dst);
  //   }

  //   cudaError_t err = cudaGetLastError();
  //   if (err != cudaSuccess) {
  //     throw std::runtime_error(cudaGetErrorString(err));
  //   }
  // }

  // void step_sewi(const std::complex<double> tau, const uint32_t step_number, std::complex<double> * host_dst) {
  //   // initial step is symmetric SS2, see publication for reference
  //   if (step_number == 1) {
  //     cudaMalloc(&d_u_prev_, n_ * sizeof(thrust::complex<double>));
  //     cudaMemcpy(d_u_prev_, d_u_, n_ * sizeof(thrust::complex<double>),
  //                cudaMemcpyDeviceToDevice);
  //     step(tau, 1, host_dst);
  //   } else {
  //     cudaMemcpy(d_buf_, d_u_, n_ * sizeof(thrust::complex<double>),
  //                cudaMemcpyDeviceToDevice); // bookkeeping prev
  //     if (!is_3d_) {
  //       B<<<grid_dim_, block_dim_>>>(d_buf2_, d_m_, d_u_, nx_, ny_); 
  //       thrust::complex<double> real_tau(tau.imag(), 0.0);
  //       matfunc_->apply(d_buf3_ /* out */, d_buf2_ /* in */, real_tau, "sinc");
  //       matfunc_->apply(d_buf2_, d_buf3_, tau, "exp"); // re-use buffer from before 
  //       matfunc_->apply(d_buf3_, d_u_prev_, 2. * tau, "exp");
  //       // u^{n+1} = e^{-2i*tau*Delta}u^{n-1} - 2i*tau*e^{-i*tau*Delta}*phi_s(tau*Delta)*B(u^n)
  //       sewi<<<grid_dim_, block_dim_>>>(d_u_, d_buf2_, d_buf3_, tau, nx_, ny_); 
  //     } else {
  //       B_3d<<<grid_dim_, block_dim_>>>(d_buf2_, d_m_, d_u_, nx_, ny_, nz_); 
  //       thrust::complex<double> real_tau(tau.imag(), 0.0);
  //       matfunc_->apply(d_buf3_ /* out */, d_buf2_ /* in */, real_tau, "sinc");
  //       matfunc_->apply(d_buf2_, d_buf3_, tau, "exp"); // re-use buffer from before 
  //       matfunc_->apply(d_buf3_, d_u_prev_, 2. * tau, "exp");
  //       // u^{n+1} = e^{-2i*tau*Delta}u^{n-1} - 2i*tau*e^{-i*tau*Delta}*phi_s(tau*Delta)*B(u^n)
  //       sewi_3d<<<grid_dim_, block_dim_>>>(d_u_, d_buf2_, d_buf3_, tau, nx_, ny_, nz_);
  //     }
  //     cudaMemcpy(d_u_prev_, d_buf_, n_ * sizeof(thrust::complex<double>),
  //                cudaMemcpyDeviceToDevice); // attach "prev" to prev
  //   }

  //   if (step_number > 1 && step_number % params_.snapshot_freq == 0) {
  //     store_snapshot_online(host_dst);
  //   }

  //   cudaError_t err = cudaGetLastError();
  //   if (err != cudaSuccess) {
  //     throw std::runtime_error(cudaGetErrorString(err));
  //   }
  // }

  // void transfer_snapshots(std::complex<double> *dst) {
  //   cudaMemcpy(dst, d_u_trajectory_,
  //              params_.num_snapshots * n_ * sizeof(thrust::complex<double>),
  //              cudaMemcpyDeviceToHost);
  // }

  void store_snapshot_online(std::complex<double> *host_dst) {
    if (current_snapshot_ >= params_.num_snapshots) {
        return;
    }
    cudaMemcpy(
        host_dst + current_snapshot_ * n_,
        d_u_,
        n_ * sizeof(thrust::complex<double>),
        cudaMemcpyDeviceToHost
    );
    current_snapshot_++;
}

private:
  // void store_snapshot(const uint32_t step_number) {
  //   cudaMemcpy(d_u_trajectory_ + current_snapshot_ * n_, d_u_,
  //              n_ * sizeof(thrust::complex<double>), cudaMemcpyDeviceToDevice);
  //   current_snapshot_++;
  // }

  MatrixFunctionApplicatorComplex *matfunc_;
  thrust::complex<double> *d_u_;
  thrust::complex<double> *d_u_prev_ =
      nullptr; // for first step set to nullptr?
  thrust::complex<double> *d_buf_;
  thrust::complex<double> *d_buf2_;
  thrust::complex<double> *d_buf3_;
  thrust::complex<double> *d_density_;
  // thrust::complex<double> *d_u_trajectory_;
  double *d_m_;

  uint32_t nx_, ny_, nz_;
  bool is_3d_ = false;
  uint32_t current_snapshot_;
  uint32_t n_;
  dim3 grid_dim_;
  dim3 block_dim_;
  Parameters params_;
};
} // namespace device
#endif
