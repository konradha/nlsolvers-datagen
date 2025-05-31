#include "laplacians.hpp"
#include "util.hpp"

#include "boundaries.cuh"
#include "sg_double_dev.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 11 && argc != 12) {
    std::cerr << "Usage: " << argv[0]
              << " nx ny Lx Ly input_u0.npy input_v0.npy output_traj.npy T nt "
                 "num_snapshots [input_m.npy]\n";
    std::cerr << "Example: " << argv[0]
              << " 256 256 10.0 10.0 initial.npy velocity.npy evolution.npy "
                 "1.5 500 100\n";
    std::cerr << "Example with m(x,y): " << argv[0]
              << " 256 256 10.0 10.0 initial.npy velocity.npy evolution.npy "
                 "1.5 500 100 coupling.npy\n";
    return 1;
  }

  const uint32_t nx = std::stoul(argv[1]);
  const uint32_t ny = std::stoul(argv[2]);
  const double Lx = std::stod(argv[3]);
  const double Ly = std::stod(argv[4]);
  const std::string input_file = argv[5];
  const std::string input_velocity = argv[6];
  const std::string output_file = argv[7];
  const double T = std::stod(argv[8]);
  const uint32_t nt = std::stoul(argv[9]);
  const uint32_t num_snapshots = std::stoul(argv[10]);

  std::optional<std::string> m_file;
  if (argc == 12) {
    m_file = argv[11];
  }

  const double dx = 2 * Lx / (nx - 1);
  const double dy = 2 * Ly / (ny - 1);
  const double dt = T / nt;
  const auto freq = nt / num_snapshots;

  std::vector<uint32_t> input_shape;
  Eigen::VectorXd u0 = read_from_npy<double>(input_file, input_shape);
  Eigen::VectorXd v0 = read_from_npy<double>(input_velocity, input_shape);

  if (input_shape.size() != 2 || input_shape[0] != ny || input_shape[1] != nx) {
    std::cerr << "Error: Input array dimensions mismatch\n";
    std::cerr << "Expected: " << ny << "x" << nx << "\n";
    std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "\n";
    return 1;
  }

  Eigen::VectorXd m;
  if (m_file) {
    try {
      std::vector<uint32_t> m_shape;
      m = read_from_npy<double>(m_file.value(), m_shape);

      if (m_shape.size() != 2 || m_shape[0] != ny || m_shape[1] != nx) {
        std::cerr << "Error: Coupling array dimensions mismatch\n";
        std::cerr << "Expected: " << ny << "x" << nx << "\n";
        std::cerr << "Got: " << m_shape[0] << "x" << m_shape[1] << "\n";
        std::cerr << "Using default m=1.0 everywhere\n";
        m = Eigen::VectorXd::Ones(nx * ny);
      }
    } catch (const std::exception &e) {
      std::cerr << "Error loading m(x,y): " << e.what() << "\n";
      std::cerr << "Using default m=1.0 everywhere\n";
      m = Eigen::VectorXd::Ones(nx * ny);
    }
  } else {
    m = Eigen::VectorXd::Ones(nx * ny);
  }

  const Eigen::SparseMatrix<double> L =
      build_laplacian_noflux<double>(nx - 2, ny - 2, dx, dy);

  int *d_row_ptr, *d_col_ind;
  double *d_values;

  cudaMalloc(&d_row_ptr, (L.rows() + 1) * sizeof(int));
  cudaMalloc(&d_col_ind, L.nonZeros() * sizeof(int));
  cudaMalloc(&d_values, L.nonZeros() * sizeof(double));

  cudaMemcpy(d_row_ptr, L.outerIndexPtr(), (L.rows() + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_ind, L.innerIndexPtr(), L.nonZeros() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, L.valuePtr(), L.nonZeros() * sizeof(double),
             cudaMemcpyHostToDevice);

  Eigen::VectorXd u_save(num_snapshots * nx * ny);

  device::SGEDoubleSolverDevice::Parameters params(num_snapshots, freq, 10);
  device::SGEDoubleSolverDevice solver(d_row_ptr, d_col_ind, d_values, m.data(),
                                       nx * ny, L.nonZeros(), u0.data(),
                                       v0.data(), dt, params);

  for (uint32_t i = 1; i < nt; ++i) {
    solver.step();
    solver.apply_bc();
    if (i % freq == 0) {
      uint32_t snapshot_idx = i / freq;
      if (snapshot_idx < num_snapshots) {
        solver.store_snapshot(snapshot_idx);
      }
    }
  }
  solver.transfer_snapshots(u_save.data());

  const std::vector<uint32_t> shape = {num_snapshots, ny, nx};
  save_to_npy(output_file, u_save, shape);

  cudaFree(d_row_ptr);
  cudaFree(d_col_ind);
  cudaFree(d_values);

  return 0;
}
