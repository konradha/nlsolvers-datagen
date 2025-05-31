#include "laplacians.hpp"
#include "util.hpp"

#include "boundaries.cuh"
#include "nlse_dev.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 10 && argc != 11) {
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
  const std::string output_file = argv[6];
  const double T = std::stod(argv[7]);
  const uint32_t nt = std::stoul(argv[8]);
  const uint32_t num_snapshots = std::stoul(argv[9]);

  std::optional<std::string> m_file;
  if (argc == 11) {
    m_file = argv[10];
  }

  const double dx = 2 * Lx / (nx - 1);
  const double dy = 2 * Ly / (ny - 1);

  const auto dt = T / nt;
  std::complex<double> dti(0, dt);
  const auto freq = nt / num_snapshots;

  std::vector<uint32_t> input_shape;
  Eigen::VectorX<std::complex<double>> u0 =
      read_from_npy<std::complex<double>>(input_file, input_shape);

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

  const Eigen::SparseMatrix<std::complex<double>> L =
      build_laplacian_noflux<std::complex<double>>(nx - 2, ny - 2, dx, dy);

  Eigen::VectorX<std::complex<double>> u_save(num_snapshots * nx * ny);

  // TODO: Redo performance benchmark. Here: Everything abstracted away.
  // TODO: Show m=15 Krylov subspace iterations are sufficient
  //       -> There's publications showing that precision becomes marginal
  //          after m=10, ie. we already land between 10^{-8} and 10^{-10}
  //          in terms of L² errors at that point.
  device::NLSESolverDevice::Parameters params(num_snapshots, freq, 15);
  device::NLSESolverDevice solver(L, u0.data(), m.data(), params);
  solver.store_snapshot_online(u_save.data()); // this should happen inside constructor ... maybe refactor later
  for (uint32_t i = 1; i < nt; ++i) {
    solver.step(dti, i, u_save.data());
    solver.apply_bc();
  }
  const std::vector<uint32_t> shape = {num_snapshots, ny, nx};
  save_to_npy(output_file, u_save, shape);
  return 0;
}
