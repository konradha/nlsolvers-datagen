#include "driver_utils.hpp"
#include "laplacians.hpp"
#include "nlse_dev.hpp"
#include "util.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <complex>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

Eigen::VectorXd read_optional_m_2d(const std::optional<std::string> &path,
                                   const driver::Grid2D &grid) {
  if (!path) {
    return Eigen::VectorXd::Ones(grid.size());
  }
  return driver::read_npy_checked<double>(*path, {grid.ny, grid.nx}, "m");
}

int run(char **argv, bool has_m_file) {
  const driver::Grid2D grid = driver::make_grid_2d(
      driver::parse_u32(argv[1], "nx"), driver::parse_u32(argv[2], "ny"),
      driver::parse_double(argv[3], "Lx"), driver::parse_double(argv[4], "Ly"));
  const std::string input_u0 = argv[5];
  const std::string output_u = argv[6];
  const double final_time = driver::parse_double(argv[7], "T");
  const uint32_t nt = driver::parse_u32(argv[8], "nt");
  const uint32_t num_snapshots = driver::parse_u32(argv[9], "num_snapshots");
  const std::optional<std::string> m_file = has_m_file
                                               ? std::optional<std::string>(argv[10])
                                               : std::nullopt;

  const double dt = final_time / static_cast<double>(nt);
  const std::complex<double> dti(0.0, dt);
  const uint32_t snapshot_freq = driver::snapshot_frequency(nt, num_snapshots);

  Eigen::VectorXcd u0 = driver::read_npy_checked<std::complex<double>>(
      input_u0, {grid.ny, grid.nx}, "u0");
  Eigen::VectorXd m = read_optional_m_2d(m_file, grid);

  const Eigen::SparseMatrix<std::complex<double>> L =
      build_laplacian_noflux<std::complex<double>>(grid.nx - 2, grid.ny - 2,
                                                    std::complex<double>(grid.dx),
                                                    std::complex<double>(grid.dy));

  Eigen::VectorXcd u_save(num_snapshots * grid.size());

  device::NLSESolverDevice::Parameters params(num_snapshots, snapshot_freq, 15);
  device::NLSESolverDevice solver(L, u0.data(), m.data(), false, params);
  solver.store_snapshot_online(u_save.data());

  driver::run_nlse_steps(
      solver, nt,
      [dti, &u_save](device::NLSESolverDevice &solver, uint32_t step) {
        solver.step(dti, step, u_save.data());
      });

  const std::vector<uint32_t> shape = {num_snapshots, grid.ny, grid.nx};
  save_to_npy(output_u, u_save, shape);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 10 && argc != 11) {
    std::cerr << "Usage: " << argv[0]
              << " nx ny Lx Ly input_u0.npy output_traj.npy T nt "
                 "num_snapshots [input_m.npy]\n";
    return 1;
  }

  try {
    return run(argv, argc == 11);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
