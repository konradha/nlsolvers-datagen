#include "driver_utils.hpp"
#include "kg_dev.hpp"
#include "laplacians.hpp"
#include "util.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

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
  const std::string input_v0 = argv[6];
  const std::string output_u = argv[7];
  const std::string output_v = argv[8];
  const double final_time = driver::parse_double(argv[9], "T");
  const uint32_t nt = driver::parse_u32(argv[10], "nt");
  const uint32_t num_snapshots = driver::parse_u32(argv[11], "num_snapshots");
  const std::optional<std::string> m_file = has_m_file
                                               ? std::optional<std::string>(argv[12])
                                               : std::nullopt;

  const double dt = final_time / static_cast<double>(nt);
  const uint32_t snapshot_freq = driver::snapshot_frequency(nt, num_snapshots);

  Eigen::VectorXd u0 = driver::read_npy_checked<double>(
      input_u0, {grid.ny, grid.nx}, "u0");
  Eigen::VectorXd v0 = driver::read_npy_checked<double>(
      input_v0, {grid.ny, grid.nx}, "v0");
  Eigen::VectorXd m = read_optional_m_2d(m_file, grid);

  const Eigen::SparseMatrix<double> L =
      build_laplacian_noflux<double>(grid.nx - 2, grid.ny - 2, grid.dx, grid.dy);
  driver::DeviceCsrMatrix d_L(L);

  Eigen::VectorXd u_save(num_snapshots * grid.size());
  Eigen::VectorXd v_save(num_snapshots * grid.size());

  device::KGESolverDevice::Parameters params(num_snapshots, snapshot_freq, 10);
  device::KGESolverDevice solver(d_L.row_ptr(), d_L.col_ind(), d_L.values(),
                                 m.data(), grid.size(), d_L.nnz(), u0.data(),
                                 v0.data(), dt, false, params);

  driver::run_kge_steps(solver, nt, snapshot_freq, num_snapshots,
                        [](device::KGESolverDevice &solver) { solver.step(); });

  solver.transfer_snapshots(u_save.data(), v_save.data());
  const std::vector<uint32_t> shape = {num_snapshots, grid.ny, grid.nx};
  save_to_npy(output_u, u_save, shape);
  save_to_npy(output_v, v_save, shape);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 12 && argc != 13) {
    std::cerr << "Usage: " << argv[0]
              << " nx ny Lx Ly input_u0.npy input_v0.npy output_traj.npy "
                 "output_vel.npy T nt num_snapshots [input_m.npy]\n";
    return 1;
  }

  try {
    return run(argv, argc == 13);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
