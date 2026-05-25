#include "driver_utils.hpp"
#include "laplacians.hpp"
#include "stochastic_phi4_dev.hpp"
#include "util.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <exception>
#include <iostream>
#include <vector>

namespace {

int run(const driver::Wave3DArgs &args) {
  const auto &grid = args.grid;
  const double dt = args.final_time / static_cast<double>(args.nt);
  const uint32_t snapshot_freq =
      driver::snapshot_frequency(args.nt, args.num_snapshots);

  Eigen::VectorXd u0 = driver::read_npy_checked<double>(
      args.input_u0, {grid.nz, grid.ny, grid.nx}, "u0");
  Eigen::VectorXd v0 = driver::read_npy_checked<double>(
      args.input_v0, {grid.nz, grid.ny, grid.nx}, "v0");
  Eigen::VectorXd m = driver::read_npy_checked<double>(
      args.m_file, {grid.nz, grid.ny, grid.nx}, "m");
  Eigen::VectorXd c = driver::read_npy_checked<double>(
      args.c_file, {grid.nz, grid.ny, grid.nx}, "c");

  const Eigen::SparseMatrix<double> L =
      (build_anisotropic_laplacian_noflux_3d<double>(
           grid.nx - 2, grid.ny - 2, grid.nz - 2, grid.dx, grid.dy, grid.dz, c))
          .eval();
  driver::DeviceCsrMatrix d_L(L);

  Eigen::VectorXd u_save(args.num_snapshots * grid.size());
  Eigen::VectorXd v_save(args.num_snapshots * grid.size());

  Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> u_save_mat(
      u_save.data(), args.num_snapshots, grid.size());
  Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> v_save_mat(
      v_save.data(), args.num_snapshots, grid.size());
  u_save_mat.row(0) = u0.transpose();
  v_save_mat.row(0) = v0.transpose();

  device::SP4SolverDevice::Parameters params(args.num_snapshots, snapshot_freq,
                                             10);
  device::SP4SolverDevice solver(d_L.row_ptr(), d_L.col_ind(), d_L.values(),
                                 m.data(), grid.size(), d_L.nnz(), u0.data(),
                                 v0.data(), dt, true, grid.Lx, params);

  driver::run_kge_steps(solver, args.nt, snapshot_freq, args.num_snapshots,
                        [](device::SP4SolverDevice &solver) { solver.step(); });

  solver.transfer_snapshots(u_save.data(), v_save.data());
  const std::vector<uint32_t> shape = {args.num_snapshots, grid.nz, grid.ny,
                                       grid.nx};
  save_to_npy(args.output_u, u_save, shape);
  save_to_npy(args.output_v, v_save, shape);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 16) {
    std::cerr << "Usage: " << argv[0] << driver::kWave3DUsage << '\n';
    return 1;
  }

  try {
    return run(driver::parse_wave_3d_args(argv));
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
