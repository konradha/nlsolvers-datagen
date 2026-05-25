#include "driver_utils.hpp"
#include "laplacians.hpp"
#include "nlse_dev.hpp"
#include "util.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <complex>
#include <exception>
#include <iostream>
#include <vector>

namespace {

int run(const driver::Complex2DArgs &args) {
  const auto &grid = args.grid;
  const double dt = args.final_time / static_cast<double>(args.nt);
  const uint32_t snapshot_freq =
      driver::snapshot_frequency(args.nt, args.num_snapshots);
  const std::complex<double> dti(0.0, dt);

  Eigen::VectorXcd u0 = driver::read_npy_checked<std::complex<double>>(
      args.input_u0, {grid.ny, grid.nx}, "u0");
  Eigen::VectorXd m = driver::read_npy_checked<double>(
      args.m_file, {grid.ny, grid.nx}, "m");
  Eigen::VectorXd c = driver::read_npy_checked<double>(
      args.c_file, {grid.ny, grid.nx}, "c");
  const Eigen::VectorXcd c_complex = c.cast<std::complex<double>>();

  const Eigen::SparseMatrix<std::complex<double>> L =
      (build_anisotropic_laplacian_noflux<std::complex<double>>(
           grid.nx - 2, grid.ny - 2, grid.dx, grid.dy, c_complex))
          .eval();

  Eigen::VectorXcd u_save(args.num_snapshots * grid.size());

  device::NLSESolverDevice::Parameters params(args.num_snapshots,
                                              snapshot_freq, 10);
  device::NLSESolverDevice solver(L, u0.data(), m.data(), false, params);
  solver.store_snapshot_online(u_save.data());

  driver::run_nlse_steps(
      solver, args.nt,
      [dti, &u_save](device::NLSESolverDevice &solver, uint32_t step) {
        solver.step_sewi(dti, step, u_save.data());
      });

  const std::vector<uint32_t> shape = {args.num_snapshots, grid.ny, grid.nx};
  save_to_npy(args.output_u, u_save, shape);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 12) {
    std::cerr << "Usage: " << argv[0] << driver::kComplex2DUsage << '\n';
    return 1;
  }

  try {
    return run(driver::parse_complex_2d_args(argv));
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
