#include "boundaries_3d.hpp"
#include "laplacians.hpp"
#include "nlse_dev.hpp"
#include "util.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// We'll assume to be taking first-order approximants for the velocity for now
// for all equations in u_tt

int main(int argc, char **argv) {
  if (argc != 14) {
    std::cerr << "Usage: " << argv[0]
              << " nx ny nz Lx Ly Lz input_u0.npy output_traj.npy T nt "
                 "num_snapshots input_m.npy input_c.npy\n";
    std::cerr << "Example: " << argv[0]
              << " 256 256 256 10.0 10.0 10.0 initial.npy evolution_u.npy "
                 "1.5 500 100\n";
    std::cerr << "Example:" << argv[0]
              << " 256 256 256 10.0 10.0 10.0 initial.npy evolution_u.npy "
                 "1.5 500 100 coupling.npy anisotropy.npy\n";
    return 1;
  }

  const uint32_t nx = std::stoul(argv[1]);
  const uint32_t ny = std::stoul(argv[2]);
  const uint32_t nz = std::stoul(argv[3]);
  const double Lx = std::stod(argv[4]);
  const double Ly = std::stod(argv[5]);
  const double Lz = std::stod(argv[6]);
  const std::string input_file = argv[7];
  const std::string output_file = argv[8];
  const double T = std::stod(argv[9]);
  const uint32_t nt = std::stoul(argv[10]);
  const uint32_t num_snapshots = std::stoul(argv[11]);
  const std::string m_file = argv[12];
  const std::string c_file = argv[13];

  const double dx = 2 * Lx / (nx - 1);
  const double dy = 2 * Ly / (ny - 1);
  const double dz = 2 * Ly / (nz - 1);
  const double dt = T / nt;
  const std::complex<double> dti(0, dt);
  const auto freq = nt / num_snapshots;

  std::vector<uint32_t> input_shape;
  Eigen::VectorXcd u0 = read_from_npy<std::complex<double>>(input_file, input_shape);

  if (input_shape.size() != 3 || input_shape[0] != nz || input_shape[1] != ny ||
      input_shape[2] != nx) {
    std::cerr << "Error: Input array dimensions mismatch\n";
    std::cerr << "Expected: " << ny << "x" << nx << "\n";
    std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "x"
              << input_shape[2] << "\n";
    return 1;
  }

  Eigen::VectorXd m;
  Eigen::VectorXd c;

  try {
    std::vector<uint32_t> m_shape;
    m = read_from_npy<double>(m_file, m_shape);
    if (m_shape.size() != 3 || m_shape[0] != nz || m_shape[1] != ny ||
        m_shape[2] != nx) {
      std::cerr << "Error: Coupling array dimensions mismatch\n";
      std::cerr << "Expected: " << nz << "x" << ny << "x" << nx << "\n";
      std::cerr << "Got: " << m_shape[0] << "x" << m_shape[1] << "x"
                << m_shape[2] << "\n";
      throw std::runtime_error(
          "Faulty m (1)"); // we don't default here from now on (3d cases only)
    }
  } catch (const std::exception &e) {
    std::cerr << "Error loading m(x, y, z): " << e.what() << "\n";
    throw std::runtime_error("Faulty m (2)");
  }

  try {
    std::vector<uint32_t> c_shape;
    c = read_from_npy<double>(c_file, c_shape);
    if (c_shape.size() != 3 || c_shape[0] != nz || c_shape[1] != ny ||
        c_shape[2] != nx) {
      std::cerr << "Error: Coupling array dimensions mismatch\n";
      std::cerr << "Expected: " << nz << "x" << ny << "x" << nx << "\n";
      std::cerr << "Got: " << c_shape[0] << "x" << c_shape[1] << "x"
                << c_shape[2] << "\n";
      throw std::runtime_error(
          "Faulty m (1)"); // we don't default here from now on (3d cases only)
    }
  } catch (const std::exception &e) {
    std::cerr << "Error loading c(x, y, z): " << e.what() << "\n";
    throw std::runtime_error("Faulty c (2)");
  }

  const Eigen::SparseMatrix<std::complex<double>> L =
      (build_anisotropic_laplacian_noflux_3d<std::complex<double>>(
           nx - 2, ny - 2, nz - 2, dx, dy, dz, c))
          .eval();
  Eigen::VectorXcd u_save(num_snapshots * nx * ny * nz);

  Eigen::Map<Eigen::Matrix<std::complex<double>, -1, -1, Eigen::RowMajor>>
      u_save_mat(u_save.data(), num_snapshots, nx * ny * nz);

  bool is_3d = true;
  device::NLSESolverDevice::Parameters params(num_snapshots, freq, 25);
  device::NLSESolverDevice solver(L, u0.data(), m.data(), is_3d, params);
  solver.store_snapshot_online(u_save.data()); // this should happen inside constructor ... maybe refactor later
  for (uint32_t i = 1; i < nt; ++i) {
    solver.step(dti, i, u_save.data());
    solver.apply_bc();
  }

  const std::vector<uint32_t> shape = {num_snapshots, nz, ny, nx};
  save_to_npy(output_file, u_save, shape);
  return 0;
}
