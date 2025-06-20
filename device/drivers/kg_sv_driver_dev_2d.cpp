#include "laplacians.hpp"
#include "kg_dev.hpp"
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
              << " nx ny Lx Ly input_u0.npy input_v0.npy output_traj.npy "
                 "output_vel.npy T nt "
                 "num_snapshots input_m.npy input_c.npy\n";
    std::cerr << "Example: " << argv[0]
              << " 256 256 256 10.0 10.0 10.0 initial.npy velocity.npy "
                 "evolution_u.npy evolution_v.npy "
                 "1.5 500 100\n";
    std::cerr << "Example:" << argv[0]
              << " 256 256 256 10.0 10.0 10.0 initial.npy velocity.npy "
                 "evolution_u.npy evolution_v.npy "
                 "1.5 500 100 coupling.npy anisotropy.npy\n";
    return 1;
  }

  const uint32_t nx = std::stoul(argv[1]);
  const uint32_t ny = std::stoul(argv[2]);
  
  const double Lx = std::stod(argv[3]);
  const double Ly = std::stod(argv[4]);
  
  const std::string input_file = argv[5];
  const std::string input_velocity = argv[6];
  const std::string output_file = argv[7];
  const std::string output_vel = argv[8];
  const double T = std::stod(argv[9]);
  const uint32_t nt = std::stoul(argv[10]);
  const uint32_t num_snapshots = std::stoul(argv[11]);
  const std::string m_file = argv[12];
  const std::string c_file = argv[13];

  const double dx = 2 * Lx / (nx - 1);
  const double dy = 2 * Ly / (ny - 1);
  
  const double dt = T / nt;
  const auto freq = nt / num_snapshots;

  std::vector<uint32_t> input_shape;
  Eigen::VectorXd u0 = read_from_npy<double>(input_file, input_shape);
  Eigen::VectorXd v0 = read_from_npy<double>(input_velocity, input_shape);

  if (input_shape.size() != 2 || input_shape[0] != nx || input_shape[1] != ny) {
    std::cerr << "Error: Input array dimensions mismatch\n";
    std::cerr << "Expected: " << ny << "x" << nx << "\n";
    std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "\n";
    return 1;
  }

  Eigen::VectorXd u_past = u0 - dt * v0;
  Eigen::VectorXd m;
  Eigen::VectorXd c;

  try {
    std::vector<uint32_t> m_shape;
    m = read_from_npy<double>(m_file, m_shape);
    if (m_shape.size() != 2 || m_shape[0] != ny || m_shape[1] != nx) {
      std::cerr << "Error: Coupling array dimensions mismatch\n";
      std::cerr << "Expected: " << nx << "x" << ny << "\n";
      std::cerr << "Got: " << m_shape[0] << "x" << m_shape[1] << "\n";
      throw std::runtime_error(
          "Faulty m (1)"); // we don't default here from now on (3d cases only)
    }
  } catch (const std::exception &e) {
    std::cerr << "Error loading m(x, y): " << e.what() << "\n";
    throw std::runtime_error("Faulty m (2)");
  }

  try {
    std::vector<uint32_t> c_shape;
    c = read_from_npy<double>(c_file, c_shape);
    if (c_shape.size() != 2 || c_shape[0] != nx || c_shape[1] != ny) {
      std::cerr << "Error: Coupling array dimensions mismatch\n";
      std::cerr << "Expected: " << ny << "x" << nx << "\n";
      std::cerr << "Got: " << c_shape[0] << "x" << c_shape[1] << "\n";
      throw std::runtime_error(
          "Faulty m (1)"); // we don't default here from now on (3d cases only)
    }
  } catch (const std::exception &e) {
    std::cerr << "Error loading c(x, y): " << e.what() << "\n";
    throw std::runtime_error("Faulty c (2)");
  }

  const Eigen::SparseMatrix<double> L =
      (build_anisotropic_laplacian_noflux<double>(nx - 2, ny - 2, dx,
                                                     dy, c))
          .eval();

  Eigen::VectorXd u_save(num_snapshots * nx * ny);
  Eigen::VectorXd v_save(num_snapshots * nx * ny);

  Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> u_save_mat(
      u_save.data(), num_snapshots, nx * ny);
  Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> v_save_mat(
      v_save.data(), num_snapshots, nx * ny);

  u_save_mat.row(0) = u0.transpose();
  v_save_mat.row(0) = v0.transpose();

  Eigen::VectorXd u = u0;
  Eigen::VectorXd v = v0;
  

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
  device::KGESolverDevice::Parameters params(num_snapshots, freq, 10);
  bool is_3d = false;
  device::KGESolverDevice solver(d_row_ptr, d_col_ind, d_values, m.data(),
                                 nx * ny, L.nonZeros(), u0.data(),
                                 v0.data(), dt, is_3d, params);

  for (uint32_t i = 1; i < nt; ++i) {
    solver.step_sv();
    solver.apply_bc();
    if (i % freq == 0) {
      uint32_t snapshot_idx = i / freq;
      if (snapshot_idx < num_snapshots) {
        solver.store_snapshot(snapshot_idx);
      }
    }
  }

  solver.transfer_snapshots(u_save.data(), v_save.data());
  const std::vector<uint32_t> shape = {num_snapshots, ny, nx};
  save_to_npy(output_file, u_save, shape);
  save_to_npy(output_vel, v_save, shape);

  cudaFree(d_row_ptr);
  cudaFree(d_col_ind);
  cudaFree(d_values);
  return 0;
}
