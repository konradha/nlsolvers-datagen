#ifndef DRIVER_UTILS_HPP
#define DRIVER_UTILS_HPP

#include "util.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cuda_runtime.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace driver {

inline constexpr const char *kWave2DUsage =
    " nx ny Lx Ly input_u0.npy input_v0.npy output_traj.npy output_vel.npy "
    "T nt num_snapshots input_m.npy input_c.npy";

inline constexpr const char *kComplex2DUsage =
    " nx ny Lx Ly input_u0.npy output_traj.npy T nt num_snapshots "
    "input_m.npy input_c.npy";

inline constexpr const char *kWave3DUsage =
    " nx ny nz Lx Ly Lz input_u0.npy input_v0.npy output_traj.npy "
    "output_vel.npy T nt num_snapshots input_m.npy input_c.npy";

inline constexpr const char *kComplex3DUsage =
    " nx ny nz Lx Ly Lz input_u0.npy output_traj.npy T nt num_snapshots "
    "input_m.npy input_c.npy";

struct Grid2D {
  uint32_t nx;
  uint32_t ny;
  double Lx;
  double Ly;
  double dx;
  double dy;

  [[nodiscard]] uint32_t size() const { return nx * ny; }
  [[nodiscard]] std::vector<uint32_t> shape() const { return {ny, nx}; }
};

struct Grid3D {
  uint32_t nx;
  uint32_t ny;
  uint32_t nz;
  double Lx;
  double Ly;
  double Lz;
  double dx;
  double dy;
  double dz;

  [[nodiscard]] uint32_t size() const { return nx * ny * nz; }
  [[nodiscard]] std::vector<uint32_t> shape() const { return {nz, ny, nx}; }
};

struct Wave2DArgs {
  Grid2D grid;
  std::string input_u0;
  std::string input_v0;
  std::string output_u;
  std::string output_v;
  double final_time;
  uint32_t nt;
  uint32_t num_snapshots;
  std::string m_file;
  std::string c_file;
};

struct Complex2DArgs {
  Grid2D grid;
  std::string input_u0;
  std::string output_u;
  double final_time;
  uint32_t nt;
  uint32_t num_snapshots;
  std::string m_file;
  std::string c_file;
};

struct Wave3DArgs {
  Grid3D grid;
  std::string input_u0;
  std::string input_v0;
  std::string output_u;
  std::string output_v;
  double final_time;
  uint32_t nt;
  uint32_t num_snapshots;
  std::string m_file;
  std::string c_file;
};

struct Complex3DArgs {
  Grid3D grid;
  std::string input_u0;
  std::string output_u;
  double final_time;
  uint32_t nt;
  uint32_t num_snapshots;
  std::string m_file;
  std::string c_file;
};

[[nodiscard]] inline uint32_t parse_u32(const char *value, const char *name) {
  errno = 0;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string("Invalid uint32 argument for ") +
                                name + ": " + value);
  }
  return static_cast<uint32_t>(parsed);
}

[[nodiscard]] inline double parse_double(const char *value, const char *name) {
  errno = 0;
  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed)) {
    throw std::invalid_argument(std::string("Invalid floating-point argument for ") +
                                name + ": " + value);
  }
  return parsed;
}

[[nodiscard]] inline Grid2D make_grid_2d(uint32_t nx, uint32_t ny, double Lx,
                                         double Ly) {
  if (nx < 3 || ny < 3) {
    throw std::invalid_argument("Grid dimensions must be at least 3 in each direction");
  }
  if (Lx <= 0.0 || Ly <= 0.0) {
    throw std::invalid_argument("Domain half-lengths must be positive");
  }

  const double dx = 2.0 * Lx / static_cast<double>(nx - 1);
  const double dy = 2.0 * Ly / static_cast<double>(ny - 1);

  if (nx != ny || std::abs(dx - dy) >= 1e-10) {
    throw std::invalid_argument(
        "The current no-flux Laplacian implementation requires a square 2D grid");
  }

  return Grid2D{nx, ny, Lx, Ly, dx, dy};
}

[[nodiscard]] inline Grid3D make_grid_3d(uint32_t nx, uint32_t ny, uint32_t nz,
                                         double Lx, double Ly, double Lz) {
  if (nx < 3 || ny < 3 || nz < 3) {
    throw std::invalid_argument("Grid dimensions must be at least 3 in each direction");
  }
  if (Lx <= 0.0 || Ly <= 0.0 || Lz <= 0.0) {
    throw std::invalid_argument("Domain half-lengths must be positive");
  }

  const double dx = 2.0 * Lx / static_cast<double>(nx - 1);
  const double dy = 2.0 * Ly / static_cast<double>(ny - 1);
  const double dz = 2.0 * Lz / static_cast<double>(nz - 1);

  if (nx != ny || ny != nz || std::abs(dx - dy) >= 1e-10 ||
      std::abs(dy - dz) >= 1e-10) {
    throw std::invalid_argument(
        "The current no-flux Laplacian implementation requires a cubic 3D grid");
  }

  return Grid3D{nx, ny, nz, Lx, Ly, Lz, dx, dy, dz};
}

[[nodiscard]] inline Wave2DArgs parse_wave_2d_args(char **argv) {
  return Wave2DArgs{
      make_grid_2d(parse_u32(argv[1], "nx"), parse_u32(argv[2], "ny"),
                   parse_double(argv[3], "Lx"), parse_double(argv[4], "Ly")),
      argv[5],
      argv[6],
      argv[7],
      argv[8],
      parse_double(argv[9], "T"),
      parse_u32(argv[10], "nt"),
      parse_u32(argv[11], "num_snapshots"),
      argv[12],
      argv[13]};
}

[[nodiscard]] inline Complex2DArgs parse_complex_2d_args(char **argv) {
  return Complex2DArgs{
      make_grid_2d(parse_u32(argv[1], "nx"), parse_u32(argv[2], "ny"),
                   parse_double(argv[3], "Lx"), parse_double(argv[4], "Ly")),
      argv[5],
      argv[6],
      parse_double(argv[7], "T"),
      parse_u32(argv[8], "nt"),
      parse_u32(argv[9], "num_snapshots"),
      argv[10],
      argv[11]};
}

[[nodiscard]] inline Wave3DArgs parse_wave_3d_args(char **argv) {
  return Wave3DArgs{
      make_grid_3d(parse_u32(argv[1], "nx"), parse_u32(argv[2], "ny"),
                   parse_u32(argv[3], "nz"), parse_double(argv[4], "Lx"),
                   parse_double(argv[5], "Ly"), parse_double(argv[6], "Lz")),
      argv[7],
      argv[8],
      argv[9],
      argv[10],
      parse_double(argv[11], "T"),
      parse_u32(argv[12], "nt"),
      parse_u32(argv[13], "num_snapshots"),
      argv[14],
      argv[15]};
}

[[nodiscard]] inline Complex3DArgs parse_complex_3d_args(char **argv) {
  return Complex3DArgs{
      make_grid_3d(parse_u32(argv[1], "nx"), parse_u32(argv[2], "ny"),
                   parse_u32(argv[3], "nz"), parse_double(argv[4], "Lx"),
                   parse_double(argv[5], "Ly"), parse_double(argv[6], "Lz")),
      argv[7],
      argv[8],
      parse_double(argv[9], "T"),
      parse_u32(argv[10], "nt"),
      parse_u32(argv[11], "num_snapshots"),
      argv[12],
      argv[13]};
}

[[nodiscard]] inline uint32_t snapshot_frequency(uint32_t nt,
                                                 uint32_t num_snapshots) {
  if (nt == 0) {
    throw std::invalid_argument("nt must be positive");
  }
  if (num_snapshots == 0) {
    throw std::invalid_argument("num_snapshots must be positive");
  }
  if (num_snapshots > nt) {
    throw std::invalid_argument("num_snapshots must not exceed nt");
  }
  return nt / num_snapshots;
}

[[nodiscard]] inline std::string shape_to_string(const std::vector<uint32_t> &shape) {
  std::ostringstream stream;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      stream << "x";
    }
    stream << shape[i];
  }
  return stream.str();
}

[[nodiscard]] inline bool shape_matches(const std::vector<uint32_t> &actual,
                                        std::initializer_list<uint32_t> expected) {
  if (actual.size() != expected.size()) {
    return false;
  }

  auto expected_it = expected.begin();
  for (uint32_t dim : actual) {
    if (dim != *expected_it) {
      return false;
    }
    ++expected_it;
  }
  return true;
}

template <typename Scalar>
[[nodiscard]] Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
read_npy_checked(const std::string &path, std::initializer_list<uint32_t> expected,
                 const char *label) {
  std::vector<uint32_t> actual;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> values = read_from_npy<Scalar>(path, actual);
  if (!shape_matches(actual, expected)) {
    std::vector<uint32_t> expected_vec(expected.begin(), expected.end());
    throw std::runtime_error(std::string("Shape mismatch for ") + label +
                             ": expected " + shape_to_string(expected_vec) +
                             ", got " + shape_to_string(actual));
  }
  return values;
}

inline void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             cudaGetErrorString(status));
  }
}

class DeviceCsrMatrix {
public:
  explicit DeviceCsrMatrix(const Eigen::SparseMatrix<double> &matrix)
      : rows_(static_cast<uint32_t>(matrix.rows())),
        nnz_(static_cast<uint32_t>(matrix.nonZeros())) {
    static_assert(
        std::is_same<typename Eigen::SparseMatrix<double>::StorageIndex, int>::value,
        "DeviceCsrMatrix expects Eigen sparse matrices with int storage indices");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&row_ptr_),
                          (rows_ + 1) * sizeof(int)),
               "cudaMalloc(row_ptr)");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&col_ind_),
                          nnz_ * sizeof(int)),
               "cudaMalloc(col_ind)");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&values_),
                          nnz_ * sizeof(double)),
               "cudaMalloc(values)");

    check_cuda(cudaMemcpy(row_ptr_, matrix.outerIndexPtr(),
                          (rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy(row_ptr)");
    check_cuda(cudaMemcpy(col_ind_, matrix.innerIndexPtr(), nnz_ * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(col_ind)");
    check_cuda(cudaMemcpy(values_, matrix.valuePtr(), nnz_ * sizeof(double),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(values)");
  }

  DeviceCsrMatrix(const DeviceCsrMatrix &) = delete;
  DeviceCsrMatrix &operator=(const DeviceCsrMatrix &) = delete;

  ~DeviceCsrMatrix() {
    cudaFree(row_ptr_);
    cudaFree(col_ind_);
    cudaFree(values_);
  }

  [[nodiscard]] int *row_ptr() const { return row_ptr_; }
  [[nodiscard]] int *col_ind() const { return col_ind_; }
  [[nodiscard]] double *values() const { return values_; }
  [[nodiscard]] uint32_t nnz() const { return nnz_; }

private:
  int *row_ptr_ = nullptr;
  int *col_ind_ = nullptr;
  double *values_ = nullptr;
  uint32_t rows_ = 0;
  uint32_t nnz_ = 0;
};

template <typename Solver, typename Step>
void run_kge_steps(Solver &solver, uint32_t nt, uint32_t freq,
                   uint32_t num_snapshots, Step step) {
  for (uint32_t i = 1; i < nt; ++i) {
    step(solver);
    solver.apply_bc();
    if (i % freq == 0) {
      const uint32_t snapshot_idx = i / freq;
      if (snapshot_idx < num_snapshots) {
        solver.store_snapshot(snapshot_idx);
      }
    }
  }
}

template <typename Solver, typename Step>
void run_nlse_steps(Solver &solver, uint32_t nt, Step step) {
  for (uint32_t i = 1; i < nt; ++i) {
    step(solver, i);
    solver.apply_bc();
  }
}

} // namespace driver

#endif // DRIVER_UTILS_HPP
