#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <npy.hpp>

#define PROGRESS_BAR(i, total)                                                 \
  if ((i + 1) % (total / 100 + 1) == 0 || i + 1 == total) {                    \
    float progress = (float)(i + 1) / total;                                   \
    int barWidth = 70;                                                         \
    std::cout << "[";                                                          \
    int pos = barWidth * progress;                                             \
    for (int i = 0; i < barWidth; ++i) {                                       \
      if (i < pos)                                                             \
        std::cout << "=";                                                      \
      else if (i == pos)                                                       \
        std::cout << ">";                                                      \
      else                                                                     \
        std::cout << " ";                                                      \
    }                                                                          \
    std::cout << "] " << int(progress * 100.0) << "% \r";                      \
    std::cout.flush();                                                         \
    if (i + 1 == total)                                                        \
      std::cout << std::endl;                                                  \
  }

struct GridInfo {
  uint32_t nx = 0, ny = 0, nz = 0;
  double Lx = 0, Ly = 0, Lz = 0;
  double dx = 0, dy = 0, dz = 0;
  uint32_t total_size = 0;
  bool is_3d = true;
  uint32_t shape[3];
};

template <typename Float>
void save_to_npy(const std::string &filename, const Eigen::VectorX<Float> &data,
                 const std::vector<uint32_t> &shape) {
  std::vector<Float> vec(data.data(), data.data() + data.size());
  std::vector<uint64_t> shape_ul;
  for (const auto dim : shape)
    shape_ul.push_back(static_cast<uint64_t>(dim));
  npy::SaveArrayAsNumpy(filename, false, shape.size(), shape_ul.data(), vec);
}

template <typename Float>
Eigen::VectorX<Float> read_from_npy(const std::string &filename,
                                    std::vector<uint32_t> &shape) {
  std::vector<Float> data;
  std::vector<uint64_t> shape_ul;
  bool fortran_order;
  npy::LoadArrayFromNumpy(filename, shape_ul, fortran_order, data);
  shape.clear();
  for (const auto dim : shape_ul) {
    shape.push_back(static_cast<uint64_t>(dim));
  }
  return Eigen::Map<Eigen::VectorX<Float>>(data.data(), data.size());
}

template <typename T,
          typename IndexType = typename Eigen::SparseMatrix<T>::StorageIndex>
void save_sparse_matrix_csr_components_npy(const std::string &filename_base,
                                           const Eigen::SparseMatrix<T> &mat) {
  if (!mat.isCompressed()) {
    throw std::runtime_error("Matrix must be compressed for CSR saving.");
  }

  long long nnz = mat.nonZeros();
  long long outer_size = mat.outerSize();
  Eigen::VectorX<T> data_vec(nnz);
  for (uint32_t i = 0; i < nnz; ++i) {
    data_vec(i) = mat.valuePtr()[i];
  }

  Eigen::VectorX<uint32_t> indices_vec(nnz);
  for (uint32_t i = 0; i < nnz; ++i) {
    indices_vec(i) = static_cast<uint32_t>(mat.innerIndexPtr()[i]);
  }

  Eigen::VectorX<uint32_t> indptr_vec(outer_size + 1);
  for (uint32_t i = 0; i < outer_size + 1; ++i) {
    indptr_vec(i) = static_cast<uint32_t>(mat.outerIndexPtr()[i]);
  }

  Eigen::VectorX<uint32_t> shape_vec = Eigen::VectorX<uint32_t>::Zero(2);
  shape_vec(0) = mat.rows(); shape_vec(1) = mat.cols();

  save_to_npy(filename_base + "_data.npy",  data_vec, {(uint32_t)mat.nonZeros()});
  save_to_npy(filename_base + "_indices.npy",  indices_vec, {(uint32_t)mat.nonZeros()});
  save_to_npy(filename_base + "_indptr.npy",  indptr_vec, {(uint32_t)mat.outerSize() + 1});
  save_to_npy(filename_base + "_shape.npy",  shape_vec, {(uint32_t)(2)});
}

// needed for testing purposes
template <typename Scalar>
Eigen::VectorX<Scalar> create_centered_gaussian_3d(const GridInfo &grid,
                                                   double width,
                                                   double amplitude = 1.0) {
  Eigen::VectorX<Scalar> u0(grid.total_size);
  double width_sq = width * width;
  if (width_sq < 1e-15)
    width_sq = 1e-15;

  for (uint32_t k = 0; k < grid.nz; ++k) {
    double z = -grid.Lz + (k + 0.5) * grid.dz;
    for (uint32_t j = 0; j < grid.ny; ++j) {
      double y = -grid.Ly + (j + 0.5) * grid.dy;
      for (uint32_t i = 0; i < grid.nx; ++i) {
        double x = -grid.Lx + (i + 0.5) * grid.dx;
        long index = k * grid.nx * grid.ny + j * grid.nx + i;
        double r_sq = x * x + y * y + z * z;
        if constexpr (std::is_same_v<Scalar, double>) {
          u0(index) = amplitude * std::exp(-r_sq / width_sq);
        } else {
          u0(index) = Scalar(amplitude * std::exp(-r_sq / width_sq), 0.0);
        }
      }
    }
  }
  double norm_u = u0.norm();
  if (norm_u > 1e-15) {
    u0 /= norm_u;
  }
  return u0;
}


#endif
