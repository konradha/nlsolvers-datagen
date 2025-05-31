#ifndef LAPLACIANS_HPP
#define LAPLACIANS_HPP

#include <Eigen/Sparse>

// TODO: finish both Sommerfeld radiation BCs and PBCs
// adapt from pytorch!

template <typename Float>
Eigen::SparseMatrix<Float> build_laplacian_noflux(uint32_t nx, uint32_t ny,
                                                  Float dx, Float dy) {
  // for this sparse solver we always assume nx == ny, dx == dy
  // which could of course be adapted in the future
  assert(nx == ny);
  assert(std::abs(dx - dy) < 1e-10);

  const uint32_t N = (nx + 2) * (nx + 2);
  const uint32_t nnz = N + 4 * (N - 1) - 4;
  using T = Eigen::Triplet<Float>;
  std::vector<T> triplets;
  triplets.reserve(nnz);

  for (uint32_t i = 0; i < N; ++i) {
    Float val = static_cast<Float>(-4.0);
    if (i < (nx + 2) || i >= N - (nx + 2) || i % (nx + 2) == 0 ||
        i % (nx + 2) == (nx + 1)) {
      val = static_cast<Float>(-3.0);
    }
    triplets.emplace_back(i, i, val);
  }

  for (uint32_t i = 0; i < N - 1; ++i) {
    if ((i + 1) % (nx + 2) != 0) {
      triplets.emplace_back(i, i + 1, static_cast<Float>(1.0));
      triplets.emplace_back(i + 1, i, static_cast<Float>(1.0));
    }
  }

  for (uint32_t i = 0; i < N - (nx + 2); ++i) {
    triplets.emplace_back(i, i + (nx + 2), static_cast<Float>(1.0));
    triplets.emplace_back(i + (nx + 2), i, static_cast<Float>(1.0));
  }

  // missing: corners!

  Eigen::SparseMatrix<Float> L(N, N);
  L.setFromTriplets(triplets.begin(), triplets.end());
  L.makeCompressed();
  L *= static_cast<Float>(1.0) / (dx * dy);

  return L;
}

template <typename Float>
Eigen::SparseMatrix<Float>
build_anisotropic_laplacian_noflux(uint32_t nx, uint32_t ny, Float dx, Float dy,
                                   const Eigen::VectorX<Float> &c) {
  /*
   * Let's hope this works: Assume div(c(x,y)grad(u))
   * Then we make use of averaged cell values for c_ij (a finite volume
   * approach).
   */
  assert(nx == ny);
  assert(std::abs(dx - dy) < 1e-10);
  assert(c.size() == (nx + 2) * (nx + 2));

  const uint32_t N = (nx + 2) * (nx + 2);
  const uint32_t nnz = N + 4 * (N - 1) - 4;
  using T = Eigen::Triplet<Float>;
  std::vector<T> triplets;
  triplets.reserve(nnz);

  std::vector<Float> diagonal_sums(N, 0.0);

  for (uint32_t i = 0; i < N - 1; ++i) {
    if ((i + 1) % (nx + 2) != 0) {
      Float avg_c = (c[i] + c[i + 1]) / 2.0;
      triplets.emplace_back(i, i + 1, avg_c);
      triplets.emplace_back(i + 1, i, avg_c);
      diagonal_sums[i] += avg_c;
      diagonal_sums[i + 1] += avg_c;
    }
  }

  for (uint32_t i = 0; i < N - (nx + 2); ++i) {
    Float avg_c = (c[i] + c[i + (nx + 2)]) / 2.0;
    triplets.emplace_back(i, i + (nx + 2), avg_c);
    triplets.emplace_back(i + (nx + 2), i, avg_c);
    diagonal_sums[i] += avg_c;
    diagonal_sums[i + (nx + 2)] += avg_c;
  }

  for (uint32_t i = 0; i < N; ++i) {
    triplets.emplace_back(i, i, -diagonal_sums[i]);
  }

  Eigen::SparseMatrix<Float> L(N, N);
  L.setFromTriplets(triplets.begin(), triplets.end());
  L.makeCompressed();
  L *= static_cast<Float>(1.0) / (dx * dy);

  return L;
}

template <typename Float>
Eigen::SparseMatrix<Float> build_laplacian_noflux_3d(uint32_t nx, uint32_t ny,
                                                     uint32_t nz, Float dx,
                                                     Float dy, Float dz) {
  assert(nx == ny && ny == nz);
  assert(std::abs(dx - dy) < 1e-10 && std::abs(dy - dz) < 1e-10);

  const uint32_t plane_size = (nx + 2) * (ny + 2);
  const uint32_t N = plane_size * (nz + 2);
  const uint32_t nnz = N + 6 * (N - 1);

  using T = Eigen::Triplet<Float>;
  std::vector<T> triplets;
  triplets.reserve(nnz);

  for (uint32_t k = 0; k < nz + 2; ++k) {
    for (uint32_t j = 0; j < ny + 2; ++j) {
      for (uint32_t i = 0; i < nx + 2; ++i) {
        const uint32_t idx = k * plane_size + j * (nx + 2) + i;
        const Float val = (k == 0 || k == nz + 1 || j == 0 || j == ny + 1 ||
                           i == 0 || i == nx + 1)
                              ? static_cast<Float>(-5.0)
                              : static_cast<Float>(-6.0);
        triplets.emplace_back(idx, idx, val);
      }
    }
  }

  for (uint32_t i = 0; i < N - 1; ++i) {
    if ((i + 1) % (nx + 2) != 0) {
      triplets.emplace_back(i, i + 1, static_cast<Float>(1.0));
      triplets.emplace_back(i + 1, i, static_cast<Float>(1.0));
    }
  }

  for (uint32_t i = 0; i < N - (nx + 2); ++i) {
    triplets.emplace_back(i, i + (nx + 2), static_cast<Float>(1.0));
    triplets.emplace_back(i + (nx + 2), i, static_cast<Float>(1.0));
  }

  for (uint32_t i = 0; i < N - plane_size; ++i) {
    triplets.emplace_back(i, i + plane_size, static_cast<Float>(1.0));
    triplets.emplace_back(i + plane_size, i, static_cast<Float>(1.0));
  }

  Eigen::SparseMatrix<Float> L(N, N);
  L.setFromTriplets(triplets.begin(), triplets.end());
  L.makeCompressed();
  L *= static_cast<Float>(1.0) / (dx * dx);

  return L;
}

template <typename Float>
Eigen::SparseMatrix<Float>
build_anisotropic_laplacian_noflux_3d(uint32_t nx, uint32_t ny, uint32_t nz,
                                      Float dx, Float dy, Float dz,
                                      const Eigen::VectorX<Float> &c) {
  /*
   * Similar to the case above. Assume div(c(x,y,z)grad(u))
   *  = (c u_x)_x + (c u_y)_y + (c u_)z
   *  {(cu_x)_x}_ijk = ((cu_x)i+1/2,j,k - (cu_x)i-1/2,j,k) / dx etc
   */
  assert(nx == ny && ny == nz);
  assert(std::abs(dx - dy) < 1e-10 && std::abs(dy - dz) < 1e-10);
  assert(c.size() == (nx + 2) * (ny + 2) * (nz + 2));

  const uint32_t plane_size = (nx + 2) * (ny + 2);
  const uint32_t N = plane_size * (nz + 2);
  const uint32_t nnz = N + 6 * (N - 1);

  using T = Eigen::Triplet<Float>;
  std::vector<T> triplets;
  triplets.reserve(nnz);

  std::vector<Float> diagonal_sums(N, 0.0);

  for (uint32_t i = 0; i < N - 1; ++i) {
    if ((i + 1) % (nx + 2) != 0) {
      Float avg_c = (c[i] + c[i + 1]) / 2.0;
      triplets.emplace_back(i, i + 1, avg_c);
      triplets.emplace_back(i + 1, i, avg_c);
      diagonal_sums[i] += avg_c;
      diagonal_sums[i + 1] += avg_c;
    }
  }

  for (uint32_t i = 0; i < N - (nx + 2); ++i) {
    Float avg_c = (c[i] + c[i + (nx + 2)]) / 2.0;
    triplets.emplace_back(i, i + (nx + 2), avg_c);
    triplets.emplace_back(i + (nx + 2), i, avg_c);
    diagonal_sums[i] += avg_c;
    diagonal_sums[i + (nx + 2)] += avg_c;
  }

  for (uint32_t i = 0; i < N - plane_size; ++i) {
    Float avg_c = (c[i] + c[i + plane_size]) / 2.0;
    triplets.emplace_back(i, i + plane_size, avg_c);
    triplets.emplace_back(i + plane_size, i, avg_c);
    diagonal_sums[i] += avg_c;
    diagonal_sums[i + plane_size] += avg_c;
  }

  for (uint32_t i = 0; i < N; ++i) {
    triplets.emplace_back(i, i, -diagonal_sums[i]);
  }

  Eigen::SparseMatrix<Float> L(N, N);
  L.setFromTriplets(triplets.begin(), triplets.end());
  L.makeCompressed();
  L *= static_cast<Float>(1.0) / (dx * dx);

  return L;
}

template <typename Float>
std::pair<Eigen::SparseMatrix<Float>, Eigen::SparseMatrix<Float>>
build_separated_laplacian_noflux(uint32_t nx, uint32_t ny, Float dx, Float dy) {
  assert(nx == ny);
  assert(std::abs(dx - dy) < 1e-10);

  const uint32_t N = (nx + 2) * (nx + 2);
  using T = Eigen::Triplet<Float>;
  std::vector<T> triplets_x, triplets_y;
  triplets_x.reserve(N * 3);
  triplets_y.reserve(N * 3);

  for (uint32_t j = 0; j < nx + 2; ++j) {
    for (uint32_t i = 0; i < nx + 2; ++i) {
      uint32_t idx = j * (nx + 2) + i;
      Float val_x = static_cast<Float>(-2.0);
      Float val_y = static_cast<Float>(-2.0);

      bool is_corner = (i == 0 || i == nx + 1) && (j == 0 || j == nx + 1);
      if (i == 0 || i == nx + 1)
        val_x = is_corner ? static_cast<Float>(-1.5) : static_cast<Float>(-1.0);
      if (j == 0 || j == nx + 1)
        val_y = is_corner ? static_cast<Float>(-1.5) : static_cast<Float>(-1.0);

      triplets_x.emplace_back(idx, idx, val_x);
      triplets_y.emplace_back(idx, idx, val_y);

      if (i < nx + 1) {
        triplets_x.emplace_back(idx, idx + 1, static_cast<Float>(1.0));
        triplets_x.emplace_back(idx + 1, idx, static_cast<Float>(1.0));
      }
      if (j < nx + 1) {
        triplets_y.emplace_back(idx, idx + nx + 2, static_cast<Float>(1.0));
        triplets_y.emplace_back(idx + nx + 2, idx, static_cast<Float>(1.0));
      }
    }
  }

  Eigen::SparseMatrix<Float> Lx(N, N), Ly(N, N);
  Lx.setFromTriplets(triplets_x.begin(), triplets_x.end());
  Ly.setFromTriplets(triplets_y.begin(), triplets_y.end());

  Lx.makeCompressed();
  Ly.makeCompressed();

  Lx *= static_cast<Float>(1.0) / (dx * dx);
  Ly *= static_cast<Float>(1.0) / (dy * dy);

  return {Lx, Ly};
}

template <typename Float>
Eigen::SparseMatrix<Float> build_xxxx_noflux(uint32_t nx, uint32_t ny,
                                             Float dx) {
  assert(nx == ny);
  const uint32_t N = (nx + 2) * (nx + 2);
  using T = Eigen::Triplet<Float>;
  std::vector<T> triplets;
  triplets.reserve(N * 5);

  for (uint32_t j = 0; j < nx + 2; ++j) {
    for (uint32_t i = 0; i < nx + 2; ++i) {
      uint32_t idx = j * (nx + 2) + i;

      if (i == 0 || i == nx + 1) {
        // boundary: no-flux
        triplets.emplace_back(idx, idx, static_cast<Float>(2.0));
        triplets.emplace_back(idx, i == 0 ? idx + 1 : idx - 1,
                              static_cast<Float>(-2.0));
      } else if (i == 1 || i == nx) {
        // close-to-boundary: second-order
        triplets.emplace_back(idx, i == 1 ? idx + 1 : idx - 1,
                              static_cast<Float>(-2.0));
        triplets.emplace_back(idx, idx, static_cast<Float>(4.0));
        triplets.emplace_back(idx, i == 1 ? idx + 2 : idx - 2,
                              static_cast<Float>(-2.0));
      } else {
        // interior: 5-point stencil
        triplets.emplace_back(idx, idx - 2, static_cast<Float>(1.0));
        triplets.emplace_back(idx, idx - 1, static_cast<Float>(-4.0));
        triplets.emplace_back(idx, idx, static_cast<Float>(6.0));
        triplets.emplace_back(idx, idx + 1, static_cast<Float>(-4.0));
        triplets.emplace_back(idx, idx + 2, static_cast<Float>(1.0));
      }
    }
  }

  Eigen::SparseMatrix<Float> L4(N, N);
  L4.setFromTriplets(triplets.begin(), triplets.end());
  L4.makeCompressed();
  L4 *= static_cast<Float>(1.0) / (dx * dx * dx * dx);

  return L4;
}

#endif
