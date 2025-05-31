#ifndef DEVICE_SPMV_HPP
#define DEVICE_SPMV_HPP

#include "pragmas.hpp"
#include <thrust/complex.h>

template <typename T> class DeviceSpMV {
private:
  cusparseHandle_t handle_;
  cusparseSpMatDescr_t mat_desc_;
  cusparseDnVecDescr_t vec_x_, vec_y_;
  T *d_x_, *d_y_;
  void *buffer_{nullptr};
  uint32_t n_;
  uint32_t nnz_;

  cudaDataType cuda_data_type() const {
    if constexpr (std::is_same_v<T, float>)
      return CUDA_R_32F;
    if constexpr (std::is_same_v<T, double>)
      return CUDA_R_64F;
    if constexpr (std::is_same_v<T, thrust::complex<float>>)
      return CUDA_C_32F;
    if constexpr (std::is_same_v<T, thrust::complex<double>>)
      return CUDA_C_64F;
    throw std::runtime_error("Unsupported dtype");
  }

public:
  DeviceSpMV(const int *d_offsets, const int *d_columns, const T *d_values,
             uint32_t n, uint32_t nnz)
      : n_(n), nnz_(nnz) {
    CHECK_CUSPARSE(cusparseCreate(&handle_));
    CHECK_CUSPARSE(cusparseCreateCsr(
        &mat_desc_, n, n, nnz, (void *)d_offsets, (void *)d_columns,
        (void *)d_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, cuda_data_type()));

    CHECK_CUDA(cudaMalloc(&d_x_, n_ * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_y_, n_ * sizeof(T)));

    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_x_, n, d_x_, cuda_data_type()));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_y_, n, d_y_, cuda_data_type()));

    size_t buffer_size;
    T alpha(1), beta(0);
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_desc_, vec_x_,
        &beta, vec_y_, cuda_data_type(), CUSPARSE_SPMV_CSR_ALG2, &buffer_size));

    CHECK_CUDA(cudaMalloc(&buffer_, buffer_size));
  }

  ~DeviceSpMV() {
    if (buffer_)
      CHECK_CUDA(cudaFree(buffer_));
    CHECK_CUDA(cudaFree(d_x_));
    CHECK_CUDA(cudaFree(d_y_));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x_));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_y_));
    CHECK_CUSPARSE(cusparseDestroySpMat(mat_desc_));
    CHECK_CUSPARSE(cusparseDestroy(handle_));
  }

  void multiply(const T *d_x, T *d_y) {
    T alpha(1), beta(0);
    CHECK_CUSPARSE(cusparseDnVecSetValues(vec_x_, (void *)d_x));
    CHECK_CUSPARSE(cusparseDnVecSetValues(vec_y_, d_y));
    CHECK_CUSPARSE(cusparseSpMV(
        handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_desc_, vec_x_,
        &beta, vec_y_, cuda_data_type(), CUSPARSE_SPMV_CSR_ALG2, buffer_));
  }
};
#endif
