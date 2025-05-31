#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <complex>

#include <Eigen/Sparse>
#include <Eigen/Dense>

// host
#include "eigen_krylov_complex.hpp"

// dev
#include "matfunc_complex.hpp"
#include "spmv.hpp"


using Clock = std::chrono::high_resolution_clock;

template <typename Scalar_t>
Eigen::SparseMatrix<Scalar_t> create_test_matrix(int n, double density = 0.01) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    Eigen::SparseMatrix<Scalar_t> L(n, n);
    L.reserve(Eigen::VectorXi::Constant(n, static_cast<int>(n * density)));
    
    for (int i = 0; i < n; ++i) {
        L.insert(i, i) = Scalar_t(dist(gen), 0.0);
       
        for (int j = i + 1; j < std::min(i + 3, n); ++j) {
            if (dist(gen) > 1.0 - density) {
                Scalar_t val(dist(gen), 0.0);
                L.insert(i, j) = val;
                L.insert(j, i) = std::conj(val);
            }
        }
    }
    
    L.makeCompressed();
    return L;
}

template <typename Scalar_t>
Eigen::VectorX<Scalar_t> create_random_vector(int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    Eigen::VectorX<Scalar_t> v(n);
    for (int i = 0; i < n; ++i) {
        v(i) = Scalar_t(dist(gen), dist(gen));
    }
    
    return v;
}

void convert_to_csr(const Eigen::SparseMatrix<std::complex<double>> &mat,
                    std::vector<thrust::complex<double>> &values,
                    std::vector<int> &row_offsets,
                    std::vector<int> &col_indices) {
    Eigen::SparseMatrix<std::complex<double>> csr = mat;
    if (!mat.isCompressed() || mat.IsRowMajor == 0) {
        csr = mat;
        csr.makeCompressed();
    }
    
    int n = csr.rows();
    int nnz = csr.nonZeros();
    values.resize(nnz);
    row_offsets.resize(n + 1);
    col_indices.resize(nnz);
    
    for (int i = 0; i < nnz; ++i) {
        auto complex_val = *(csr.valuePtr() + i);
        values[i] = thrust::complex<double>(complex_val.real(), complex_val.imag());
    }
    
    for (int i = 0; i <= n; ++i) {
        row_offsets[i] = *(csr.outerIndexPtr() + i);
    }
    
    for (int i = 0; i < nnz; ++i) {
        col_indices[i] = *(csr.innerIndexPtr() + i);
    }
}

std::vector<thrust::complex<double>> eigen_to_thrust(const Eigen::VectorXcd &vec) {
    std::vector<thrust::complex<double>> result(vec.size());
    for (int i = 0; i < vec.size(); ++i) {
        result[i] = thrust::complex<double>(vec(i).real(), vec(i).imag());
    }
    return result;
}

Eigen::VectorXcd thrust_to_eigen(const std::vector<thrust::complex<double>> &vec) {
    Eigen::VectorXcd result(vec.size());
    for (int i = 0; i < vec.size(); ++i) {
        result(i) = std::complex<double>(vec[i].real(), vec[i].imag());
    }
    return result;
}

double relative_error(const Eigen::VectorXcd &v1, const Eigen::VectorXcd &v2) {
    return (v1 - v2).norm() / v1.norm();
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " matrix_size krylov_dim time_step num_tests" << std::endl;
        std::cerr << "Example: " << argv[0] << " 1000 30 0.1 5" << std::endl;
        return 1;
    }
    
    const int n = std::stoi(argv[1]);          // Matrix size
    const int m = std::stoi(argv[2]);          // Krylov dimension
    const double dt = std::stod(argv[3]);      // Time step
    const int num_tests = std::stoi(argv[4]);  // Number of test cases
    unsigned int seed = std::random_device{}();
    std::mt19937 gen(seed);
    std::cout << "Using seed: " << seed << std::endl;
    
    std::complex<double> complex_dt(0.0, dt);
    thrust::complex<double> thrust_dt(0.0, dt);
    
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    double total_error = 0.0;
    double max_error = 0.0;
    double avg_host_time = 0.0;
    double avg_device_time = 0.0;
    
    std::cout << "Running " << num_tests << " comparison tests with matrix size " << n << " and Krylov dimension " << m << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    for (int test = 0; test < num_tests; ++test) {
        std::cout << "Test " << (test + 1) << ":" << std::endl;
        auto L = create_test_matrix<std::complex<double>>(n, 0.01);
        auto u = create_random_vector<std::complex<double>>(n);
       
        u = u / u.norm();
        std::vector<thrust::complex<double>> values;
        std::vector<int> row_offsets;
        std::vector<int> col_indices;
        convert_to_csr(L, values, row_offsets, col_indices);
        
        auto u_thrust = eigen_to_thrust(u);
        thrust::complex<double> *d_values, *d_u, *d_result;
        int *d_row_offsets, *d_col_indices;
        
        cudaMalloc(&d_values, values.size() * sizeof(thrust::complex<double>));
        cudaMalloc(&d_row_offsets, row_offsets.size() * sizeof(int));
        cudaMalloc(&d_col_indices, col_indices.size() * sizeof(int));
        cudaMalloc(&d_u, n * sizeof(thrust::complex<double>));
        cudaMalloc(&d_result, n * sizeof(thrust::complex<double>));
        
        cudaMemcpy(d_values, values.data(), values.size() * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_offsets, row_offsets.data(), row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices, col_indices.data(), col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_u, u_thrust.data(), n * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
        DeviceSpMV<thrust::complex<double>> spmv(d_row_offsets, d_col_indices, d_values, n, L.nonZeros());
	MatrixFunctionApplicatorComplex matfunc_applicator(L, n, m, L.nonZeros());
        
        auto host_start = Clock::now();
        Eigen::VectorXcd host_result = expm_multiply(L, u, complex_dt, m);
        auto host_end = Clock::now();
        auto host_time = std::chrono::duration_cast<std::chrono::milliseconds>(host_end - host_start).count();
        avg_host_time += host_time;
        
        auto device_start = Clock::now();
        matfunc_applicator.apply(d_result, d_u, thrust_dt, "exp");
        cudaDeviceSynchronize();
        auto device_end = Clock::now();
        auto device_time = std::chrono::duration_cast<std::chrono::milliseconds>(device_end - device_start).count();
        avg_device_time += device_time;
        
        std::vector<thrust::complex<double>> device_result_thrust(n);
        cudaMemcpy(device_result_thrust.data(), d_result, n * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);
        Eigen::VectorXcd device_result = thrust_to_eigen(device_result_thrust);
        double error = relative_error(host_result, device_result);
        total_error += error;
        max_error = std::max(max_error, error);
        
        std::cout << "  Host time:   " << host_time << " ms" << std::endl;
        std::cout << "  Device time: " << device_time << " ms" << std::endl;
        std::cout << "  Speedup:     " << static_cast<double>(host_time) / device_time << "x" << std::endl;
        std::cout << "  Rel. error:  " << error << std::endl;

        int num_to_print = std::min(5, n);
        std::cout << "  First " << num_to_print << " elements comparison:" << std::endl;
        for (int i = 0; i < num_to_print; ++i) {
            std::cout << "    host[" << i << "] = (" << host_result(i).real() << ", " << host_result(i).imag() << ")" << std::endl;
            std::cout << "    dev [" << i << "] = (" << device_result(i).real() << ", " << device_result(i).imag() << ")" << std::endl;
        }
        std::cout << std::endl;
        cudaFree(d_values);
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_u);
        cudaFree(d_result);
    }
    
    std::cout << "=========================================================" << std::endl;
    std::cout << "Test summary:" << std::endl;
    std::cout << "  Average relative error: " << total_error / num_tests << std::endl;
    std::cout << "  Maximum relative error: " << max_error << std::endl;
    std::cout << "  Average host time:   " << avg_host_time / num_tests << " ms" << std::endl;
    std::cout << "  Average device time: " << avg_device_time / num_tests << " ms" << std::endl;
    std::cout << "  Average speedup:     " << (avg_host_time / avg_device_time) << "x" << std::endl;
    cublasDestroy(cublas_handle);
    
    return 0;
}
