#ifndef STOCHASTIC_PHI4_CUH
#define STOCHASTIC_PHI4_CUH

#include "common_kernels.cuh"
#include "matfunc_real.hpp"
#include "pragmas.hpp"
#include "spmv.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <ctime>

namespace device {
	namespace SP4Solver {
		void step(double *d_v, double *d_u, double *d_u_past,
                         double *d_buf, double *d_buf2, double *d_buf3,
                         MatrixFunctionApplicatorReal *matfunc, const double *d_m,
                         const double tau, const uint32_t n, const double scaling_factor, const dim3 grid,
                         const dim3 block) {
    const double noise_strength = scaling_factor;
    const unsigned int seed = static_cast<unsigned int>(time(nullptr));

    thrust::device_ptr<double> d_u_ptr(d_u);
    thrust::device_ptr<double> d_buf3_ptr(d_buf3);
    thrust::copy(d_u_ptr, d_u_ptr + n, d_buf3_ptr);

    matfunc->expose_spmv()->multiply(d_u, d_buf);

    thrust::counting_iterator<unsigned int> index_seq(0);
    thrust::device_ptr<const double> d_m_ptr(d_m);
    thrust::device_ptr<double> d_buf2_ptr(d_buf2);

    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                         d_u_ptr, d_m_ptr, index_seq)),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         d_u_ptr + n, d_m_ptr + n, index_seq + n)),
                     d_buf2_ptr,
                     [noise_strength, seed] __device__ (const thrust::tuple<double, double, unsigned int>& t) {
                         double u_val = thrust::get<0>(t);
                         double m_val = thrust::get<1>(t);
                         unsigned int idx = thrust::get<2>(t);

                         thrust::default_random_engine rng(seed + idx);
                         thrust::normal_distribution<double> dist(0.0, 1.0);

                         double noise = noise_strength * dist(rng);

                         return -m_val * (u_val - u_val*u_val*u_val + noise);
                     });

    double dt_squared = tau * tau;
    thrust::device_ptr<double> d_buf_ptr(d_buf);
    thrust::transform(d_buf_ptr, d_buf_ptr + n, d_buf_ptr,
                    [dt_squared] __device__ (double val) { return dt_squared * val; });
    thrust::transform(d_buf2_ptr, d_buf2_ptr + n, d_buf2_ptr,
                    [dt_squared] __device__ (double val) { return dt_squared * val; });

    thrust::device_ptr<double> d_u_past_ptr(d_u_past);
    thrust::transform(d_u_ptr, d_u_ptr + n, d_u_past_ptr, d_u_ptr,
                    [] __device__ (double u, double u_past) { return 2.0 * u - u_past; });

    thrust::transform(d_u_ptr, d_u_ptr + n, d_buf_ptr, d_u_ptr,
                    [] __device__ (double u, double dt2_Lu) { return u + dt2_Lu; });
    thrust::transform(d_u_ptr, d_u_ptr + n, d_buf2_ptr, d_u_ptr,
                    [] __device__ (double u, double dt2_terms) { return u + dt2_terms; });

    thrust::copy(d_buf3_ptr, d_buf3_ptr + n, d_u_past_ptr);
    thrust::device_ptr<double> d_v_ptr(d_v);
    thrust::transform(d_u_ptr, d_u_ptr + n, d_u_past_ptr, d_v_ptr,
                    [tau] __device__ (double u, double u_past) {
                        return (u - u_past) / tau;
                    });
}
	}
}
#endif
