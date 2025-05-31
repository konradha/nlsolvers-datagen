#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "util.hpp"
#include "laplacians.hpp"
#include "boundaries_3d.hpp"
#include "nlse_dev.hpp"
#include "eigen_krylov_complex.hpp"
#include "nlse_cubic_gautschi_solver.hpp" // Assuming host sEWI step is here
#include "nlse_cubic_solver_3d.hpp"     // Assuming host SS2 step is here


Eigen::VectorXcd create_gaussian_3d(int nx, int ny, int nz, double Lx, double Ly, double Lz, double width, double amplitude = 1.0) {
    Eigen::VectorXcd u0(nx * ny * nz);
    double dx = 2.0 * Lx / nx;
    double dy = 2.0 * Ly / ny;
    double dz = 2.0 * Lz / nz;
    double width_sq = width * width;

    #pragma omp parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double x = -Lx + i * dx;
                double y = -Ly + j * dy;
                double z = -Lz + k * dz;
                long index = k * nx * ny + j * nx + i;
                double r_sq = x * x + y * y + z * z;
                u0(index) = amplitude * exp(-r_sq / width_sq);
            }
        }
    }
    return u0;
}

double relative_error(const Eigen::VectorXcd& v1, const Eigen::VectorXcd& v2) {
    double norm_v1 = v1.norm();
    if (norm_v1 < 1e-15) {
        return (v1 - v2).norm();
    }
    return (v1 - v2).norm() / norm_v1;
}


int main(int argc, char **argv) {
    const int N = 80;
    const double L_domain = 0.1;
    const double T_final = 0.01;
    const int num_steps_total = 5;
    const std::vector<int> krylov_dims = {10, 20, 30};

    const uint32_t nx = N, ny = N, nz = N;
    const double Lx = L_domain, Ly = L_domain, Lz = L_domain;
    const double T = T_final;
    const uint32_t nt = num_steps_total;

    const double dx = 2.0 * Lx / nx;
    const double dy = 2.0 * Ly / ny;
    const double dz = 2.0 * Lz / nz;
    const double dt = T / nt;
    const std::complex<double> dti(0.0, dt);

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Grid: " << nx << "x" << ny << "x" << nz
              << ", Domain: [-" << Lx << ", " << Lx << "]^3" << std::endl;
    std::cout << "Time: T=" << T << ", nt=" << nt << ", dt=" << dt << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    Eigen::VectorXd m_coeff = Eigen::VectorXd::Constant(nx * ny * nz, -1.0);
    Eigen::VectorXd c_coeff = Eigen::VectorXd::Constant(nx * ny * nz, 1.0);

    const Eigen::SparseMatrix<std::complex<double>> L_op =
       (build_anisotropic_laplacian_noflux_3d<std::complex<double>>(
            nx, ny, nz, dx, dy, dz, c_coeff))
           .eval();

    double initial_width = Lx / 4.0;
    Eigen::VectorXcd u0 = create_gaussian_3d(nx, ny, nz, Lx, Ly, Lz, initial_width);
    double initial_mass = u0.squaredNorm() * dx * dy * dz;
    if (initial_mass > 1e-15) {
         u0 /= std::sqrt(initial_mass);
    }
     std::cout << "Initial Condition: 3D Gaussian (width=" << initial_width
               << "), Normalized to Mass=" << u0.squaredNorm() * dx * dy * dz << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;


    for (int m_krylov : krylov_dims) {
        std::cout << "Testing Krylov Dimension m = " << m_krylov << std::endl;

        std::cout << "  Running Host Simulation..." << std::flush;
        Eigen::VectorXcd u_host = u0;
        Eigen::VectorXcd u_prev_host = u0;
        Eigen::VectorXcd buf_host(u0.size());
        Eigen::VectorXcd rho_buf_host(u0.size());

        host::NLSESolver3d::step<std::complex<double>>(buf_host, rho_buf_host, u_host, L_op, m_coeff, dti);
        host::neumann_bc_no_velocity_3d<std::complex<double>>(u_host, nx, ny, nz);
        u_prev_host = u0;

        for (uint32_t i = 2; i <= nt; ++i) {
            host::NLSECubicGautschiSolver::step(buf_host, rho_buf_host, u_host, u_prev_host, L_op, m_coeff, dti, m_krylov);
            host::neumann_bc_no_velocity_3d<std::complex<double>>(u_host, nx, ny, nz);
        }
        std::cout << " Done." << std::endl;


        std::cout << "  Running Device Simulation..." << std::flush;
        device::NLSESolverDevice::Parameters params(1, 1, m_krylov);
        device::NLSESolverDevice solver(L_op, u0.data(), m_coeff.data(), params);


        for (uint32_t i = 1; i <= nt; ++i) {
            solver.step_sewi(dti, i, nullptr);
            solver.apply_bc();
        }

        Eigen::VectorXcd u_device(nx * ny * nz);
        // Add this method to NLSESolverDevice:
        // const thrust::complex<double>* expose_current_u() const { return d_u_; }
        cudaMemcpy(u_device.data(), solver.expose_current_u(),
                   nx * ny * nz * sizeof(std::complex<double>),
                   cudaMemcpyDeviceToHost);
        std::cout << " Done." << std::endl;


        double error = relative_error(u_host, u_device);
        std::cout << "  Relative Error (Host vs Device) after " << nt << " steps: "
                  << std::scientific << error << std::endl;

        int num_to_print = std::min(5, (int)(nx*ny*nz));
        std::cout << "  First " << num_to_print << " elements comparison:" << std::endl;
        for (int i = 0; i < num_to_print; ++i) {
             std::cout << "    host[" << i << "] = (" << u_host(i).real() << ", " << u_host(i).imag() << ")" << std::endl;
             std::cout << "    dev [" << i << "] = (" << u_device(i).real() << ", " << u_device(i).imag() << ")" << std::endl;
        }
         std::cout << "-------------------------------------------------" << std::endl;

    }

    return 0;
}
