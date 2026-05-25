from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVERS = [
    "kg_driver_dev_2d.cpp",
    "kg_sv_driver_dev_2d.cpp",
    "kg_driver_dev_3d.cpp",
    "kg_sv_driver_dev_3d.cpp",
    "nlse_cubic_driver_2d.cpp",
    "nlse_cubic_sewi_driver_2d.cpp",
    "nlse_cubic_driver_3d.cpp",
    "nlse_cubic_sewi_driver_3d.cpp",
    "sp4_sv_driver_dev_2d.cpp",
    "sp4_sv_driver_dev_3d.cpp",
    "kg_driver_dev.cpp",
    "nlse_driver_dev.cpp",
]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_refactored_drivers_use_shared_utilities():
    for driver in DRIVERS:
        source = read_text(f"device/drivers/{driver}")
        assert '#include "driver_utils.hpp"' in source, driver
        assert "read_npy_checked" in source, driver
        assert "DeviceCsrMatrix" in source or driver.startswith("nlse"), driver
        assert "cudaMalloc(&d_row_ptr" not in source, driver
        assert "cudaFree(d_row_ptr" not in source, driver


def test_nlse_sewi_drivers_use_sewi_step():
    for driver in ["nlse_cubic_sewi_driver_2d.cpp", "nlse_cubic_sewi_driver_3d.cpp"]:
        source = read_text(f"device/drivers/{driver}")
        assert "solver.step_sewi(dti, step, u_save.data());" in source, driver
        assert "solver.step(dti" not in source, driver


def test_nlse_sewi_update_uses_exp_prev_first():
    source = read_text("device/include/nlse_dev.hpp")
    assert "thrust::complex<double>* exp_phi_B = d_buf2_;" in source
    assert "thrust::complex<double>* exp_prev = d_buf3_;" in source
    assert "apply_sewi(d_u_, exp_prev, exp_phi_B, tau, n_);" in source


def test_kge_updates_are_fused():
    source = read_text("device/include/kg_single.cuh")
    assert source.count("thrust::make_zip_iterator") >= 2
    assert "void cubic_force(" in source
    assert "void update_velocity(" in source
    assert "struct GautschiUpdate" in source
    assert "struct StormerVerletUpdate" in source


def test_three_dimensional_paths_are_refactored_and_enabled():
    root_cmake = read_text("CMakeLists.txt")
    drivers_cmake = read_text("device/drivers/CMakeLists.txt")
    utilities = read_text("device/include/driver_utils.hpp")
    assert 'option(BUILD_3D_DRIVERS "Build three-dimensional CUDA driver executables" ON)' in root_cmake
    assert 'option(BUILD_STOCHASTIC_PHI4 "Build stochastic phi^4 CUDA driver executables" ON)' in root_cmake
    assert "if(BUILD_3D_DRIVERS)" in drivers_cmake
    assert "if(BUILD_STOCHASTIC_PHI4)" in drivers_cmake
    assert "parse_wave_3d_args" in utilities
    assert "parse_complex_3d_args" in utilities
    assert "2.0 * Lz" in utilities
    for driver in DRIVERS:
        source = read_text(f"device/drivers/{driver}")
        assert "2 * Ly / (nz - 1)" not in source, driver


def test_stochastic_phi4_drivers_are_in_build():
    drivers_cmake = read_text("device/drivers/CMakeLists.txt")
    assert "add_cuda_driver(sp4_sv_2d_dev sp4_sv_driver_dev_2d.cpp)" in drivers_cmake
    assert "add_cuda_driver(sp4_sv_3d_dev sp4_sv_driver_dev_3d.cpp)" in drivers_cmake


if __name__ == "__main__":
    tests = [
        test_refactored_drivers_use_shared_utilities,
        test_nlse_sewi_drivers_use_sewi_step,
        test_nlse_sewi_update_uses_exp_prev_first,
        test_kge_updates_are_fused,
        test_three_dimensional_paths_are_refactored_and_enabled,
        test_stochastic_phi4_drivers_are_in_build,
    ]
    for test in tests:
        test()
    print(f"{len(tests)} static CUDA refactor checks passed")
