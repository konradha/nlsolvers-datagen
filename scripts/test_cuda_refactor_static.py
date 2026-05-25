from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_nlse_2d_sewi_driver_uses_sewi_step():
    source = read_text("device/drivers/nlse_cubic_sewi_driver_2d.cpp")
    assert "solver.step_sewi(dti, i, u_save.data());" in source
    loop_body = re.search(r"for \(uint32_t i = 1; i < nt; \+\+i\) \{(?P<body>.*?)\n  \}", source, re.S)
    assert loop_body is not None
    assert "solver.step(" not in loop_body.group("body")


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


def test_default_cuda_driver_set_stays_two_dimensional():
    root_cmake = read_text("CMakeLists.txt")
    drivers_cmake = read_text("device/drivers/CMakeLists.txt")
    assert 'option(BUILD_2D_DRIVERS "Build two-dimensional CUDA driver executables" ON)' in root_cmake
    assert 'option(BUILD_3D_DRIVERS "Build three-dimensional CUDA driver executables" OFF)' in root_cmake
    assert 'option(BUILD_AUXILIARY_DRIVERS "Build auxiliary CUDA check executables" OFF)' in root_cmake
    assert "if(BUILD_3D_DRIVERS)" in drivers_cmake
    assert "if(BUILD_AUXILIARY_DRIVERS)" in drivers_cmake


if __name__ == "__main__":
    tests = [
        test_nlse_2d_sewi_driver_uses_sewi_step,
        test_nlse_sewi_update_uses_exp_prev_first,
        test_kge_updates_are_fused,
        test_default_cuda_driver_set_stays_two_dimensional,
    ]
    for test in tests:
        test()
    print(f"{len(tests)} static CUDA refactor checks passed")
