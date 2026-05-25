set(DEVICE_HEADERS
    include/boundaries.cuh
    include/common_kernels.cuh
    include/driver_utils.hpp
    include/kg_dev.hpp
    include/kg_single.cuh
    include/lanczos.hpp
    include/lanczos_complex.hpp
    include/matfunc.hpp
    include/matfunc_complex.hpp
    include/matfunc_complex_small.hpp
    include/matfunc_real.hpp
    include/nlse.cuh
    include/nlse_dev.hpp
    include/pragmas.hpp
    include/spmv.hpp
    include/stochastic_phi4.cuh
    include/stochastic_phi4_dev.hpp
)

add_library(device_solvers INTERFACE)
target_include_directories(device_solvers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(device_solvers INTERFACE common)

add_subdirectory(drivers)

install(FILES ${DEVICE_HEADERS}
    DESTINATION include/device
)
