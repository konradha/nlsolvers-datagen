# TODO add SV/Gautschi includes for all
set(DEVICE_HEADERS
    include/boundaries.cuh # templated boundary update
    include/common_kernels.cuh # mostly to collect velocity in real-wave integrators
    # others such as spmv.hpp omitted to avoid cluttering

    include/sg_single.cuh
    include/sg_single_dev.hpp
 
    include/kg_dev.hpp
    include/kg_single.cuh

    include/nlse.cuh
    include/nlse_dev.hpp

    include/nlse_cubic_quintic.cuh
    include/nlse_cubic_quintic_dev.hpp

    include/nlse_saturating.cuh
    include/nlse_saturating_dev.hpp

    include/phi4_dev.hpp
    include/phi4_single.cuh

    include/sg_double.cuh
    include/sg_double_dev.hpp

    include/sg_hyperbolic.cuh
    include/sg_hyperbolic_dev.hpp                                                        
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
