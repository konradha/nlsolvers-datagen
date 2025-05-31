set(COMMON_HEADERS
    include/laplacians.hpp
    include/util.hpp
)

add_library(common INTERFACE)
target_include_directories(common INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(common INTERFACE Eigen3::Eigen)

install(FILES ${COMMON_HEADERS}
    DESTINATION include/common
)
