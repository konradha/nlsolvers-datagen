# TODO add gautschi/sv drivers for all

function(add_cuda_driver name source)
    add_executable(${name} ${source})
    set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${name} PRIVATE device_solvers ${CUDA_LIBRARIES})
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    install(TARGETS ${name} RUNTIME DESTINATION bin)
endfunction()



if(BUILD_KG)
  add_cuda_driver(kg_gautschi_2d_dev kg_driver_dev_2d.cpp )
  add_cuda_driver(kg_sv_2d_dev kg_sv_driver_dev_2d.cpp)
  add_cuda_driver(kg_gautschi_3d_dev kg_driver_dev_3d.cpp)
  add_cuda_driver(kg_sv_3d_dev kg_sv_driver_dev_3d.cpp)
endif()

if(BUILD_NLSE_CUBIC)
  add_cuda_driver(nlse_3d_dev nlse_cubic_driver_3d.cpp)
  add_cuda_driver(nlse_sewi_3d_dev nlse_cubic_sewi_driver_3d.cpp)
  add_cuda_driver(nlse_2d_dev nlse_cubic_driver_2d.cpp)
  add_cuda_driver(nlse_sewi_2d_dev nlse_cubic_sewi_driver_2d.cpp)
endif()

# simple checks here!
add_cuda_driver(sp4_sv_2d_dev sp4_sv_driver_dev_2d.cpp)
add_cuda_driver(sp4_sv_3d_dev sp4_sv_driver_dev_3d.cpp)
