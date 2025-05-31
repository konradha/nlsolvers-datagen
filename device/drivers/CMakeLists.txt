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

#if(BUILD_SG_SINGLE)
#    add_cuda_driver(sg_single_dev sg_single_driver_dev.cpp)
#endif()
#
#if(BUILD_SG_DOUBLE)
#    add_cuda_driver(sg_double_dev sg_double_driver_dev.cpp)
#endif()
#
#if(BUILD_SG_HYPERBOLIC)
#    add_cuda_driver(sg_hyperbolic_dev sg_hyperbolic_driver_dev.cpp)
#endif()
#
#if(BUILD_KG)
#    add_cuda_driver(kg_dev kg_driver_dev.cpp)
#    add_cuda_driver(kg_3d_dev kg_driver_dev_3d.cpp)
#    add_cuda_driver(kg_sv_3d_dev kg_sv_driver_dev_3d.cpp)
#endif()
#
#if(BUILD_PHI4)
#    add_cuda_driver(phi4_dev phi4_driver_dev.cpp)
#endif()
#
#if(BUILD_NLSE_CUBIC)
#    add_cuda_driver(nlse_dev nlse_driver_dev.cpp)
#
#    add_cuda_driver(nlse_3d_dev nlse_cubic_driver_3d.cpp)
#    add_cuda_driver(nlse_sewi_3d_dev nlse_cubic_sewi_driver_3d.cpp)
#endif()
#
#if(BUILD_NLSE_CUBIC_QUINTIC)
#    add_cuda_driver(nlse_cubic_quintic_dev nlse_cubic_quintic_driver_dev.cpp)
#endif()
#
#if(BUILD_NLSE_SATURATING)
#    add_cuda_driver(nlse_saturating_dev nlse_saturating_driver_dev.cpp)
#endif()
#
## add_cuda_driver(complex_test_matfunc test_matfunc_complex.cpp) 
## add_cuda_driver(nlse_step_test test_nlse_step.cpp)
#add_cuda_driver(scipy_comparison_dev test_scipy_matfunc.cpp)


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

add_cuda_driver(sp4_sv_2d_dev sp4_sv_driver_dev_2d.cpp)
add_cuda_driver(sp4_sv_3d_dev sp4_sv_driver_dev_3d.cpp)

add_cuda_driver(sge_sv_2d_dev sg_sv_driver_dev_2d.cpp)
add_cuda_driver(sge_sv_3d_dev sg_sv_driver_dev_3d.cpp)



