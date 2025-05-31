#ifndef BOUNDARIES_CUH
#define BOUNDARIES_CUH

#include <cuda.h>
#include <cuda_runtime.h>

// benchmarked to be fastest for small (nx, ny <= 5000) grid sizes; after that
// considerable overhead and we should think about how to better employ warps
template <typename Float>
void neumann_bc_no_velocity_blocking(Float *u_dev, uint32_t nx, uint32_t ny) {
  cudaMemcpy(u_dev + 0 * ny + 1, u_dev + 1 * ny + 1, (ny - 2) * sizeof(Float),
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(u_dev + (nx - 1) * ny + 1, u_dev + (nx - 2) * ny + 1,
             (ny - 2) * sizeof(Float), cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(u_dev + 0, ny * sizeof(Float), u_dev + 1, ny * sizeof(Float),
               sizeof(Float), nx, cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(u_dev + (ny - 1), ny * sizeof(Float), u_dev + (ny - 2),
               ny * sizeof(Float), sizeof(Float), nx, cudaMemcpyDeviceToDevice);
}

// Experimental 3d version of the above. Needs testing and maybe benchmarking
// against other implementations.
template <typename Float>
void neumann_bc_no_velocity_blocking_3d(Float *u_dev, uint32_t nx, uint32_t ny,
                                      uint32_t nz) {
  cudaExtent extent = make_cudaExtent((nz - 2) * sizeof(Float), ny - 2, 1);
  
  cudaMemcpy3DParms xMinParams = {0};
  xMinParams.srcPtr = make_cudaPitchedPtr(u_dev + 1 * ny * nz + 1 * nz + 1,
                                          nz * sizeof(Float), nz - 2, ny - 2);
  xMinParams.dstPtr = make_cudaPitchedPtr(u_dev + 0 * ny * nz + 1 * nz + 1,
                                          nz * sizeof(Float), nz - 2, ny - 2);
  xMinParams.extent = extent;
  xMinParams.kind = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&xMinParams);

  cudaMemcpy3DParms xMaxParams = {0};
  xMaxParams.srcPtr = make_cudaPitchedPtr(u_dev + (nx - 2) * ny * nz + 1 * nz + 1,
                                          nz * sizeof(Float), nz - 2, ny - 2);
  xMaxParams.dstPtr = make_cudaPitchedPtr(u_dev + (nx - 1) * ny * nz + 1 * nz + 1,
                                          nz * sizeof(Float), nz - 2, ny - 2);
  xMaxParams.extent = extent;
  xMaxParams.kind = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&xMaxParams);

  cudaMemcpy3DParms yMinParams = {0};
  yMinParams.srcPtr = make_cudaPitchedPtr(u_dev + 0 * ny * nz + 1 * nz + 1,
                                          nz * sizeof(Float), nz - 2, nx);
  yMinParams.dstPtr = make_cudaPitchedPtr(u_dev + 0 * ny * nz + 0 * nz + 1,
                                          nz * sizeof(Float), nz - 2, nx);
  yMinParams.extent = make_cudaExtent((nz - 2) * sizeof(Float), 1, nx);
  yMinParams.kind = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&yMinParams);

  cudaMemcpy3DParms yMaxParams = {0};
  yMaxParams.srcPtr = make_cudaPitchedPtr(u_dev + 0 * ny * nz + (ny - 2) * nz + 1,
                                          nz * sizeof(Float), nz - 2, nx);
  yMaxParams.dstPtr = make_cudaPitchedPtr(u_dev + 0 * ny * nz + (ny - 1) * nz + 1,
                                          nz * sizeof(Float), nz - 2, nx);
  yMaxParams.extent = make_cudaExtent((nz - 2) * sizeof(Float), 1, nx);
  yMaxParams.kind = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&yMaxParams);

  cudaMemcpy3DParms zMinParams = {0};
  zMinParams.srcPtr = make_cudaPitchedPtr(u_dev + 0 * ny * nz + 0 * nz + 1,
                                          nz * sizeof(Float), 1, ny);
  zMinParams.dstPtr = make_cudaPitchedPtr(u_dev + 0 * ny * nz + 0 * nz + 0,
                                          nz * sizeof(Float), 1, ny);
  zMinParams.extent = make_cudaExtent(sizeof(Float), ny, nx);
  zMinParams.kind = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&zMinParams);

  cudaMemcpy3DParms zMaxParams = {0};
  zMaxParams.srcPtr = make_cudaPitchedPtr(u_dev + 0 * ny * nz + 0 * nz + (nz - 2),
                                          nz * sizeof(Float), 1, ny);
  zMaxParams.dstPtr = make_cudaPitchedPtr(u_dev + 0 * ny * nz + 0 * nz + (nz - 1),
                                          nz * sizeof(Float), 1, ny);
  zMaxParams.extent = make_cudaExtent(sizeof(Float), ny, nx);
  zMaxParams.kind = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&zMaxParams);
}

#endif
