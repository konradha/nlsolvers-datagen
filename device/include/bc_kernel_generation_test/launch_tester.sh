#!/bin/bash

#SBATCH --job-name=simple_torch_test  
#SBATCH --time=00:03:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=simple_torch_test_%j.out     
#SBATCH --error=simple_torch_test_%j.err       

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load python_cuda/3.11.6
module load gcc/12.2.0  openmpi/4.1.6
module load py-mpi4py/3.1.4

#echo "Some info:"
#nvidia-smi

export PTXAS_VERBOSE=3
export TORCH_COMPILE_DEBUG=1
export TORCH_TRITON_DUMP_CUDA=1

python bc_update_kernel_fusion.py

nsys profile -t cuda python bc_update_kernel_fusion.py
