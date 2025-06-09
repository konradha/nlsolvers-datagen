#!/bin/bash

#SBATCH --job-name=build_nlsolvers  
#SBATCH --time=00:05:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --output=build_%j.out     
#SBATCH --error=build_%j.err       

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load python_cuda/3.11.6
module load gcc/12.2.0  openmpi/4.1.6
module load cmake/3.27.7

mkdir -p build
cd build

cmake .. -DENABLE_GPU=ON
cmake --build . --parallel $SLURM_CPUS_PER_TASK
