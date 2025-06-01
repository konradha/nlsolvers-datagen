#!/bin/bash
#SBATCH --job-name=abstract_stats_nlse_2d
#SBATCH --array=0-5
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/abstract_%A_%a.out
#SBATCH --error=logs/abstract_%A_%a.err

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load gcc/12.2.0 openmpi/4.1.6
module load py-mpi4py/3.1.4
module load cmake/3.27.7

c_types=(constant periodic_structure piecewise_constant sign_changing_mass layered waveguide)
c=${c_types[$SLURM_ARRAY_TASK_ID]}

mkdir -p nlse_2d_analysis/abstracted/$c

find_dirs=$(find $SCRATCH/nlse_2d/c_$c -path "*/*/hdf5" -type d)
if [ -n "$find_dirs" ]; then
    mpirun -n $SLURM_NTASKS python ensemble_processing.py "$SCRATCH/nlse_2d/c_$c" nlse_2d_analysis/abstracted/$c --pattern="**/hdf5/*.h5"
else
    echo "No HDF5 directories found for c_type=$c"
fi
