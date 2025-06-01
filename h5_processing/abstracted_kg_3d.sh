#!/bin/bash
#SBATCH --job-name=abstract_stats_kge_3d
#SBATCH --array=0-6
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=32G
#SBATCH --time=00:05:00
#SBATCH --output=logs/abstract_%A_%a.out
#SBATCH --error=logs/abstract_%A_%a.err

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load gcc/12.2.0 openmpi/4.1.6
module load py-mpi4py/3.1.4
module load cmake/3.27.7

c_types=(constant layered piecewise_constant quasiperiodic sign_changing_mass turbulent waveguide)
c=${c_types[$SLURM_ARRAY_TASK_ID]}

mkdir -p kge_3d_analysis/abstracted/$c

find_dirs=$(find $SCRATCH/kge_3d/c_$c -path "*/*/hdf5" -type d)
if [ -n "$find_dirs" ]; then
    mpirun -n $SLURM_NTASKS python ensemble_processing.py "$SCRATCH/kge_3d/c_$c" kge_3d_analysis/abstracted/$c --pattern="**/hdf5/*.h5"
else
    echo "No HDF5 directories found for c_type=$c"
fi
