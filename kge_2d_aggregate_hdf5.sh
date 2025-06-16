#!/bin/bash

#SBATCH --job-name=kge_2d_aggregate
#SBATCH --time=09:59:00                
#SBATCH --ntasks=1                    
#SBATCH --mem-per-cpu=8G               
#SBATCH --output=logs/%j/aggregate_kge_2d.out 
#SBATCH --error=logs/%j/aggregate_kge_2d.err

mkdir -p logs/$SLURM_JOB_ID
mkdir -p $SCRATCH/h5-scratch-kge-2d/
python scripts/aggregate_single.py \
    --base_path /cluster/work/math/camlab-data/konradha/ \
    --pde_type kge_2d \
    --output_dir $SCRATCH/kge-2d-global
    #--num_files 5 \

