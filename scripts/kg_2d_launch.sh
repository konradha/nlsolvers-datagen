#!/bin/bash

#SBATCH --job-name=kge_2d_dev
#SBATCH --time=07:30:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --output=logs/%j/kge2d_solver_dev.out
#SBATCH --error=logs/%j/kge2d_solver_dev.err
#SBATCH --partition=gpu

BASE_DIR="$1"

if [ -z "$BASE_DIR" ]; then
    echo "sbatch kg_2d_launch.sh exe-dir"
    exit 1
fi


module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load python_cuda/3.11.6
module load gcc/12.2.0 
module load ffmpeg

L=3.

n=250
nt_breather=15000 # using SV we're really _fast_
nt_other=6000

dr=128
exe=${BASE_DIR}

T_breather=20.
T_other=5.

snap_breather=100
snap_other=50

# This is unfortunately really dense, but expresses very exactly the fold we're doing.
for c_type in constant periodic_structure piecewise_constant sign_changing_mass layered waveguide quasiperiodic turbulent; do
	for m_type in constant piecewise gradient phase topological defects quasiperiodic multiscale; do
		time { # inner loop should be about (hopefully) 5-7 minutes		
			for p in kink_field kink_array_field multi_breather_field breather_field; do
				for v in fitting; do
			        	python real_launcher_2d.py \
			        	        --system-type klein_gordon --phenomenon ${p} --velocity-type ${v} \
						--anisotropy-type ${c_type} --m_type ${m_type} \
			        	        --nx $n --ny $n --Lx $L --Ly $L --nt ${nt_breather} --T ${T_breather} --snapshots ${snap_breather} \
			        	        --dr-x ${dr} --dr-y ${dr} --delete-intermediates \
			        	        --exe ${exe} --num-runs 2 \
			        	        --output-dir=$SCRATCH/kge_2d/c_${c_type}/m_${m_type}/${p}\
						--seed $SLURM_JOB_ID
				done
			done
			
			for p in multi_spiral_state colliding_rings \
			        multi_ring_state  skyrmion_like_field  \
			        multi_q_ball elliptical_soliton grf_modulated_soliton_field; do
				for v in fitting; do	
			        	python real_launcher_2d.py \
			                	--system-type klein_gordon --phenomenon ${p} --velocity-type ${v} \
						--anisotropy-type ${c_type} --m_type ${m_type} \
			                	--nx $n --ny $n --Lx $L --Ly $L --nt ${nt_other} --T ${T_other} --snapshots ${snap_other} \
			                	--dr-x ${dr} --dr-y ${dr} --delete-intermediates \
			                	--exe ${exe} --num-runs 2 \
			                	--output-dir=$SCRATCH/kge_2d/c_${c_type}/m_${m_type}/${p}\
						--seed $SLURM_JOB_ID
				done
			done
		}
	done
done
