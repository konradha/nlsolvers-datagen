#!/bin/bash

#SBATCH --job-name=nlse_3d_dev
#SBATCH --time=01:15:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --output=logs/%A_%a/nlse3d_solver_dev.out
#SBATCH --error=logs/%A_%a/nlse3d_solver_dev.err
#SBATCH --partition=gpu
#SBATCH --array=0-111

BASE_DIR="$1"

if [ -z "$BASE_DIR" ]; then
    echo "sbatch nlse_3d_launch.sh exe-dir"
    exit 1
fi

C_TYPES=(constant periodic_structure piecewise_constant sign_changing_mass layered waveguide quasiperiodic turbulent)
M_TYPES=(constant piecewise gradient topological defects quasiperiodic multiscale)
P_TYPES=(multi_soliton_state skyrmion_tube)

NUM_M_TYPES=${#M_TYPES[@]}
NUM_P_TYPES=${#P_TYPES[@]}

c_type_idx=$((SLURM_ARRAY_TASK_ID / (NUM_M_TYPES * NUM_P_TYPES)))
m_type_idx=$(((SLURM_ARRAY_TASK_ID / NUM_P_TYPES) % NUM_M_TYPES))
p_type_idx=$((SLURM_ARRAY_TASK_ID % NUM_P_TYPES))

c_type=${C_TYPES[$c_type_idx]}
m_type=${M_TYPES[$m_type_idx]}
p=${P_TYPES[$p_type_idx]}

mkdir -p logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# just to be safe sequentially checking if dir exists
mkdir -p $SCRATCH/nlse_3d/
mkdir -p $SCRATCH/nlse_3d/
mkdir -p $SCRATCH/nlse_3d/c_${c_type}
mkdir -p $SCRATCH/nlse_3d/c_${c_type}/m_${m_type}
mkdir -p $SCRATCH/nlse_3d/c_${c_type}/m_${m_type}/$p

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load python_cuda/3.11.6
module load gcc/12.2.0 
module load ffmpeg

L=6.
n=100
nt=4000

dr=64
exe=${BASE_DIR}

T=1.2
snap=100

echo "Processing c_type=${c_type}, m_type=${m_type}, phenomenon=${p}"

# we don't have "system_type" as a param for now
python complex_launcher_3d.py \
        --phenomenon ${p} \
        --anisotropy-type ${c_type} --m_type ${m_type} \
        --nx $n --ny $n --nz $n --Lx $L --Ly $L --Lz $L \
       	--nt ${nt} --T ${T} --snapshots ${snap} \
        --dr-x ${dr} --dr-y ${dr} --dr-z ${dr} --delete-intermediates \
        --exe ${exe} --num-runs 5 \
        --output-dir=$SCRATCH/nlse_3d/c_${c_type}/m_${m_type}/${p} \
        --seed $SLURM_JOB_ID
