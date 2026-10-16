#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=LCEFMv2
#SBATCH --output=batch_out/LCEFMv2_%J_stdout.txt
#SBATCH --error=batch_out/LCEFMv2_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=64GB
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/1_data_cleaning/1_EFM/
#SBATCH --array=6
#SBATCH --dependency=
#SBATCH --exclude=

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python -u 1e_EFM_concat.py --year=$SLURM_ARRAY_TASK_ID
