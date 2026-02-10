#!/bin/bash
#
#SBATCH --partition=ai2es_a100
#SBATCH -w c829
#SBATCH --job-name=LCOutFig
#SBATCH --output=batch_out/LCOutFig_%J_stdout.txt
#SBATCH --error=batch_out/LCOutFig_%J_stderr.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/3_model_analysis/
#SBATCH --gres=gpu:0
#SBATCH --array=0
#SBATCH --dependency=

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python 1c_LC_long_out_warning.py @txt_exp_eval.txt