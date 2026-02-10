#!/bin/bash
#
#SBATCH --partition=ai2es_a100
#SBATCH -w c733
#SBATCH --job-name=LCPaper
#SBATCH --output=batch_out/LCPaper_%J_stdout.txt
#SBATCH --error=batch_out/LCPaper_%J_stderr.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/4_Paper_Figures/
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --dependency=

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python Table_num_samples.py