#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH -w c731
#SBATCH --job-name=LCData
#SBATCH --output=batch_out/LCData_%J_stdout.txt
#SBATCH --error=batch_out/LCData_%J_stderr.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/2_model_training/
#SBATCH --gres=gpu:0
#SBATCH --array=0
#SBATCH --dependency=

module load Graphviz/8.1.0-GCCcore-12.3.0
module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python -u LC_data_loader_forecast.py @txt_exp.txt @txt_model.txt