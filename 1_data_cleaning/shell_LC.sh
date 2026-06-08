#!/bin/bash
#
#SBATCH --partition=sooner_test
#SBATCH --container=el9hw
#SBATCH --job-name=LCRots
#SBATCH --output=batch_out/LCRots_%J_stdout.txt
#SBATCH --error=batch_out/LCRots_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/1_data_cleaning/
#SBATCH --time=24:00:00

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python slicendice_build_datasets.py
