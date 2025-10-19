#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=LCDAT
#SBATCH --output=batch_out/LC_Data_Augmentation_Test/LCDAT_%J_stdout.txt
#SBATCH --error=batch_out/LC_Data_Augmentation_Test/LCDAT_%J_stderr.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/2_model_training/
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python LC_train_eval.py @txt_exp.txt @txt_model.txt --exp=$SLURM_ARRAY_TASK_ID