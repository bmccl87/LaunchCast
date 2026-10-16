#!/bin/bash
#
#SBATCH --partition=ai2es_v100
#SBATCH --job-name=LCTrain
#SBATCH --output=batch_out/LCTrain_%A_%a_stdout.txt
#SBATCH --error=batch_out/LCTrain_%A_%a_stderr.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/2_model_training/
#SBATCH --gres=gpu:0
#SBATCH --array=0
#SBATCH --dependency=
#SBATCH --exclude=
#SBATCH --time=1:00:00

echo "Starting job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Host: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Date: $(date)"

module load Graphviz/8.1.0-GCCcore-12.3.0
module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

python -u LC_model_64216.py @txt_exp.txt @txt_model.txt --cpus_per_task=${SLURM_CPUS_PER_TASK}

echo "Finished job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"