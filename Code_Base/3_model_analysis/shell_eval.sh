#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=LCMaps
#SBATCH --output=batch_out/LCMaps_%A_%a_stdout.txt
#SBATCH --error=batch_out/LCMaps_%A_%a_stderr.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/3_model_analysis/
#SBATCH --gres=gpu:0
#SBATCH --array=0
#SBATCH --exclude=
#SBATCH --dependency=

echo "Starting job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Host: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Date: $(date)"

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export CUDA_DIR=${CUDA_HOME}

# python -u 0_LC_calibrate.py @txt_exp_eval.txt --cpus_per_task=${SLURM_CPUS_PER_TASK} --exp=$SLURM_ARRAY_TASK_ID
# python -u 1a_LC_quick_out.py @txt_exp_eval.txt --cpus_per_task=${SLURM_CPUS_PER_TASK} --exp=$SLURM_ARRAY_TASK_ID
# python -u 1b_LC_long_out.py @txt_exp_eval.txt --cpus_per_task=${SLURM_CPUS_PER_TASK} --exp=$SLURM_ARRAY_TASK_ID
python -u 5_LC_case_studies.py @txt_exp_eval.txt --cpus_per_task=${SLURM_CPUS_PER_TASK} --exp=$SLURM_ARRAY_TASK_ID
# python -u 7_LC_ablation.py @txt_exp_eval.txt --cpus_per_task=${SLURM_CPUS_PER_TASK} --exp=$SLURM_ARRAY_TASK_ID
# python -u 8_LC_ALE.py @txt_exp_eval.txt --cpus_per_task=${SLURM_CPUS_PER_TASK} --exp=$SLURM_ARRAY_TASK_ID

echo "Finished job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"
