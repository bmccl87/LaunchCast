#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=LCMrgv3
#SBATCH --output=batch_out/LCMrgv3_%A_%a_stdout.txt
#SBATCH --error=batch_out/LCMrgv3_%A_%a_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/1_data_cleaning/5_Merge/
#SBATCH --array=0
#SBATCH --exclude=

echo "Starting job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Host: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Date: $(date)"

module load Graphviz/8.1.0-GCCcore-12.3.0
module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate
module load cuDNN/8.9.2.26-CUDA-12.2.0

python -u LC_calc_min_max.py --exp=$SLURM_ARRAY_TASK_ID

echo "Finished job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"
