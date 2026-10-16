#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=HR3Ltg
#SBATCH --output=batch_out/HR3Ltg_%A_%a_stdout.txt
#SBATCH --error=batch_out/HR3Ltg_%A_%a_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/1_data_cleaning/0_HRRR/
#SBATCH --time=24:00:00
#SBATCH --array=0


echo "Starting job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Host: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Date: $(date)"

source /opt/oscer/software/Mamba/23.1.0-4/etc/profile.d/conda.sh
conda activate /home/bmac87/.conda/envs/xe2

python -u 0c_LC_hrrr_ltg_forecast.py

echo "Finished job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"
