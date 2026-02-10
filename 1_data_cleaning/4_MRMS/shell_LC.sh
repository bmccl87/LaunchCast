#!/bin/bash
#
#SBATCH --partition=sooner_test
#SBATCH --container=el9hw
#SBATCH --job-name=MRMSDown
#SBATCH --output=batch_out/MRMSDown_%J_stdout.txt
#SBATCH --error=batch_out/MRMSDown_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=48G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/1_data_cleaning/4_MRMS/
#SBATCH --time=24:00:00
#SBATCH --array=1-38
#SBATCH --dependency=

python 4d_concat_month.py --var=$SLURM_ARRAY_TASK_ID