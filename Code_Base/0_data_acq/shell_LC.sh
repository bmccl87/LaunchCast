#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --job-name=ZdrDwn
#SBATCH --output=/home/bmac87/LaunchCast/0_data_acq/batch_out_mrms/zdr_down_%J_stdout.txt
#SBATCH --error=/home/bmac87/LaunchCast/0_data_acq/batch_out_mrms/zdr_down_%J_stderr.txt
#SBATCH --nodes=2
#SBATCH --ntasks=40
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-31

python MRMS_Parallelized_v3_brandon.py --idx=$SLURM_ARRAY_TASK_ID
