#!/bin/bash
#
#SBATCH --partition=sooner_test
#SBATCH --container=el9hw
#SBATCH --job-name=EFMNoDJF
#SBATCH --output=batch_out/EFMNoDJF/EFMNoDJF_%J_stdout.txt
#SBATCH --error=batch_out/EFMNoDJF/EFMNoDJF_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=30GB
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/1_data_cleaning/1_EFM/
#SBATCH --array=0
#SBATCH --time=12:00:00
#SBATCH --dependency=
#SBATCH --exclude=

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate

python 1e_EFM_concat_annual.py
