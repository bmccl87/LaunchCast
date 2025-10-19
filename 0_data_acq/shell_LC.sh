#!/bin/bash
#
#SBATCH --partition=sooner_test
#SBATCH --container=el9hw
#SBATCH --job-name=hr3dwn
#SBATCH --output=batch_out/hr3dwn_%J_stdout.txt
#SBATCH --error=batch_out/hr3dwn_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/0_data_acq/
#SBATCH --time=48:00:00

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate

python 0d_HRRR_grib_subhourly.py
