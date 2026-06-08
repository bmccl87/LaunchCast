#!/bin/bash
#
#SBATCH --partition=sooner_test
#SBATCH --container=el9hw
#SBATCH --job-name=MRL2FLSH
#SBATCH --output=batch_out/MRL2FLSH_%J_stdout.txt
#SBATCH --error=batch_out/MRL2FLSH_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=30G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/1_data_cleaning/2_MERLIN/
#SBATCH --time=8:00:00
#SBATCH --array=1-84
#SBATCH --exclude=
#SBATCH --dependency=

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate

python 2d_MERLIN_binary_fed_target.py --exp=$SLURM_ARRAY_TASK_ID
