#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH --job-name=LC_train
#SBATCH --output=batch_out/LC_train_%J_stdout.txt
#SBATCH --error=batch_out/LC_train_%J_stderr.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/
#SBATCH --time=12:00:00

. /home/fagg/tf_setup.sh
conda activate dnn_2024_02
module load cuDNN/8.9.2.26-CUDA-12.2.0

python LC_train.py @txt_exp.txt @txt_unet.txt


