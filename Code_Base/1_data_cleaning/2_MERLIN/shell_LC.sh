#!/bin/bash
#
#SBATCH --partition=ai2es
#SBATCH -w c733
#SBATCH --job-name=MERLINKD
#SBATCH --output=batch_out/MERLINKD_%J_stdout.txt
#SBATCH --error=batch_out/MERLINKD_%J_stderr.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --mail-user=bmac7167@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/bmac87/LaunchCast/1_data_cleaning/2_MERLIN/
#SBATCH --time=00:30:00
#SBATCH --array=1-84%10
#SBATCH --exclude=
#SBATCH --dependency=

echo "Starting job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Host: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Date: $(date)"

module load Python/3.10.8-GCCcore-12.2.0
source /home/bmac87/BoltCast/BC_env/bin/activate

python -u 2f_flashes2kdtree_df.py --exp=$SLURM_ARRAY_TASK_ID

echo "Finished job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Date: $(date)"
