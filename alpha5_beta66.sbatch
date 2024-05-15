#!/bin/bash

#BATCH --job-name=alpha5_beta66
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=alpha5_beta66_%j.out
#SBATCH --account=nvz@v100

module purge

module load anaconda-py3/2023.09

conda activate dcase
python run_training.py --project_name structure_alpha_beta --experiment_name alpha5_beta66 --subset 10 --alpha 5 --beta 66 --sav_dir ./structure_alpha_beta --fast_dev_run True --batch_size 1

