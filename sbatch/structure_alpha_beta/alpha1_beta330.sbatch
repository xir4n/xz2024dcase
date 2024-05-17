#!/bin/bash

#BATCH --job-name=alpha1_beta330
#SBATCH --nodes=1
#SBATCH -C v100-32g
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --account=nvz@v100
#SBATCH --output=alpha1_beta330_%j.out

module purge

module load anaconda-py3/2023.09

export PATH=$WORK/.local/bin:$PATH
conda activate dcase
python /gpfsssd/scratch/rech/nvz/ufy95rl/dcase/xz2024dcase/run_training.py --project_name structure_alpha_beta --experiment_name alpha1_beta330 --subset 5 --alpha 1 --beta 330 --sav_dir ./log/structure_alpha_beta --batch_size 512 --warmup_steps 0

