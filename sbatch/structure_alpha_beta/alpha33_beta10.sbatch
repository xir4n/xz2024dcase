#!/bin/bash

#BATCH --job-name=alpha33_beta10
#SBATCH --nodes=1
#SBATCH -C v100-32g
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --account=nvz@v100
#SBATCH --output=alpha33_beta10_%j.out

module purge

module load anaconda-py3/2023.09

export PATH=$WORK/.local/bin:$PATH
conda activate dcase
python /gpfsssd/scratch/rech/nvz/ufy95rl/dcase/xz2024dcase/run_training.py --project_name structure_alpha_beta --experiment_name alpha33_beta10 --subset 5 --alpha 33 --beta 10 --sav_dir ./log/structure_alpha_beta --batch_size 512 --warmup_steps 0

