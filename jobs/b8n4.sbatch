#!/bin/bash
#SBATCH --job-name=dcase24
#SBATCH --comment="run training for dcase24 challenge"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --qos=short

# Loading Software/Libraries
module purge
module load python

# Micromamba
source ~/.bashrc
micromamba activate dcase

# Run code
cd /scratch/nautilus/users/cz7tygkr@cnrs.fr/xz2024dcase
python run_training.py --experiment_name b8n4 --num_workers 30 --alpha 40 --beta 8 --m 1 --n 4 --n_epochs 150