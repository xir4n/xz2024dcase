#!/bin/bash
#SBATCH --job-name=dcase24
#SBATCH --comment="run training for dcase24 challenge"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=0-1:00:00
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
cd /scratch/nautilus/users/x21zhang@imta.fr/xz2024dcase
python run_training.py --ckpt_id 1 --fast_dev_run True 