#!/bin/bash
#SBATCH --job-name=nerf
#SBATCH --time=0-20:00:00
#SBATCH --nodes=1
#SBATCH --account=plgplgnerfvideo-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --output='logs.out'
#SBATCH --error='errors.out'

python main.py --config config.txt
