#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00                                 
#SBATCH --output slurm/%j.out
#SBATCH --error slurm/%j.err
#SBATCH --mail-user=vincentmillions@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --comment="diffusion"

python train.py gflownet.optimizer.n_train_steps=3 env=photo proxy=photo

