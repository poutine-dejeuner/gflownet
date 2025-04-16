#!/bin/bash                                          
#SBATCH --array=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --error=slurm/%A/serr/unethpo.%a.err
#SBATCH --output=slurm/%A/sout/unethpo.%a.out

module load miniconda/3
conda activate /home/mila/l/letournv/miniconda3/envs/cphoto

python /home/mila/l/letournv/repos/gflownets/train.py env=photo proxy=photo
#orion hunt -n topn python treesearchmeep.py \
# --top_n~'uniform(16, 256, discrete=True)' \
