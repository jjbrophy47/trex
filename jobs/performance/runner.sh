#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=performance
#SBATCH --output=jobs/logs/performance
#SBATCH --error=jobs/errors/performance
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
model=$2

python3 scripts/experiments/performance.py \
  --dataset $1 \
  --model $2
