#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=roar
#SBATCH --output=jobs/logs/roar/churn
#SBATCH --error=jobs/errors/roar/churn
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.6.1

dataset='churn'
n_estimators=100
max_depth=3

tree_kernel='leaf_output'

python3 experiments/scripts/roar.py \
  --dataset $dataset \
  --tree_kernel $tree_kernel \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --trex \
  --kernel_model 'lr' \
  --teknn
  # --maple
