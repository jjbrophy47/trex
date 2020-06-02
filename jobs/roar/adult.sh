#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=roar
#SBATCH --output=jobs/logs/roar/adult
#SBATCH --error=jobs/errors/roar/adult
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.6.1

dataset='adult'
n_estimators=250
max_depth=10

tree_kernel='leaf_output'

python3 experiments/scripts/roar.py \
  --dataset $dataset \
  --tree_kernel $tree_kernel \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --trex \
  --kernel_model 'lr' \
  --teknn \
  --maple
