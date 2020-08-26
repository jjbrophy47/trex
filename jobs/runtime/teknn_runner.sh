#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.6.1

dataset=$1
n_estimators=$2
max_depth=$3
tree_kernel=$4
rs=$5

python3 experiments/scripts/runtime.py \
  --teknn \
  --dataset $dataset \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --tree_kernel $tree_kernel \
  --rs $rs
