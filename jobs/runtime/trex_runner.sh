#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.6.1

dataset=$1
n_estimators=$2
max_depth=$3
kernel_model=$4
tree_kernel=$5
rs=$6

python3 scripts/experiments/runtime.py \
  --trex \
  --dataset $dataset \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --tree_kernel $tree_kernel \
  --kernel_model $kernel_model \
  --rs $rs
