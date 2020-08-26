#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.6.1

dataset=$1
n_estimators=$2
max_depth=$3
check_pct=$4
train_frac=$5
kernel_model=$6
tree_kernel=$7
rs=$8

python3 experiments/scripts/cleaning.py \
  --trex \
  --dataset $dataset \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --check_pct $check_pct \
  --train_frac $train_frac \
  --tree_kernel $tree_kernel \
  --kernel_model $kernel_model \
  --rs $rs
