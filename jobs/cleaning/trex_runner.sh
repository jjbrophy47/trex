#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.6.1

dataset=$1
tree_type$2
n_estimators=$3
max_depth=$4
check_pct=$5
train_frac=$6
kernel_model=$7
tree_kernel=$8
rs=$9

python3 scripts/experiments/cleaning.py \
  --trex \
  --dataset $dataset \
  --tree_type $tree_type \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --check_pct $check_pct \
  --train_frac $train_frac \
  --tree_kernel $tree_kernel \
  --kernel_model $kernel_model \
  --rs $rs
