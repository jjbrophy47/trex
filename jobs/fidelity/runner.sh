#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
model=$2
n_estimators=$3
max_depth=$4
surrogate=$5
tree_kernel=$6
rs=$7

python3 scripts/experiments/fidelity.py \
  --dataset $dataset \
  --model $model \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --surrogate $surrogate \
  --kernel_model $surrogate \
  --tree_kernel $tree_kernel \
  --rs $rs
