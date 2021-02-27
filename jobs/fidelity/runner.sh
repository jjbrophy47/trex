#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
model=$2
preprocessing=$3
n_estimators=$4
max_depth=$5
surrogate=$6
tree_kernel=$7
metric=$8
rs=$9

python3 scripts/experiments/fidelity.py \
  --dataset $dataset \
  --model $model \
  --preprocessing $preprocessing \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --surrogate $surrogate \
  --kernel_model $surrogate \
  --tree_kernel $tree_kernel \
  --metric $metric \
  --rs $rs
