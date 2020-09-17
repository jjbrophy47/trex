#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.6.1

dataset=$1
n_estimators=$2
max_depth=$3
tree_kernel=$4

if [ $dataset = 'census' ]
then
    python3 scripts/experiments/fidelity.py \
      --teknn \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --tree_kernel $tree_kernel \
      --train_frac 0.1 \
      --pred_size 20
else
    python3 scripts/experiments/fidelity.py \
      --teknn \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --tree_kernel $tree_kernel
fi