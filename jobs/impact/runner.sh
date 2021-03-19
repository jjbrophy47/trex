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
method=$6
setting=$7
n_test=$8
klr_C=$9
knn_tree_kernel=${10}
knn_n_neighbors=${11}
rs=${12}

tree_kernel=$knn_tree_kernel

python3 scripts/experiments/impact.py \
  --dataset $dataset \
  --model $model \
  --preprocessing $preprocessing \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --method $method \
  --setting $setting \
  --n_test $n_test \
  --tree_kernel $tree_kernel \
  --C $klr_C \
  --n_neighbors $knn_n_neighbors \
  --rs $rs
