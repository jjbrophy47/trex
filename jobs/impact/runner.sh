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
start_pred=$7
setting=$8
n_test=$9
klr_C=${10}
knn_tree_kernel=${11}
knn_n_neighbors=${12}
rs=${13}

tree_kernel=$knn_tree_kernel

python3 scripts/experiments/impact.py \
  --dataset $dataset \
  --model $model \
  --preprocessing $preprocessing \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --method $method \
  --start_pred $start_pred \
  --setting $setting \
  --n_test $n_test \
  --tree_kernel $tree_kernel \
  --C $klr_C \
  --n_neighbors $knn_n_neighbors \
  --rs $rs
