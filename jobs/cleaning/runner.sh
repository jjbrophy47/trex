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
klr_tree_kernel=$7
klr_C=$8
knn_tree_kernel=$9
knn_n_neighbors=${10}
check_pct=${11}
train_frac=${12}
rs=${13}

if [ $method = 'klr' ]; then
    tree_kernel=$klr_tree_kernel
else
    tree_kernel=$knn_tree_kernel
fi

python3 scripts/experiments/cleaning.py \
  --dataset $dataset \
  --model $model \
  --preprocessing $preprocessing \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --method $method \
  --check_pct $check_pct \
  --train_frac $train_frac \
  --tree_kernel $tree_kernel \
  --C $klr_C \
  --n_neighbors $knn_n_neighbors \
  --rs $rs
