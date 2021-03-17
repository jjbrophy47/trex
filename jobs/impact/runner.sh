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
desired_pred=$7
n_test=$8
klr_tree_kernel=$9
klr_C=${10}
knn_tree_kernel=${11}
knn_n_neighbors=${12}
rs=${13}

if [ $method = 'klr' ]; then
    tree_kernel=$klr_tree_kernel
else
    tree_kernel=$knn_tree_kernel
fi

python3 scripts/experiments/impact.py \
  --dataset $dataset \
  --model $model \
  --preprocessing $preprocessing \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --method $method \
  --desired_pred $desired_pred \
  --n_test $n_test \
  --tree_kernel $tree_kernel \
  --C $klr_C \
  --n_neighbors $knn_n_neighbors \
  --rs $rs
