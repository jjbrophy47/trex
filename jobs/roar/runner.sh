#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
model=$2
n_estimators=$3
max_depth=$4
method=$5
train_frac_to_remove=$6
n_checkpoints=$7
rs=$8

python3 scripts/experiments/roar.py \
  --dataset $dataset \
  --model $model \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --method $method \
  --train_frac_to_remove $train_frac_to_remove \
  --n_checkpoints $n_checkpoints \
  --rs $rs
