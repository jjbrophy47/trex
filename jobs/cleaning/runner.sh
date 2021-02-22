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
check_pct=$6
train_frac=$7
rs=$8

python3 scripts/experiments/cleaning.py \
  --dataset $dataset \
  --model $model \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --method $method \
  --check_pct $check_pct \
  --train_frac $train_frac \
  --rs $rs
