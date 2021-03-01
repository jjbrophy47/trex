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
train_frac_to_remove=$7
n_checkpoints=$8
rs=$9

python3 scripts/experiments/roar.py \
  --dataset $dataset \
  --model $model \
  --preprocessing $preprocessing \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --method $method \
  --train_frac_to_remove $train_frac_to_remove \
  --n_checkpoints $n_checkpoints \
  --rs $rs
