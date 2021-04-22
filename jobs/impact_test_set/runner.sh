#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
model=$2
n_estimators=$3
max_depth=$4
setting=$5
method=$6
frac_remove=$7

python3 scripts/experiments/impact_test_set.py \
  --dataset $dataset \
  --model $model \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --setting $setting \
  --method $method \
  --train_frac_to_remove $frac_remove
