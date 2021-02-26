#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
model=$2
rs=$3
preprocessing=$4
tune_frac=$5

python3 scripts/experiments/performance.py \
  --dataset $dataset \
  --model $model \
  --rs $rs \
  --preprocessing $preprocessing \
  --tune_frac $tune_frac
