#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=cleaning
#SBATCH --output=jobs/logs/cleaning/census8
#SBATCH --error=jobs/errors/cleaning/census8
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.6.1

dataset='census'
n_estimators=250
max_depth=5
check_pct=0.15

rs_list=(3)
train_frac=0.1

verbose=1
inf_k=-1

for i in ${!rs_list[@]}; do
    python3 experiments/scripts/cleaning.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --check_pct $check_pct \
      --save_results \
      --rs ${rs_list[$i]} \
      --verbose $verbose \
      --inf_k $inf_k \
      --train_frac $train_frac
done