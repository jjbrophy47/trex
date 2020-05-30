#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=cleaning
#SBATCH --output=jobs/logs/cleaning/churn8
#SBATCH --error=jobs/errors/cleaning/churn8
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.6.1

dataset='churn'
n_estimators=100
max_depth=3
check_pct=0.3
rs_list=(3)

verbose=1
inf_k=-1

for i in ${!rs_list[@]}; do
    python3 experiments/scripts/cleaning.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --check_pct $check_pct \
      --rs ${rs_list[$i]} \
      --verbose $verbose \
      --inf_k $inf_k
done