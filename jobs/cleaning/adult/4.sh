#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=cleaning
#SBATCH --output=jobs/logs/cleaning/adult4
#SBATCH --error=jobs/errors/cleaning/adult4
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.6.1

dataset='adult'
n_estimators=250
max_depth=10
check_pct=0.25

rs_list=(2)

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
      --inf_k $inf_k
done