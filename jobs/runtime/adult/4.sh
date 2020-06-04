#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=runtime
#SBATCH --output=jobs/logs/runtime/adult4
#SBATCH --error=jobs/errors/runtime/adult4
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --account=uoml
module load python3/3.6.1

dataset='adult'
n_estimators=100
max_depth=5

rs_list=(1 2 3 4 5)

for i in ${!rs_list[@]}; do
    python3 experiments/scripts/runtime.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --rs ${rs_list[$i]} \
      --inf_k 0
done