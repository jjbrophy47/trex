#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=runtime
#SBATCH --output=jobs/logs/runtime/adult2
#SBATCH --error=jobs/errors/runtime/adult2
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.6.1

dataset='adult'
n_estimators=250
max_depth=10

rs_list=(1 2 3 4 5)

for i in ${!rs_list[@]}; do
    python3 experiments/scripts/runtime.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --rs ${rs_list[$i]} \
      --teknn
done