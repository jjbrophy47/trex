#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=runtime
#SBATCH --output=jobs/logs/runtime/census1
#SBATCH --error=jobs/errors/runtime/census1
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.6.1

dataset='census'
n_estimators=250
max_depth=5

rs_list=(1 2 3 4 5)

for i in ${!rs_list[@]}; do
    python3 experiments/scripts/runtime.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --rs ${rs_list[$i]} \
      --trex \
      --kernel_model 'svm'

    # python3 experiments/scripts/runtime.py \
    #   --dataset $dataset \
    #   --n_estimators $n_estimators \
    #   --max_depth $max_depth \
    #   --rs ${rs_list[$i]} \
    #   --trex \
    #   --kernel_model 'lr'
done