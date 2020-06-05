#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=runtime
#SBATCH --output=jobs/logs/runtime/census1
#SBATCH --error=jobs/errors/runtime/census1
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --account=uoml
module load python3/3.6.1

dataset='census'
n_estimators=250
max_depth=5

tree_kernel='tree_output'

rs_list=(1 2 3 4 5)

for i in ${!rs_list[@]}; do
    python3 experiments/scripts/runtime.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --rs ${rs_list[$i]} \
      --trex \
      --tree_kernel $tree_kernel \
      --kernel_model 'svm' \
      --val_frac 0.01

    python3 experiments/scripts/runtime.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --rs ${rs_list[$i]} \
      --trex \
      --tree_kernel $tree_kernel \
      --kernel_model 'lr' \
      --val_frac 0.01
done