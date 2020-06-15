#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=runtime
#SBATCH --output=jobs/logs/runtime/census3
#SBATCH --error=jobs/errors/runtime/census3
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --account=uoml
module load python3/3.6.1

dataset='census'
n_estimators=250
max_depth=5

tree_kernels=('leaf_output')
rs_list=(3 4 5)

for tree_kernel in ${tree_kernels[@]}; do
    for rs in ${rs_list[@]}; do
        python3 experiments/scripts/runtime.py \
          --dataset $dataset \
          --n_estimators $n_estimators \
          --max_depth $max_depth \
          --rs $rs \
          --tree_kernel $tree_kernel \
          --teknn
    done
done
