#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=roar
#SBATCH --output=jobs/logs/roar/census
#SBATCH --error=jobs/errors/roar/census
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=11
#SBATCH --account=uoml
module load python3/3.6.1

dataset='census'
n_estimators=250
max_depth=5

tree_kernels=('tree_output' 'leaf_path' 'leaf_output')
rs_list=(1 2 3 4 5 6 7 8 9 10)
test_frac=0.05

for tree_kernel in ${tree_kernels[@]}; do
    for rs in ${!rs_list[@]}; do

        python3 experiments/scripts/roar.py \
          --dataset $dataset \
          --tree_kernel $tree_kernel \
          --n_estimators $n_estimators \
          --max_depth $max_depth \
          --rs $rs \
          --test_frac $test_frac \
          --trex \
          --kernel_model 'klr'

        python3 experiments/scripts/roar.py \
          --dataset $dataset \
          --tree_kernel $tree_kernel \
          --n_estimators $n_estimators \
          --max_depth $max_depth \
          --rs $rs \
          --test_frac $test_frac \
          --teknn

        python3 experiments/scripts/roar.py \
          --dataset $dataset \
          --tree_kernel $tree_kernel \
          --n_estimators $n_estimators \
          --max_depth $max_depth \
          --rs $rs \
          --test_frac $test_frac \
          --maple
    done
done
