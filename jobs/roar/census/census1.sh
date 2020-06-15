#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=roar
#SBATCH --output=jobs/logs/roar/census1
#SBATCH --error=jobs/errors/roar/census1
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
rs_list=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
test_frac=1.0
n_test=50

for tree_kernel in ${tree_kernels[@]}; do
    python3 experiments/scripts/roar.py \
      --dataset $dataset \
      --tree_kernel $tree_kernel \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --rs 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
      --test_frac $test_frac \
      --n_test $n_test \
      --trex \
      --kernel_model 'klr'
done
