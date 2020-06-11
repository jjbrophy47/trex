#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=roar
#SBATCH --output=jobs/logs/roar/adult
#SBATCH --error=jobs/errors/roar/adult
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.6.1

dataset='adult'
n_estimators=100
max_depth=5

tree_kernels=('tree_output' 'leaf_path' 'leaf_output')
rs_list=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
test_frac=1.0
n_test=50

for tree_kernel in ${tree_kernels[@]}; do
    for rs in ${!rs_list[@]}; do

        python3 experiments/scripts/roar.py \
          --dataset $dataset \
          --tree_kernel $tree_kernel \
          --n_estimators $n_estimators \
          --max_depth $max_depth \
          --rs $rs \
          --test_frac $test_frac \
          --n_test $n_test \
          --trex \
          --kernel_model 'klr'

        python3 experiments/scripts/roar.py \
          --dataset $dataset \
          --tree_kernel $tree_kernel \
          --n_estimators $n_estimators \
          --max_depth $max_depth \
          --rs $rs \
          --test_frac $test_frac \
          --n_test $n_test \
          --teknn

        python3 experiments/scripts/roar.py \
          --dataset $dataset \
          --tree_kernel $tree_kernel \
          --n_estimators $n_estimators \
          --max_depth $max_depth \
          --rs $rs \
          --test_frac $test_frac \
          --n_test $n_test \
          --maple
    done
done
