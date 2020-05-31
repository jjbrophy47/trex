#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=fidelity
#SBATCH --output=jobs/logs/fidelity/census
#SBATCH --error=jobs/errors/fidelity/census
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --account=uoml
module load python3/3.6.1

dataset='census'
n_estimators=250
max_depth=5

# tree_kernels=('leaf_output' 'leaf_path')
tree_kernels=('leaf_path')


for tree_kernel in ${tree_kernels[@]}; do
    python3 experiments/scripts/fidelity.py \
      --dataset $dataset \
      --tree_kernel $tree_kernel \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --trex \
      --kernel_model 'svm'

    python3 experiments/scripts/fidelity.py \
      --dataset $dataset \
      --tree_kernel $tree_kernel \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --trex \
      --kernel_model 'lr' \
      --teknn
done
