#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=cleaning
#SBATCH --output=jobs/logs/cleaning/census100_5
#SBATCH --error=jobs/errors/cleaning/census100_5
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.6.1

dataset='census'
n_estimators=250
max_depth=5
check_pct=0.15

rs_list=(5)

verbose=1
tree_kernel='leaf_output'
kernel_model_kernel='linear'

for i in ${!rs_list[@]}; do
    python3 experiments/scripts/cleaning.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --check_pct $check_pct \
      --rs ${rs_list[$i]} \
      --verbose $verbose \
      --trex \
      --tree_kernel $tree_kernel \
      --kernel_model 'svm' \
      --kernel_model_kernel $kernel_model_kernel \
      --kernel_model_loss

    python3 experiments/scripts/cleaning.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --check_pct $check_pct \
      --rs ${rs_list[$i]} \
      --verbose $verbose \
      --trex \
      --tree_kernel $tree_kernel \
      --kernel_model 'lr' \
      --kernel_model_kernel $kernel_model_kernel \
      --kernel_model_loss \
      --maple \
      --teknn \
      --teknn_loss
done