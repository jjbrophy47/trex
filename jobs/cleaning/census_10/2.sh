#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=cleaning
#SBATCH --output=jobs/logs/cleaning/census2
#SBATCH --error=jobs/errors/cleaning/census2
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --account=uoml
module load python3/3.6.1

dataset='census'
n_estimators=250
max_depth=5
check_pct=0.15

rs_list=(2)
train_frac=0.1

verbose=1
tree_kernel='tree_output'
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
      --kernel_model_loss \
      --train_frac $train_frac

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
      --teknn \
      --teknn_loss \
      --train_frac $train_frac
      # --maple \
done