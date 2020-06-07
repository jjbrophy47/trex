#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=cleaning
#SBATCH --output=jobs/logs/cleaning/census2
#SBATCH --error=jobs/errors/cleaning/census2
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --account=uoml
module load python3/3.6.1

dataset='census'
n_estimators=250
max_depth=5
check_pct=0.15

rs=2
train_frac=0.1
verbose=1
tree_kernels=('tree_output' 'leaf_path' 'leaf_output')

python3 experiments/scripts/cleaning.py \
  --dataset $dataset \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --check_pct $check_pct \
  --rs ${rs_list[$i]} \
  --verbose $verbose \
  --maple \
  --train_frac $train_frac

for tree_kernel in ${tree_kernels[@]}; do
    python3 experiments/scripts/cleaning.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --check_pct $check_pct \
      --rs $rs \
      --verbose $verbose \
      --trex \
      --tree_kernel $tree_kernel \
      --kernel_model 'svm' \
      --train_frac $train_frac

    python3 experiments/scripts/cleaning.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --check_pct $check_pct \
      --rs $rs \
      --verbose $verbose \
      --trex \
      --tree_kernel $tree_kernel \
      --kernel_model 'klr' \
      --train_frac $train_frac

    python3 experiments/scripts/cleaning.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --check_pct $check_pct \
      --rs $rs \
      --verbose $verbose \
      --teknn \
      --tree_kernel $tree_kernel \
      --train_frac $train_frac
done
