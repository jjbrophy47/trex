#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=cleaning
#SBATCH --output=jobs/logs/cleaning/adult4
#SBATCH --error=jobs/errors/cleaning/adult4
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --account=uoml
module load python3/3.6.1

dataset='adult'
n_estimators=100
max_depth=5
check_pct=0.25

rs=4
verbose=1
tree_kernels=('tree_output' 'leaf_path' 'leaf_output')

python3 experiments/scripts/cleaning.py \
  --dataset $dataset \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --check_pct $check_pct \
  --rs ${rs_list[$i]} \
  --verbose $verbose \
  --maple

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
      --kernel_model 'svm'

    python3 experiments/scripts/cleaning.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --check_pct $check_pct \
      --rs $rs \
      --verbose $verbose \
      --trex \
      --tree_kernel $tree_kernel \
      --kernel_model 'klr'

    python3 experiments/scripts/cleaning.py \
      --dataset $dataset \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --check_pct $check_pct \
      --rs $rs \
      --verbose $verbose \
      --teknn \
      --tree_kernel $tree_kernel
done