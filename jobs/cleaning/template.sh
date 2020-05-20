#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=cleaning
#SBATCH --output=jobs/logs/addition/cleaning
#SBATCH --error=jobs/errors/addition/cleaning
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.6.1

python3 experiments/scripts/cleaning.py \
  --dataset census \
  --train_frac 0.05 \
  --linear_model lr \
  --check_pct 0.15 \
  --rs 1 \
  --repeats 5 \
  --verbose 2 \
