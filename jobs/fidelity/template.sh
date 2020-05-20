#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=fidelity
#SBATCH --output=jobs/logs/addition/fidelity
#SBATCH --error=jobs/errors/addition/fidelity
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.6.1

python3 experiments/scripts/fidelity.py \
  --dataset census