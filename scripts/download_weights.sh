#!/usr/bin/env bash

#SBATCH --job-name=download-weights
#SBATCH --output=/projets/thesepizenberg/deep-learning/logs/download-%j.out
#SBATCH --error=/projets/thesepizenberg/deep-learning/logs/download-%j.out

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=24CPUNodes
#SBATCH --mem-per-cpu=5000M

wait

set -e

SCRIPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/utils"

echo "Launching download for model ${1}..."
srun -n1 -N1 /projets/thesepizenberg/deep-learning/deeplab-generic/matlab/venv/bin/python3.4 \
        "$SCRIPT_PATH/get_pretrained_checkpoints.py" --model=${1}


wait