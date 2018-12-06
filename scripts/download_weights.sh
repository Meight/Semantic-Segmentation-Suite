#!/usr/bin/env bash

#SBATCH --job-name=download-weights
#SBATCH --output=/projets/thesepizenberg/deep-learning/logs/download.out
#SBATCH --error=/projets/thesepizenberg/deep-learning/logs/download.out

#SBATCH --ntasks=7
#SBATCH --cpus-per-task=3
#SBATCH --partition=24CPUNodes
#SBATCH --mem-per-cpu=5000M

wait

set -e

SCRIPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/utils"

MODEL_NAMES=("ResNet50", "ResNet101", "ResNet152", "MobileNetV2", "InceptionV4", "SEResNeXt50", "SEResNeXt101")

for model_name in "${MODEL_NAMES[@]}"; do
    echo "Launching download for model ${model_name}..."
    srun -n1 -N1 /projets/thesepizenberg/deep-learning/deeplab-generic/matlab/venv/bin/python3.4 \
        "$SCRIPT_PATH/get_pretrained_checkpoints.py" --model=${model_name} &

done;

wait