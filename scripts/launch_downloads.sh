#!/usr/bin/env bash

SCRIPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/scripts"
MODEL_NAMES=("ResNet50", "ResNet101", "ResNet152", "MobileNetV2", "InceptionV4", "SEResNeXt50", "SEResNeXt101")

for model_name in "${MODEL_NAMES[@]}"; do
    sbatch "$SCRIPT_PATH"/download_weights.sh ${model_name}
done;
