#!/usr/bin/env bash

SCRIPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/scripts"
MODEL_NAMES=("DeepLabV3_plus" "PSPNet" "BiSeNet")

for model_name in "${MODEL_NAMES[@]}"; do
    sbatch "$SCRIPT_PATH"/train.sh ${model_name}
done;
