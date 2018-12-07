#!/usr/bin/env bash

SCRIPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/scripts"
MODEL_NAMES=("DeepLabV3_plus" "BiSeNet" "GCN" "FC-DenseNet103")
FRONTEND_NAMES=("ResNet101" "ResNet152" "InceptionV4")

for model_name in "${MODEL_NAMES[@]}"; do
    for frontend_name in "${FRONTEND_NAMES[@]}"; do
        sbatch "$SCRIPT_PATH"/train.sh ${model_name} ${frontend_name}
    done;
done;
