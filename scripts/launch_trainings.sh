#!/usr/bin/env bash

SCRIPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/scripts"
MODEL_NAMES=("BiSeNet") # "FC-DenseNet103" "GCN" "DeepLabV3_plus" "RefineNet" "DenseASPP" "PSPNet" "DDSC" "AdapNet")
FRONTEND_NAMES=("ResNet101")
INPUT_SIZE=(256)
BATCH_SIZE=(1)
DATASET="voc-chh"

for input_size in "${INPUT_SIZE[@]}"; do
    for batch_size in "${BATCH_SIZE[@]}"; do
        for model_name in "${MODEL_NAMES[@]}"; do
            for frontend_name in "${FRONTEND_NAMES[@]}"; do
                sbatch "$SCRIPT_PATH"/train.sh ${model_name} ${frontend_name} ${input_size} ${batch_size} ${DATASET}
            done;
        done;
    done;
done;
