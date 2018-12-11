#!/usr/bin/env bash

SCRIPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/scripts"
MODEL_NAMES=("BiSeNet")
FRONTEND_NAMES=("ResNet101")
INPUT_SIZE=(256 384 512)
BATCH_SIZE=(1)
DATASET="voc-chh"

for model_name in "${MODEL_NAMES[@]}"; do
    for frontend_name in "${FRONTEND_NAMES[@]}"; do
        for input_size in "${INPUT_SIZE[@]}"; do
            for batch_size in "${BATCH_SIZE[@]}"; do
                sbatch "$SCRIPT_PATH"/train.sh ${model_name} ${frontend_name} ${input_size} ${batch_size} ${DATASET}
            done;
        done;
    done;
done;
