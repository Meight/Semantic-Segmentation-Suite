#!/usr/bin/env bash

SCRIPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/scripts"
MODEL_NAMES=("BiSeNet")
FRONTEND_NAMES=("ResNet101")
INPUT_SIZE=(384)
BATCH_SIZE=1
DATASET="voc-ha"

for model_name in "${MODEL_NAMES[@]}"; do
    for frontend_name in "${FRONTEND_NAMES[@]}"; do
        for input_size in "${INPUT_SIZE[@]}"; do
            sbatch "$SCRIPT_PATH"/train.sh ${model_name} ${frontend_name} ${input_size} ${BATCH_SIZE} ${DATASET}
        done;
    done;
done;
