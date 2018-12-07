#!/usr/bin/env bash

SCRIPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/scripts"
MODEL_NAMES=("FC-DenseNet56" "DeepLabV3_plus" "PSPNet" "BiSeNet" "RefineNet" "FRRN-A" "FRRN-B" "Encoder-Decoder" "MobileUNet" "GCN" "DenseASPP" "DDSC" "AdapNet" "FC-DenseNet103")
FRONTEND_NAMES=("InceptionV4" "ResNet152" "ResNet101" "MobileNetV2")

for model_name in "${MODEL_NAMES[@]}"; do
    for frontend_name in "${FRONTEND_NAMES[@]}"; do
        sbatch "$SCRIPT_PATH"/train.sh ${model_name} ${frontend_name}
    done;
done;
