#!/usr/bin/env bash

SCRIPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/scripts"

IMAGE="/projets/thesepizenberg/deep-learning/segmentation-suite/voc-ha/val"
CKPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/checkpoints/BiSeNet/ResNet101/non-augmented/0058/model.ckpt"

sbatch "$SCRIPT_PATH"/predict.sh ${IMAGE} ${CKPT_PATH} BiSeNet voc-ha