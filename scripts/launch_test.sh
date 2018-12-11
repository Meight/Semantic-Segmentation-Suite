#!/usr/bin/env bash

SCRIPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/scripts"

CKPT_PATH="/projets/thesepizenberg/deep-learning/segmentation-suite/checkpoints/BiSeNet/ResNet101/non-augmented/0058/model.ckpt"

sbatch "$SCRIPT_PATH"/predict.sh ${CKPT_PATH} BiSeNet voc-ha