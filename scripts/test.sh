#!/usr/bin/env bash
# Options SBATCH :
#SBATCH --job-name=test
#SBATCH --output=/projets/thesepizenberg/deep-learning/logs/test.out
#SBATCH --error=/projets/thesepizenberg/deep-learning/logs/test.out

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=9000M

set -e

# Various script and dataset paths.
TRAIN_SCRIPT_DIR="/projets/thesepizenberg/deep-learning/segmentation-suite/"
DATASETS_DIR="/projets/thesepizenberg/deep-learning/datasets/VOC2012"
TENSORBOARD_LOGS_DIR="/projets/thesepizenberg/deep-learning/logs/tensorboard"
WEIGHTS_SAVE_DIR="/projets/thesepizenberg/deep-learning/models"
# Misc.
CHECK_DEPENDENCIES=false

# Begin script.

# Create a virtual environment from the Docker container.

srun keras-py3-tf virtualenv --system-site-packages /users/thesepizenberg/mlebouch/venv
wait

srun keras-py3-tf /users/thesepizenberg/mlebouch/venv/bin/python "$TRAIN_SCRIPT_DIR/test.py" \
                --checkpoint_path=${1} \
                --model=${2} \
                --dataset=${3} \
                --input-size=384
wait