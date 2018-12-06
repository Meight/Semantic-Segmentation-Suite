#!/usr/bin/env bash
# Options SBATCH :
#SBATCH --job-name=Training
#SBATCH --output=/projets/thesepizenberg/deep-learning/logs/densenet-%j.out
#SBATCH --error=/projets/thesepizenberg/deep-learning/logs/densenet-%j.out

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding
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

if [ "$CHECK_DEPENDENCIES" = true ] ; then
    echo 'Checking for dependencies...'

    # Install required packages.
    srun keras-py3-tf /users/thesepizenberg/mlebouch/venv/bin/pip3 install tensorflow-gpu==1.10.0
    wait
    srun keras-py3-tf /users/thesepizenberg/mlebouch/venv/bin/pip3 install scikit-image --upgrade
    srun keras-py3-tf /users/thesepizenberg/mlebouch/venv/bin/pip3 install tqdm
    wait
    srun keras-py3-tf /users/thesepizenberg/mlebouch/venv/bin/pip3 install tensorboard==1.10.0
    srun keras-py3-tf /users/thesepizenberg/mlebouch/venv/bin/pip3 install tensorflow==1.10.0
    srun keras-py3-tf /users/thesepizenberg/mlebouch/venv/bin/pip3 install keras==2.2.2
    srun keras-py3-tf /users/thesepizenberg/mlebouch/venv/bin/pip3 install matplotlib
    srun keras-py3-tf /users/thesepizenberg/mlebouch/venv/bin/pip3 install -U scikit-learn
    srun keras-py3-tf /users/thesepizenberg/mlebouch/venv/bin/pip3 list
fi

wait


srun keras-py3-tf /users/thesepizenberg/mlebouch/venv/bin/python "$TRAIN_SCRIPT_DIR/train.py" \
                --num_epochs=200 \
                --checkpoint_step=2 \
                --validation_step=1 \
                --dataset=voc-ha \
                --crop_height=332 \
                --crop_width=332 \
                --model=FC-DenseNet56 \
                --frontend=ResNet101
wait