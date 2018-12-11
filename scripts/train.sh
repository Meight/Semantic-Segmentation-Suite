#!/usr/bin/env bash
# Options SBATCH :
#SBATCH --job-name=training-full-val
#SBATCH --output=/projets/thesepizenberg/deep-learning/logs/densenet-%j.out
#SBATCH --error=/projets/thesepizenberg/deep-learning/logs/densenet-%j.out

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
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
                --num_epochs=60 \
                --checkpoint_step=10 \
                --validation_step=1 \
                --num_val_images=1449 \
                --model=${1} \
                --frontend=${2} \
                --crop_height=${3} \
                --crop_width=${3} \
                --input_size=${3} \
                --batch_size=${4} \
                --dataset=${5}
                #--h_flip=yes \
                #--brightness=0.1 \
                #--rotation=5 \
wait