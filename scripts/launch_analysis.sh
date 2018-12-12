#!/usr/bin/env bash

#SBATCH --job-name=analysis
#SBATCH --output=/projets/thesepizenberg/deep-learning/logs/analysis.out
#SBATCH --error=/projets/thesepizenberg/deep-learning/logs/analysis.out

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=24CPUNodes
#SBATCH --mem-per-cpu=7500M

set -e

# Various script and dataset paths.
DATASET_NAMES=("voc-chh" "voc-ha")
GROUND_TRUTH_MASKS_DIR="/projets/thesepizenberg/deep-learning/deeplab-generic/matlab/Masques"
ANALYSIS_SCRIPT="/projets/thesepizenberg/deep-learning/segmentation-suite"

cd ${ANALYSIS_SCRIPT}

for dataset_name in "${DATASET_NAMES[@]}"; do
    echo "Launching analysis for dataset ${dataset_name}."
    srun -n1 -N1 /projets/thesepizenberg/deep-learning/deeplab-generic/matlab/venv/bin/python3.4 \
            "${ANALYSIS_SCRIPT}/analysis.py" --ground-truth-masks-directory=${GROUND_TRUTH_MASKS_DIR} \
            --dataset-name=${dataset_name} &
done;

wait
