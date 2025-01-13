#!/bin/bash
# This script runs the evaluation module for eDiff using Python.
# Please change the "dataset" parameter to the desired dataset before running the script.

echo "Starting evaluation for eDiff..."

python -m evaluation.eDiff \
    --dataset spatial_prompts \

echo "Evaluation completed."