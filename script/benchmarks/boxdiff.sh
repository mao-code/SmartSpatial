#!/bin/bash
# This script runs the evaluation module for BoxDiffusion using Python.
# Please change the "dataset" parameter to the desired dataset before running the script.

echo "Starting evaluation for BoxDiffusion..."

python -m evaluation.benchmarks.boxdiff \
    --dataset spatial_prompts

echo "Evaluation completed."