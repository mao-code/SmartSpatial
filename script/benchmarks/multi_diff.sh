#!/bin/bash
# This script runs the evaluation module for MultiDiffusion using Python.
# Please change the "dataset" parameter to the desired dataset before running the script.

echo "Starting evaluation for MultiDiffusion..."

python -m evaluation.benchmarks.multi_diff \
    --dataset spatial_prompts

echo "Evaluation completed."