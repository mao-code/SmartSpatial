#!/bin/bash
# This script runs the evaluation for generation results of benchmarks with traditional metrics.
# Please change the "dataset" parameter to the desired dataset before running the script.

echo "Starting evaluation..."

python -m evaluation.eval_trad \
    --img_root_path /path/to/img_root \
    --dataset spatial_prompts \
    --output_path /path/to/output

echo "Evaluation completed.