#!/bin/bash
# This script runs the evaluation module for SmartSpatial using Python.
# Please change the "dataset" parameter to the desired dataset before running the script.

echo "Starting evaluation for SmartSpatial..."

python -m evaluation.smart_spatial \
    --dataset spatial_prompts \
    --config_path conf/base_config.yaml \
    --use_random_seed \
    --use_save_simple_result \
    --use_attention_guide \
    --use_controlnet \
    --use_controlnet_term \
    --controlnet_scale 0.2 \
    --use_momentum \
    --momentum_value 0.7

echo "Evaluation completed."