#!/bin/bash
# This script runs the evaluation module for SmartSpatial using Python.
# Please change the "dataset" parameter to the desired dataset before running the script.

# For COCO2017, please execure the following command first:
# 1. %mkdir coco
# 2. %cd coco
# 3. !wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# 4. !unzip annotations_trainval2017.zip
# 5. %cd ..

echo "Starting evaluation for SmartSpatial..."

python -m evaluation.benchmarks.smart_spatial \
    --dataset spatial_prompts \
    --save_path results/spatial_prompts/smart_spatial \
    --config_path conf/base_config.yaml \
    --use_random_seed \
    --use_save_simple_result \
    --use_attention_guide \
    --use_special_token_guide \
    --use_controlnet \
    --use_controlnet_term \
    --controlnet_scale 0.2 \
    --use_momentum \
    --momentum_value 0.7 \
    --noise_scheduler_type ddim 

echo "Evaluation completed."



