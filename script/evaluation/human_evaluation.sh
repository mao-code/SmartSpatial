#!/bin/bash
# This script runs the evaluation module for eDiff using Python.
# Please change the "dataset" parameter to the desired dataset before running the script.

echo "Starting statistic analysis for human evaluation..."

python -m HumanEvaluation.statistic_analysis

echo "Evaluation completed."