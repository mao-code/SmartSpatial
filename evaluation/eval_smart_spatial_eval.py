from SmartSpatialEval.pipeline import SmartSpatialEvalPipeline
from omegaconf import OmegaConf
import os
import argparse

from dataset.spatial_prompt import (
    prompt_datas_front,
    prompt_datas_behind,
    prompt_datas_left,
    prompt_datas_right,
    prompt_datas_on,
    prompt_datas_under,
    prompt_datas_above,
    prompt_datas_below
)
from dataset.coco2017 import COCO2017
from dataset.visor import VISOR

import torch

from evaluation.eval_utils import (
    compute_statistics
)

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Evaluation"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="conf/base_config.yaml",
        help="Path to the YAML config file."
    )

    parser.add_argument(
        "--img_root_path",
        type=str,
        default="results/spatial_prompts/smart_spatial",
        help="Path to the result image files."
    )

    parser.add_argument(
        "--dataset",
        choices=["spatial_prompts", "coco2017", "visor"],
        default="spatial_prompts",
        help="Types of dataset to generate images"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="results/smart_spatial_eval/result.txt",
        help="Path to save the evaluation results."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Load config and initialize pipeline
    conf = OmegaConf.load(args.config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smart_spatial_eval = SmartSpatialEvalPipeline(conf, device)

    # Gather image paths
    img_root_path = args.img_root_path
    valid_extensions = ('.png', '.jpg', '.jpeg')

    image_paths = [
        os.path.join(img_root_path, file)
        for file in os.listdir(img_root_path)
        if file.lower().endswith(valid_extensions)
    ]

    # Prepare data for your chosen dataset
    if args.dataset == "spatial_prompts":
        prompt_datas = {
            "front": prompt_datas_front,
            "behind": prompt_datas_behind,
            "left": prompt_datas_left,
            "right": prompt_datas_right,
            "on": prompt_datas_on,
            "under": prompt_datas_under,
            "above": prompt_datas_above,
            "below": prompt_datas_below
        }
        # Flatten prompt_datas
        all_prompt_datas = []
        for spatial_type in prompt_datas:
            all_prompt_datas += prompt_datas[spatial_type]
        prompt_datas = all_prompt_datas
    elif args.dataset == "coco2017":
        coco2017 = COCO2017()
        prompt_datas = coco2017.get_data()
    elif args.dataset == "visor":
        visor = VISOR()
        prompt_datas = visor.get_data()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # Sort to match the order in which results are processed
    sorted_paths = sorted(
        image_paths,
        key=lambda x: int(x.split('/')[-1].split('_')[0])
    )

    # Evaluate results
    smart_spatial_data = smart_spatial_eval.evaluate(sorted_paths, prompt_datas)
    
    # ----------------------------------------------------------------
    # 1) Add the image path as a column to the DataFrame
    #    so that each row can be associated with its file name.
    # 2) Parse out the position word from the filename (e.g., "front").
    # ----------------------------------------------------------------
    # Assume the evaluate(...) returns a DataFrame with the same ordering 
    # as sorted_paths, so we can assign them directly:
    smart_spatial_data["image_path"] = sorted_paths
    
    # Example filenames: "123_front_something.jpg"
    # To extract "front":
    def get_position_word(path):
        file_name = os.path.basename(path)   # e.g. "123_front_something.jpg"
        parts = file_name.split('_')         # ["123", "front", "something.jpg"]
        if len(parts) > 1:
            return parts[1]
        return "unknown"

    smart_spatial_data["position_word"] = smart_spatial_data["image_path"].apply(get_position_word)

    # 3) Overall Stats (all images together)
    numeric_cols = ["D+O", "D", "O", "SR"]  # columns you want stats for
    stats = {}

    for col in numeric_cols:
        mean_val, std_val, ci_lower, ci_upper = compute_statistics(
            smart_spatial_data[col].values, confidence=0.95
        )
        stats[col] = {
            "mean": mean_val,
            "std_dev": std_val,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper
        }

    # 4) Per-Position-Word Stats
    # Group by "position_word" and compute the same stats for each group
    position_groups = smart_spatial_data.groupby("position_word")

    per_position_stats = {}  # { position_word: { metric: { "mean": ..., "std": ..., ...}}}

    for position_word, group_df in position_groups:
        per_position_stats[position_word] = {}
        for col in numeric_cols:
            mean_val, std_val, ci_lower, ci_upper = compute_statistics(
                group_df[col].values, confidence=0.95
            )
            per_position_stats[position_word][col] = {
                "mean": mean_val,
                "std_dev": std_val,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper
            }

    # ----------------------------------------------------------------
    # Build a textual summary of results
    # ----------------------------------------------------------------
    result_txt = f"Result for {args.img_root_path}\n"
    result_txt += "=== Overall Summary Stats (All Images) ===\n"
    for metric_name, info in stats.items():
        result_txt += f"""
{metric_name} =>
    mean: {info['mean']:.4f}
    std: {info['std_dev']:.4f}
    95% CI: [{info['ci_95_lower']:.4f}, {info['ci_95_upper']:.4f}]
    margin of error: {info['ci_95_upper'] - info['mean']:.4f}
"""

    result_txt += "\n=== Per-Position-Word Stats ===\n"
    for position_word, metrics_dict in per_position_stats.items():
        result_txt += f"\nPosition Word: {position_word}\n"
        for metric_name, info in metrics_dict.items():
            result_txt += f"""    {metric_name} =>
        mean: {info['mean']:.4f}
        std: {info['std_dev']:.4f}
        95% CI: [{info['ci_95_lower']:.4f}, {info['ci_95_upper']:.4f}]
        margin of error: {info['ci_95_upper'] - info['mean']:.4f}\n
"""

    print(result_txt)

    # Save to file
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(result_txt + "\n")

if __name__ == "__main__":
    main()