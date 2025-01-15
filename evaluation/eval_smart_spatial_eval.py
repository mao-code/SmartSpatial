from SmartSpatialEval import SmartSpatialEvalPipeline
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
from .coco2017.prepare import COCO2017

import torch

from evaluation.eval_utils import (
    compute_statistics
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Evaluation"
    )
    parser.add_argument(
        "--img_root_path",
        type=str,
        default="results/spatial_prompts/smart_spatial",
        help="Path to the result image files."
    )

    parser.add_argument(
        "--dataset",
        choices=["spatial_prompts", "coco2017", "flikr30k"],
        default="spatial_prompts",
        help="Types of dataset to generate images"
    )

    return parser.parse_args()

def main():
    conf = OmegaConf.load(args.config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smart_spatial_eval = SmartSpatialEvalPipeline(conf, device)

    args = parse_args()
    img_root_path = args.img_root_path
    valid_extensions = ('.png', '.jpg', '.jpeg')

    image_paths = [
        os.path.join(img_root_path, file)
        for file in os.listdir(img_root_path)
        if file.lower().endswith(valid_extensions)
    ]

    prompt_datas = None
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

        # Flatten the prompt_datas
        all_prompt_datas = []
        for spatial_type in prompt_datas:
            all_prompt_datas += prompt_datas[spatial_type]

        prompt_datas = all_prompt_datas
    elif args.dataset == "coco2017":
        coco2017 = COCO2017()
        prompt_datas = coco2017.get_data()
    elif args.dataset == "flikr30k":
        raise NotImplementedError("Flikr30k dataset is not supported yet.")
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # Orders Matter !!!
    sorted_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split('_')[0]))

    smart_spatial_data = smart_spatial_eval.evaluate(sorted_paths, prompt_datas)

    stats = {}
    numeric_cols = ["D+O", "D", "O", "SR"]  # columns you want stats for
    
    for col in numeric_cols:
        mean_val, std_val, ci_lower, ci_upper = compute_statistics(smart_spatial_data[col].values, confidence=0.95)
        stats[col] = {
            "mean": mean_val,
            "std_dev": std_val,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper
        }

    print("=== Results DataFrame ===")
    print(smart_spatial_data)

    print("\n=== Summary Stats ===")
    for metric_name, info in stats.items():
        print(f"{metric_name} => mean: {info['mean']:.4f}, "
            f"std: {info['std_dev']:.4f}, "
            f"95% CI: [{info['ci_95_lower']:.4f}, {info['ci_95_upper']:.4f}]")