from utils import bbox_ref_mapping
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

from evaluation.eval_utils import (
    run_iou_clip,
    average_results,
    compute_statistics
)

import os
import argparse
from dataset.coco2017 import COCO2017
from dataset.visor import VISOR

from pathlib import Path
import numpy as np  # For computing mean, etc.

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

def parse_args():
    parser = argparse.ArgumentParser(description="Run Evaluation")
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
        default="results/trad/result.txt",
        help="Path to save the evaluation results."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    img_root_path = args.img_root_path
    valid_extensions = ('.png', '.jpg', '.jpeg')

    image_paths = [
        os.path.join(img_root_path, file)
        for file in os.listdir(img_root_path)
        if file.lower().endswith(valid_extensions)
    ]

    if args.dataset == "spatial_prompts":
        # Flatten the prompt_datas
        all_prompt_datas = []
        for spatial_type, data_list in prompt_datas.items():
            all_prompt_datas += data_list
        final_prompt_datas = all_prompt_datas
    elif args.dataset == "coco2017":
        coco2017 = COCO2017()
        final_prompt_datas = coco2017.get_data()
    elif args.dataset == "visor":
        visor = VISOR()
        final_prompt_datas = visor.get_data()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # Orders Matter !!! 
    # (make sure the sort key matches how you enumerated image filenames)
    sorted_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split('_')[0]))

    results_list = run_iou_clip(sorted_paths, final_prompt_datas, bbox_ref_mapping)
    avg_iou, avg_map, avg_clip_score = average_results(results_list)

    # -------------------------------------------
    # Gather all IoUs, mAPs, and CLIP scores
    # into separate lists for more detailed stats
    iou_vals = [res["per_image_mean_iou"] for res in results_list]
    map_vals = [res["per_image_map"] for res in results_list]
    clip_vals = [res["clip_score"] for res in results_list]

    mean_iou, std_iou, ci_low_iou, ci_high_iou = compute_statistics(iou_vals, confidence=0.95)
    mean_map, std_map, ci_low_map, ci_high_map = compute_statistics(map_vals, confidence=0.95)
    mean_clip, std_clip, ci_low_clip, ci_high_clip = compute_statistics(clip_vals, confidence=0.95)
    # -------------------------------------------

    # -------------------------------------------
    # Compute metrics for each position word
    # -------------------------------------------

    position_metrics = {}  # key = position_word, val = dict of lists { "iou": [], "map": [], "clip": [] }

    for res in results_list:
        img_path = res["image_path"]
        # e.g.   somepath/123_front_something.jpg
        # split("/")[-1] => "123_front_something.jpg"
        # split("_")[1] => "front" 
        # Adjust if needed
        file_name = img_path.split("/")[-1]
        parts = file_name.split("_")

        if len(parts) < 2:
            # In case there's no underscore or the naming is unexpected
            position_word = "unknown"
        else:
            position_word = parts[1]

        if position_word not in position_metrics:
            position_metrics[position_word] = {
                "iou": [],
                "map": [],
                "clip": []
            }
        position_metrics[position_word]["iou"].append(res["per_image_mean_iou"])
        position_metrics[position_word]["map"].append(res["per_image_map"])
        position_metrics[position_word]["clip"].append(res["clip_score"])

    # Now compute stats per position
    position_stats_text = []
    for position, vals in position_metrics.items():
        ious = vals["iou"]
        maps = vals["map"]
        clips = vals["clip"]

        # Use your compute_statistics or simple mean if you want
        mean_iou_p, std_iou_p, ci_low_iou_p, ci_high_iou_p = compute_statistics(ious, confidence=0.95)
        mean_map_p, std_map_p, ci_low_map_p, ci_high_map_p = compute_statistics(maps, confidence=0.95)
        mean_clip_p, std_clip_p, ci_low_clip_p, ci_high_clip_p = compute_statistics(clips, confidence=0.95)

        stats_str = f"""
Position: {position}
    #Images: {len(ious)}
    IoU  - Mean: {mean_iou_p:.4f}, 95% CI: [{ci_low_iou_p:.4f}, {ci_high_iou_p:.4f}]
    mAP  - Mean: {mean_map_p:.4f}, 95% CI: [{ci_low_map_p:.4f}, {ci_high_map_p:.4f}]
    CLIP - Mean: {mean_clip_p:.4f}, 95% CI: [{ci_low_clip_p:.4f}, {ci_high_clip_p:.4f}]
"""
        position_stats_text.append(stats_str)

    # Combine all position-based stats
    position_stats_summary = "\n".join(position_stats_text)
    # -------------------------------------------

    txt_result = f"""
    ---------------------------------------
    Total Images Number: {len(sorted_paths)}
    Total Results Calculated: {len(results_list)}
    Evaluation Results for {img_root_path}:
        - Average per-image mean IoU: {avg_iou:.4f}
        - Average per-image mAP@0.5: {avg_map:.4f}
        - Average CLIP score: {avg_clip_score:.4f}

    ------------- Additional Stats (All Images) -------------
    IoU:
        - Mean: {mean_iou:.4f}
        - Std Dev: {std_iou:.4f}
        - 95% CI: [{ci_low_iou:.4f}, {ci_high_iou:.4f}]
    mAP:
        - Mean: {mean_map:.4f}
        - Std Dev: {std_map:.4f}
        - 95% CI: [{ci_low_map:.4f}, {ci_high_map:.4f}]
    CLIP Score:
        - Mean: {mean_clip:.4f}
        - Std Dev: {std_clip:.4f}
        - 95% CI: [{ci_low_clip:.4f}, {ci_high_clip:.4f}]
    ---------------------------------------

    ================ Per-Position Metrics ================
    {position_stats_summary}
    -----------------------------------------------------
    """

    print(txt_result)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "a") as f:
        f.write(txt_result + "\n")

if __name__ == "__main__":
    main()