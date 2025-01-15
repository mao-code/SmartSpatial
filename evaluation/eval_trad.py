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
from .coco2017.prepare import COCO2017

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

    results_list = run_iou_clip(sorted_paths, prompt_datas, bbox_ref_mapping)
    avg_iou, avg_map, avg_clip_score = average_results(results_list)

    # -------------------------------------------
    # Gather all IOUs, mAPs, and CLIP scores 
    # into separate lists for more detailed stats:
    iou_vals = [res["per_image_mean_iou"] for res in results_list]
    map_vals = [res["per_image_map"] for res in results_list]
    clip_vals = [res["clip_score"] for res in results_list]

    # Compute mean, std, and 95% CI for each
    mean_iou, std_iou, ci_low_iou, ci_high_iou = compute_statistics(iou_vals, confidence=0.95)
    mean_map, std_map, ci_low_map, ci_high_map = compute_statistics(map_vals, confidence=0.95)
    mean_clip, std_clip, ci_low_clip, ci_high_clip = compute_statistics(clip_vals, confidence=0.95)
    # -------------------------------------------

    print(f"""
    ---------------------------------------
    Total Images Number: {len(sorted_paths)}
    Total Results Calculated: {len(results_list)}
    Evaluation Results for {img_root_path}:
        - Average per-image mean IoU: {avg_iou:.4f}
        - Average per-image mAP@0.5: {avg_map:.4f}
        - Average CLIP score: {avg_clip_score:.4f}

    ------------- Additional Stats -------------
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
    """)

if __name__ == "__main__":
    main()