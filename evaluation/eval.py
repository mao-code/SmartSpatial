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

from eval_utils import (
    run_iou_clip,
    average_results
)

import os
import argparse

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

    # Orders Matter !!!
    sorted_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split('_')[0]))

    results_list = run_iou_clip(sorted_paths, prompt_datas, bbox_ref_mapping)
    avg_iou, avg_map, avg_clip_score = average_results(results_list)

    print(f"""
    ---------------------------------------
    Total Images Number: {len(sorted_paths)}
    Total Results Calculated: {len(results_list)}
    Evaluation Results for {img_root_path}:
        - Average per-image mean IoU: {avg_iou:.4f}
        - Average per-image mAP@0.5: {avg_map:.4f}
        - Average CLIP score: {avg_clip_score:.4f}
    ---------------------------------------
    """)    

if __name__ == "__main__":
    main()