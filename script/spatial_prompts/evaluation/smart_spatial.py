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

def main():
    folder_path = 'results/spatial_prompts/smart_spatial'
    valid_extensions = ('.png', '.jpg', '.jpeg')

    # List comprehension to iterate over files in the folder and filter by extension
    image_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(valid_extensions)
    ]

    results_list = run_iou_clip(image_paths, prompt_datas, bbox_ref_mapping)
    avg_iou, avg_map, avg_clip_score = average_results(results_list)

    avg_iou, avg_map, avg_clip_score

if __name__ == "__main__":
    main()