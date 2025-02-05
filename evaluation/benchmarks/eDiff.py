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
from utils import load_image

import numpy as np
import torch

import os

import importlib

# Please rename "paint-with-words-sd" to "paint_with_words_sd"
from paint_with_words_sd.paint_with_words import paint_with_words

from PIL import Image
from tqdm import tqdm
import argparse
from dataset.coco2017 import COCO2017
from dataset.visor import VISOR

def create_mask_from_bbox(bbox, image_width, image_height):
    """
    Create a binary mask (as a NumPy array) for a given bounding box.
    - bbox: dict with 'x', 'y', 'w', 'h'
    - image_width, image_height: final dimensions for the mask
    """
    mask = np.zeros((image_height, image_width), dtype=np.float32)
    
    x_min = int(bbox['x'])
    y_min = int(bbox['y'])
    x_max = int(bbox['x'] + bbox['w'])
    y_max = int(bbox['y'] + bbox['h'])

    # Clip the bounding box values so they don't go out of image boundaries
    x_min = max(0, min(x_min, image_width))
    x_max = max(0, min(x_max, image_width))
    y_min = max(0, min(y_min, image_height))
    y_max = max(0, min(y_max, image_height))

    mask[y_min:y_max, x_min:x_max] = 1.0
    return mask

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run eDiff"
    )

    parser.add_argument(
        "--dataset",
        choices=["spatial_prompts", "coco2017", "visor"],
        default="spatial_prompts",
        help="Types of dataset to generate images"
    )

    return parser.parse_args()

def main():
    # Please first clone the eDiff repo
    # git clone https://github.com/cloneofsimo/paint-with-words-sd.git

    args = parse_args()

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
    elif args.dataset == "visor":
        visor = VISOR()
        prompt_datas = visor.get_data()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # 1. Setup config/paths
    save_path = f"results/{args.dataset}/eDiff"
    os.makedirs(save_path, exist_ok=True)
    
    for (idx, prompt_data) in tqdm(enumerate(prompt_datas), bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}', total=len(prompt_datas)):
        prompt = prompt_data['prompt']
        classes = [prompt_data['prompt_meta']['objects'][0]['obj'], prompt_data['prompt_meta']['center']]

        background = prompt_data['prompt_meta']['background']
        spatial_type = prompt_data['prompt_meta']['objects'][0]['pos']
        depth_boxes = bbox_ref_mapping[spatial_type]
        W, H = 512, 512

        # Create color map as a NumPy array
        color_map_np = np.zeros((H, W, 3), dtype=np.uint8)
        color_map_np[:] = (0, 0, 0)  # Background

        object_colors = [
            (255, 255, 255),
            (0, 255, 255)
        ]

        # Fill entire color map with "background color" first
        bg_color = (74, 18, 1)
        color_map_np[:, :] = bg_color

        # Create a mask image
        object_masks = []
        for i, bbox_item in enumerate(depth_boxes):
            obj_color = object_colors[i % len(object_colors)]
            mask_np = create_mask_from_bbox(
                bbox_item['box'],
                image_width=W,
                image_height=H
            )

            # Fill bounding box region in color_map_np
            # We only color the pixels where mask_np == 1
            # We'll do something like:
            rows, cols = np.where(mask_np == 1)
            color_map_np[rows, cols, 0] = obj_color[0]
            color_map_np[rows, cols, 1] = obj_color[1]
            color_map_np[rows, cols, 2] = obj_color[2]

        # Create a PIL image from the color_map_np
        color_map_image = Image.fromarray(color_map_np, mode='RGB')

        settings = {
            "color_context": {
                object_colors[0]: f"{classes[0]},1.0",
                object_colors[1]: f"{classes[1]},1.0",
                bg_color: f"{background},0.2",
            },
            "input_prompt": prompt,
        }

        color_context = settings["color_context"]
        input_prompt = settings["input_prompt"]

        img = paint_with_words(
            color_context=color_context,
            color_map_image=color_map_image,
            input_prompt=input_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            device="cuda"
        )

        # save image
        short_prompt = prompt.replace(" ", "_")[:10]  # up to 10 chars
        current_save_path = f"{save_path}/{idx}_{spatial_type}_{short_prompt}"
        img.save(current_save_path+"eDiff.png")

if __name__ == "__main__":
    main()