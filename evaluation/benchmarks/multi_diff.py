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
from MultiDiffusion.region_based import (
    seed_everything,
    MultiDiffusion
)
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
        description="Run MultiDiffusion"
    )

    parser.add_argument(
        "--dataset",
        choices=["spatial_prompts", "coco2017", "visor"],
        default="spatial_prompts",
        help="Types of dataset to generate images"
    )

    return parser.parse_args()

def main():
    # Please first clone the MultiDiffusion repo
    # git clone https://github.com/omerbt/MultiDiffusion.git

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
    save_path = f"results/{args.dataset}/multi_diff"
    os.makedirs(save_path, exist_ok=True)

    for (idx, prompt_data) in tqdm(enumerate(prompt_datas), bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}', total=len(prompt_datas)):
        prompt = prompt_data['prompt']
        classes = [prompt_data['prompt_meta']['objects'][0]['obj'], prompt_data['prompt_meta']['center']]

        background = prompt_data['prompt_meta']['background']
        spatial_type = prompt_data['prompt_meta']['objects'][0]['pos']
        depth_boxes = bbox_ref_mapping[spatial_type]
        W, H = 512, 512

        seed_everything(42)
        device = torch.device('cuda')
        sd = MultiDiffusion(device, '1.5')

        # Create a mask image
        object_masks = []
        for bbox_item in depth_boxes:
            mask_np = create_mask_from_bbox(
                bbox_item['box'],
                image_width=W,
                image_height=H
            )

            # Convert to a torch tensor [1, H, W]
            mask_torch = torch.from_numpy(mask_np)[None, ...].to(device)  # shape = [1, H, W]
            object_masks.append(mask_torch)
        
        # Stack them: shape = [num_objects, 1, H, W]
        fg_masks = torch.cat(object_masks, dim=0).unsqueeze(1)  # each object is separate channel
        
        bg_mask = 1 - fg_masks.sum(dim=0, keepdim=True)  # shape = [1, 1, H, W]
        bg_mask = torch.clamp(bg_mask, 0, 1)
        masks = torch.cat([bg_mask, fg_masks])

        prompts = [background] + classes
        neg_prompts = ["low quality"] + ["low quality" for i in range(len(classes))]

        img = sd.generate(masks, prompts, neg_prompts, H, W, 50, bootstrapping=20)

        # save image
        short_prompt = prompt.replace(" ", "_")[:10]  # up to 10 chars
        current_save_path = f"{save_path}/{idx}_{spatial_type}_{short_prompt}"
        img.save(current_save_path+"multi_diff.png")

if __name__ == "__main__":
    main()