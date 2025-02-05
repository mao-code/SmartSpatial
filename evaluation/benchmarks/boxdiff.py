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
from PIL import Image

from tqdm import tqdm
import argparse
from dataset.coco2017 import COCO2017
from dataset.visor import VISOR

# --------- BoxDiff imports ----------
# modify the import in BoxDiff (to relative path) for 2 files (run_sd_boxdiff.py and pipeline)
from BoxDiff.config import RunConfig
from BoxDiff.pipeline.sd_pipeline_boxdiff import BoxDiffPipeline
from BoxDiff.utils.ptp_utils import AttentionStore
from BoxDiff.run_sd_boxdiff import load_model, run_on_prompt
from pathlib import Path

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

def get_token_indices_for_classes(stable, prompt: str, classes: list[str]):
    # -- 1) Tokenize the prompt (no special tokens) --
    prompt_enc = stable.tokenizer(prompt, add_special_tokens=False)
    prompt_tokens = prompt_enc["input_ids"]

    # Prepare the result dict. Enumerate to keep duplicates separate.
    class2indices = {}
    for idx, cls in enumerate(classes):
        unique_key = f"{cls}_{idx}"
        class2indices[unique_key] = []  # store (start, end) pairs

    # -- 2) For each class, do a sliding-window match in the prompt tokens --
    i = 0 # non-overlapping index
    for idx, cls in enumerate(classes):
        unique_key = f"{cls}_{idx}"

        # Tokenize this class string
        cls_enc = stable.tokenizer(cls, add_special_tokens=False)
        cls_tokens = cls_enc["input_ids"]
        cls_len = len(cls_tokens)

        # Slide over the prompt tokens to find matches
        while i <= len(prompt_tokens) - cls_len:
            # Check if the sequence matches
            if prompt_tokens[i : i + cls_len] == cls_tokens:
                # Record the start/end of this match
                start_idx = i
                end_idx = i + cls_len - 1
                class2indices[unique_key].append(start_idx)

                # Move past this match (so we can find subsequent matches)
                i += cls_len

                break
            else:
                i += 1

    return class2indices

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BoxDiffusion"
    )

    parser.add_argument(
        "--dataset",
        choices=["spatial_prompts", "coco2017", "visor"],
        default="spatial_prompts",
        help="Types of dataset to generate images"
    )

    return parser.parse_args()

def main():
    # Please first clone the BoxDiff repo
    # git clone https://github.com/showlab/BoxDiff.git
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

    # Setup config/paths
    save_path = f"results/{args.dataset}/boxdiff"
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda')

    # Prepare a default config object for BoxDiff
    config = RunConfig(
        prompt="",
        sd_2_1=False,              
        seeds=[42],                
        output_path=Path(save_path),
        token_indices=None,        
        bbox=[],                 
        color=[(255,255,0),(255,0,255)],    
        guidance_scale=7.5,
        n_inference_steps=50,
        max_iter_to_alter=10,
        thresholds={0: 0.05, 10: 0.5, 20: 0.8},
        scale_factor=1.0,
        scale_range=(1.0, 0.5),
        smooth_attentions=False,
        sigma=0.5,
        kernel_size=3,
        run_standard_sd=False
    )
    stable = load_model(config)

    for (idx, prompt_data) in tqdm(enumerate(prompt_datas), bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}', total=len(prompt_datas)):
        prompt = prompt_data['prompt']
        classes = [prompt_data['prompt_meta']['objects'][0]['obj'], prompt_data['prompt_meta']['center']]
        class2idx = get_token_indices_for_classes(stable, prompt, classes)
        
        # Flatten all the indices if you want to alter them all at once:
        token_indices_to_alter = []
        for cls in class2idx:
            token_indices_to_alter += class2idx[cls]
        token_indices_to_alter = list(set(token_indices_to_alter))  # remove duplicates if any

        background = prompt_data['prompt_meta']['background']
        spatial_type = prompt_data['prompt_meta']['objects'][0]['pos']
        depth_boxes = bbox_ref_mapping[spatial_type] # 512x512
        W, H = 512, 512

        for depth_box in depth_boxes:
            depth_box = depth_box['box']
            x1, y1 = depth_box['x'], depth_box['y']
            x2, y2 = x1 + depth_box['w'], y1 + depth_box['h']

            config.bbox.append([x1, y1, x2, y2])

        config.prompt = prompt
        config.token_indices = token_indices_to_alter

        seed = torch.Generator(device=device).manual_seed(42)

        controller = AttentionStore()
        img = run_on_prompt(
            prompt=[prompt], 
            model=stable,
            controller=controller,
            token_indices=token_indices_to_alter,
            seed=seed,
            config=config
        )

        # save image
        short_prompt = prompt.replace(" ", "_")[:10]  # up to 10 chars
        current_save_path = f"{save_path}/{idx}_{spatial_type}_{short_prompt}"
        img.save(current_save_path+"box_diff.png")

if __name__ == "__main__":
    main()