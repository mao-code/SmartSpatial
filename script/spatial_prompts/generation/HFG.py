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

import numpy as np
import torch

import os
from PIL import Image

# Rename "GradOP-Guided-Image-Synthesis" to "GradOP_Guided_Image_Synthesis"
from GradOP_Guided_Image_Synthesis.pipeline_gradop_stroke2img import GradOPStroke2ImgPipeline
from tqdm import tqdm

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

def main():
    # Please first clone the HFG repo
    # git clone https://github.com/1jsingh/GradOP-Guided-Image-Synthesis

    # 1. Setup config/paths
    save_path = "results/spatial_prompts/hfg"
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda')

    # Flatten the prompt_datas
    all_prompt_datas = []
    for spatial_type in prompt_datas:
      all_prompt_datas += prompt_datas[spatial_type]

    for (idx, prompt_data) in tqdm(enumerate(all_prompt_datas), bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}', total=len(all_prompt_datas)):
        prompt = prompt_data['prompt']
        classes = [prompt_data['prompt_meta']['objects'][0]['obj'], prompt_data['prompt_meta']['center']]

        background = prompt_data['prompt_meta']['background']
        spatial_type = prompt_data['prompt_meta']['objects'][0]['pos']
        depth_boxes = bbox_ref_mapping[spatial_type]
        W, H = 512, 512

        pipeline = GradOPStroke2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32
        ).to(device)

        # define the guidance inputs: 1) text prompt and 2) guidance image containing coarse scribbles
        seed = 0
        
        # Create color map as a NumPy array (Stroke Image)
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
        stroke_img = color_map_image

        # perform img2img guided synthesis using gradop+
        generator = torch.Generator(device=device).manual_seed(seed)
        img = pipeline.gradop_plus_stroke2img(prompt, stroke_img, strength=0.8, num_iterative_steps=3, grad_steps_per_iter=12, generator=generator)

        # save image
        short_prompt = prompt.replace(" ", "_")[:10]  # up to 10 chars
        current_save_path = f"{save_path}/{idx}_{spatial_type}_{short_prompt}"
        img.save(current_save_path+"hfg.png")

if __name__ == "__main__":
    main()