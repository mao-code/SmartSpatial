import numpy as np
import torch
import os
import importlib

# Please rename "paint-with-words-sd" to "paint_with_words_sd"
from paint_with_words_sd.paint_with_words import paint_with_words

from PIL import Image
from tqdm import tqdm
import pickle

def create_mask_from_bbox(bbox, image_width, image_height):
    """
    Create a binary mask (as a NumPy array) for a given bounding box.
    """
    mask = np.zeros((image_height, image_width), dtype=np.float32)
    
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = int(bbox[2])
    y_max = int(bbox[3])

    # Clip the bounding box values so they don't go out of image boundaries
    x_min = max(0, min(x_min, image_width))
    x_max = max(0, min(x_max, image_width))
    y_min = max(0, min(y_min, image_height))
    y_max = max(0, min(y_max, image_height))

    mask[y_min:y_max, x_min:x_max] = 1.0
    return mask

def main():
    # Please first clone the eDiff repo
    # git clone https://github.com/cloneofsimo/paint-with-words-sd.git

    # !gdown 14e9BODX0PEe-FpCORvB1bdwZOb7hInxB -O coco2017.pkl
    with open('coco2017.pkl', 'rb') as f:
        data = pickle.load(f)

    # 1. Setup config/paths
    save_path = "results/coco2017/eDiff"
    os.makedirs(save_path, exist_ok=True)
    
    for (idx, data_dict) in tqdm(enumerate(data), bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}', total=len(data)):
        # Basic information
        caption = data_dict['caption']
        bboxes = [boxes.squeeze(0).tolist() for boxes in data_dict['bounding_boxes']]
        classes = data_dict["classes"]

        prompt = caption

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
        for i, bbox_item in enumerate(bboxes):
            obj_color = object_colors[i % len(object_colors)]
            mask_np = create_mask_from_bbox(
                bbox_item,
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
        current_save_path = f"{save_path}/{idx}_{short_prompt}"
        img.save(current_save_path+"eDiff.png")

if __name__ == "__main__":
    main()