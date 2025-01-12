import numpy as np
import torch

import os
from MultiDiffusion.region_based import (
    seed_everything,
    MultiDiffusion
)
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
    # Please first clone the MultiDiffusion repo
    # git clone https://github.com/omerbt/MultiDiffusion.git

    # !gdown 14e9BODX0PEe-FpCORvB1bdwZOb7hInxB -O coco2017.pkl
    with open('coco2017.pkl', 'rb') as f:
        data = pickle.load(f)

    # 1. Setup config/paths
    save_path = "results/coco2017/multi_diff"
    os.makedirs(save_path, exist_ok=True)

    for (idx, data_dict) in tqdm(enumerate(data), bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}', total=len(data)):
        # Basic information
        caption = data_dict['caption']
        bboxes = [boxes.squeeze(0).tolist() for boxes in data_dict['bounding_boxes']]
        classes = data_dict["classes"]

        prompt = caption

        W, H = 512, 512

        seed_everything(42)
        device = torch.device('cuda')
        sd = MultiDiffusion(device, '1.5')

        # Create a mask image
        object_masks = []
        for bbox_item in bboxes:
            mask_np = create_mask_from_bbox(
                bbox_item,
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

        prompts = classes
        neg_prompts = ["low quality"] + ["low quality" for i in range(len(classes))]

        img = sd.generate(masks, prompts, neg_prompts, H, W, 50, bootstrapping=20)

        # save image
        short_prompt = prompt.replace(" ", "_")[:10]  # up to 10 chars
        current_save_path = f"{save_path}/{idx}_{short_prompt}"
        img.save(current_save_path+"multi_diff.png")

if __name__ == "__main__":
    main()