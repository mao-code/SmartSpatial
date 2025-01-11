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

from tqdm import tqdm

# --------- BoxDiff imports ----------
from BoxDiff.config import RunConfig
from BoxDiff.pipeline.sd_pipeline_boxdiff import BoxDiffPipeline
from BoxDiff.utils.ptp_utils import AttentionStore
from BoxDiff.run_boxdiff import load_model, run_on_prompt

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
    """
    Return a dict mapping each class in `classes` to the indices of the tokens
    in `prompt` that contain that class string.
    """
    # Encode the prompt using the pipeline's tokenizer.
    encoded = stable.tokenizer(prompt, add_special_tokens=False)
    # For human readability, we also get the list of tokens (strings)
    tokens = stable.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    
    # Prepare a dict to hold token indices for each class
    class2indices = {cls: [] for cls in classes}
    
    # Find token indices for each class
    for i, token in enumerate(tokens):
        # Because tokens can be subwords like "cat" vs "ca", "##t", 
        # do a naive substring check (lowercased).
        for cls in classes:
            if cls.lower() in token.lower():
                class2indices[cls].append(i)
    
    return class2indices

def main():
    # Please first clone the BoxDiff repo
    # git clone https://github.com/showlab/BoxDiff.git

    # Setup config/paths
    save_path = "results/spatial_prompts/boxdiff"
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda')

    # Flatten the prompt_datas
    all_prompt_datas = []
    for spatial_type in prompt_datas:
      all_prompt_datas += prompt_datas[spatial_type]

    # Prepare a default config object for BoxDiff
    config = RunConfig(
        prompt=prompt,
        sd_2_1=False,              
        seeds=[42],                
        output_path=os.path.abspath(save_path),
        token_indices=None,        
        bbox=[],                 
        color=[(255,255,0),(255,0,255)]       
        guidance_scale=7.5,
        n_inference_steps=50,
        max_iter_to_alter=10,
        thresholds=[0.3],
        scale_factor=1.0,
        scale_range=1,
        smooth_attentions=False,
        sigma=0.5,
        kernel_size=3,
        run_standard_sd=False
    )
    stable = load_model(config)

    for (idx, prompt_data) in tqdm(enumerate(all_prompt_datas), bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}', total=len(all_prompt_datas)):
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