from utils import load_image
import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
import pickle

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

    # !gdown 14e9BODX0PEe-FpCORvB1bdwZOb7hInxB -O coco2017.pkl
    with open('coco2017.pkl', 'rb') as f:
        data = pickle.load(f)

    # Setup config/paths
    save_path = "results/coco2017/boxdiff"
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

    for (idx, data_dict) in tqdm(enumerate(data), bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}', total=len(data)):
        # Basic information
        caption = data_dict['caption']
        bboxes = [boxes.squeeze(0).tolist() for boxes in data_dict['bounding_boxes']]
        classes = data_dict["classes"]

        prompt = caption
        class2idx = get_token_indices_for_classes(stable, prompt, classes)
        
        # Flatten all the indices if you want to alter them all at once:
        token_indices_to_alter = []
        for cls in class2idx:
            token_indices_to_alter += class2idx[cls]
        token_indices_to_alter = list(set(token_indices_to_alter))  # remove duplicates if any

        for box in bboxes:
            config.bbox.append(box)

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
        current_save_path = f"{save_path}/{idx}_{short_prompt}"
        img.save(current_save_path+"box_diff.png")

if __name__ == "__main__":
    main()