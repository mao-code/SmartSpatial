import pickle
import argparse
import shutil
from tqdm import tqdm
import os
import skimage.io as io

import torch
from omegaconf import OmegaConf

from SmartSpatial.pipeline import SmartSpatialPipeline
from utils import load_image
from SmartSpatial.utils import convert_bbox_data, pil_to_numpy, numpy_to_pt
from utils import bbox_ref_mapping

from ..prepare import COCO2017

front_depth_img = load_image("reference_images/depth_maps/front.png")
behind_depth_img = load_image("reference_images/depth_maps/behind.png")
left_depth_img = load_image("reference_images/depth_maps/left.png")
right_depth_img = load_image("reference_images/depth_maps/right.png")
on_depth_img = load_image("reference_images/depth_maps/on.png")
under_depth_img = load_image("reference_images/depth_maps/under.png")
above_depth_img = load_image("reference_images/depth_maps/above.png")
below_depth_img = load_image("reference_images/depth_maps/below.png")

depth_maps = {
    "front": front_depth_img,
    "behind": behind_depth_img,
    "left": left_depth_img,
    "right": right_depth_img,
    "on": on_depth_img,
    "under": under_depth_img,
    "above": above_depth_img,
    "below": below_depth_img
}

def generation_pipeline_coco2017(
    smart_spatial_pipeline,
    data,
    device,

    is_use_random_seed=True,
    is_save_simple_result=True,
    is_save_result=False,
    save_path="results/coco2017/smart_spatial",

    is_used_attention_guide=True,

    is_used_controlnet=False,
    is_used_controlnet_term=False,
    controlnet_scale=0.2,

    is_used_momentum=False,
    momentum=0.7,
):
    num_tested_images = 0
    root_save_path = save_path

    for (idx, prompt_data) in tqdm(enumerate(data), bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt}', total=len(data)):
        
        prompt = prompt_data['prompt']
        classes = [prompt_data['prompt_meta']['objects'][0]['obj'], prompt_data['prompt_meta']['center']]

        background = prompt_data['prompt_meta']['background']
        spatial_type = prompt_data['prompt_meta']['objects'][0]['pos']
        depth_boxes = bbox_ref_mapping[spatial_type]

        depth_map_tensor = None
        if is_used_controlnet:
            # depth_maps is a dict of PIL images
            depth_img = depth_maps[spatial_type]
            depth_map_tensor = pil_to_numpy(depth_img)
            depth_map_tensor = numpy_to_pt(depth_map_tensor)
            depth_map_tensor = depth_map_tensor.to(device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Depth map for spatial type {spatial_type} not found.")

        # Prepare bounding box data
        bboxes = convert_bbox_data(depth_boxes)
        bbox_datas = []
        for obj_name, bbox in zip(classes, bboxes):
            if obj_name.lower() in prompt.lower():
                bbox_data = {
                    "caption": obj_name,
                    "box": bbox
                }
                bbox_datas.append(bbox_data)

        # Call the pipeline function
        imgs = smart_spatial_pipeline.generate(
            prompt=prompt,
            bbox_datas=bbox_datas,

            # Save logic
            is_save_images=is_save_result,
            is_save_losses=is_save_result,
            save_path=save_path,

            # Attention guide
            is_used_attention_guide=is_used_attention_guide,

            # ControlNet
            is_used_controlnet=is_used_controlnet,
            is_used_controlnet_term=is_used_controlnet_term,
            control_image=depth_map_tensor,
            controlnet_scale=controlnet_scale,

            # Momentum
            is_used_momentum=is_used_momentum,
            momentum=momentum,

            # Additional placeholders:
            is_process_bbox_data=False,
            is_random_seed=is_use_random_seed
        )
        num_tested_images += 1

        if is_save_simple_result:
            img = imgs[0]
            short_prompt = prompt.replace(" ", "_")[:10]  # up to 10 chars
            current_save_path = f"{root_save_path}/{idx}_{spatial_type}_{short_prompt}"
            img.save(current_save_path+"smart_spatial.png")

    return num_tested_images

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SmartSpatial on COCO2017"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="conf/base_config.yaml",
        help="Path to the YAML config file."
    )
    parser.add_argument(
        "--use_random_seed",
        action="store_true",
        help="Whether to use random seed or not."
    )
    parser.add_argument(
        "--use_save_simple_result",
        action="store_true",
        help="Whether to save only the output images."
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="Whether to save the generated outputs."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/coco2017/smart_spatial",
        help="Directory to save the generated outputs."
    )
    parser.add_argument(
        "--use_attention_guide",
        action="store_true",
        help="Use attention guidance if provided."
    )
    parser.add_argument(
        "--use_controlnet",
        action="store_true",
        help="Use ControlNet if provided."
    )
    parser.add_argument(
        "--use_controlnet_term",
        action="store_true",
        help="Use ControlNet termination if provided."
    )
    parser.add_argument(
        "--controlnet_scale",
        type=float,
        default=0.2,
        help="Scaling factor for ControlNet."
    )
    parser.add_argument(
        "--use_momentum",
        action="store_true",
        help="Use momentum if provided."
    )
    parser.add_argument(
        "--momentum_value",
        type=float,
        default=0.2,
        help="Momentum value, if momentum is used."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load config via OmegaConf
    conf = OmegaConf.load(args.config_path)
    prompt_datas = COCO2017.get_data()

    # Decide on device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Initialize your pipeline
    smart_spatial = SmartSpatialPipeline(conf, device)

    # Create save directory if not exists
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Example usage of the generation pipeline
    generation_pipeline_coco2017(
        smart_spatial_pipeline=smart_spatial,
        data=prompt_datas,
        device=device,

        is_use_random_seed=args.use_random_seed,
        is_save_simple_result=args.use_save_simple_result,
        is_save_result=args.save_result,
        save_path=args.save_path,

        is_used_attention_guide=args.use_attention_guide,

        is_used_controlnet=args.use_controlnet,
        is_used_controlnet_term=args.use_controlnet_term,
        controlnet_scale=args.controlnet_scale,

        is_used_momentum=args.use_momentum,
        momentum=args.momentum_value,
    )

if __name__ == "__main__":
    main()
    
    """
        Example usage:
            python -m evaluation.coco2017.generation.smart_spatial \
            --config_path conf/base_config.yaml \
            --use_random_seed \
            --use_save_simple_result \
            --num_test -1 \
            --save_every 20 \
            --use_attention_guide \
            --use_controlnet \
            --use_controlnet_term \
            --controlnet_scale 0.2 \
            --use_momentum \
            --momentum_value 0.7
    """

