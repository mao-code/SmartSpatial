import os
import argparse
import shutil
from tqdm import tqdm

import torch
from omegaconf import OmegaConf

from SmartSpatial.pipeline import SmartSpatialPipeline
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
from SmartSpatial.utils import convert_bbox_data, pil_to_numpy, numpy_to_pt
from dataset.coco2017 import COCO2017
from dataset.visor import VISOR

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

def generation_pipeline_spatial_prompt(
    smart_spatial_pipeline,
    prompt_datas,
    device,
    is_use_random_seed=True,
    is_save_simple_result=True,
    is_save_result=False,
    save_path="results/spatial_prompts/smart_spatial",

    is_used_attention_guide=True,
    is_special_token_guide=True,

    is_used_controlnet=False,
    is_used_controlnet_term=False,
    controlnet_scale=1.0,
    depth_maps=None,
    bbox_ref_mapping=None,
    spatial_types=None,

    is_used_momentum=False,
    momentum=0.2,

    start_index=0,
    num_test=-1,
    save_every=10,

    noise_scheduler_type="ddim"
):
    """
    This function runs the generation pipeline with or without ControlNet,
    momentum, attention guidance, etc.
    """

    if spatial_types is None:
        spatial_types = ["front"]

    num_tested_images = 0
    root_save_path = save_path

    if start_index < 0 or start_index >= len(prompt_datas):
        raise ValueError("start_index is out of bounds.")

    end_index = (len(prompt_datas) if num_test == -1 
                 else min(start_index + num_test, len(prompt_datas)))

    with tqdm(total=end_index - start_index, desc="Generating Images") as pbar:
        for idx in range(start_index, end_index):
            data_dict = prompt_datas[idx]

            # Basic information
            prompt = data_dict['prompt']
            prompt_meta = data_dict['prompt_meta']
            center = prompt_meta['center']
            obj_pos_pairs = prompt_meta['objects']
            # E.g., 'front', 'behind', ...
            spatial_type = obj_pos_pairs[0]['pos']

            # Create a shortened prompt to avoid super-long folder names
            short_prompt = prompt.replace(" ", "_")[:10]  # up to 10 chars
            current_save_path = f"{root_save_path}/{idx}_{spatial_type}_{short_prompt}"

            # Get the corresponding bbox data for different reference images
            bboxes = bbox_ref_mapping[spatial_type]  # example: [ball(obj), box(center)]
            classes = [obj_pos_pairs[0]['obj'], center]  # assume only 2
            bboxes = convert_bbox_data(bboxes)

            # Prepare a depth_map if controlnet is used
            depth_map_tensor = None
            if is_used_controlnet:
                if depth_maps and spatial_type in depth_maps:
                    # depth_maps is a dict of PIL images
                    depth_img = depth_maps[spatial_type]
                    depth_map_tensor = pil_to_numpy(depth_img)
                    depth_map_tensor = numpy_to_pt(depth_map_tensor)
                    depth_map_tensor = depth_map_tensor.to(device=device, dtype=torch.float32)
                else:
                    raise ValueError(f"Depth map for spatial type {spatial_type} not found.")

            # Prepare bounding box data
            bbox_datas = []
            for obj_name, bbox in zip(classes, bboxes):
                if obj_name.lower() in prompt.lower():
                    bbox_data = {
                        "caption": obj_name,
                        "box": bbox
                    }
                    bbox_datas.append(bbox_data)

            # =============== CALL THE SMART SPATIAL PIPELINE ===============
            imgs = smart_spatial_pipeline.generate(
                prompt=prompt,
                bbox_datas=bbox_datas,

                # Save logic
                is_save_images=is_save_result,
                is_save_losses=is_save_result,
                is_save_attn_maps=is_save_result,
                save_path=current_save_path,

                # Attention guide
                is_used_attention_guide=is_used_attention_guide,
                is_special_token_guide=is_special_token_guide,

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
                is_random_seed=is_use_random_seed,

                noise_scheduler_type=noise_scheduler_type,

                is_stay_close=False,
                stay_close_weight=0.1,
        
                is_smoothness=False,
                smoothness_weight=0.01,

                is_grad_clipping=False,
                grad_clip_threshold=1.0,
            )
            num_tested_images += 1

            # Check if it's time to create a zip file
            if is_save_result:
                if (num_tested_images % save_every == 0 
                    or (num_tested_images == num_test and num_test != -1)
                    or idx == end_index - 1):
                    zip_filename = f"{root_save_path}"
                    try:
                        existing_zip = f"{zip_filename}.zip"
                        if os.path.exists(existing_zip):
                            os.remove(existing_zip)
                        shutil.make_archive(zip_filename, 'zip', root_save_path)
                        print(f"\nCreated zip file: {existing_zip} after processing {num_tested_images} images.")
                    except Exception as e:
                        print(f"\nFailed to create zip file: {e}")

            if is_save_simple_result:
                img = imgs[0]
                short_prompt = prompt.replace(" ", "_")[:10]  # up to 10 chars
                current_save_path = f"{save_path}/{idx}_{spatial_type}_{short_prompt}"
                img.save(current_save_path+"smart_spatial.png")

            pbar.update(1)

    return num_tested_images

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SmartSpatialPipeline"
    )

    parser.add_argument(
        "--dataset",
        choices=["spatial_prompts", "coco2017", "visor"],
        default="spatial_prompts",
        help="Types of dataset to generate images"
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
        help="Whether to save the generated outputs including attention maps and loss."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/spatial_prompts/smart_spatial",
        help="Directory to save the generated outputs."
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index for prompt data."
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=-1,
        help="Number of items to test. Use -1 to process all."
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Number of processed items after which to create a zip file."
    )
    parser.add_argument(
        "--use_attention_guide",
        action="store_true",
        help="Use attention guidance if provided."
    )
    parser.add_argument(
        "--use_special_token_guide",
        action="store_true",
        help="Use attention guidance for special tokens if provided."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Cross-Attention loss term for UNet."
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
        "--beta",
        type=float,
        default=0.3,
        help="Cross-Attention loss term for ControlNet."
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
    parser.add_argument(
        "--use_salt",
        action="store_true",
        help="Use 'salt' if your pipeline supports it."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: 'cpu', 'cuda', or 'auto' to detect automatically."
    )

    parser.add_argument(
        "--noise_scheduler_type",
        choices=["ddim", "lmsdiscrete"],
        default="ddim",
        help="Types of noise scheduler to generate images"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load config via OmegaConf
    conf = OmegaConf.load(args.config_path)

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
        print("COCO 2017 dataset preview: ", prompt_datas[:3])
    elif args.dataset == "visor":
        visor = VISOR()
        prompt_datas = visor.get_data()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

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

    # save_path = f"results/{args.dataset}/smart_spatial"
    # os.makedirs(save_path, exist_ok=True)

    # Example usage of the generation pipeline
    generation_pipeline_spatial_prompt(
        smart_spatial_pipeline=smart_spatial,
        prompt_datas=prompt_datas,
        device=device,
        is_use_random_seed=args.use_random_seed,
        is_save_simple_result=args.use_save_simple_result,
        is_save_result=args.save_result,
        save_path=args.save_path,

        is_used_attention_guide=args.use_attention_guide,
        is_special_token_guide=args.use_special_token_guide,
        alpha=args.alpha,

        is_used_controlnet=args.use_controlnet,
        is_used_controlnet_term=args.use_controlnet_term,
        controlnet_scale=args.controlnet_scale,
        beta=args.beta,
        depth_maps=depth_maps,
        bbox_ref_mapping=bbox_ref_mapping,
        spatial_types=["front", "behind", "left", "right", "on", "under", "above", "below"],

        is_used_momentum=args.use_momentum,
        momentum=args.momentum_value,

        start_index=args.start_index,
        num_test=args.num_test,
        save_every=args.save_every,

        noise_scheduler_type=args.noise_scheduler_type
    )

if __name__ == "__main__":
    main()

    """
        Example usage:
            python -m evaluation.benchmarks.smart_spatial \
            --dataset spatial_prompts \
            --config_path conf/base_config.yaml \
            --use_random_seed \
            --use_save_simple_result \
            --use_attention_guide \
            --use_controlnet \
            --use_controlnet_term \
            --controlnet_scale 0.2 \
            --use_momentum \
            --momentum_value 0.7
    """