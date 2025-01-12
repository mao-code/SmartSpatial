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

def generation_pipeline_coco2017(
    smart_spatial_pipeline,
    data,
    device,

    start_image_index=0,
    end_index=1000,

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

    with tqdm(total=end_index - start_image_index, desc="Generating Images") as pbar:
      for idx in range(start_image_index, end_index):
        data_dict = data[idx]

        # Basic information
        img_url = data_dict['image_url']
        image = io.imread(img_url) # numpy
        caption = data_dict['caption']
        bboxes = [boxes.squeeze(0) for boxes in data_dict['bounding_boxes']]
        classes = data_dict["classes"]

        save_path = f"{root_save_path}/{idx}_{caption[:10]}"

        # Normalize the coordinate of bounding boxes
        original_height, original_width = image.shape[:2] # numpy with shape HxWxC
        # Define resized depth map dimensions
        resized_width, resized_height = 512, 512

        # Calculate scaling factors
        scale_x = resized_width / original_width
        scale_y = resized_height / original_height

        # Scale the bounding boxes to match the resized depth map
        scaled_bboxes = [
            [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            for [x1, y1, x2, y2] in bboxes
        ]

        # Normalize the scaled bounding boxes based on resized dimensions
        normalized_bboxes = [
            [[x1 / resized_width, y1 / resized_height, x2 / resized_width, y2 / resized_height]]
            for [x1, y1, x2, y2] in scaled_bboxes
        ]

        # Use depth helper function to convert it to a depth map
        depth_map = None
        if is_used_controlnet:
            depth_map = smart_spatial_pipeline.preprocess_depth_map(img_url)
            depth_map = depth_map.resize((512, 512))

            depth_map = pil_to_numpy(depth_map)
            depth_map = numpy_to_pt(depth_map)
            depth_map = depth_map.to(device=device, dtype=torch.float32)

        # Organize the data and pass it to the attention guidance function for different settings
        # Prepare the bounding box data
        bbox_datas = []
        for obj_name, bbox in zip(classes, normalized_bboxes):
            if obj_name.lower() in caption.lower():
                bbox_data = {
                    "caption": obj_name,
                    "box": bbox
                }
                bbox_datas.append(bbox_data)

        # Call the pipeline function
        imgs = smart_spatial_pipeline.generate(
            prompt=caption,
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
            control_image=depth_map,
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
            short_prompt = caption.replace(" ", "_")[:10]  # up to 10 chars
            current_save_path = f"{save_path}/{idx}_{short_prompt}"
            img.save(current_save_path+"smart_spatial.png")

        # Update the progress bar
        pbar.update(1)

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
        "--start_index",
        type=int,
        default=0,
        help="Starting index for prompt data."
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=1000,
        help="Ending index for prompt data."
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
    return parser.parse_args()

def main():
    args = parse_args()

    # !gdown 14e9BODX0PEe-FpCORvB1bdwZOb7hInxB -O coco2017.pkl
    """
    [
        {
            'image_url': 'http://images.cocodataset.org/val2017/000000397133.jpg',
            'caption': 'The person is standing to the right of the oven.',
            'bounding_boxes': [
                tensor([[385.0713,  68.8378, 498.9966, 347.7491]]),
                tensor([[233.2383, 191.3256, 396.4366, 312.5436]])
            ],
            'classes': ['person', 'oven']
        },
        ...
    ]
    """

    with open('coco2017.pkl', 'rb') as f:
        data = pickle.load(f)

    # Load config via OmegaConf
    conf = OmegaConf.load(args.config_path)

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
        data=data,
        device=device,

        start_image_index=args.start_index,
        end_index=args.end_index,

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
            python -m script.spatial_prompts.generation.smart_spatial \
            --config_path conf/base_config.yaml \
            --use_random_seed \
            --use_save_simple_result \
            --start_index 0 \
            --num_test -1 \
            --save_every 20 \
            --use_attention_guide \
            --use_controlnet \
            --use_controlnet_term \
            --controlnet_scale 0.2 \
            --use_momentum \
            --momentum_value 0.7
    """

