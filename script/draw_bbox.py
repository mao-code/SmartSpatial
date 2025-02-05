from utils import draw_box
import os
from PIL import Image
from utils import bbox_ref_mapping
from SmartSpatial.utils import convert_bbox_data

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
from dataset.coco2017 import COCO2017
from dataset.visor import VISOR

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Evaluation"
    )

    parser.add_argument(
        "--dataset",
        choices=["spatial_prompts", "coco2017", "visor"],
        default="spatial_prompts",
        help="Types of dataset to generate images"
    )

    return parser.parse_args()

def main():
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
        print("COCO 2017 dataset preview: ", prompt_datas[:3])
    elif args.dataset == "visor":
        visor = VISOR()
        prompt_datas = visor.get_data()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    imgs_path = ""
    if args.dataset == "spatial_prompts":
        imgs_path = "qualitive/spatial_prompts"
    elif args.dataset == "coco2017":
        imgs_path = "qualitive/coco2017"
    elif args.dataset == "visor":
        imgs_path = "qualitive/visor"

    # Interate over the spatial prompts
    for data_id in os.listdir(imgs_path):   
        # Iterate over the images
        for img_id in os.listdir(os.path.join(imgs_path, data_id)):
            img_path = os.path.join(imgs_path, data_id, img_id)
            pil_img = Image.open(img_path)
            spatial_term = img_id.split("_")[1]
            data_idx = int(img_id.split("_")[0])
            prompt_data = prompt_datas[data_idx]

            bboxes = bbox_ref_mapping[spatial_term]
            bboxes = convert_bbox_data(bboxes)

            prompt = prompt_data['prompt']
            prompt_meta = prompt_data['prompt_meta']
            center = prompt_meta['center']
            obj_pos_pairs = prompt_meta['objects']
            classes = [obj_pos_pairs[0]['obj'], center]
            phrases = ";".join(classes)

            save_path = os.path.join(imgs_path, data_id, img_id.replace(".jpg", "_bbox.jpg"))
            print(f"Save path: {save_path}")
            draw_box(pil_img, bboxes, phrases, save_path)

if __name__ == "__main__":
    main()
    print("Done!")

    """
    python -m script.draw_bbox \
    --dataset visor
    """