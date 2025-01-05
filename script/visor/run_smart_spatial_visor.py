from omegaconf import OmegaConf
import torch
from ...SmartSpatial.pipeline import SmartSpatialPipeline
import json

conf = OmegaConf.load('conf/base_config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
smart_spatial = SmartSpatialPipeline(conf, device)

save_path = 'output/visor'
model_name = "smart_spatial"

with open('./text_spatial_rel_phrases.json', 'r') as f:
    text_data = json.load(f)

    for data in text_data:
        uniq_id = data["unique_id"]
        free_form_prompt = text_data[uniq_id]["text"]
        obj1 = text_data[uniq_id]["obj_1_attributes"][0]
        obj2 = text_data[uniq_id]["obj_2_attributes"][0]
        rel = text_data[uniq_id]["rel_type"]

        if rel == "to the left of":
            rel = "left"
        elif rel == "to the right of":
            rel = "right"
        else:
            rel = rel

        depth_map = f"../../reference_images/depth_maps/{rel}.png"

        for i in range(4): # 4 images per prompt
            img_id = "{}_{}".format(uniq_id, i)
            impath = save_path + model_name + "/{}.png".format(img_id)
   
            img = smart_spatial.generate(
                is_save_images=False,
                save_path=save_path,

                is_used_attention_guide=True,

                is_used_controlnet=True,
                control_image=depth_map,
                controlnet_scale=0.5,

                is_used_momentum=True,
                momentum=0.5,

                is_process_bbox_data=False
            )

            img.save(impath)