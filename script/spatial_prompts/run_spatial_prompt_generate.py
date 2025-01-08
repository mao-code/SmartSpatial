# from omegaconf import OmegaConf
# import torch
# from SmartSpatial.pipeline import SmartSpatialPipeline
# import os

# conf = OmegaConf.load('conf/base_config.yaml')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# smart_spatial = SmartSpatialPipeline(conf, device)

# def generation_pipeline_spatial_prompt(
#     cfg,

#     prompt_datas,

#     save_path="output",

#     is_used_attention_guide=True,

#     is_used_controlnet=False,
#     is_used_controlnet_term=False,
#     controlnet=None,
#     controlnet_scale = 1.0,
#     single_control_image=None,
#     depth_maps={},
#     bbox_ref_mapping=None,
#     spatial_types=["front"],

#     is_used_momentum=False,
#     momentum=0.2,

#     is_used_salt=False,
#     obj_set=None,
#     bg_set=None,

#     start_index=0,
#     num_test=-1,
#     save_every=10
# ):
#     num_tested_images = 0
#     root_save_path = save_path

#     spatial_type_prompt_datas = []
#     for spatial_type in spatial_types:
#       spatial_type_prompt_datas += prompt_datas[spatial_type]

#     if start_index < 0 or start_index >= len(spatial_type_prompt_datas):
#       raise ValueError("start_index is out of bounds.")

#     end_index = len(spatial_type_prompt_datas) if num_test==-1 else min(start_index + num_test, len(spatial_type_prompt_datas))

#     with tqdm(total=end_index - start_index, desc="Generating Images") as pbar:
#       for idx in range(start_index, end_index):
#           data_dict = spatial_type_prompt_datas[idx]

#           # Basic information
#           prompt = data_dict['prompt']
#           prompt_meta = data_dict['prompt_meta']
#           center = prompt_meta['center']
#           obj_pos_pairs = prompt_meta['objects']
#           spatial_type = obj_pos_pairs[0]['pos']

#           save_path = f"{root_save_path}/{idx}_{spatial_type}_{prompt[:10]}"

#           # Get the corresponding bbox data for different reference image
#           bboxes = bbox_ref_mapping[spatial_type] # ball(obj), box(center)
#           classes = [obj_pos_pairs[0]['obj'], center] # assume only 2
#           bboxes = convert_bbox_data(bboxes)

#           # Step 2: ControlNet
#           depth_map = None
#           if is_used_controlnet:
#             if depth_maps:
#               depth_map = depth_maps[spatial_type]
#               depth_map = pil_to_numpy(depth_map)
#               depth_map = numpy_to_pt(depth_map)
#               depth_map = depth_map.to(device=device, dtype=torch.float32)
#             else:
#               depth_map = preprocess_depth_map(single_control_image)
#               depth_map = depth_map.resize((512, 512))

#               # Store the depth map
#               if not os.path.exists(save_path):
#                 os.makedirs(save_path)

#               depth_map_path = os.path.join(save_path, "depth_map.png")
#               depth_map.save(depth_map_path)
#               depth_map = pil_to_numpy(depth_map)
#               depth_map = numpy_to_pt(depth_map)
#               depth_map = depth_map.to(device=device, dtype=torch.float32)

#         # Step 4: Organize the data and pass it to the attention guidance function for different settings
#           # Prepare the bounding box data
#           bbox_datas = []
#           for obj_name, bbox in zip(classes, bboxes):
#             if obj_name.lower() in prompt.lower():

#               bbox_data = {
#                   "caption": obj_name,
#                   "box": bbox
#               }
#               bbox_datas.append(bbox_data)

#         # Step5: Call the attention guidance function
#         # Store: images(including attention maps), losses, iteration steps
#         smart_spatial.generate(
#                 prompt=free_form_prompt,
#                 bbox_datas=bbox_datas,

#                 is_save_images=False,
#                 is_save_losses=False,
#                 save_path=save_path,

#                 is_used_attention_guide=True,

#                 is_used_controlnet=True,
#                 is_used_controlnet_term=True,
#                 control_image=depth_map,
#                 controlnet_scale=0.2,

#                 is_used_momentum=True,
#                 momentum=0.5,

#                 is_process_bbox_data=False,
#                 is_random_seed=False
#             )
#           num_tested_images += 1

#           # Check if it's time to create a zip file
#           if num_tested_images % save_every == 0 or num_tested_images == num_test or idx == end_index-1:
#               zip_filename = f"{root_save_path}" # Base name without extension
#               try:
#                   # Remove existing zip file if it exists
#                   existing_zip = f"{zip_filename}.zip"
#                   if os.path.exists(existing_zip):
#                       os.remove(existing_zip)

#                   # Create a new zip file using shutil.make_archive
#                   shutil.make_archive(zip_filename, 'zip', root_save_path)
#                   print(f"\nCreated zip file: {existing_zip} after processing {num_tested_images} images.")
#               except Exception as e:
#                   print(f"\nFailed to create zip file: {e}")

#           # Update the progress bar
#           pbar.update(1)

#     return num_tested_images


# from utils import bbox_ref_mapping
# from prompt import (
#     prompt_datas_front,
#     prompt_datas_behind,
#     prompt_datas_left,
#     prompt_datas_right,
#     prompt_datas_on,
#     prompt_datas_under,
#     prompt_datas_above,
#     prompt_datas_below
# )

# prompt_datas = {
#     "front": prompt_datas_front,
#     "behind": prompt_datas_behind,
#     "left": prompt_datas_left,
#     "right": prompt_datas_right,
#     "on": prompt_datas_on,
#     "under": prompt_datas_under,
#     "above": prompt_datas_above,
#     "below": prompt_datas_below
# }

# from utils import load_image

# front_depth_img = load_image("reference_images/depth_maps/front.png")
# behind_depth_img = load_image("reference_images/depth_maps/behind.png")
# left_depth_img = load_image("reference_images/depth_maps/left.png")
# right_depth_img = load_image("reference_images/depth_maps/right.png")
# on_depth_img = load_image("reference_images/depth_maps/on.png")
# under_depth_img = load_image("reference_images/depth_maps/under.png")
# above_depth_img = load_image("reference_images/depth_maps/above.png")
# below_depth_img = load_image("reference_images/depth_maps/below.png")

# depth_maps = {
#     "front": front_depth_img,
#     "behind": behind_depth_img,
#     "left": left_depth_img,
#     "right": right_depth_img,
#     "on": on_depth_img,
#     "under": under_depth_img,
#     "above": above_depth_img,
#     "below": below_depth_img
# }

# save_path = 'output/visor'
# model_name = "smart_spatial"
# save_path = save_path + "/" + model_name
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# # ControlNet & AG & Momentum
# evaluation_pipeline_prompt(
#     conf,
#     unet,
#     tokenizer,
#     text_encoder,
#     vae,

#     prompt_datas=prompt_datas,

#     save_path="output_controlnet_obj_mom",

#     is_used_attention_guide=True,

#     is_used_controlnet=True,
#     is_used_controlnet_term=True,
#     controlnet=controlnet,
#     controlnet_scale = 0.2,
#     depth_maps=depth_maps,
#     bbox_ref_mapping=bbox_ref_mapping,
#     spatial_types=[
#         'front',
#         'behind',
#         'left',
#         'right',
#         'on',
#         'under',
#         'above',
#         'below'
#     ],

#     is_used_momentum=True,
#     momentum=0.7,

#     is_used_salt=False,
#     obj_set=None,
#     bg_set=None,

#     start_index=0,
#     num_test=-1, # set to -1 for all data
#     save_every=20
# )