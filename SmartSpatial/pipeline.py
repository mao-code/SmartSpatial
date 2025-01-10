from transformers import pipeline, CLIPTextModel, CLIPTokenizer
from my_model import controlnet, unet_2d_condition
from diffusers import DDIMScheduler, AutoencoderKL, LMSDiscreteScheduler
import json

import numpy as np
from PIL import Image

from SmartSpatial.utils import (
    pil_to_numpy, 
    numpy_to_pt, 
    save_to_pkl, 
    compute_ca_loss, 
    visualize_attention_maps,
    convert_bbox_data
)
from utils import Pharse2idx, sentence_to_list, draw_box, setup_logger
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm

class SmartSpatialPipeline():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        
        self.depth_estimator = pipeline(
            cfg.general.depth_estimator.type, 
            model=cfg.general.depth_estimator.path
        )

        with open(cfg.general.unet_config) as f:
            unet_config = json.load(f)

        self.unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path, subfolder="unet")
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")

        self.ddim_scheduler = DDIMScheduler.from_pretrained(cfg.general.model_path, subfolder="scheduler")

        # controlnet
        self.controlnet = controlnet.ControlNetModel.from_pretrained(cfg.general.controlnet_path)

        self.unet.to(device)
        self.text_encoder.to(device)
        self.vae.to(device)
        self.controlnet.to(device)

    def preprocess_depth_map(self, ref_img):
        """
        Use the depth estimator to generate a depth map from a reference image.
        """
        depth_image = self.depth_estimator(ref_img)['depth']  # Get depth map
        depth_image = np.array(depth_image)
        depth_image = depth_image[:, :, None]  # Add channel dimension
        depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)  # Convert to 3 channels
        depth_map = Image.fromarray(depth_image)

        return depth_map

    def inference(
        self,

        prompt,
        bboxes,
        phrases,

        center,
        position,

        logger,

        is_save_attn_maps=False,
        is_save_losses=False,
        save_path="output",

        is_used_attention_guide=True,

        is_used_controlnet=False,
        is_used_controlnet_term=False,
        control_image=None,
        controlnet_scale = 0.2,

        is_used_momentum=False,
        momentum = 0.5,

        is_used_salt=False,
        prompt_data=None,
        obj_set=None,
        bg_set=None,
        bbox_ref_mapping=None,

        is_random_seed=False,

        negative_prompt=""
    ):
        logger.info("Inference")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Phrases: {phrases}")

        # Get Object Positions
        # logger.info("Convert Phrases to Object Positions")
        object_positions = Pharse2idx(prompt, phrases)

        # Get position words indices
        position_word_indices = []
        position_word_indices = [prompt.find(position) + 1]

        # Encode Classifier Embeddings
        if negative_prompt:
            uncond_input = self.tokenizer(
                [negative_prompt] * self.cfg.inference.batch_size, 
                padding="max_length", 
                max_length=self.tokenizer.model_max_length, 
                return_tensors="pt"
            )
        else:
            uncond_input = self.tokenizer(
                [""] * self.cfg.inference.batch_size, 
                padding="max_length", 
                max_length=self.tokenizer.model_max_length, 
                return_tensors="pt"
            )

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Encode Prompt
        input_ids = self.tokenizer(
                [prompt] * self.cfg.inference.batch_size,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )

        cond_embeddings = self.text_encoder(input_ids.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        if is_random_seed:
            generator = torch.manual_seed(self.cfg.inference.rand_seed)
        else:
            generator = None

        # noise_scheduler = LMSDiscreteScheduler(
        #     beta_start=cfg.noise_schedule.beta_start,
        #     beta_end=cfg.noise_schedule.beta_end,
        #     beta_schedule=cfg.noise_schedule.beta_schedule,
        #     num_train_timesteps=cfg.noise_schedule.num_train_timesteps
        # )

        # Initialize DDIMScheduler
        noise_scheduler = self.ddim_scheduler

        noise_scheduler.set_timesteps(self.cfg.inference.timesteps)
        # noise_scheduler.set_timesteps(61)

        latents = torch.randn(
            (self.cfg.inference.batch_size, 4, 64, 64),
            generator=generator,
        ).to(self.device)
        latents = latents * noise_scheduler.init_noise_sigma

        # if is_used_salt:
        #     latents = prepare_salt(
        #         cfg,
        #         prompt_data=prompt_data,

        #         vae=vae,
        #         unet=unet,
        #         tokenizer=tokenizer,
        #         text_encoder=text_encoder,
        #         scheduler=noise_scheduler,

        #         bbox_ref_mapping=bbox_ref_mapping,

        #         objects_set=obj_set, # dcit: name: path
        #         background_set=bg_set
        #     )[-1].unsqueeze(0).to(device)
        #     latents = latents * noise_scheduler.init_noise_sigma

        loss = torch.tensor(10000)
        losses_each_step = []
        iterations_each_step = []

        # Denoising loop
        # for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        for index, t in enumerate(noise_scheduler.timesteps):
            # MODIFIED: Adjust guidance scale based on step for finer control
            guidance_scale = 10 if index < len(noise_scheduler.timesteps) // 2 else 7.5

            if is_used_attention_guide:
                iteration = 0
                losses_each_iteration = []

                v = torch.zeros_like(latents)

                # Guidance
                loss_threshold = self.cfg.inference.loss_threshold
                # loss_threshold = 0.5
                max_iter = self.cfg.inference.max_iter
                # max_iter = 5
                # while loss.item() / cfg.inference.loss_scale > cfg.inference.loss_threshold and iteration < cfg.inference.max_iter and index < cfg.inference.max_index_step:
                while loss.item() / self.cfg.inference.loss_scale > loss_threshold and iteration < max_iter:
                    # print(f"Timestep: {index}, iteration: {iteration+1}-------------")

                    latents = latents.requires_grad_(True)
                    latent_model_input = latents
                    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

                    # ControlNet AG Guidance
                    attn_down_control=None
                    attn_mid_control=None
                    if is_used_controlnet:
                        controlnet_input = latent_model_input
                        controlnet_prompt_embeds = cond_embeddings

                        # Get ControlNet outputs
                        control_output, attn_down_control, attn_mid_control = self.controlnet(
                            controlnet_input,
                            t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=control_image,
                            conditioning_scale=controlnet_scale,
                            return_dict=True,
                        )

                        down_block_res_samples, mid_block_res_sample = control_output['down_block_res_samples'], control_output['mid_block_res_sample']

                        noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                            self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=cond_embeddings,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                                return_dict=True
                            )
                    else:
                        noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=cond_embeddings
                            )

                    # Update latents with guidance
                    loss = compute_ca_loss(
                        attn_map_integrated_mid,
                        attn_map_integrated_up,
                        bboxes=bboxes,
                        object_positions=object_positions,
                        position_word_indices=position_word_indices,
                        alpha=1,

                        is_used_controlnet_term=is_used_controlnet_term,
                        beta=0.3,
                        attn_maps_down_control=attn_down_control,
                        attn_maps_mid_control=attn_mid_control

                    ) * self.cfg.inference.loss_scale # scale factor

                    # Store the loss for monitoring
                    losses_each_iteration.append(loss.item())

                    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

                    # variance = noise_scheduler.sigmas[index] ** 2 # Original: LMSD scheduler, variance
                    # variance = noise_scheduler.betas[index]
                    variance = 1
                    if is_used_momentum:
                        v = momentum * v - grad_cond * variance
                        latents = latents + v
                    else:
                        latents = latents - grad_cond * variance

                    iteration += 1
                    torch.cuda.empty_cache()

                # Store the iteration step for monotoring
                if iteration == 0:
                    iterations_each_step.append(0)

                    last_loss = losses_each_step[-1][-1]
                    losses_each_step.append([last_loss])
                else:
                    iterations_each_step.append(iteration)
                    losses_each_step.append(losses_each_iteration)

                if is_save_attn_maps and index % 10 == 0:
                    # Visualize the attention map
                    visualize_attention_maps(
                        attention_maps=attn_map_integrated_mid,
                        prompt=prompt,
                        object_positions=object_positions,
                        timestep=index,
                        is_save_attn_maps=is_save_attn_maps,
                        save_path=f"{save_path}/attention_maps_mid",
                        is_all_tokens=False
                    )

                    visualize_attention_maps(
                        attention_maps=attn_map_integrated_up[0],
                        prompt=prompt,
                        object_positions=object_positions,
                        timestep=index,
                        is_save_attn_maps=is_save_attn_maps,
                        save_path=f"{save_path}/attention_maps_up",
                        is_all_tokens=False
                    )

                    if is_used_controlnet:
                        # visualize_attention_maps(
                        #     attention_maps=attn_down_control[0],
                        #     prompt=prompt,
                        #     object_positions=object_positions,
                        #     timestep=index,
                        #     is_save_attn_maps=is_save_attn_maps,
                        #     save_path=f"{save_path}/attention_maps_down_control",
                        #     is_all_tokens=False
                        # )

                        visualize_attention_maps(
                            attention_maps=attn_mid_control,
                            prompt=prompt,
                            object_positions=object_positions,
                            timestep=index,
                            is_save_attn_maps=is_save_attn_maps,
                            save_path=f"{save_path}/attention_maps_mid_control",
                            is_all_tokens=False
                        )

            # After getting the optimized latent
            with torch.no_grad():
                latent_model_input = torch.cat([latents] * 2) # CFG
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

                if is_used_controlnet:
                    control_image_cfg = torch.cat([control_image] * 2) # if CFG
                    # Prepare the input for ControlNet
                    # controlnet_input = noise_scheduler.scale_model_input(latents.clone(), t)
                    controlnet_input = latent_model_input
                    controlnet_prompt_embeds = text_embeddings

                    # Get ControlNet outputs
                    control_output, attn_down_control, attn_mid_control = self.controlnet(
                        controlnet_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=control_image_cfg,
                        conditioning_scale=controlnet_scale,
                        return_dict=True,
                    )

                    down_block_res_samples, mid_block_res_sample = control_output['down_block_res_samples'], control_output['mid_block_res_sample']

                    # Pass through UNet with ControlNet conditioning
                    noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                        self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=text_embeddings,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            return_dict=True
                        )

                    noise_pred = noise_pred.sample

                else:
                    noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                        self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=text_embeddings,
                            return_dict=True
                        )
                    noise_pred = noise_pred.sample

                # MODIFIED: Dynamically scale guidance for flexibility in complex prompts
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
                torch.cuda.empty_cache()

                if is_save_attn_maps and not is_used_attention_guide and index % 10 == 0:
                    # Visualize the attention map without guidance
                    visualize_attention_maps(
                        attention_maps=attn_map_integrated_mid,
                        prompt=prompt,
                        object_positions=object_positions,
                        timestep=index,
                        is_save_attn_maps=is_save_attn_maps,
                        save_path=f"{save_path}/attention_maps_mid",
                        is_all_tokens=False
                    )

                    visualize_attention_maps(
                        attention_maps=attn_map_integrated_up[0],
                        prompt=prompt,
                        object_positions=object_positions,
                        timestep=index,
                        is_save_attn_maps=is_save_attn_maps,
                        save_path=f"{save_path}/attention_maps_up",
                        is_all_tokens=False
                    )

        # Save losses and iterations to a pickle file
        if is_used_attention_guide:
            logger.info(f"\nFirst Loss: {losses_each_step[0][0]}")
            logger.info(f"Last Loss: {losses_each_step[-1][-1]}")

            if is_save_losses:
                save_to_pkl(losses_each_step, f"{save_path}/losses.pkl")
                save_to_pkl(iterations_each_step, f"{save_path}/iterations.pkl")

        # Decode to pixel space
        with torch.no_grad():
            logger.info("Decode Image...")
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]

            return pil_images

    def generate(
        self,

        prompt,
        bbox_datas,

        is_save_images=False,
        is_save_attn_maps=False,
        is_save_losses=False,
        save_path="output",

        is_used_attention_guide=True,

        is_used_controlnet=False,
        is_used_controlnet_term=False,
        control_image=None,
        controlnet_scale = 0.2,

        is_used_momentum=False,
        momentum = 0.5,

        is_used_salt=False,
        prompt_data=None,
        obj_set=None,
        bg_set=None,
        bbox_ref_mapping=None,

        is_process_bbox_data=True,
        is_random_seed=False,
        negative_prompt="ball,box,low resolution,bad composition,blurry image,bad anatomy"
    ):

        # ------------------ example input ------------------
        # examples = {"prompt": "A hello kitty toy is playing with a purple ball.",
        #             "phrases": "hello kitty; ball",
        #             "bboxes": [[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]],
        #             'save_path': cfg.general.save_path
        #             }

        position_words = [
            "on",
            "above",
            "under",
            "below",
            "left",
            "right",
            "behind",
            "front"
        ]

        if is_process_bbox_data:
            bboxes = convert_bbox_data(bbox_datas)
        else:
            bboxes = [bbox_data['box'] for bbox_data in bbox_datas]

        objects = []
        objects_in_prompt = []
        bboxes_in_prompt = []
        for idx, bbox_data in enumerate(bbox_datas):
            obj_name = bbox_data['caption']
            objects.append(obj_name)

            if obj_name in prompt:
                objects_in_prompt.append(obj_name)
                bboxes_in_prompt.append(bboxes[idx])


        position = ""
        for pos in position_words:
            if prompt.find(pos) != -1:
                position = pos
                break

        center_obj = objects[1] if len(objects) == 2 else objects[0]
        examples = {
            "prompt": prompt,
            "phrases": "; ".join(objects_in_prompt),
            "bboxes": bboxes_in_prompt,
            'save_path': save_path,

            'center': center_obj,
            'position': position
        }

        # ---------------------------------------------------
        # Prepare the save path
        if not os.path.exists(self.cfg.general.save_path):
            os.makedirs(self.cfg.general.save_path)
        logger = setup_logger(self.cfg.general.save_path, __name__)

        # logger.info(cfg)
        # Save cfg
        # logger.info("save config to {}".format(os.path.join(cfg.general.save_path, 'config.yaml')))
        OmegaConf.save(self.cfg, os.path.join(self.cfg.general.save_path, 'config.yaml'))

        # Inference
        pil_images = self.inference(
            examples['prompt'],
            examples['bboxes'],
            examples['phrases'],
            examples['center'],
            examples['position'],

            logger,

            is_save_attn_maps=is_save_attn_maps,
            is_save_losses=is_save_losses,
            save_path=save_path,

            is_used_attention_guide=is_used_attention_guide,

            is_used_controlnet=is_used_controlnet,
            is_used_controlnet_term=is_used_controlnet_term,
            control_image=control_image,
            controlnet_scale=controlnet_scale,

            is_used_momentum=is_used_momentum,
            momentum=momentum,

            is_used_salt=is_used_salt,
            prompt_data=prompt_data,
            obj_set=obj_set,
            bg_set=bg_set,
            bbox_ref_mapping=bbox_ref_mapping,

            is_random_seed=is_random_seed,
            negative_prompt=negative_prompt
        )

        if is_save_images:
            """
            Store:
            1. images
            2. attention maps
            3. losses
            4. iteration steps
            """

            # Save images
            for index, pil_image in enumerate(pil_images):
                image_path = os.path.join(save_path, 'output_{}.png'.format(index))
                pil_image.save(image_path)
                # logger.info('save example image to {}'.format(image_path))

                bbox_image_path = os.path.join(save_path, 'output_bboxes_{}.png'.format(index))
                draw_box(
                    pil_image,
                    examples['bboxes'],
                    examples['phrases'],
                    bbox_image_path
                ) # save the bbox output image

        torch.cuda.empty_cache()

        return pil_images
