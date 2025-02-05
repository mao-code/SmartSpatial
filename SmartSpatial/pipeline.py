from transformers import pipeline, CLIPTextModel, CLIPTokenizer
from my_model import controlnet, unet_2d_condition
from diffusers import DDIMScheduler, LMSDiscreteScheduler, AutoencoderKL
import json

import numpy as np
from PIL import Image

from SmartSpatial.utils import (
    save_to_pkl, 
    compute_ca_loss, 
    visualize_attention_maps,
    convert_bbox_data,

    get_special_token_indices,
    Pharse2idx
)
from utils import draw_box, setup_logger
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

        self.lmsdiscrete_scheduler = LMSDiscreteScheduler(
            beta_start=cfg.noise_schedule.beta_start, 
            beta_end=cfg.noise_schedule.beta_end,
            beta_schedule=cfg.noise_schedule.beta_schedule, 
            num_train_timesteps=cfg.noise_schedule.num_train_timesteps
        )

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
        logger,
        is_save_attn_maps=False,
        is_save_losses=False,
        save_path="output",

        is_used_attention_guide=True,
        is_special_token_guide=False,
        alpha=1.0,

        is_used_controlnet=False,
        is_used_controlnet_term=False,
        control_image=None,
        controlnet_scale=0.2,
        beta=0.3,

        is_used_momentum=False,
        momentum=0.5,

        is_stay_close=False,
        stay_close_weight=0.1,
 
        is_smoothness=False,
        smoothness_weight=0.01,

        is_grad_clipping=False,
        grad_clip_threshold=1.0,

        is_random_seed=False,
        negative_prompt="",

        noise_scheduler_type="ddim"
    ):
        logger.info("Inference")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Phrases: {phrases}")

        # Get object words indices in the output of the tokenizer for the prompt
        object_positions = Pharse2idx(prompt, phrases)

        # Encode classifier embeddings
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

        # Encode the prompt
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

        # Special token indices
        sot_idx, eot_idx = None, None
        if is_special_token_guide:
            sot_idx, eot_idx = get_special_token_indices(self.tokenizer, prompt)

        # Initialize DDIMScheduler
        # You can also try other schedulers like "LMSDiscreteScheduler"
        if noise_scheduler_type == "ddim":
            noise_scheduler = self.ddim_scheduler
        elif noise_scheduler_type == "lmsdiscrete":
            noise_scheduler = self.lmsdiscrete_scheduler
        else:
            raise ValueError(f"Unknown noise scheduler type: {noise_scheduler_type}")

        noise_scheduler.set_timesteps(self.cfg.inference.timesteps)

        latents = torch.randn(
            (self.cfg.inference.batch_size, 4, 64, 64),
            generator=generator,
        ).to(self.device)
        latents = latents * noise_scheduler.init_noise_sigma

        loss = torch.tensor(10000)
        losses_each_step = []
        iterations_each_step = []

        # -----------------------------
        # Denoising loop
        # -----------------------------
        for index, t in enumerate(noise_scheduler.timesteps):
            # Dynamic guidance scale based on step for finer control
            guidance_scale = 10 if index < len(noise_scheduler.timesteps) // 2 else 7.5

            # ----------------------------------------------------------------------
            #  A) Get the "Original" attention maps for this step (before guidance)
            # ----------------------------------------------------------------------
            with torch.no_grad():
                # We'll feed the current latents (unmodified by the iterative loop)
                latents_temp = latents.detach().clone()
                latents_temp = noise_scheduler.scale_model_input(latents_temp, t)

                if is_used_controlnet:
                    # Pass latents_temp through ControlNet
                    control_output, attn_down_control, attn_mid_control = self.controlnet(
                        latents_temp,
                        t,
                        encoder_hidden_states=cond_embeddings,  # or text_embeddings if CFG
                        controlnet_cond=control_image,
                        conditioning_scale=controlnet_scale,
                        return_dict=True,
                    )
                    down_block_res_samples = control_output['down_block_res_samples']
                    mid_block_res_sample = control_output['mid_block_res_sample']

                    # Then pass through UNet
                    unet_out = self.unet(
                        latents_temp,
                        t,
                        encoder_hidden_states=cond_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=True
                    )
                    noise_pred_orig, attn_map_up_orig, attn_map_mid_orig, attn_map_down_orig = unet_out
                    # ControlNet's mid attn maps are in attn_mid_control
                else:
                    unet_out = self.unet(
                        latents_temp,
                        t,
                        encoder_hidden_states=cond_embeddings,
                        return_dict=True
                    )
                    noise_pred_orig, attn_map_up_orig, attn_map_mid_orig, attn_map_down_orig = unet_out
                    attn_mid_control = None

            # If needed for "Stay Close", store them in lists or clones
            # NOTE: You only need to store them if is_stay_close=True, but storing
            # them unconditionally is also fine.
            orig_attn_maps_mid = [x.detach().clone() for x in attn_map_mid_orig]  # mid-block
            # up block might be a list-of-lists, so handle carefully:
            if len(attn_map_up_orig) > 0 and isinstance(attn_map_up_orig[0], list):
                # e.g. attn_map_up_orig[0] is a list of Tensors for each up-block
                orig_attn_maps_up = [
                    [tensor.detach().clone() for tensor in block] for block in attn_map_up_orig
                ]
            else:
                # e.g. if you only store first up-block
                orig_attn_maps_up = [
                    [tensor.detach().clone() for tensor in attn_map_up_orig]
                ]

            # For ControlNet's mid block
            orig_attn_maps_mid_control = None
            if is_used_controlnet and (attn_mid_control is not None):
                orig_attn_maps_mid_control = [
                    c_map.detach().clone() for c_map in attn_mid_control
                ]

            # -------------------------------------------------------------------
            #  B) Iterative Guidance to refine latents
            # -------------------------------------------------------------------
            if is_used_attention_guide:
                iteration = 0
                losses_each_iteration = []

                v = torch.zeros_like(latents)

                loss_threshold = self.cfg.inference.loss_threshold
                max_iter = self.cfg.inference.max_iter

                while (loss.item() / self.cfg.inference.loss_scale > loss_threshold) and \
                    (iteration < max_iter):

                    latents = latents.requires_grad_(True)
                    latent_model_input = noise_scheduler.scale_model_input(latents, t)

                    # Forward pass for the guided step
                    if is_used_controlnet:
                        control_output, attn_down_control, attn_mid_control = self.controlnet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=cond_embeddings,
                            controlnet_cond=control_image,
                            conditioning_scale=controlnet_scale,
                            return_dict=True,
                        )
                        down_block_res_samples = control_output['down_block_res_samples']
                        mid_block_res_sample = control_output['mid_block_res_sample']

                        unet_out = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=cond_embeddings,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            return_dict=True
                        )
                        noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = unet_out
                    else:
                        unet_out = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=cond_embeddings,
                            return_dict=True
                        )
                        noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = unet_out

                    # ---- Compute your cross-attention loss ----
                    loss = compute_ca_loss(
                        attn_maps_mid=attn_map_integrated_mid,
                        attn_maps_up=attn_map_integrated_up,
                        bboxes=bboxes,
                        object_positions=object_positions,
                        alpha=alpha,

                        # ControlNet
                        is_used_controlnet_term=is_used_controlnet_term,
                        beta=beta,
                        attn_maps_mid_control=attn_mid_control,

                        # Special tokens
                        is_special_token_guide=is_special_token_guide,
                        sot_idx=sot_idx,
                        eot_idx=eot_idx,
                        gamma=1.0,

                        is_stay_close=is_stay_close,
                        stay_close_weight=stay_close_weight,
                        orig_attn_maps_mid=orig_attn_maps_mid,
                        orig_attn_maps_up=orig_attn_maps_up,
                        orig_attn_maps_mid_control=orig_attn_maps_mid_control,

                        is_smoothness=is_smoothness,
                        smoothness_weight=smoothness_weight,
                    ) * self.cfg.inference.loss_scale

                    # Store for monitoring
                    losses_each_iteration.append(loss.item())

                    # Gradient w.r.t. latents
                    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

                    # Optional gradient clipping for latents
                    if is_grad_clipping:
                        grad_norm = grad_cond.norm(p=2)
                        if grad_norm > grad_clip_threshold:
                            grad_cond = grad_cond * (grad_clip_threshold / (grad_norm + 1e-8))

                    # Update latents (momentum or direct)
                    variance = 1.0
                    if noise_scheduler_type == "ddim":
                        variance = 1.0
                    elif noise_scheduler_type == "lmsdiscrete":
                        variance = noise_scheduler.sigmas[index] ** 2

                    if is_used_momentum:
                        v = momentum * v - grad_cond * variance
                        latents = latents + v
                    else:
                        latents = latents - grad_cond * variance

                    iteration += 1
                    torch.cuda.empty_cache()

                    # Store iteration info
                    if iteration == 0:
                        iterations_each_step.append(0)
                        if len(losses_each_step) > 0:
                            last_loss = losses_each_step[-1][-1]
                        else:
                            last_loss = loss.item()
                        losses_each_step.append([last_loss])
                    else:
                        iterations_each_step.append(iteration)
                        losses_each_step.append(losses_each_iteration)

                    # Optional: save attn maps
                    if is_save_attn_maps and (index % 10 == 0):
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
                            visualize_attention_maps(
                                attention_maps=attn_mid_control,
                                prompt=prompt,
                                object_positions=object_positions,
                                timestep=index,
                                is_save_attn_maps=is_save_attn_maps,
                                save_path=f"{save_path}/attention_maps_mid_control",
                                is_all_tokens=False
                            )

            # ----------------------------------------------------------
            #  C) Now do the normal CFG step (uncond + cond pass)
            # ----------------------------------------------------------
            with torch.no_grad():
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

                if is_used_controlnet:
                    control_image_cfg = torch.cat([control_image] * 2)
                    control_output, attn_down_control, attn_mid_control = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=control_image_cfg,
                        conditioning_scale=controlnet_scale,
                        return_dict=True,
                    )
                    down_block_res_samples = control_output['down_block_res_samples']
                    mid_block_res_sample = control_output['mid_block_res_sample']

                    unet_out = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=True
                    )
                    noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = unet_out
                    noise_pred = noise_pred.sample
                else:
                    unet_out = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        return_dict=True
                    )
                    noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = unet_out
                    noise_pred = noise_pred.sample

                # Guidance
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

        # ----------------------------------------------------------------
        # After the loop: decode latents
        # ----------------------------------------------------------------
        if is_used_attention_guide:
            logger.info(f"\nFirst Loss: {losses_each_step[0][0] if len(losses_each_step) > 0 else None}")
            logger.info(f"Last Loss: {losses_each_step[-1][-1] if len(losses_each_step) > 0 else None}")
            if is_save_losses:
                save_to_pkl(losses_each_step, f"{save_path}/losses.pkl")
                save_to_pkl(iterations_each_step, f"{save_path}/iterations.pkl")

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
        is_special_token_guide=False,
        alpha=1.0,

        is_used_controlnet=False,
        is_used_controlnet_term=False,
        control_image=None,
        controlnet_scale = 0.2,
        beta=0.3,

        is_used_momentum=False,
        momentum = 0.5,

        is_stay_close=False,
        stay_close_weight=0.1,
 
        is_smoothness=False,
        smoothness_weight=0.01,

        is_grad_clipping=False,
        grad_clip_threshold=1.0,

        is_process_bbox_data=True,
        is_random_seed=False,
        negative_prompt="ball,box,low resolution,bad composition,blurry image,bad anatomy",

        noise_scheduler_type="ddim"
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

            logger,

            is_save_attn_maps=is_save_attn_maps,
            is_save_losses=is_save_losses,
            save_path=save_path,

            is_used_attention_guide=is_used_attention_guide,
            is_special_token_guide=is_special_token_guide,
            alpha=alpha,

            is_used_controlnet=is_used_controlnet,
            is_used_controlnet_term=is_used_controlnet_term,
            control_image=control_image,
            controlnet_scale=controlnet_scale,
            beta=beta,

            is_used_momentum=is_used_momentum,
            momentum=momentum,

            is_stay_close=is_stay_close,
            stay_close_weight=stay_close_weight,
    
            is_smoothness=is_smoothness,
            smoothness_weight=smoothness_weight,

            is_grad_clipping=is_grad_clipping,
            grad_clip_threshold=grad_clip_threshold,

            is_random_seed=is_random_seed,
            negative_prompt=negative_prompt,

            noise_scheduler_type=noise_scheduler_type
        )

        if is_save_images:
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
