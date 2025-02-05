import torch
import numpy as np
import pickle
import math
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import re
import string

def compute_ca_loss(
    attn_maps_mid, # List of tensors
    attn_maps_up,  # List of tensors
    bboxes,
    object_positions,
    alpha=1,

    is_used_controlnet_term=False,
    beta=0.3, # Controlnet term loss weight
    attn_maps_mid_control=None,

    is_special_token_guide=False,
    sot_idx=None,
    eot_idx=None,
    gamma=0.5, # Special token term loss weight

    # --- Regularization Flags and Weights ---
    # 1) Stay-Close to original attention
    is_stay_close=False,
    stay_close_weight=0.1,
    # Originals for UNet
    orig_attn_maps_mid=None,
    orig_attn_maps_up=None,
    # Originals for ControlNet
    orig_attn_maps_mid_control=None,

    # 2) Smoothness (e.g. total variation)
    is_smoothness=False,
    smoothness_weight=0.01
):
    """
    Optional regularizations:
      - 'Stay Close': Encourages new attention to remain close to original/previous attention
      - 'Smoothness': Encourages spatial smoothness in attention maps
      - 'Gradient Clipping': Restricts the gradient norm to avoid large updates (commonly done outside)
    """

    # -----------------------
    # Base cross-attention loss
    # -----------------------
    device = attn_maps_mid[0].device if len(attn_maps_mid) > 0 else 'cpu'
    loss = torch.tensor(0.0, device=device)
    controlnet_loss = torch.tensor(0.0, device=device)
    special_token_total_loss = torch.tensor(0.0, device=device)

    object_number = len(bboxes)
    if object_number == 0:
        return loss  # Zero
    
    # =============== Mid-Block Loss ===============
    for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated  # shape: (b, h*w, tokens)
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))

        # For each object's bounding box
        for obj_idx in range(object_number):
            obj_loss = 0.0

            # Create bounding box mask
            mask = torch.zeros(size=(H, W), device=device)
            for obj_box in bboxes[obj_idx]:
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                                             int(obj_box[1] * H), \
                                             int(obj_box[2] * W), \
                                             int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            # For each token that composes the object
            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                activation_value = (ca_map_obj * mask).view(b, -1).sum(dim=-1) / ca_map_obj.view(b, -1).sum(dim=-1)
                obj_loss += torch.mean((1 - activation_value) ** 2)

            loss += (obj_loss / len(object_positions[obj_idx]))

    # =============== Up-Block Loss (only first up-block) ===============
    if len(attn_maps_up) > 0:
        for attn_map_integrated in attn_maps_up[0]:
            attn_map = attn_map_integrated
            b, i, j = attn_map.shape
            H = W = int(math.sqrt(i))

            for obj_idx in range(object_number):
                obj_loss = 0.0

                mask = torch.zeros(size=(H, W), device=device)
                for obj_box in bboxes[obj_idx]:
                    x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                                                 int(obj_box[1] * H), \
                                                 int(obj_box[2] * W), \
                                                 int(obj_box[3] * H)
                    mask[y_min: y_max, x_min: x_max] = 1

                for obj_position in object_positions[obj_idx]:
                    ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                    activation_value = (ca_map_obj * mask).view(b, -1).sum(dim=-1) / ca_map_obj.view(b, -1).sum(dim=-1)
                    obj_loss += torch.mean((1 - activation_value) ** 2)

                loss += (obj_loss / len(object_positions[obj_idx]))

    # =============== Special Token Loss ===============
    if is_special_token_guide and (sot_idx is not None) and (eot_idx is not None):
        # 1) Create union / background masks
        b, i, j = attn_maps_mid[0].shape
        H = W = int(math.sqrt(i))
        union_mask = create_union_mask(bboxes, H, W).to(device)
        background_mask = create_background_mask(union_mask).to(device)

        # 2) Compute special token loss on mid blocks
        special_token_loss = 0.0
        for attn_map_integrated in attn_maps_mid:
            st_loss = compute_special_token_loss(
                attn_map_integrated, sot_idx, eot_idx,
                union_mask, background_mask
            )
            special_token_loss += st_loss

        # 3) Compute special token loss on up blocks
        if len(attn_maps_up) > 0:
            for attn_map_integrated in attn_maps_up[0]:
                st_loss = compute_special_token_loss(
                    attn_map_integrated, sot_idx, eot_idx,
                    union_mask, background_mask
                )
                special_token_loss += st_loss

        special_token_total_loss += special_token_loss

    # =============== ControlNet term ===============
    if is_used_controlnet_term and (attn_maps_mid_control is not None):
        for attn_map_integrated in attn_maps_mid_control:
            attn_map = attn_map_integrated
            b, i, j = attn_map.shape
            H = W = int(math.sqrt(i))

            for obj_idx in range(object_number):
                obj_loss = 0.0

                mask = torch.zeros(size=(H, W), device=device)
                for obj_box in bboxes[obj_idx]:
                    x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                                                 int(obj_box[1] * H), \
                                                 int(obj_box[2] * W), \
                                                 int(obj_box[3] * H)
                    mask[y_min: y_max, x_min: x_max] = 1

                for obj_position in object_positions[obj_idx]:
                    ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                    activation_value = (ca_map_obj * mask).view(b, -1).sum(dim=-1) / ca_map_obj.view(b, -1).sum(dim=-1)
                    obj_loss += torch.mean((1 - activation_value) ** 2)

                controlnet_loss += (obj_loss / len(object_positions[obj_idx]))

    # -----------------------
    # Averaging baseline loss
    # -----------------------
    num_mid_maps = len(attn_maps_mid)
    num_up_maps_first = len(attn_maps_up[0]) if len(attn_maps_up) > 0 else 0
    base_divisor = object_number * (num_mid_maps + num_up_maps_first)

    if base_divisor > 0:
        loss = loss / base_divisor

    if is_used_controlnet_term and (attn_maps_mid_control is not None):
        # If you have more control net layers, adjust the divisor as needed
        controlnet_loss = controlnet_loss / len(attn_maps_mid_control)
    else:
        beta = 0.0

    if not is_special_token_guide:
        gamma = 0.0

    # -------------------------------------------------------
    # "Stay Close" Regularization (UNet + ControlNet)
    # -------------------------------------------------------
    stay_close_total = torch.tensor(0.0, device=device)

    if is_stay_close:
        # --- Stay close for UNet mid ---
        if (orig_attn_maps_mid is not None) and (len(orig_attn_maps_mid) == len(attn_maps_mid)):
            for curr_map, orig_map in zip(attn_maps_mid, orig_attn_maps_mid):
                stay_close_total += stay_close_loss(curr_map, orig_map)

        # --- Stay close for UNet up (just first up-block for example) ---
        if (orig_attn_maps_up is not None) and len(orig_attn_maps_up) > 0 \
           and len(attn_maps_up) > 0 and len(attn_maps_up[0]) > 0:
            up_block_curr = attn_maps_up[0]
            up_block_orig = orig_attn_maps_up[0]
            for curr_map, orig_map in zip(up_block_curr, up_block_orig):
                stay_close_total += stay_close_loss(curr_map, orig_map)

        # --- Stay close for ControlNet mid (if enabled) ---
        if is_used_controlnet_term and (orig_attn_maps_mid_control is not None) \
           and (attn_maps_mid_control is not None) \
           and (len(orig_attn_maps_mid_control) == len(attn_maps_mid_control)):
            for curr_map, orig_map in zip(attn_maps_mid_control, orig_attn_maps_mid_control):
                stay_close_total += stay_close_loss(curr_map, orig_map)

    stay_close_total = stay_close_weight * stay_close_total

    # -------------------------------------------------------
    # "Smoothness" Regularization (UNet + ControlNet)
    # -------------------------------------------------------
    smoothness_total = torch.tensor(0.0, device=device)

    if is_smoothness:
        # ---- Smoothness for UNet mid ----
        for attn_map_integrated in attn_maps_mid:
            b, hw, t = attn_map_integrated.shape
            H = W = int(math.sqrt(hw))
            # sum over tokens
            for token_idx in range(t):
                attn_map_token = attn_map_integrated[:, :, token_idx].reshape(b, H, W)
                smoothness_total += total_variation_loss(attn_map_token)

        # ---- Smoothness for UNet up (first block) ----
        if len(attn_maps_up) > 0 and len(attn_maps_up[0]) > 0:
            for attn_map_integrated in attn_maps_up[0]:
                b, hw, t = attn_map_integrated.shape
                H = W = int(math.sqrt(hw))
                for token_idx in range(t):
                    attn_map_token = attn_map_integrated[:, :, token_idx].reshape(b, H, W)
                    smoothness_total += total_variation_loss(attn_map_token)

        # ---- Smoothness for ControlNet mid (if enabled) ----
        if is_used_controlnet_term and (attn_maps_mid_control is not None):
            for attn_map_integrated in attn_maps_mid_control:
                b, hw, t = attn_map_integrated.shape
                H = W = int(math.sqrt(hw))
                for token_idx in range(t):
                    attn_map_token = attn_map_integrated[:, :, token_idx].reshape(b, H, W)
                    smoothness_total += total_variation_loss(attn_map_token)

    smoothness_total = smoothness_weight * smoothness_total    

    # -------------------------------------------------------
    # Combine everything
    # -------------------------------------------------------
    total_loss = alpha * loss + beta * controlnet_loss + gamma * special_token_total_loss
    total_loss += stay_close_total
    total_loss += smoothness_total

    return total_loss

def total_variation_loss(attn_map):
    """
    A simple total variation regularization for smoothness.
    attn_map shape: (b, h, w)
    """
    # Horizontal and vertical differences
    loss = torch.mean(torch.abs(attn_map[:, :, :-1] - attn_map[:, :, 1:])) + \
           torch.mean(torch.abs(attn_map[:, :-1, :] - attn_map[:, 1:, :]))
    return loss

def stay_close_loss(current_attn, orig_attn):
    """
    L2 difference between the current attention map and the original.
    Both shapes: (b, h*w, tokens) or (b, h, w).
    Adapt as needed if shapes differ.
    """
    return F.mse_loss(current_attn, orig_attn)

def get_special_token_indices(tokenizer, prompt):
    """
    Given a tokenizer and prompt, returns the indices of the [SoT] and [EoT] tokens 
    within the cross-attention dimension.
    
    NOTE: This is an example. You need to adapt it to however your pipeline 
    or custom tokens are actually named or appended.
    """
    # Tokenize the prompt
    # For example, if using CLIPTokenizer or similar:
    # encoded = tokenizer(prompt, truncation=True, max_length=77, return_tensors="pt")
    
    # Example placeholder: pretend tokenizer returns a list of tokens
    tokens = tokenizer.tokenize(prompt)  # or your actual usage
    
    sot_idx = None
    eot_idx = None
    
    # Loop through the tokens to find [SoT], [EoT]
    for i, tok in enumerate(tokens):
        if tok == "[SoT]":
            sot_idx = i
        elif tok == "[EoT]":
            eot_idx = i
            
    return sot_idx, eot_idx

def create_union_mask(bboxes, H, W):
    """
    Given all bounding boxes for the image, create a union mask of shape (H, W).
    """
    union_mask = torch.zeros(size=(H, W))
    for box in bboxes:
        x_min, y_min, x_max, y_max = (
            int(box[0] * W),
            int(box[1] * H),
            int(box[2] * W),
            int(box[3] * H)
        )
        union_mask[y_min: y_max, x_min: x_max] = 1  # set inside bounding box to 1
    
    return union_mask

def create_background_mask(union_mask):
    return 1 - union_mask

def compute_special_token_loss(attn_map, sot_idx, eot_idx, union_mask, background_mask):
    """
    Compute the contribution to the loss for [SoT] and [EoT] from one attention map.
    
    attn_map: Tensor of shape [batch_size, (H*W), num_tokens]
    sot_idx: int index of [SoT] in token dimension
    eot_idx: int index of [EoT] in token dimension
    union_mask, background_mask: (H, W)
    """
    b, i, j = attn_map.shape
    H = W = int(math.sqrt(i))

    # Reshape the attn map for the special tokens [SoT], [EoT]
    sot_ca_map = attn_map[:, :, sot_idx].reshape(b, H, W)
    eot_ca_map = attn_map[:, :, eot_idx].reshape(b, H, W)

    # Move masks to same device if needed
    device = attn_map.device
    union_mask = union_mask.to(device)
    background_mask = background_mask.to(device)

    # Calculate activation values
    # For [EoT] -> union of boxes
    eot_numerator   = (eot_ca_map * union_mask).reshape(b, -1).sum(dim=-1)
    eot_denominator = eot_ca_map.reshape(b, -1).sum(dim=-1) + 1e-8
    eot_activation  = eot_numerator / eot_denominator

    # For [SoT] -> background
    sot_numerator   = (sot_ca_map * background_mask).reshape(b, -1).sum(dim=-1)
    sot_denominator = sot_ca_map.reshape(b, -1).sum(dim=-1) + 1e-8
    sot_activation  = sot_numerator / sot_denominator

    # Example: we want [EoT] to *have* attention in union mask -> activation_value ~ 1
    #          we want [SoT] to *avoid* objects -> activation_value ~ 1 in background
    # 
    # So a typical MSE to push them to 1 could be:
    #   [EoT] loss: (1 - eot_activation)^2
    #   [SoT] loss: (1 - sot_activation)^2

    eot_loss = torch.mean((1 - eot_activation)**2)
    sot_loss = torch.mean((1 - sot_activation)**2)

    return eot_loss + sot_loss

def Pharse2idx(prompt, phrases):
    phrases = [x.strip() for x in phrases.split(';')]
    prompt_list = sentence_to_list(prompt)

    object_positions = []
    for obj in phrases:
        obj_position = []

        # For multi-token objects (e.g. "wine glass")
        for word in obj.split(' '):
            # Add 1 to match the index in the attention map ([SoT] + words + [EoT])
            obj_first_index = prompt_list.index(word) + 1 
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)

    return object_positions

def sentence_to_list(sentence):
    """
    Converts a sentence into a list of words by removing punctuation and handling contractions.
    
    Parameters:
        sentence (str): The input sentence to be processed.
        
    Returns:
        List[str]: A list of cleaned, lowercase words.
    """
    # Step 1: Remove possessive 's (e.g., "here's" -> "here")
    sentence = re.sub(r"'s\b", "", sentence)
    
    # Step 2: Remove all other punctuation
    # Create a translation table for string.punctuation
    translator = str.maketrans('', '', string.punctuation)
    sentence = sentence.translate(translator)
    
    # Step 3: Convert to lowercase
    sentence = sentence.lower()
    
    # Step 4: Split into words
    word_list = sentence.split()
    
    return word_list

def idx2phrase(prompt, indices):
    """
    Retrieve the phrase from a prompt based on the provided indices.

    Parameters:
    - prompt (str): The input prompt string.
    - indices (list of lists): A list of lists where each inner list contains indices of tokens to retrieve.

    Returns:
    - phrases (list): A list of phrases corresponding to the provided indices.
    """
    # Split the prompt into words
    prompt_list = prompt.strip('.').split(' ')

    # Initialize an empty list to store the phrases
    phrases = []

    # Loop through the provided indices
    for phrase_indices in indices:
        # Subtract 1 from each index because the original function added 1 to the index
        words = [prompt_list[idx - 1] for idx in phrase_indices]
        # Join the words to form the phrase
        phrase = ' '.join(words)
        phrases.append(phrase)

    # Example usage:
    # prompt = "A bird is perched to the right of a cup on a wooden picnic table in a sunny park."
    # indices = [[2, 3], [10]]  # Example indices for "bird is" and "cup"
    # retrieved_phrases = idx2phrase(prompt, indices)
    # print(retrieved_phrases)  # Output: ['bird is', 'cup']

    return phrases

def visualize_attention_maps(
    attention_maps,
    prompt,
    object_positions,
    timestep,
    is_save_attn_maps=False,
    save_path="",
    is_all_tokens=True
):
    """
    Visualize and save attention maps for all tokens in object_positions.

    Parameters:
    - attention_maps (list): a list of Attention maps, each of shape (b, H*W, num_tokens).
    - prompt (str): The input prompt string.
    - object_positions (list): A list of lists where each inner list contains indices of tokens to visualize.
    - timestep (int): The current timestep in the process.
    - save_path (str): Base path to save the attention maps. Defaults to "attention_maps/{token}/{timestep}/".
    """

    # Firstly, average multi-head attention maps
    attention_map = integrate_attention_maps(attention_maps)

    # Calculate the height and width from the attention map size (H*W)
    b, hw, num_tokens = attention_map.shape  # Assuming shape (b, H*W, num_tokens) # ERROR cuz attention_maps is a list)
    height = width = int(math.sqrt(hw))  # Assuming square attention map (H = W)

    # Visualize every tokens in the prompt
    tokens = prompt.strip('.').split(' ')
    tokens = ["[SoT]"] + tokens + ["[EoT]"]

    if is_all_tokens:
      for idx, token in enumerate(tokens):
          # Create the directory for saving attention maps if it doesn't exist
          if save_path == "":
              save_dir = f"attention_maps/{token}/{timestep}/"
          else:
              save_dir = os.path.join(save_path, f"{token}/{timestep}/")
          os.makedirs(save_dir, exist_ok=True)

          attn_prob = attention_map[0, :, idx].reshape(height, width)  # Assuming batch size b=1

          # Set the individual save path for each token attention map
          token_save_path = os.path.join(save_dir, f"token_{idx}.png")

          # make sure the attn_prob be moved to cpu and convert to numpy
          attn_prob_array = attn_prob.cpu().detach().numpy()

          # Plot the attention probabilities as a heatmap
          plt.figure(figsize=(6, 6))
          plt.imshow(attn_prob_array, cmap='viridis', interpolation='nearest')
          # plt.colorbar(label='Attention Probability')
          plt.title(f'"{token}"')
          plt.axis('off')  # Optionally, you can remove the axis ticks and labels

          # Save the figure
          if is_save_attn_maps:
            plt.savefig(token_save_path)
          plt.close()
          # print(f'Attention map for token "{token}" (index: {idx}) saved at: {token_save_path}')

          # Upscaling attention map
          if is_save_attn_maps:
            upscale_and_save_attention_map(
                attn_prob_array,
                prompt,
                token,
                timestep,
                target_size=(512, 512),
                save_dir=f"{save_path}_upscale"
            )

    else:
      # Loop through each token index in object_positions
      for obj_idx, token_indices in enumerate(object_positions):
          # Get the token phrase for the current set of token indices
          token = idx2phrase(prompt, [token_indices])[0]

          # Create the directory for saving attention maps if it doesn't exist
          if save_path == "":
              save_dir = f"attention_maps/{token}/{timestep}/"
          else:
              save_dir = os.path.join(save_path, f"{token}/{timestep}/")
          os.makedirs(save_dir, exist_ok=True)

          # Visualize and save the attention map for each token in the token_indices
          # This token index corresponds to the position of the token in the output of the tokenizer
          for token_idx in token_indices:
              attn_prob = attention_map[0, :, token_idx].reshape(height, width)  # Assuming batch size b=1

              # Set the individual save path for each token attention map
              token_save_path = os.path.join(save_dir, f"token_{token_idx}.png")

              # make sure the attn_prob be moved to cpu and convert to numpy
              attn_prob_array = attn_prob.cpu().detach().numpy()

              # Plot the attention probabilities as a heatmap
              plt.figure(figsize=(6, 6))
              plt.imshow(attn_prob_array, cmap='viridis', interpolation='nearest')
              # plt.colorbar(label='Attention Probability')
              plt.title(f'"{token}"')
              plt.axis('off')  # Optionally, you can remove the axis ticks and labels

              # Save the figure
              if is_save_attn_maps:
                plt.savefig(token_save_path)
              plt.close()
              # print(f'Attention map for token "{token}" (index: {token_idx}) saved at: {token_save_path}')

          # Upscaling attention map
          if is_save_attn_maps:
            upscale_and_save_attention_map(
                attn_prob_array,
                prompt,
                token,
                timestep,
                target_size=(512, 512),
                save_dir=f"{save_path}_upscale"
            )

      # for special tokens
      for idx, token in enumerate(tokens):
          if idx != 0 and idx != len(tokens) - 1:
            continue  
        
          # Create the directory for saving attention maps if it doesn't exist
          if save_path == "":
              save_dir = f"attention_maps/{token}/{timestep}/"
          else:
              save_dir = os.path.join(save_path, f"{token}/{timestep}/")
          os.makedirs(save_dir, exist_ok=True)

          attn_prob = attention_map[0, :, idx].reshape(height, width)  # Assuming batch size b=1

          # Set the individual save path for each token attention map
          token_save_path = os.path.join(save_dir, f"token_{idx}.png")

          # make sure the attn_prob be moved to cpu and convert to numpy
          attn_prob_array = attn_prob.cpu().detach().numpy()

          # Plot the attention probabilities as a heatmap
          plt.figure(figsize=(6, 6))
          plt.imshow(attn_prob_array, cmap='viridis', interpolation='nearest')
          # plt.colorbar(label='Attention Probability')
          plt.title(f'"{token}"')
          plt.axis('off')  # Optionally, you can remove the axis ticks and labels

          # Save the figure
          if is_save_attn_maps:
            plt.savefig(token_save_path)
          plt.close()
          # print(f'Attention map for token "{token}" (index: {idx}) saved at: {token_save_path}')

          # Upscaling attention map
          if is_save_attn_maps:
            upscale_and_save_attention_map(
                attn_prob_array,
                prompt,
                token,
                timestep,
                target_size=(512, 512),
                save_dir=f"{save_path}_upscale"
            )

    # Example usage:
    # Assuming attention_map is the (b, H*W, num_tokens) attention map from the model
    # and object_positions are the token indices for objects
    # prompt = "A bird is perched to the right of a cup on a wooden picnic table in a sunny park."
    # attention_map = np.random.rand(1, 32*32, num_tokens)  # Example random attention map
    # object_positions = [[2, 3], [10]]  # Example positions for tokens
    # visualize_attention_maps(attention_map, prompt, object_positions, timestep=50)

def integrate_attention_maps(attention_maps):
    """
    Integrate all attention maps by averaging them.

    Parameters:
    - attention_maps (list): a list of Attention maps, each of shape (b, H*W, num_tokens).

    Returns:
    - integrated_map (np.array): Averaged attention map of shape (b, H*W, num_tokens).
    """
    # Sum all attention maps
    sum_attention_map = None
    for attention_map in attention_maps:
        if sum_attention_map is None:
            sum_attention_map = attention_map.clone()
        else:
            sum_attention_map += attention_map

    # Average the sum
    integrated_map = sum_attention_map / len(attention_maps)
    return integrated_map

def pil_to_numpy(images):
    """
    Convert a PIL image or a list of PIL images to NumPy arrays.

    Args:
        images (`PIL.Image.Image` or `List[PIL.Image.Image]`):
            The PIL image or list of images to convert to NumPy format.

    Returns:
        `np.ndarray`:
            A NumPy array representation of the images.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images

# Upscale the attention map
def upscale_and_save_attention_map(
    attn_map,
    prompt,
    token,
    timestep,
    target_size=(512, 512),
    save_dir="attention_map_upscale",
    colormap='seismic',
    interpolation_method='bicubic'
):
    """
    Upscale a 64x64 attention map, convert it to a heatmap, and save it to a specified path.

    Parameters:
        attn_map (np.ndarray): The original 64x64 attention map.
        prompt (str): The prompt associated with the attention map.
        token (str): The token associated with the attention map.
        target_size (tuple): The target size for upscaling (default is (512, 512)).
        save_dir (str): The base directory where the heatmap will be saved.
        colormap (str): The colormap to use for visualization ('seismic' for cold-hot).
        interpolation_method (str): Interpolation method ('bilinear', 'bicubic', or 'nearest').
    """

    # Choose the interpolation method
    if interpolation_method == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    elif interpolation_method == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif interpolation_method == 'nearest':
        interpolation = cv2.INTER_NEAREST
    else:
        raise ValueError("Unknown interpolation method. Use 'bicubic', 'bilinear', or 'nearest'.")

    # Upscale the attention map to the target size
    upscaled_map = cv2.resize(attn_map, target_size, interpolation=interpolation)

    # Normalize the upscaled map to [0, 1] range for better visualization
    upscaled_map = (upscaled_map - np.min(upscaled_map)) / (np.max(upscaled_map) - np.min(upscaled_map))

    # Create the save directory if it doesn't exist
    save_path = f"{save_dir}/{timestep}"
    os.makedirs(save_path, exist_ok=True)

    # File path for the saved heatmap
    file_path = os.path.join(save_path, f"{token}.png")

    # Plot and save the heatmap without any elements
    plt.figure(figsize=(8, 8))
    plt.imshow(upscaled_map, cmap=colormap)

    # Remove all axes, borders, and additional elements
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Save the figure without any borders or additional space
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # print(f"Saved heatmap to: {file_path}")

def convert_bbox_data(bbox_datas, image_width=512, image_height=512):
    bboxes = []
    for i in range(len(bbox_datas)):
        obj = bbox_datas[i]
        box = obj['box']
        x_min = box['x']
        y_min = box['y']
        x_max = box['x'] + box['w']
        y_max = box['y'] + box['h']

        # Normalize the coordinates
        x_min_norm = x_min / image_width
        y_min_norm = y_min / image_height
        x_max_norm = x_max / image_width
        y_max_norm = y_max / image_height

        # Append to bboxes in the required format
        bboxes.append([[x_min_norm, y_min_norm, x_max_norm, y_max_norm]])

    return bboxes

def numpy_to_pt(images):
    """
    Convert a NumPy image to a PyTorch tensor.

    Args:
        images (`np.ndarray`):
            The NumPy image array to convert to PyTorch format.

    Returns:
        `torch.Tensor`:
            A PyTorch tensor representation of the images.
    """
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

def save_to_pkl(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
