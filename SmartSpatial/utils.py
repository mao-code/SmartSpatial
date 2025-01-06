import torch
import numpy as np
import pickle
import math
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_ca_loss(
    attn_maps_mid, # List of tensors
    attn_maps_up,  # List of tensors
    bboxes,
    object_positions,
    position_word_indices=[],
    alpha=1,

    is_used_controlnet_term=False,
    beta=0.3,    # controlnet term loss weight
    attn_maps_down_control=None,
    attn_maps_mid_control=None
):
    loss = 0
    controlnet_loss = 0

    object_number = len(bboxes)
    # object_number = len(object_positions)

    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()

    # For attention maps mid
    for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated
        b, i, j = attn_map.shape # batch, i(HxW), j(tokens)
        H = W = int(math.sqrt(i))

        ###### Loss for Objects ######
        for obj_idx in range(object_number):
            obj_loss = 0

            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))

            ###### Object mask ######
            for obj_box in bboxes[obj_idx]:

                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            ###### Loss Calculation ######
            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)

            loss += (obj_loss/len(object_positions[obj_idx]))


    # For attention maps up
    for attn_map_integrated in attn_maps_up[0]: # first up block
        attn_map = attn_map_integrated

        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))

        for obj_idx in range(object_number):
            obj_loss = 0

            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            for obj_box in bboxes[obj_idx]:
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                # ca_map_obj = attn_map[:, :, object_positions[obj_position]].reshape(b, H, W)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(
                    dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)

            loss += (obj_loss / len(object_positions[obj_idx]))

    if is_used_controlnet_term:
      # For attention maps down
      # The paper said that down block sampling did not effective in backward guidance
      # for attn_map_integrated in attn_maps_down_control[0]: # so that we can get the tensor
      #     attn_map = attn_map_integrated

      #     b, i, j = attn_map.shape
      #     H = W = int(math.sqrt(i))

      #     for obj_idx in range(object_number):
      #         obj_loss = 0

      #         mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
      #         for obj_box in bboxes[obj_idx]:
      #             x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
      #                 int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
      #             mask[y_min: y_max, x_min: x_max] = 1

      #         for obj_position in object_positions[obj_idx]:
      #             ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)

      #             activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(
      #                 dim=-1)

      #             obj_loss += torch.mean((1 - activation_value) ** 2)

      #         controlnet_loss += (obj_loss / len(object_positions[obj_idx]))

      # For attention maps mid
      for attn_map_integrated in attn_maps_mid_control:
          attn_map = attn_map_integrated
          b, i, j = attn_map.shape # batch, i(HxW), j(tokens)
          H = W = int(math.sqrt(i))

          ###### Loss for Objects ######
          for obj_idx in range(object_number):
              obj_loss = 0

              mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))

              ###### Object mask ######
              for obj_box in bboxes[obj_idx]:

                  x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                      int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                  mask[y_min: y_max, x_min: x_max] = 1

              ###### Loss Calculation ######
              for obj_position in object_positions[obj_idx]:
                  ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                  activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)

                  obj_loss += torch.mean((1 - activation_value) ** 2)

              controlnet_loss += (obj_loss/len(object_positions[obj_idx]))

    # Average the losses (total number of attention maps)
    loss = loss / (object_number * (len(attn_maps_up[0]) + len(attn_maps_mid)))
    if is_used_controlnet_term:
      # controlnet_loss = controlnet_loss / (len(attn_maps_down_control[0]) + len(attn_maps_mid_control))
      controlnet_loss = controlnet_loss / len(attn_maps_mid_control)

    # Add new terms to the total loss
    if not is_used_controlnet_term:
      beta = 0

    total_loss = alpha * loss + beta * controlnet_loss

    return total_loss

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

    if is_all_tokens:
      for idx, token in enumerate(tokens):
          idx = idx+1 # skip the special token

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
  with open(path, 'wb') as pickle_file:
    pickle.dump(data, pickle_file)