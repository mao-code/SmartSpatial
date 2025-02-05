import torch
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import os

bbox_ref_mapping = {
    "front": [    
        {'caption': 'ball', 'box': {'x': 179.93, 'y': 288.83, 'w': 135.23, 'h': 134.28}},
        {'caption': 'box', 'box': {'x': 111, 'y': 87, 'w': 300.5, 'h': 256.5}}
    ],
    "behind": [
        {'caption': 'ball', 'box': {'x': 104.5, 'y': 78, 'w': 311, 'h': 292}},
        {'caption': 'box', 'box': {'x': 155.5, 'y': 251.5, 'w': 215, 'h': 185.5}}
    ],
    "left": [
        {'caption': 'ball', 'box': {'x': 30, 'y': 208, 'w': 167.5, 'h': 165}},
        {'caption': 'box', 'box': {'x': 212, 'y': 106, 'w': 259, 'h': 268}}
    ],
    "right": [
        {'caption': 'ball', 'box': {'x': 330, 'y': 217, 'w': 137, 'h': 133}},
        {'caption': 'box', 'box': {'x': 75, 'y': 167, 'w': 191, 'h': 202.5}}
    ],
    "on": [
        {'caption': 'ball', 'box': {'x': 165, 'y': 80.5, 'w': 182.5, 'h': 182.5}},
        {'caption': 'box', 'box': {'x': 86, 'y': 220, 'w': 343.5, 'h': 224}}
    ],
    "under": [
        {'caption': 'ball', 'box': {'x': 115.5, 'y': 181, 'w': 272, 'h': 265}},
        {'caption': 'box', 'box': {'x': 175, 'y': 43.5, 'w': 159, 'h': 166.5}}
    ],
    "above": [
        {'caption': 'ball', 'box': {'x': 194, 'y': 82, 'w': 117, 'h': 119}},
        {'caption': 'box', 'box': {'x': 154, 'y': 275, 'w': 193.5, 'h': 162.5}}
    ],
    "below": [
        {'caption': 'ball', 'box': {'x': 181.5, 'y': 261, 'w': 144.5, 'h': 147.5}},
        {'caption': 'box', 'box': {'x': 164, 'y': 66.5, 'w': 181.5, 'h': 157}}
    ]
}

def load_image(image_path):
    """Load image from file."""
    return Image.open(image_path)

    loss = 0
    object_number = len(bboxes)
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated

        #
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

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)
            loss += (obj_loss/len(object_positions[obj_idx]))

    for attn_map_integrated in attn_maps_up[0]:
        attn_map = attn_map_integrated
        #
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
    loss = loss / (object_number * (len(attn_maps_up[0]) + len(attn_maps_mid)))
    return loss

def draw_box(pil_img, bboxes, phrases, save_path):
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype('./FreeMono.ttf', 25)
    phrases = [x.strip() for x in phrases.split(';')]
    for obj_bboxes, phrase in zip(bboxes, phrases):
        for obj_bbox in obj_bboxes:
            x_0, y_0, x_1, y_1 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
            draw.rectangle([int(x_0 * 512), int(y_0 * 512), int(x_1 * 512), int(y_1 * 512)], outline='red', width=5)
            draw.text((int(x_0 * 512) + 5, int(y_0 * 512) + 5), phrase, font=font, fill=(255, 0, 0))
    pil_img.save(save_path)

def setup_logger(save_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all logs

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    log_file = os.path.join(save_path, f"{logger_name}.log")
    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler for WARNING and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optionally, prevent logs from propagating to ancestor loggers
    logger.propagate = False

    return logger

def load_text_inversion(text_encoder, tokenizer, placeholder_token, embedding_ckp_path):
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    learned_embedding = torch.load(embedding_ckp_path)
    token_embeds[placeholder_token_id] = learned_embedding[placeholder_token]
    return text_encoder, tokenizer