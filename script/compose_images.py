#!/usr/bin/env python3

import os
import argparse
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compose images (side-by-side) from multiple folders, with labeling."
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        required=True,
        help="List of folder paths containing images."
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        required=True,
        help="List of model names corresponding to each folder."
    )
    parser.add_argument(
        "--output-folder",
        default="output_images",
        help="Folder where composed images will be saved."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Check that the number of folders matches the number of model names
    if len(args.folders) != len(args.model_names):
        raise ValueError("The number of folders must match the number of model names.")

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Read image filenames in each folder, sort them by the integer before the underscore
    all_folders_images = []
    for folder in args.folders:
        # Get image files only
        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        
        # Sort by the integer part before the first underscore
        def sort_key(filename):
            base = filename.split("_")[0]
            return int(base) if base.isdigit() else 999999999

        files_sorted = sorted(files, key=sort_key)

        # Store the full paths
        full_paths = [os.path.join(folder, f) for f in files_sorted]
        all_folders_images.append(full_paths)

    # Assume all folders have the same number of images
    # If not, handle accordingly (e.g. min length or raise an error).
    num_images_per_folder = len(all_folders_images[0])
    for idx, folder_images in enumerate(all_folders_images):
        if len(folder_images) != num_images_per_folder:
            raise ValueError(
                f"Folder '{args.folders[idx]}' has a different number of images "
                f"({len(folder_images)}) than expected ({num_images_per_folder})."
            )

    num_folders = len(args.folders)

    # Prepare a font for labeling (try to load a TTF font, fallback to default if not found)
    font_size = 40
    stroke_width = 2
    try:
        font = ImageFont.truetype("FreeMono.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # For each index i, compose one image by stacking horizontally
    for i in range(num_images_per_folder):
        # Load each folder's i-th image
        images = []
        for j in range(num_folders):
            img_path = all_folders_images[j][i]
            model_name = args.model_names[j]

            # Open and convert to RGB
            img = Image.open(img_path).convert("RGB")

            # Draw model name
            draw = ImageDraw.Draw(img)
            text_x, text_y = 10, 10
            draw.text(
                (text_x, text_y),
                model_name,
                font=font,
                fill=(255, 0, 0),
                stroke_width=stroke_width,
                stroke_fill=(0, 0, 0)
            )

            images.append(img)

        # Determine total width and max height for this row
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        # Create a blank canvas
        composed_img = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

        # Paste images side by side
        current_x = 0
        for img in images:
            composed_img.paste(img, (current_x, 0))
            current_x += img.width

        # Construct filename
        # Example: 0_compose_SD_AG_CN.png
        index_str = str(i)
        model_names_str = "_".join(args.model_names)
        filename = f"{index_str}_compose_{model_names_str}.png"
        output_path = os.path.join(args.output_folder, filename)

        # Save
        composed_img.save(output_path)
        print(f"Saved composed image: {output_path}")

if __name__ == "__main__":
    main()

    """
    python -m script.compose_images \
    --folders results/spatial_prompts/SD results/spatial_prompts/AG results/spatial_prompts/CN results/spatial_prompts/CN_AG results/spatial_prompts/smart_spatial results/spatial_prompts/smart_spatial_rs_445_lt_06 \
    --model-names vanilla_SD AG CN CN_AG SmartSpatial1 SmartSpatial2 \
    --output-folder results/composed_images/spatial_prompts
    """
