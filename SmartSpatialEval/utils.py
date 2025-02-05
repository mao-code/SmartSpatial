from io import BytesIO
import base64
import numpy as np
import pandas as pd

position_words = {
    "front", "on", "above", "under", "below", "left",
    "behind", "right"
}

simple_position_mapping = {
    "under": "under",

    "below": "below",

    "on": "on",
    "on top of": "on",
    "atop": "on",

    "above": "above",

    "in front of": "front",
    "behind": "behind",

    "to the left of": "left",
    "to the right of": "right"
}

reverse_position_mapping = {
    "on": "under",
    "atop": "under",
    "on top of": "under",
    "above": "below",
    "below": "above",
    "under": "on",
    "front": "behind",
    "behind": "front",
    "left": "right",
    "right": "left"
}

# Handling multi-word prepositions and various tags for position words
position_phrases = [
    "on",
    "on top of",
    "atop",
    "above",
    "under",
    "below",
    "in front of",
    "behind",
    "to the left of",
    "to the right of"
]

# Based on the spatial sphere
# For now, we only consider: on, above, under, below, front, behind, left, right, in
# To make it simpler, we lemmanized them
spatial_ball_map = {
    "in": (0,0,0),
    "on": (0,0,1),
    "above": (0,0,2),
    "under": (0,0,-1),
    "below": (0,0,-2),
    "front": (1,0,0),
    "behind": (-1,0,0),
    "left": (0,-1,0),
    "right": (0,1,0)
}

def encode_image(image):
    """
    Encodes a PIL image to a base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")

    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    if norm_A * norm_B == 0:
        return 0  # Return 0 similarity when dividing by 0 would occur

    return dot_product / (norm_A * norm_B)