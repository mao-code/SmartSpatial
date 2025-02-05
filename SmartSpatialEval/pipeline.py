import openai
import pandas as pd
import numpy as np
from PIL import Image

from SmartSpatialEval.utils import (
  simple_position_mapping,
  spatial_ball_map
)

from SmartSpatialEval.utils import (
    encode_image
)

from SmartSpatialEval.vlm_response_analyzer import VLMResponseAnalyzer

class SmartSpatialEvalPipeline:
  def __init__(self, cfg, device):
    self.cfg = cfg
    self.device = device

    self.openai_api_key = cfg.general.openai_api_key
    self.vlm_response_analyzer = VLMResponseAnalyzer()

  def build_vlm_instruction_multi_objs(self, prompt_meta):
      center = prompt_meta['center']
      objects = prompt_meta['objects']

      objects_str = "the "
      example_str = ""
      example_str2 = ""

      pws = [
          "on", "above", "under", "below", "to the left of", "to the right of",
          "behind", "in front of",
      ]
      pos_arr = np.array(pws)

      # For only one object
      if len(objects) == 1:
          obj_pos_pair = objects[0]
          obj, pos = obj_pos_pair['obj'], obj_pos_pair['pos']
          objects_str = f"the {center} and the {obj}"

          rand_pos = np.random.choice(pos_arr)
          example_str += f"the {obj} is {rand_pos} the {center}."

          rand_pos = np.random.choice(pos_arr)
          example_str2 += f"the {obj} is {rand_pos} the {center}."
      else:
          objects_str = f"the {center}, "
          for obj_pos_pairs in objects[:-1]:
              obj, pos = obj_pos_pairs['obj'], obj_pos_pairs['pos']
              objects_str += f"the {obj}, "

              rand_pos = np.random.choice(pos_arr)
              example_str += f"the {obj} is {rand_pos} the {center} and "

              rand_pos = np.random.choice(pos_arr)
              example_str2 += f"the {obj} is {rand_pos} the {center} and "

          rand_pos = np.random.choice(pos_arr)
          objects_str += f"and the {objects[-1]['obj']}"
          example_str += f"the {objects[-1]['obj']} is {rand_pos} the {center}"

          rand_pos = np.random.choice(pos_arr)
          example_str2 += f"the {objects[-1]['obj']} is {rand_pos} the {center}"

      instruction = f"Simply describe the spatial arrangement of {objects_str} in this image. \n"
      instruction += f"Use only the following position words: {', '.join(pws)}. \n"
      instruction += f"Don't add any redundant words. Just output the simple spatial relationships among {objects_str}. \n"
      instruction += f"Note: Consider the left side of the object as the left side of the observer or the entire image, and the right side of the object as the right side of the observer or the entire image. No need to change the perspective. \n"
      instruction += f"Use '{center}' as the center reference object. Describe other objects that is relative to the '{center}'"

      instruction += f"For example: (these examples are just for example format)\n"
      instruction += f"Example1: '{example_str}'. \n"
      instruction += f"Example2: '{example_str2}'. \n"
      instruction += f"Answer: "

      return instruction

  def generate_description_of_image(self, prompt, img, is_batch=True):
    img_url = ""
    if isinstance(img, Image.Image):
      # bg_img is already a PIL image
      img_url = encode_image(img)
    elif isinstance(img, str):
      # bg_img is a URL
      img_url = img
    else:
        raise ValueError("img must be a PIL image or a string URL.")

    client = openai.OpenAI(api_key=self.openai_api_key)

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                          "type": "image_url",
                          "image_url": {
                            "url": f"data:image/jpeg;base64,{img_url}",
                          },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
            model="gpt-4o",
        )
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return []

    response_text = response.choices[0].message.content.strip()

    return response_text

  def compute_object_recognition(self, response, prompt_meta):
    score = 0
    underrepresented_objs = []
    is_undetect_center = False

    # objects
    for pair in prompt_meta["objects"]:
      obj, pos = pair["obj"], pair["pos"]
      if obj.lower() in response.lower():
        score += 1
      else:
        underrepresented_objs.append(obj)

    # center object
    if prompt_meta["center"].lower() not in response.lower():
      is_undetect_center = True
    else:
      score += 1

    # normalize: 0~1
    score = score / (len(prompt_meta["objects"])+1)

    # print("<DEBUGGING>")
    # print("Undetectted objects: ", underrepresented_objs)
    # print("<DEBUGGING>")

    return score, underrepresented_objs, is_undetect_center

  def compute_spatial_accuracy_multi_objs(
    self,
    response,
    prompt_metadata,
    undetectted_objs=[],
    is_undetect_center=False
  ):
      if is_undetect_center:
        return 0, 0, None

      # Setup
      doc = self.vlm_response_analyzer.nlp(response)
      center, obj_pos_pairs = prompt_metadata["center"], prompt_metadata["objects"]

      # Understand and purify the spatial relation
      position_phrases = self.vlm_response_analyzer.find_position_phrases(doc)
      relations = self.vlm_response_analyzer.find_relation(doc, position_phrases)
      purified_perceptions = self.vlm_response_analyzer.chain_relation(prompt_metadata, relations)

      # Remove undetectted objects
      for obj in undetectted_objs:
        purified_perceptions = [pp for pp in purified_perceptions if pp["obj"] != obj]

      # print("<DEBUGGING>")
      # print("Prompt Meta: ", prompt_metadata)
      # print("Response: ", response)
      # print("Undetected Objects: ", undetectted_objs)
      # print("Undetected Center: ", is_undetect_center)
      # print("Position phrases: ", position_phrases)
      # print("Relations: ", relations)
      # print("Purified perception: ", purified_perceptions)
      # print("<DEBUGGING>")

      if len(purified_perceptions) == 0:
          return 0, 0, None

      # Real point of objs on the spatial sphere
      obj_real_points = []
      for pair in obj_pos_pairs:
          obj, pos = pair["obj"], pair["pos"]
          obj_real_point = spatial_ball_map[simple_position_mapping.get(pos, pos)]
          obj_real_points.append((obj, obj_real_point))

      # Compute the position of the objects on the spatial ball
      observed_points = {} # obj: [points]
      for pp in purified_perceptions:
          obj, pos, ref = pp["obj"], pp["pos"], pp["ref"]
          # Find the observed points
          # multi pos situation -> use the middle point
          if obj not in observed_points:
              observed_points[obj] = []

          for key in spatial_ball_map:
              if key in pos:
                  observed_points[obj].append(np.array(spatial_ball_map[key]))

          # obj: point
          observed_point = tuple(np.mean(np.array(observed_points[obj]), axis=0))
          observed_points[obj] = observed_point

      # print("Observed points: ", observed_points)

      # Calculate each metrics
      spatial_relationship_scroe = 0 # SR
      distance_score = 1000 # D # A far distance for no objects found
      cos_sim = 0 # Cosine Similarity
      mp_distance = 1000 # Middle Point

      if len(observed_points) > 0:
          # A far distance point to represent unfound points
          unfound_far_point = (100, 100, 100)

          sparse_correctness = 0
          distance = 0
          # cos_sim = 0
          # expected_mp = np.array([0,0,0], dtype='float64') # center: (0,0,0)
          # observed_mp = np.array([0,0,0], dtype='float64')
          for (ro, rp) in obj_real_points:
            op = observed_points.get(ro, unfound_far_point)

            ##### SPARSE CORRECTNESS #####
            sparse_correctness += 1 if op == rp else 0

            ##### DISTANCE #####
            distance += np.linalg.norm(np.array(rp) - np.array(op))

            ##### COSINE SIMILARITY #####
            # [-1,1] -> [0,1]
            # cos_sim += (1 + cosine_similarity(np.array(rp), np.array(op))) / 2

            ##### MIDDLE POINT #####
            # expected_mp += np.array(rp, dtype='float64')
            # observed_mp += np.array(op, dtype='float64')

          # expected_mp = expected_mp / len(obj_real_points)
          # observed_mp = observed_mp / len(obj_real_points)
          # mp_distance = np.linalg.norm(expected_mp - observed_mp)

      # normalize the score to (0,1]
      spatial_relationship_scroe = sparse_correctness / len(obj_real_points)
      dist_score = 1 / (1 + distance)
      # cos_sim_score = cos_sim / len(obj_real_points)
      # mp_dist_score = 1 / (1 + mp_distance)

      # print("<DEBUGGING>")
      # print("observed objects: ", observed_points)
      # print(f"dist_score: {dist_score}, total_cos_sim: {cos_sim_score}, mp_dist: {mp_dist_score}")
      # print("<DEBUGGING>")

      return dist_score, spatial_relationship_scroe, purified_perceptions

  def evaluate(
    self, 
    images, 
    prompt_datas,

    is_obj_recognition=True,
  ):
    w1, w2 = 0.5, 0.5

    combined_scores = []
    D_scores = []
    object_recognitions = []
    SR_scores = []  
    positions = []

    for i, data in enumerate(zip(images, prompt_datas)):
      image, prompt_data = data

      if isinstance(image, str):
        image = Image.open(image)

      print(f"Processing data {i}...")

      """
      'prompt_data': {
        'prompt': 'A scissors is positioned directly under a hot dog in a hallway.',
        'prompt_meta': {
          'center': 'hot dog',
          'objects': [{'obj': 'scissors', 'pos': 'under'}]
        }
      }
      """
      prompt, prompt_meta = prompt_data["prompt"], prompt_data["prompt_meta"]

      ########## Object Recognition ##########
      if is_obj_recognition:
          # object response
          center = prompt_meta["center"]
          objs = ""
          poss = []
          for pair in prompt_meta["objects"]:
              objs += f"and {pair['obj']}"
              poss.append(pair["pos"])
          positions.append("_".join(poss))

          object_instruction = (
              f"Please list the objects present in this image from the following: {center}, {objs}. "
              "If an object does not appear in the image, do not include it in the response. "
              "Only list the objects that are present, separated by commas."
              "If there are no metioned objects in this image, just output an empty string."
          )

          object_response = self.generate_description_of_image(object_instruction, image)
          object_recognition, undetectted_objs, is_undetect_center = self.compute_object_recognition(
              object_response,
              prompt_meta
          )

      ########## Spatial Arrangement ##########
      instruction = self.build_vlm_instruction_multi_objs(prompt_meta)
      spatial_response = self.generate_description_of_image(instruction, image)
      D, SR, purification = self.compute_spatial_accuracy_multi_objs(
          spatial_response,
          prompt_meta,
          undetectted_objs,
          is_undetect_center
      )

      ########## SR ##########
      SR_scores.append(SR)

      ########## D ##########
      D_scores.append(D)
      object_recognitions.append(object_recognition)

      combined_score = w1*D + w2*object_recognition
      combined_scores.append(combined_score)

      # print("instruction: ", instruction)
      # print("prompt: ", prompt)
      # print("prompt_meta: ", prompt_meta)
      # print("object_response: ", object_response)
      # print("spatial_response: ", spatial_response)
      print("purification: ", purification)
      print(f"D: {D}, SR: {SR},  O:{object_recognition}, Combined Scores: {combined_score}")
      print("="*20)

    # Create the DataFrame
    data = {
      'D+O': combined_scores,
      'D': D_scores,
      'O': object_recognitions,
      'SR': SR_scores,
      "Position": positions
    }

    return pd.DataFrame(data)