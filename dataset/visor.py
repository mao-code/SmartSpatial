import json
import random
from dataset.coco2017 import COCO2017

class VISOR:
    def __init__(self):
        self.background_set = [
            "in a park", 
            "on a beach", 
            "in an urban area", 
            "in a forest", 
            "at a farm", 
            "in a kitchen", 
            "in a living room", 
            "under a clear sky", 
            "in a snowy field", 
            "at sunset"
        ]
        self.simple_spatial_relationships = [
            "front",
            "behind",
            "left",
            "right",
            "on",
            "under",
            "above",
            "below"
        ]
        self.simple_spatial_relationships_mapping = {
            "front": "in front of", 
            "behind": "behind", 
            "left": "to the left of", 
            "right": "to the right of", 
            "above": "above", 
            "below": "below", 
            "on": "on", 
            "under": "under"
        }

        file_path = 'dataset/visor_2d.json'
        with open(file_path, 'r') as file:
            data = json.load(file) 
        self.data = data

        coco2017 = COCO2017() # For filtering categories
        self.coco_cat_names = set(coco2017.cat_names)

    def is_coco_category(self, obj_attributes, coco_cat_set):
        """
        Returns True if the last attribute in `obj_attributes` is found
        in the set of COCO categories.

        Better for YOLO to detect objects
        """
        if not obj_attributes:
            return False
        return obj_attributes[-1] in coco_cat_set

    def get_data(self):
        """
        1. Sample 1000 instances from the VISOR dataset.
           1.1  For the 2D relationships (left, right, above, below),
                we need 166 each, total 664.
           1.2  For 3D relationships (front, behind),
                we sample 168 each, total 336, but these come
                from the same original 2D data: we replace their 'rel_type'
                with front/behind.
           2. Add background to each instance.
           3. Organize them into prompt_data format.
           
           Return: A list of dictionaries in the format:
               {
                   'prompt': 'A book is on a dining table in the library.',
                   'prompt_meta': {
                       'center': 'dining table',
                       'objects': [{'obj': 'book', 'pos': 'on'}],
                       'background': 'library'
                   }
               }
        """
        random.seed(42) 

        filtered_data = []
        for item in self.data:
            if (
                self.is_coco_category(item["obj_1_attributes"], self.coco_cat_names) and
                self.is_coco_category(item["obj_2_attributes"], self.coco_cat_names)
            ):
                filtered_data.append(item)

        # -----------------------------------------------------------
        # 1) Collect items for the 4 two-dimensional relationships
        #    We want exactly 166 for each: left, right, above, below
        # -----------------------------------------------------------
        phrase_to_simple = {
            "to the left of": "left",
            "to the right of": "right",
            "above": "above",
            "below": "below"
        }
        
        # Filter out items for each relationship
        rel_buckets = {r: [] for r in phrase_to_simple}  # "to the left of", "above", etc.
        for item in filtered_data:
            rel_type = item["rel_type"]
            if rel_type in rel_buckets:
                rel_buckets[rel_type].append(item)

        # Now sample 166 from each bucket
        two_d_samples = []
        for rel_phrase, bucket in rel_buckets.items():
            # If the dataset is large enough, sample 166
            # If it might not be, you might do min(len(bucket), 166)
            chosen = random.sample(bucket, 166)
            two_d_samples.extend(chosen)

        # -----------------------------------------------------------
        # 2) Collect items for the 3D relationships (front, behind)
        #    We'll sample 336 from the entire dataset, ignoring
        #    their original 2D relation, because we'll override it.
        # -----------------------------------------------------------
        # Let's do 168 with "front" and 168 with "behind"
        three_d_replacements = random.sample(filtered_data, 336)
        
        front_half = three_d_replacements[:168]
        behind_half = three_d_replacements[168:]
        
        def override_relation(items, new_rel):
            for it in items:
                it["rel_type"] = new_rel
            return items
        
        front_items = override_relation(front_half, "in front of")
        behind_items = override_relation(behind_half, "behind")
        three_d_samples = front_items + behind_items
        
        # -----------------------------------------------------------
        # Combine them all: total 664 + 336 = 1000
        # -----------------------------------------------------------
        final_samples = two_d_samples + three_d_samples
        random.shuffle(final_samples) 

        # -----------------------------------------------------------
        # 3) Convert to prompt_data format
        #    We want 'prompt' plus 'prompt_meta':
        #
        #    prompt: "A <obj1> is <pos> a <obj2> in <background>."
        #    prompt_meta.center: <obj2>
        #    prompt_meta.objects: [ { "obj": <obj1>, "pos": <pos> } ]
        #    prompt_meta.background: <background> (minus the "in"/"at" prefix)
        # -----------------------------------------------------------

        phrase_or_simple_to_simple = {
            "to the left of": "left",
            "to the right of": "right",
            "above": "above",
            "below": "below",
            "in front of": "front",
            "behind": "behind"
        }

        prompt_data_list = []
        
        for item in final_samples:
            rel_phrase = item["rel_type"]
            if rel_phrase in phrase_or_simple_to_simple:
                final_rel = phrase_or_simple_to_simple[rel_phrase]
                final_rel_phrase = rel_phrase
            else:
                continue
            
            # 2) Compose the two objects
            obj1_str = " ".join(item["obj_1_attributes"]) 
            obj2_str = " ".join(item["obj_2_attributes"]) 
            
            # 3) Random background
            background_full = random.choice(self.background_set)  

            # 4) Construct the final prompt. 
            prompt = f"A {obj1_str} is {final_rel_phrase} a {obj2_str} {background_full}."

            prompt_meta = {
                "center": obj2_str,
                "objects": [
                    {
                        "obj": obj1_str,
                        "pos": final_rel
                    }
                ],
                "background": background_full
            }

            prompt_data_list.append({
                "prompt": prompt,
                "prompt_meta": prompt_meta
            })

        return prompt_data_list