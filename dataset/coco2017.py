from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

import random

class COCO2017:
    """
    It is unfair to
    """

    def __init__(self):
        """
            Please first download the COCO 2017 dataset with command:
            1. %mkdir coco
            2. %cd coco
            3. !wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
            4. !unzip annotations_trainval2017.zip
            5. %cd ..
        """
        dataDir='coco'
        dataType='val2017'
        annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
        
        # initialize COCO api for instance annotations
        self.coco=COCO(annFile)

        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_names=[cat['name'] for cat in self.cats]
        self.sup_cat_names = set([cat['supercategory'] for cat in self.cats])
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
        self.spatial_relationships = {
            "front": "in front of", 
            "behind": "behind", 
            "left": "to the left of", 
            "right": "to the right of", 
            "above": "above", 
            "below": "below", 
            "on": "on", 
            "under": "under"
        }
       
    def get_data(self):
        """
            Sample 1000 image prompts from the COCO 2017 dataset.
            Each prompt has 2 objects and a background.
        """
        random.seed(42)

        limit = 1000
        print(f"""
            Total number of classes: {len(self.cat_names)}
            We select 2 objects from the above classes and 1 background from the following set.
            Total combination of prompts: {len(self.cat_names)} x {len(self.cat_names)-1} x {len(self.background_set)} = {len(self.cat_names) * (len(self.cat_names) - 1) * len(self.background_set)}
            
            We only select {limit} prompts.
        """)
        
        prompt_datas = []
        seen_prompt = set()
        count = 0
        while count < limit:
            selected_objects = random.sample(self.cat_names, 2)
            selected_background = random.sample(self.background_set, 1)[0]
            selected_spatial = random.sample(self.simple_spatial_relationships, 1)[0]

            prompt = f"A {selected_objects[0]} is {self.spatial_relationships[selected_spatial]} a {selected_objects[1]} {selected_background}."

            if prompt in seen_prompt:
                continue
            seen_prompt.add(prompt)

            prompt_data = {
                'prompt': prompt,
                'prompt_meta': {
                    'center': selected_objects[1],
                    'objects': [{'obj': selected_objects[0], 'pos': selected_spatial}],
                    'background': selected_background.split(" ")[-1]
                }
            }

            prompt_datas.append(prompt_data)
            count += 1

        return prompt_datas

