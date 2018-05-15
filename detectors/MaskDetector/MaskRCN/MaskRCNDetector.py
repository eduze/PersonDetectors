import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import detectors.MaskDetector.MaskRCN.coco as coco
import detectors.MaskDetector.MaskRCN.utils as utils
import detectors.MaskDetector.MaskRCN.model as modellib
import detectors.MaskDetector.MaskRCN.visualize as visualize

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush']



class MaskRCNDetector:
    def __init__(self, preview = False):
        self.COCO_MODEL_PATH = os.path.join(os.path.dirname(__file__),"trained_model","mask_rcnn_coco.h5")

        # Directory to save logs and trained trained_model
        self.MODEL_DIR = os.path.join(os.path.dirname(__file__), "logs")

        self.config = InferenceConfig()
        # Create trained_model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)


        # Load weights trained on MS-COCO
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)

        self.preview = preview

    def _preview(self, frame, rois, masks, class_ids, scores):
        visualize.display_instances(frame, rois, masks, class_ids, class_names, scores)

    def process_frame(self, frame):
        results = self.model.detect([frame], verbose=1)
        r = results[0]

        if self.preview:
            self._preview(frame, r['rois'], r['masks'], r['class_ids'], r['scores'])

        return r['rois'], r['masks'], r['class_ids'],r['scores']






