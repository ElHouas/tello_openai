#! /usr/bin/env python

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
from scipy.spatial import distance as dist
import rospy
import rospkg
rospack = rospkg.RosPack()

from .encoder import create_box_encoder

model = os.path.join(rospack.get_path("tracking"), "scripts/helpers/mars/model/", "mars-small128.pb")

class DeepFeatures:
    def __init__(self):
        self.encoder = create_box_encoder(model)

    def __preProcess(self, bboxes):
        #Convert tlrb to tlwh
        boxes = np.array(bboxes)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        return boxes
    
    def extractBBoxFeatures(self, img, bboxes, tracking_id=0):
        bbox = self.__preProcess([bboxes[tracking_id]])
        bbox_features = self.encoder(img, bbox)
        return bbox_features
    
    def extractBBoxesFeatures(self, img, bboxes):
        bboxes = self.__preProcess(bboxes)
        bboxes_features = self.encoder(img, bboxes)
        return bboxes_features