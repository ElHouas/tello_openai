#! /usr/bin/env python3

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

class MarsFeatures:
    def __init__(self, model=model, img_shape=(720, 960, 3), feature_dist=0.4, neighbor_dist=0.15):
        self.img_shape = img_shape
        self.features = []
        self.encoder = create_box_encoder(model)
        self.feature_dist = feature_dist
        self.neighbor_dist = neighbor_dist
        # warm up
        blank_image = np.zeros(self.img_shape, np.uint8)
        bbs = np.array([[455, 201, 521, 400], [180, 213, 249, 396]])
        for i in range(4):
            self.extractBBoxFeatures(blank_image, bbs, 0)
        self.resetDeepFeatures()

    def __preProcess(self, bboxes):
        #Convert tlrb to tlwh
        boxes = np.array(bboxes)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        return boxes
    
    def extractBBoxFeatures(self, img, bboxes, tracking_id=0):
        bbox = self.__preProcess([bboxes[tracking_id]])
        self.features  = self.encoder(img, bbox)

    def __calcFeaturesDistance(self, frame, bboxes):
        bboxes_features = self.__extractBBoxesFeatures(frame, bboxes)
        features_distance = dist.cdist(self.features, bboxes_features, "cosine")[0]
        return features_distance
    
    def __extractBBoxesFeatures(self, img, bboxes):
        bboxes = self.__preProcess(bboxes)
        bboxes_features = self.encoder(img, bboxes)
        return bboxes_features

    def __assignNewTrackingId(self, distance, threshold):
        # Logic: 
        # 1. If detect only one and the distance is less than 0.3, assign id;
        # 2. If detect more than one, but the first two closest distances' difference is lesss than 0.1, don't assign id;
        # 3. if the first two closest distances' difference is more than 0.1, and the closest distance is less than 0.3, assign id; 
        tracking_id = -1
        dist_sort = np.sort(distance)
        if len(dist_sort) == 0:
            tracking_id = -1
        elif len(dist_sort) == 1:
            if distance[0] < threshold:
                tracking_id = 0
        else:
            if (dist_sort[1]-dist_sort[0]) < self.neighbor_dist:
                tracking_id = -1
            else:
                min_position = np.argmin(distance)
                if distance[min_position] < threshold:
                    tracking_id = min_position
        return tracking_id

    
    def matchBoundingBoxes(self, frame, bboxes):
        if len(bboxes) == 0:
            target_id = -1
        else:
            distances = self.__calcFeaturesDistance(frame, bboxes)
            target_id = self.__assignNewTrackingId(distances, self.feature_dist)
        return target_id

    def resetDeepFeatures(self):
        self.features = []