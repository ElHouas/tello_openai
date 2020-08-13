#! /usr/bin/env python3

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import time
from scipy.spatial import distance as dist

from .encoder import create_box_encoder

class DeepFeatures():
    def __init__(self, img_shape=(720, 960, 3), feature_dist=0.4, neighbor_dist=0.14):
        self.encoder = create_box_encoder()
        self.img_shape = img_shape
        self.height = self.img_shape[0]
        self.width = self.img_shape[1]
        self.features = []
        self.feature_dist = feature_dist
        self.neighbor_dist = neighbor_dist
        self.resetDeepFeatures()  

    def extractBBoxFeatures(self, img, bboxes, tracking_id=0):
        self.features  = self.encoder(img, [bboxes[tracking_id]]) 

    def __calcFeaturesDistance(self, frame, bboxes):
        bboxes_features = self.__extractBBoxesFeatures(frame, bboxes)
        features_distance = dist.cdist(self.features, bboxes_features, "cosine")[0]
        print('features_distance', features_distance)
        return features_distance
           
    def __extractBBoxesFeatures(self, img, bboxes):
        bboxes_features = self.encoder(img, bboxes)
        return bboxes_features

    def __assignNewTrackingId(self, distance, threshold):
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
                    print('TRACKED')
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

