#! /usr/bin/env python3

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import os
import time
import rospy
import rospkg
rospack = rospkg.RosPack()
from scipy.spatial import distance as dist

from .model import Net

model_path = os.path.join(rospack.get_path("tracking"), "scripts/helpers/deep/checkpoint/", "ckpt.t7")

class DeepFeatures():
    def __init__(self, model_path=model_path , use_cuda=True, img_shape=(720, 960, 3), feature_dist=0.4, neighbor_dist=0.15):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.img_shape = img_shape
        self.height = self.img_shape[0]
        self.width = self.img_shape[1]
        self.features = []
        self.feature_dist = feature_dist
        self.neighbor_dist = neighbor_dist
         # warm up
        blank_image = np.zeros(self.img_shape, np.uint8)
        bbs = np.array([[455, 201, 521, 400], [180, 213, 249, 396]])
        for i in range(4):
            self.extractBBoxFeatures(blank_image, bbs, 0)
        self.resetDeepFeatures()
    
    def __xywh_to_xyxy(self, bboxes):
        x,y,w,h = bboxes
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def __preprocess(self, img, bboxes):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        im_crops = []
        for bbox in bboxes:
            x1,y1,x2,y2 = self.__xywh_to_xyxy(bbox)
            im = img[y1:y2,x1:x2]
            im_crops.append(im)

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def extractBBoxFeatures(self, img, bboxes, tracking_id=0):
        im_batch = self.__preprocess(img, bboxes)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
            self.features = features.cpu().numpy() 

    def __calcFeaturesDistance(self, frame, bboxes):
        bboxes_features = self.__extractBBoxesFeatures(frame, bboxes)
        features_distance = dist.cdist(self.features, bboxes_features, "cosine")[0]
        return features_distance
    
    def __extractBBoxesFeatures(self, img, bboxes):
        im_batch = self.__preprocess(img, bboxes)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
            features = features.cpu().numpy() 
            return features

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

