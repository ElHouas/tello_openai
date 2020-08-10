#!/usr/bin/env python3
import time
import numpy as np
from PIL import Image
import torch
import torchvision
from scipy.spatial import distance as dist
import tensorrt as trt
import torch2trt
from torch2trt import TRTModule
from statistics import mean
import cv2


class OSNet():
    def __init__(self,
                 model,
                 osnet_history_length=1,
                 osnet_weight_upper=1.0,
                 osnet_weight_lower=1.0,
                 batch_size=8,
                 feature_thresh=0.25,
                 neighbor_dist=0.05,
                 img_shape=(480, 640, 3)):
        self.tracked_bbox_features = []
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(model))
        self.dists_buffer = []
        self.history_length = osnet_history_length
        self.weight_upper = osnet_weight_upper
        self.weight_lower = osnet_weight_lower
        self.batch_size = batch_size
        self.feature_thresh = feature_thresh
        self.neighbor_dist = neighbor_dist
        self.img_shape = img_shape
        self.patch_shape = [256, 128]
        self.transforms = torchvision.transforms.Compose([
            #torchvision.transforms.Resize((256, 128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])
        # Warm up run
        # blank_image = np.zeros(self.img_shape, np.uint8)
        # for _ in range(4):
        #     self.__extractBboxFeatures(blank_image, [10, 10, 110, 110])
        self.resetDeepFeatures()

    def extract_image_patch(self, image, bbox, patch_shape):
        #Convert tlrb to tlwh
        bbox = np.array(bbox)
        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]
        
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image

    def __preProcess(self, frame, crop_dim):
        patch = self.extract_image_patch(frame, crop_dim, self.patch_shape)
        img = Image.fromarray(patch)
        img = self.transforms(img).cuda()
        img.unsqueeze_(0)
        return img

    def __extractBboxFeatures(self, frame, bbox):
        crop = self.__preProcess(frame, bbox)
        features = self.model_trt(crop).cpu().detach().numpy()
        return features

    def __extractBBoxesFeatures(self, frame, bboxes):
        bboxes_features = []
        for bbox in bboxes:
            crop = self.__preProcess(frame, bbox)
            features = self.model_trt(
                crop).cpu().detach().numpy()  # output a 1d array [1, 512]
            bboxes_features.append(list(features[0]))
        return np.array(bboxes_features)

    def calcFeaturesDistance(self, frame, bboxes):
        bboxes_features = self.__extractBBoxesFeatures(frame, bboxes)
        features_distance = dist.cdist(self.tracked_bbox_features,
                                       bboxes_features, "cosine")[0]
        return features_distance

    def extractTrackedBBoxFeatures(self, frame, bbox):
        self.tracked_bbox_features = self.__extractBboxFeatures(frame, bbox)

    def resetDeepFeatures(self):
        self.tracked_bbox_features = []
        self.dists_buffer = []

    def weighted_mean(self, array):
        coefs = np.linspace(self.weight_upper,
                            self.weight_lower,
                            num=array.shape[0])
        array = (array.T * coefs).T
        return np.nanmean(array, axis=0)

    def __assignNewTrackingId(self, distances):
        # Logic: 
        # 1. If detect only one and the mean distance is less than feature_thresh, assign id;
        # 2. If detect more than one, but the first two closest distances' difference is less than neighbor_dist, don't assign id;
        # 3. If the first two closest distances' difference is more than neighbor_dist, and the closest distance is less than feature_thresh, assign id; 
        tracking_id = -1
        if len(distances) == 1:
            if distances[0] < self.feature_thresh:
                tracking_id = 0
        else:
            dists_sort = np.sort(distances)
            if (dists_sort[1]-dists_sort[0]) < self.neighbor_dist:
                tracking_id = -1
            else:
                min_position = np.argmin(distances)
                if distances[min_position] < self.feature_thresh:
                    tracking_id = min_position
        return tracking_id


    def matchBoundingBoxes(self, frame, bboxes):
        # if bboxes.size == 0: return -1
        distances = self.calcFeaturesDistance(frame, bboxes)
        new_id = self.__assignNewTrackingId(distances)
        return new_id