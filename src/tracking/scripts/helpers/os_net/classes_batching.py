#!/usr/bin/env python3
import time
import numpy as np
from PIL import Image as PIm
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
        blank_image = np.zeros(self.img_shape, np.uint8)
        for _ in range(4):
            self.__extractBboxFeatures(blank_image, [10, 10, 110, 110])
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
        img = PIm.fromarray(patch)
        img = self.transforms(img).cuda()
        img.unsqueeze_(0)
        return img

    def __extractBboxFeatures(self, frame, bbox):
        crop = self.__preProcess(frame, bbox)
        features = self.model_trt(crop).cpu().detach().numpy()
        return features

    def __extractBBoxesFeatures(self, frame, bboxes):
        bboxes_features = np.empty([0, 512])
        t0 = time.time()
        crops = []
        for bbox in bboxes:
            crop = self.__preProcess(frame, bbox)
            crops.append(crop[0])
        if len(crops) <= self.batch_size:
            crops = torch.stack(crops, dim=0)
            bboxes_features = self.model_trt(
                crops).cpu().detach().numpy()  # output array [n, 512]
        else:
            iters = int( len(crops) / self.batch_size )
            last_loop = len(crops) % self.batch_size
            if last_loop != 0:
                iters += 1
            for i in range(iters):
                start = self.batch_size*i
                end = start + self.batch_size
                num = self.batch_size
                if end > len(crops):
                    end = len(crops)
                    num = len(crops) - start
                crops_stacked = torch.stack(crops[start:end], dim=0)
                features = self.model_trt(crops_stacked).cpu().detach().numpy()[0:num]  # output array [n, 512]
                bboxes_features = np.concatenate((bboxes_features, features))
        return np.array(bboxes_features)

    def calcFeaturesDistance(self, frame, bboxes):
        bboxes_features = self.__extractBBoxesFeatures(frame, bboxes)
        features_distance = dist.cdist(self.tracked_bbox_features,
                                       bboxes_features, "cosine")
        return features_distance

    def extractTrackedBBoxFeatures(self, frame, bbox):
        self.tracked_bbox_features = self.__extractBboxFeatures(frame, bbox)

    def resetDeepFeatures(self):
        self.tracked_bbox_features = []
        self.dists_buffer = []

    def __weightedMean(self, array):
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

    def __runBufferSystem(self, distances):
        if len(self.dists_buffer) >= self.history_length:
            self.dists_buffer.pop(0)
        if distances.shape[1] == 1:
            self.dists_buffer.append(distances.flatten().tolist())
        else:
            self.dists_buffer.append(np.squeeze(distances).tolist())
        max_entries = max([len(x) for x in self.dists_buffer])
        last_entries = len(self.dists_buffer[-1])
        for row in self.dists_buffer:
            if len(row) < max_entries:
                row.extend(np.nan for _ in range(max_entries - len(row)))
        array = np.array(self.dists_buffer)
        array = np.delete(array, np.s_[last_entries:max_entries], axis=1)
        means = self.__weightedMean(array)
        return means

    def matchBoundingBoxes(self, frame, bboxes):
        if bboxes.size == 0: return -1, []
        distances = self.calcFeaturesDistance(frame, bboxes)
        if self.history_length != 1:
            distances = self.__runBufferSystem(distances)
        else:
            distances = distances[0]
        new_id = self.__assignNewTrackingId(distances)
        return new_id