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
                 osnet_weight_lower=1.0):
        self.tracked_bbox_features = []
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(model))
        self.dists_buffer = []
        self.history_length = osnet_history_length
        self.weight_upper = osnet_weight_upper
        self.weight_lower = osnet_weight_lower
        self.patch_shape = [256, 128]
        self.transforms = torchvision.transforms.Compose([
            #torchvision.transforms.Resize((256, 128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])

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
                                       bboxes_features, "cosine")
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

    def __assignNewTrackingId(self):
        tracking_id = -1
        max_entries = max([len(x) for x in self.dists_buffer])
        last_entries = len(self.dists_buffer[-1])
        for row in self.dists_buffer:
            if len(row) < max_entries:
                row.extend(np.nan for _ in range(max_entries - len(row)))
        array = np.array(self.dists_buffer)
        array = np.delete(array, np.s_[last_entries:max_entries], axis=1)
        means = self.weighted_mean(array)
        min_position = np.argmin(means)
        tracking_id = min_position
        return tracking_id, means

    def __euclidean_squared_distance(self, input1, input2):
        """Computes euclidean squared distance.
        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.
        Returns:
            torch.Tensor: distance matrix.
        """
        m, n = input1.size(0), input2.size(0)
        mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
        mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat = mat1 + mat2
        distmat.addmm_(1, -2, input1, input2.t())
        return distmat

    def reduce_bboxes_to_nearby(self, bboxes):
        reduced_bboxes = []
        for bbox in bboxes:
            dx = abs((bbox[2] + bbox[0]) / 2 - self.target_cent[0, 0])
            dy = abs((bbox[3] + bbox[1]) / 2 - self.target_cent[0, 1])
            if dx < 100 and dy < 80:
                reduced_bboxes.append(bbox)
        return np.array(reduced_bboxes)

    def __updateBuffer(self, distances):
        if len(self.dists_buffer) >= self.history_length:
            self.dists_buffer.pop(0)
        if distances.shape[1] == 1:
            self.dists_buffer.append(distances.flatten().tolist())
        else:
            self.dists_buffer.append(np.squeeze(distances).tolist())

    def matchBoundingBoxes(self, frame, bboxes):
        if bboxes.size == 0: return -1, []
        # bboxes = self.reduce_bboxes_to_nearby(bboxes)
        if bboxes.size == 0: return -1, []
        distances = self.calcFeaturesDistance(frame, bboxes)
        self.__updateBuffer(distances)
        new_id, confs = self.__assignNewTrackingId()
        return new_id, confs
