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
import rospkg


def diffBetweenSmallest(a):
    if a.size == 1: return 0
    a = np.sort(a)
    return a[1] - a[0]

def bbox_to_center(bbox):
    '''Return center of box.

    Args:
        bbox in corner format [x1 y1 x2 y2] where x/yi are the corner pts.

    Returns:
        coordinates, [x y], of center.
    '''
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    return [x, y]

def scale_bbox(bbox, factor):
    ''' Scale bbox size by factor.

    Args:
        bbox in corner format [x1 y1 x2 y2] where x/yi are the corner pts.
        factor: Multiplication factor to times bbox by.

    Returns:
        bbox scaled by factor.
    '''
    bb = np.empty(4)
    center = bbox_to_center(bbox)
    half_width = ( bbox[2] - bbox[0] ) / 2
    half_height = ( bbox[3] - bbox[1] ) / 2
    bb[0] = center[0] - factor * half_width
    bb[1] = center[1] - factor * half_height
    bb[2] = center[0] + factor * half_width
    bb[3] = center[1] + factor * half_height
    return bb.astype(int)


osnet_path = rospkg.RosPack().get_path('tracking') \
                            +"/scripts/helpers/deep_feature_tracking/models/osnet_trt_fp16.pth"
mn_path = rospkg.RosPack().get_path('tracking') \
                            +"/scripts/helpers/deep_feature_tracking/models/mobilenetv2_x10_trt_fp16.pth"


class DeepFeatures():
    def __init__(self,
                 model_o = osnet_path ,
                 model_m= mn_path,
                 img_shape=(720, 960, 3),
                 feature_thresh=0.9,
                 neighbor_dist=0.2,
                 histogram_multiplier=1.5):
        self.img_shape = img_shape
        self.tracked_bbox_features = []
        self.model_trt_o = TRTModule()
        self.model_trt_o.load_state_dict(torch.load(model_o))
        self.model_trt_m = TRTModule()
        self.model_trt_m.load_state_dict(torch.load(model_m))
        self.feature_thresh = feature_thresh
        self.neighbor_dist = neighbor_dist
        self.histogram_multiplier = histogram_multiplier
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
        features_o = self.model_trt_o(crop).cpu().detach().numpy()
        features_m = self.model_trt_m(crop).cpu().detach().numpy()
        features_h = self.__calcHist(frame, bbox)
        return features_o, features_m, features_h

    def __calcHist(self, frame, bbox):
        bbox = scale_bbox(bbox, 0.5)
        hsv_frame = frame
        bbox_frame = hsv_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        hist = cv2.calcHist([bbox_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def __extractBBoxesFeatures(self, frame, bboxes):
        bboxes_features_o = []
        bboxes_features_m = []
        bboxes_features_h = []
        for bbox in bboxes:
            crop = self.__preProcess(frame, bbox)
            features_o = self.model_trt_o(
                crop).cpu().detach().numpy()  # output a 1d array [1, 512]
            features_m = self.model_trt_m(
                crop).cpu().detach().numpy()  # output a 1d array [1, 1280]
            features_h = self.__calcHist(frame, bbox)
            bboxes_features_o.append(list(features_o[0]))
            bboxes_features_m.append(list(features_m[0]))
            bboxes_features_h.append(features_h)
        return np.array(bboxes_features_o), np.array(bboxes_features_m), np.array(bboxes_features_h)

    def calcFeaturesDistance(self, frame, bboxes):
        bboxes_features_o, bboxes_features_m, bboxes_features_h = self.__extractBBoxesFeatures(frame, bboxes)
        features_distance_o = dist.cdist(self.features_o, bboxes_features_o, "cosine")
        features_distance_m = dist.cdist(self.features_m, bboxes_features_m, "cosine")
        features_distance_h = dist.cdist([self.features_h], bboxes_features_h, "cosine")
        features_distance_h *= self.histogram_multiplier
        return features_distance_o.flatten(), features_distance_m.flatten(), features_distance_h.flatten()

    def extractTrackedBBoxFeatures(self, frame, bbox):
        self.features_o, self.features_m, self.features_h = self.__extractBboxFeatures(frame, bbox)

    def resetDeepFeatures(self):
        self.features_o = []
        self.features_m = []
        self.features_h = []
    
    def scale_and_add_distances(self, distances_o, distances_m, distances_h):
        if len(distances_o) == 0: return []
        o_diff = diffBetweenSmallest(distances_o)
        m_diff = diffBetweenSmallest(distances_m)
        h_diff = diffBetweenSmallest(distances_h)
        ods = (( 1 + o_diff ) ** 2) * np.array(distances_o)
        mds = (( 1 + m_diff ) ** 2) * np.array(distances_m)
        hds = (( 1 + h_diff ) ** 2) * np.array(distances_h)
        distances = np.add(np.add(ods, mds), hds) 
        return distances
    
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
        if bboxes.size == 0: return -1
        distances_o, distances_m, distances_h = self.calcFeaturesDistance(frame, bboxes)
        distances = self.scale_and_add_distances(distances_o, distances_m, distances_h)
        new_id = self.__assignNewTrackingId(distances)
        return new_id
