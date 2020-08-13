#! /usr/bin/env python3

import os
import logging
import errno
import argparse
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import rospy
import rospkg
rospack = rospkg.RosPack()

model_path = os.path.join(rospack.get_path("tracking"), "scripts/helpers/deep/checkpoint/", "original_ckpt.t7")

from .original_model import Net


class ImageEncoder(object):

    def __init__(self, model_path=model_path, use_cuda=True, img_shape=(720, 960, 3)):
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
        

    def __tlrb_to_tlwh(self, bboxes):
        boxes = np.array(bboxes)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        return boxes
    

    def __extract_image_patch(self, image, bbox, patch_shape):
        bbox = np.array(bbox)
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

    def __preprocess(self, img, bboxes):
        im_crops = []
        bboxes = self.__tlrb_to_tlwh(bboxes)
        for bbox in bboxes:
            im = self.__extract_image_patch(img, bbox, self.img_shape[:2])
            im_crops.append(im)    
        return im, im_crops    

    def __run_in_batches(self, im, im_crops):
        #TODO:CECK IF NOT IM AND SOLVE EXCEPTION
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, img, bboxes):
        im, im_crops = self.__preprocess(img, bboxes)
        im_batch = self.__run_in_batches(im, im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


def create_box_encoder():
    image_encoder = ImageEncoder()

    def encoder(img, bboxes):        
        feature = image_encoder(img, bboxes)
        return feature

    return encoder

def main():
    encoder = create_box_encoder()

if __name__ == "__main__":
    main()
