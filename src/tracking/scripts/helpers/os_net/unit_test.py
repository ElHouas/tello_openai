#!/usr/bin/env python3

#import system modules
from PIL import Image
import time
import unittest
import os
import numpy as np

#import tested custom module
from classes import OSNet

class TestTarget(unittest.TestCase):

    def setUp(self):
        # load model      
        model = "./model/osnet_trt_fp16.pth" #'/media/nvida/AI/ai_modules_store/tracking/os_net/model/gpu.engine' 
        self.os_net = OSNet(model=model)

        # load test image
        PATH = "./input/image_2_people.png"
        img = Image.open(PATH).convert('RGB')
        img = img.resize((640, 480))
        self.frame = np.asarray(img, dtype=np.uint8) # input image in format np arrray uint8

        self.tracked_features = np.load('./input/tracked_features.npy')
        self.bboxes = np.array([[455, 201, 521, 400], [180, 213, 249, 396]])

    def test_extractTrackedBBoxFeaturess(self):
        self.os_net.extractTrackedBBoxFeatures(self.frame, [180, 213, 249, 396])
        np.testing.assert_array_almost_equal(self.os_net.tracked_bbox_features, self.tracked_features, 6, "Failed on extracting bbox features.")

    def test_matchBoundingBoxes(self):
        self.os_net.extractTrackedBBoxFeatures(self.frame, [180, 213, 249, 396]) 
        tracked_id, _  = self.os_net.matchBoundingBoxes(self.frame, self.bboxes)
        self.assertEqual(tracked_id, 1, 'Failed matching Id')

if __name__ == '__main__':
    unittest.main()
