#! /usr/bin/env python3

import numpy as np
import time
import rospy
import jetson.inference
import jetson.utils
import argparse
import sys
import os
import cv2
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()
#Import custom modules

class Detection:
    """ Thread run for logic to do with bboxes; inference etc. """
    def __init__(self):
        parser = argparse.ArgumentParser(description="Locate objects in an image using an object detection DNN.", formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())
        parser.add_argument("--network", type=str, default="pednet", help="pre-trained model to load (see below for options)")
        parser.add_argument("--overlay", type=str, default="none", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
        parser.add_argument("--threshold", type=float, default=0.2, help="minimum detection threshold to use")
        parser.add_argument("--device", type=str, default="GPU", help="Device to use. Either GPU or DLA")
        parser.add_argument("--precision", type=str, default="FP16", help="Either INT8, FP16, FP32")
        try:
            opt = parser.parse_known_args()[0]
            rospy.loginfo(opt)
        except:
            print("")
            parser.print_help()
            sys.exit(0)
        #TODO: More efficient arg parser and argv
        argv = ['--network=pednet', '--precision=FP16', '--device=GPU', '--allowGPUFallback', '--threshold=0.2']
        self.net = jetson.inference.detectNet(opt.network, argv, opt.threshold)
        self.opt = opt

    def detect(self, image):
        if image is None: return np.array([])
        cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        cuda_mem = jetson.utils.cudaFromNumpy(cv_image)
        detections = self.net.Detect(cuda_mem, image.shape[1], image.shape[0], self.opt.overlay)
        centroids, bboxes = self._detections2bboxes(detections)
        return centroids, bboxes

    def _detections2bboxes(self, detections):
        bboxes = []
        centroids = []
        for d in detections:
            bbox = [d.Left, d.Top, d.Right, d.Bottom]
            bboxes.append(bbox)
            cent = self._bbox_to_center(bbox)
            centroids.append(cent)
        return np.array(centroids), np.array(bboxes)

    def _bbox_to_center(self, bbox):
        '''Return center of box.
        Args:
            bbox in corner format [x1 y1 x2 y2] where x/yi are the corner pts.
        Returns:
            coordinates, [x y], of center.
        '''
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        return [int(x), int(y)]