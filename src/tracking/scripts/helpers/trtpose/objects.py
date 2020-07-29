#!/usr/bin/env python3
import os
import sys
import rospy
import rospkg
import json
import trt_pose.coco
import trt_pose.models
import torch
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import numpy as np
import torch2trt
from torch2trt import TRTModule
np.set_printoptions(threshold=sys.maxsize)
from math import *
from helpers.utils.bbox import *


class TrtPose():
    def __init__(self):
        self.goal = 0.0  # [angle]

        rospy.loginfo("Getting Path to package...")
        self.follow_people_configfiles_path = rospkg.RosPack().get_path('tracking')+"/scripts/helpers/trtpose/models"

        rospy.loginfo("We get the human pose json file that described the human pose")
        humanPose_file_path = os.path.join(rospkg.RosPack().get_path('tracking')+"/scripts/helpers/trtpose/models/", 'human_pose.json')

        rospy.loginfo("Opening json file")
        with open(humanPose_file_path, 'r') as f:
            self.human_pose = json.load(f)

        rospy.loginfo("Creating topology")
        self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)
        #print("Topology====>", self.topology)

        self.WIDTH = 960
        self.HEIGHT = 720

        OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        #OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
        optimized_model_weights_path = os.path.join(self.follow_people_configfiles_path, OPTIMIZED_MODEL)

        if not os.path.exists(optimized_model_weights_path):
            self.__create_optimodel(optimized_model_weights_path)

        rospy.loginfo("Load the saved model using Torchtrt")
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(optimized_model_weights_path))

        rospy.loginfo("Define the Image processing variables")
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device("cuda")

        rospy.loginfo("Classes to parse the object of the NeuralNetwork and draw on the image")
        self.parse_objects = ParseObjects(self.topology)
        self.draw_objects = DrawObjects(self.topology)
    
    def __create_optimodel(self, optimized_model_weights_path):
        rospy.loginfo("** No optimised model found. **")
        num_parts = len(self.human_pose['keypoints'])
        num_links = len(self.human_pose['skeleton'])

        rospy.loginfo("Creating Model")
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
        #model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
        rospy.loginfo("Load the weights from the eight files predownloaded to this package")
        MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
        #MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
        model_weights_path = os.path.join(self.follow_people_configfiles_path, MODEL_WEIGHTS)

        rospy.loginfo("Load state dict")
        model.load_state_dict(torch.load(model_weights_path))

        rospy.loginfo("Creating empty data")
        data = torch.zeros((1, 3, self.HEIGHT, self.WIDTH)).cuda()

        rospy.loginfo("Use tortchtrt to go from Torch to TensorRT to generate an optimised model")
        self.model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

        rospy.loginfo("Saving new optimodel")
        torch.save(self.model_trt.state_dict(), optimized_model_weights_path)


    def __preprocess(self, image):
        self.device = torch.device("cuda")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]


    def __keypoints_conversion(self, peaks):
        """TODO: convert 18 keypoints from TRT pose estimation to the 15 needed for GCN"""
        posetrack_order = [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9, 17, 0, 1]
        posetrack = [peaks[i] for i in posetrack_order]
        pose = np.array(posetrack).flatten()
        return pose 


    def __poses_to_bboxes(self, poses):
        '''Calculates bounding boxes from poses.
           Adapted from CIE logic 
        Args:
            poses: List of poses. 
        Return: 
            List of bboxes.
        '''
        bboxes = []
        for i in range(len(poses)):
            points = []
            for j in range(15):
                x, y = poses[i][j*2] , poses[i][j*2+1]
                if x == -1 or y == -1:
                    continue
                points.append([x,y])
            if points == []:
                continue
            points = np.array(points)
            min_x, min_y = np.min(points, axis=0)
            max_x, max_y = np.max(points, axis=0)
            bbox_tight = [min_x, min_y, max_x, max_y]
            bboxes.append(bbox_tight)
        return np.array(bboxes).astype(int)

    def detect(self, frame):

        data = self.__preprocess(frame)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        object_counts, objects, normalized_peaks = self.parse_objects(cmap, paf)

        height = self.HEIGHT
        width = self.WIDTH
  
        count = int(object_counts[0])
        K = self.topology.shape[0]
        
        poses = []
        
        for i in range(count):
            color = (0, 255, 0)
            obj = objects[0][i]
            C = obj.shape[0]
            pose = []
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    #cv2.circle(frame, (x, y), 3, color, 2)
                    kp = [x, y]
                else:
                    kp = [-1, -1]
                pose.append(kp)

            # for k in range(K):
            #     c_a = self.topology[k][2]
            #     c_b = self.topology[k][3]
            #     if obj[c_a] >= 0 and obj[c_b] >= 0:
            #         peak0 = normalized_peaks[0][c_a][obj[c_a]]
            #         peak1 = normalized_peaks[0][c_b][obj[c_b]]
            #         x0 = round(float(peak0[1]) * width)
            #         y0 = round(float(peak0[0]) * height)
            #         x1 = round(float(peak1[1]) * width)
            #         y1 = round(float(peak1[0]) * height)
            #         cv2.line(frame, (x0, y0), (x1, y1), color, 2)


            pose = self.__keypoints_conversion(pose)
            if(np.count_nonzero(pose==-1) < 10):
                poses.append(pose)
        
        bboxes = self.__poses_to_bboxes(poses)
        centroids = [bbox_to_center(bbox) for bbox in bboxes]

        
        return centroids, bboxes

