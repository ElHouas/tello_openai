#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, "/usr/local/lib/python3.6/dist-packages/jetcam-0.0.0-py3.6.egg/")
import rospy
import rospkg
from sensor_msgs.msg import Image
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
import numpy
import torch2trt
from torch2trt import TRTModule
numpy.set_printoptions(threshold=sys.maxsize)
from math import *


nPoints = 18

class TrtPose():
    def __init__(self):
        self.goal = 0.0  # [angle]

        print("Getting Path to package...")
        self.follow_people_configfiles_path = rospkg.RosPack().get_path('tracking')+"/scripts/helpers/trtpose/models"

        print("We get the human pose json file that described the human pose")
        humanPose_file_path = os.path.join(rospkg.RosPack().get_path('tracking')+"/scripts/helpers/trtpose/models/", 'human_pose.json')

        print("Opening json file")
        with open(humanPose_file_path, 'r') as f:
            self.human_pose = json.load(f)

        print("Creating topology")
        self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)
        #print("Topology====>", self.topology)

        self.WIDTH = 960
        self.HEIGHT = 720

        OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        #OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
        optimized_model_weights_path = os.path.join(self.follow_people_configfiles_path, OPTIMIZED_MODEL)

        if not os.path.exists(optimized_model_weights_path):
            self.__create_optimodel(optimized_model_weights_path)

        print("Load the saved model using Torchtrt")
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(optimized_model_weights_path))

        print("Define the Image processing variables")
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device("cuda")

        print("Classes to parse the object of the NeuralNetwork and draw on the image")
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

    def detect(self, image):

        data = self.__preprocess(image)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)

        return counts, objects, peaks, self.topology

