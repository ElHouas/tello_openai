#!/usr/bin/env python

import av
import cv2
from copy import deepcopy
from math import *
import numpy as np
import os
from scipy.spatial import distance as dist
import threading
import tellopy
import time


# ros
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8, Empty, Float64, String
from tracking.msg import BBox, BBoxes

# helpers
#from helpers.cvlib import Detection
#detection = Detection()

from helpers.detection import Detection
detection = Detection()

from helpers.mars import DeepFeatures
mars = DeepFeatures()
roi_dist = 400 # To-do: dynamic
feature_dist = 0.4
neighbor_dist = 0.15
height = 720
width = 960 

class Yaw(object):
    def __init__(self):
        rospy.init_node('yaw_node', anonymous=False)
        rate = rospy.Rate(30)

        # yaw cmd
        self.yaw_speed = 50
        self.yaw_cmd = ""
        self.target = [480,360] #???
        self.frame = None
        self.data = None

        # tracking history
        self.tracking_bbox_features = None
        self.prev_target_cent = None
        self.prev_target_features = None
        target_id = -1        # Connect to the drone
        
        # connect to the drone

        self.drone = tellopy.Tello()
        self.drone.connect()
        self.drone.wait_for_connection(60.0)
        rospy.loginfo('connected to drone')
        self.drone.takeoff()
        self.container = av.open(self.drone.get_video_stream())

        # keypress for selection
        self.prev_keypress = -1
        self.keypress = -1
        rospy.Subscriber('/keypress', String, self.key_callback)
        
        self.__createThreads()
        

        while not rospy.is_shutdown():
            if self.data is not None:
                self.frame = np.array(self.data.to_image())
                centroids, bboxes = detection.detect(self.frame) # arrays

                if len(centroids) > 0:
                    # select target id using keypress
                    if self.keypress != -1 and self.keypress != self.prev_keypress:
                        target_id = self.keypress
                        self.tracking_bbox_features = mars.extractBBoxFeatures(self.frame, bboxes, target_id)
                        self.prev_target_cent = centroids[target_id]
                        self.prev_keypress = self.keypress
                        print("catch once")
                    elif self.prev_target_cent is not None:
                        print("start tracking")

                        # extract features of bboxes
                        bboxes_features = mars.extractBBoxesFeatures(self.frame, bboxes)
                        features_distance = dist.cdist(self.tracking_bbox_features, bboxes_features, "cosine")[0]
                        tracking_id = self.__assignNewTrackingId(features_distance, threshold=feature_dist)

                        if tracking_id != -1:
                            print(tracking_id)
                            target_cent = centroids[tracking_id]
                            self.prev_target_cent = target_cent # for roi 
                            cv2.rectangle(self.frame, (target_cent[0]-20, target_cent[1]-40), (target_cent[0]+20, target_cent[1]+40), (0,0, 255), 1)

                            xoff = int(target_cent[0] - width/2)
                            self.__yaw(xoff)
                
                    i = 0
                    for cent in centroids:                   
                        cv2.circle(self.frame, (cent[0], cent[1]), 3, [0,0,255], -1, cv2.LINE_AA)
                        cv2.putText(self.frame, str(i), (cent[0], cent[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
                        i = i + 1

                cv2.imshow("", self.frame)
                cv2.waitKey(1)

            rate.sleep()
        
        rospy.spin()
        rospy.on_shutdown(self.shutdown)

    
    def frameCallback(self):
        frame_skip = 300
        for frame in self.container.decode(video=0):
            if 0 < frame_skip:
                     frame_skip = frame_skip - 1
                     continue
            start_time = time.time()
            if frame.time_base < 1.0/60:
                    time_base = 1.0/60
            else:
                    time_base = frame.time_base
                    frame_skip = int((time.time() - start_time)/time_base)
            if frame is None:
                continue
            self.data = frame


    def key_callback(self, data):
        if data.data != "":
            self.keypress = int(data.data)


    # def video_worker(self):
    #     while True:
    #         rospy.loginfo('starting video pipeline')
    #         if self.frame is None:
    #             continue
    #         # self.frame = np.array(self.data.to_image())
    #         cv2.circle(self.frame, (self.target[0], self.target[1]), 3, [0,0,255], -1, cv2.LINE_AA)
    #         cv2.imshow("", self.frame)
    #         cv2.waitKey(1)

    #     cv2.destroyAllWindows


    def __createThreads(self):
        """ Run AI in thread. """

        self.stop_request = threading.Event()
        self.frame_thread = threading.Thread(target=self.frameCallback)
        self.frame_thread.start()
        
        # self.video_thread = threading.Thread(target=self.video_worker)
        # self.video_thread.daemon = True
        # self.video_thread.start()

    #Tracking functions
    def __roi(self, centroids, bboxes):
        # Logic: 
        # 1. Only compare features of targets within centroids ROI

        centroids_dist = np.array(abs(centroids[:, [0]] - self.prev_target_cent[0])).flatten()
        position_roi = np.where(centroids_dist < roi_dist)[0]
        centroids_roi = centroids[position_roi, :]
        bboxes_roi = bboxes[position_roi, :]
        return centroids_roi, bboxes_roi

    def __assignNewTrackingId(self, distance, threshold):
        # Logic: 
        # 1. If detect only one and the distance is less than 0.3, assign id;
        # 2. If detect more than one, but the first two closest distances' difference is lesss than 0.1, don't assign id;
        # 3. if the first two closest distances' difference is more than 0.1, and the closest distance is less than 0.3, assign id; 

        tracking_id = -1
        dist_sort = np.sort(distance)
        if len(dist_sort) == 1:
            if distance[0] < threshold:
                tracking_id = 0
        else:
            if (dist_sort[1]-dist_sort[0]) < neighbor_dist:
                tracking_id = -1
            else:
                min_dist = np.argsort(distance.min(axis=0))
                min_position = np.where(min_dist==0)
                if distance[min_position[0][0]] < threshold:
                    tracking_id = min_position[0][0]

        return tracking_id

    # yaw
    def __yaw(self, xoff):
        distance = 100
        cmd = ""
        
        if xoff < -distance:
            cmd = "counter_clockwise"
        elif xoff > distance:
            cmd = "clockwise"
        else:
            if self.yaw_cmd is not "":
                getattr(self.drone, self.yaw_cmd)(0)
                self.yaw_cmd = ""

        if cmd is not self.yaw_cmd:
            if cmd is not "":
                getattr(self.drone, cmd)(self.yaw_speed)
                self.yaw_cmd = cmd

    def shutdown(self):
        self.stop_request.set()
        self.drone.land()
        self.drone.quit()
        self.drone = None


def main():
    try:
        Yaw()
    except KeyboardInterrupt:
        pass
    

if __name__ == '__main__':
    main()
