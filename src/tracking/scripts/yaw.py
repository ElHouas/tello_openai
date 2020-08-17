#!/usr/bin/env python3

import av
import cv2
from math import *
import numpy as np
import os
from scipy.spatial import distance as dist
import threading
import tellopy
import time
import pygame
import pygame.locals
import yaml

from helpers.rc import JoystickPS4

# ros
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int8, Empty, Float64, String
from tracking.msg import BBox, BBoxes

# helpers

from helpers.detection import Detection
detection = Detection()

from helpers.deep_feature_tracking import *
dft = DeepFeatures()

roi_dist = 400 # To-do: dynamic
feature_dist = 0.4
neighbor_dist = 0.15
height = 720
width = 960 

speed = 100
throttle = 0.0
yaw = 0.0
pitch = 0.0
roll = 0.0


class Yaw(object):
    def __init__(self):
        rospy.init_node('yaw_node', anonymous=False)
        rate = rospy.Rate(30)

        # yaw cmd
        self.yaw_speed = 50
        self.yaw_cmd = ""
        self.target = [480,360] 
        self.frame = None
        self.data = None

        # tracking history
        self.tracking_bbox_features = None
        self.prev_target_cent = None
        self.prev_target_features = None
        target_id = -1       
        
        # Connect to the drone

        self.drone = tellopy.Tello()
        self.drone.connect()
        self.drone.wait_for_connection(60.0)
        rospy.loginfo('connected to drone')

        self.container = av.open(self.drone.get_video_stream())

        #Drone controller PS4
        self.ps4_js = JoystickPS4()
        self.__joystick()
        rospy.loginfo("PS4 control loaded")

        # keypress for selection
        self.prev_keypress = -1
        self.keypress = -1
        rospy.Subscriber('/keypress', String, self.key_callback)
        
        self.__createThreads()

        #self.drone.takeoff()

        while not rospy.is_shutdown():
            for e in pygame.event.get():
                self.handle_input_event(self.drone, e)            
            if self.data is not None:
                self.frame = np.array(self.data.to_image())
                centroids, bboxes = detection.detect(self.frame) # arrays

                if len(centroids) > 0:
                    # select target id using keypress
                    if self.keypress != -1 and self.keypress != self.prev_keypress:
                        target_id = self.keypress
                        dft.extractTrackedBBoxFeatures(self.frame, bboxes[target_id])
                        self.prev_target_cent = centroids[target_id]
                        self.prev_keypress = self.keypress
                        print("catch once")
                    elif self.prev_target_cent is not None:
                        # print("start tracking")

                        # extract features of bboxes
                        tracking_id = dft.matchBoundingBoxes(self.frame, bboxes)
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
                        cv2.putText(self.frame, str(i), (cent[0], cent[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
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


    def __createThreads(self):
        """ Run AI in thread. """

        self.stop_request = threading.Event()
        self.frame_thread = threading.Thread(target=self.frameCallback)
        self.frame_thread.start()
        
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
                min_position = np.argmin(distance)
                if distance[min_position] < threshold:
                    tracking_id = min_position

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

    def __joystick(self):
        global buttons
        pygame.init()
        pygame.joystick.init()

        try: 
            js = pygame.joystick.Joystick(0)
            js.init()
            js_name = js.get_name()

            buttons = self.ps4_js
        except pygame.error:
            pass                

    def update(self,old, new, max_delta=0.3):
        if abs(old - new) <= max_delta:
            res = new
        else:
            res = 0.0
        return res

    def handle_input_event(self, drone, e):
        global speed
        global throttle
        global yaw
        global pitch
        global roll
        if e.type == pygame.locals.JOYAXISMOTION:
            # ignore small input values (Deadzone)
            if -buttons.DEADZONE <= e.value and e.value <= buttons.DEADZONE:
                e.value = 0.0
            if e.axis == buttons.LEFT_Y:
                throttle = self.update(throttle, e.value * buttons.LEFT_Y_REVERSE)
                drone.set_throttle(throttle)
            if e.axis == buttons.LEFT_X:
                yaw = self.update(yaw, e.value * buttons.LEFT_X_REVERSE)
                drone.set_yaw(yaw)
            if e.axis == buttons.RIGHT_Y:
                pitch = self.update(pitch, e.value * buttons.RIGHT_Y_REVERSE)
                drone.set_pitch(pitch)
            if e.axis == buttons.RIGHT_X:
                roll = self.update(roll, e.value * buttons.RIGHT_X_REVERSE)
                drone.set_roll(roll)
        elif e.type == pygame.locals.JOYHATMOTION:
            if e.value[0] < 0:
                drone.counter_clockwise(speed)
            if e.value[0] == 0:
                drone.clockwise(0)
            if e.value[0] > 0:
                drone.clockwise(speed)
            if e.value[1] < 0:
                drone.down(speed)
            if e.value[1] == 0:
                drone.up(0)
            if e.value[1] > 0:
                drone.up(speed)
        elif e.type == pygame.locals.JOYBUTTONDOWN:
            if e.button == buttons.LAND:
                drone.land()
            elif e.button == buttons.SELECTION: 
                pass
                # rospy.loginfo("JOYSTICK SELECTION BUTTON PRESSED")
                # self.ai.update_id(None)
            elif e.button == buttons.UP:
                drone.up(speed)
            elif e.button == buttons.DOWN:
                drone.down(speed)
            elif e.button == buttons.ROTATE_RIGHT:
                drone.clockwise(speed)
            elif e.button == buttons.ROTATE_LEFT:
                drone.counter_clockwise(speed)
            elif e.button == buttons.FORWARD:
                drone.forward(speed)
            elif e.button == buttons.BACKWARD:
                drone.backward(speed)
            elif e.button == buttons.RIGHT:
                drone.right(speed)
            elif e.button == buttons.LEFT:
                drone.left(speed)
        elif e.type == pygame.locals.JOYBUTTONUP:
            if e.button == buttons.TAKEOFF:
                if throttle != 0.0:
                    print('###')
                    print('### throttle != 0.0 (This may hinder the drone from taking off)')
                    print('###')
                drone.takeoff()
            elif e.button == buttons.OFFBOARD:
                # rospy.loginfo("JOYSTICK: OFFBOARD BUTTON PRESSED")
                # self.ai.start_deep_tracking(None)
                pass
            elif e.button == buttons.UP:
                drone.up(0)
            elif e.button == buttons.DOWN:
                drone.down(0)
            elif e.button == buttons.ROTATE_RIGHT:
                drone.clockwise(0)
            elif e.button == buttons.ROTATE_LEFT:
                drone.counter_clockwise(0)
            elif e.button == buttons.FORWARD:
                drone.forward(0)
            elif e.button == buttons.BACKWARD:
                drone.backward(0)
            elif e.button == buttons.RIGHT:
                drone.right(0)
            elif e.button == buttons.LEFT:
                drone.left(0)

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
