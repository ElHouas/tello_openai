#!/usr/bin/env python

#Ros utilites
import threading
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from tello_msgs.msg import FlightData
from cv_bridge import CvBridge, CvBridgeError

#Python Utilities
import av
import cv2
import numpy as np
import tellopy
from copy import deepcopy
from math import *
import time

fpv = [960, 720]

#Classes
from helpers.control import Control
control = Control()

class TelloDriver(object):
    def __init__(self):
       
        # Connect to the drone
        self._drone = tellopy.Tello()
        self._drone.connect()
        self._drone.wait_for_connection(60.0)

        # Init 
        rospy.init_node('tello_driver_node', anonymous=False)
        self.current_yaw = 0.0
        self.rate = rospy.Rate(10)
        self._cv_bridge = CvBridge()
        self.frame = None
        self.centroids = []

        # ROS publishers
        self._flight_data_pub = rospy.Publisher('/tello/flight_data', FlightData, queue_size=10)
        self._image_pub = rospy.Publisher('/tello/camera/image_raw', Image, queue_size=10)

        #  ROS subscribers
        self._drone.subscribe(self._drone.EVENT_FLIGHT_DATA, self.flight_data_callback)
        #rospy.Subscriber('/tello/odom', Pose, self.pose_callback)
        rospy.Subscriber("/aiming/target_point", Point, self.point_callback)


        # Drone start fly
        #self._drone.takeoff()

        # Start video thread
        self._stop_request = threading.Event()
        video_thread = threading.Thread(target=self.video_worker)
        video_thread.start()
        
        rospy.on_shutdown(self.shutdown)
        
        
        while not rospy.is_shutdown():
            if self.frame is not None:
                start_time = time.time()
                frame = deepcopy(self.frame)
                current_yaw = deepcopy(self.current_yaw)
                
                self.centroids = [480,360]

                if len(self.centroids)==0: 
                    continue
                else:
                    cent = self.centroids
                    print("cent", cent)
                    yaw_angle = control.yaw(cent)

                    try:
                        #self._drone.clockwise(yaw_angle)
                        print("yaw_angle", yaw_angle)

                    except rospy.ServiceException:
                        pass

                    cv2.circle(frame, (320, cent[1]), 3, [0,0,255], -1, cv2.LINE_AA)
                    cv2.circle(frame, (cent[0], cent[1]), 3, [0,255,0], -1, cv2.LINE_AA)

                cv2.imshow("", frame)
                cv2.waitKey(1)
                 
                # print("%s seconds" % (time.time() - start_time))
                #time.sleep((time.time() - start_time)) #slows down twice dont do it
                
            self.rate.sleep()



    def video_worker(self):
        container = av.open(self._drone.get_video_stream())
        rospy.loginfo('starting video pipeline')

        for frame in container.decode(video=0):
            try:
                color = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                color_mat = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                self._image_pub.publish(self._cv_bridge.cv2_to_imgmsg(color_mat, 'bgr8'))
            except CvBridgeError as e:
                print(e)
            self.frame = color_mat
            if self._stop_request.isSet():
                return
            
    def point_callback(self,data):
        self.centroids = [int(data.x), int(data.y)]

    def flight_data_callback(self, event, sender, data, **args):
        flight_data = FlightData()

        flight_data.battery_percent = data.battery_percentage
        flight_data.estimated_flight_time_remaining = data.drone_fly_time_left / 10.
        flight_data.flight_mode = data.fly_mode
        flight_data.flight_time = data.fly_time
        flight_data.east_speed = -1. if data.east_speed > 30000 else data.east_speed / 10.
        flight_data.north_speed = -1. if data.north_speed > 30000 else data.north_speed / 10.
        flight_data.ground_speed = -1. if data.ground_speed > 30000 else data.ground_speed / 10.
        flight_data.altitude = -1. if data.height > 30000 else data.height / 10.
        flight_data.equipment = data.electrical_machinery_state
        flight_data.high_temperature = data.temperature_height

        self._flight_data_pub.publish(flight_data)

    def shutdown(self):
        self._stop_request.set()
        self._drone.land()
        self._drone.quit()
        self._drone = None

def main():
    try:
        TelloDriver()
    except KeyboardInterrupt:
        pass
    

if __name__ == '__main__':
    main()
