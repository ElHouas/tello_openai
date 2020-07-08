#!/usr/bin/env python

#Ros utilites
import threading
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from sensor_msgs.msg import Image, CompressedImage, Imu
from std_msgs.msg import Empty
from tello_msgs.msg import FlightData
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

#Python Utilities
import av
import cv2
import numpy as np
import tellopy
from copy import deepcopy
from math import *
import time
import pygame
import pygame.locals

fpv = [960, 720]

# Helpers
from helpers.control import Control
control = Control()

from helpers.rc import JoystickPS4
ps4_js = JoystickPS4()

buttons = None
speed = 100
throttle = 0.0
yaw = 0.0
pitch = 0.0
roll = 0.0


class TelloDriver(object):
    def __init__(self):
       
        # Connect to the drone
        self._drone = tellopy.Tello()
        self._drone.connect()
        self._drone.wait_for_connection(10.0)

        # Init 
        rospy.init_node('tello_driver_node', anonymous=False)
        self.current_yaw = 0.0
        self.rate = rospy.Rate(30)
        self._cv_bridge = CvBridge()
        self.pose = Pose()
        self.frame = None
        self.centroids = []
        self.drone_position = None
        self.height = 0

        # ROS publishers
        self._flight_data_pub = rospy.Publisher('/tello/flight_data', FlightData, queue_size=10)
        self._image_pub = rospy.Publisher('/tello/camera/image_raw', Image, queue_size=10)
        self.pub_odom = rospy.Publisher('/tello/odom', Odometry, queue_size=10, latch=True)
        self.pub_imu= rospy.Publisher('/tello/imu', Imu, queue_size=10, latch=True)

        #  ROS subscribers
        self._drone.subscribe(self._drone.EVENT_FLIGHT_DATA, self.flight_data_callback)
        self._drone.subscribe(self._drone.EVENT_LOG_DATA, self.cb_data_log)
        rospy.Subscriber("/aiming/target_point", Point, self.point_callback)


        # Drone start fly
        #self._drone.takeoff()

        #Drone controller PS4
        global buttons
        pygame.init()
        pygame.joystick.init()
        
        try:
            js = pygame.joystick.Joystick(0)
            js.init()
            js_name = js.get_name()

            buttons = ps4_js

        except pygame.error:
              pass

        # Start video thread
        self._stop_request = threading.Event()
        video_thread = threading.Thread(target=self.video_worker)
        video_thread.start()
        
        rospy.on_shutdown(self.shutdown)
        
        
        while not rospy.is_shutdown():
            
            for e in pygame.event.get():
                    self.handle_input_event(self._drone, e)

            if self.frame is not None:
                start_time = time.time()
                frame = deepcopy(self.frame)
                #drone_position = deepcopy(self.drone_position)

                #self.centroids = [480, 360]
                
                if len(self.centroids)==0: 
                    continue
                else:
                    cent = self.centroids
                    rospy.loginfo('cent %s', cent)
                    
                    yaw_angle = control.yaw(cent)

                    try:
                        rospy.loginfo('yaw_angle %s', yaw_angle)
                        self._drone.clockwise(yaw_angle)

                        #self.pose.position =  drone_position
                        #self.pose.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0, yaw_angle*pi/180))
                        
                        #print(self.pose)
                        #self.pub_odom.pose.publish(self.pose)

                    except rospy.ServiceException:
                        pass

                    cv2.circle(frame, (480, cent[1]), 3, [0,0,255], -1, cv2.LINE_AA) #red
                    cv2.circle(frame, (cent[0], cent[1]), 3, [0,255,0], -1, cv2.LINE_AA) #green

                cv2.imshow("", frame)
                cv2.waitKey(1)
                 
                #print("%s seconds" % (time.time() - start_time))
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

    def cb_data_log(self, event, sender, data, **args):
        time_cb = rospy.Time.now()

        odom_msg = Odometry()
        odom_msg.child_frame_id = rospy.get_namespace() + 'base_link'
        odom_msg.header.stamp = time_cb
        odom_msg.header.frame_id = rospy.get_namespace() + 'local_origin'        

        # Height from MVO received as negative distance to floor
        odom_msg.pose.pose.position.z = -data.mvo.pos_z #self.height #-data.mvo.pos_z
        odom_msg.pose.pose.position.x = data.mvo.pos_x
        odom_msg.pose.pose.position.y = data.mvo.pos_y
        odom_msg.pose.pose.orientation.w = data.imu.q0
        odom_msg.pose.pose.orientation.x = data.imu.q1
        odom_msg.pose.pose.orientation.y = data.imu.q2
        odom_msg.pose.pose.orientation.z = data.imu.q3

        #self.drone_position = odom_msg.pose.pose.position

        # Linear speeds from MVO received in dm/sec
        odom_msg.twist.twist.linear.x = data.mvo.vel_y/10
        odom_msg.twist.twist.linear.y = data.mvo.vel_x/10
        odom_msg.twist.twist.linear.z = -data.mvo.vel_z/10
        odom_msg.twist.twist.angular.x = data.imu.gyro_x
        odom_msg.twist.twist.angular.y = data.imu.gyro_y
        odom_msg.twist.twist.angular.z = data.imu.gyro_z
                
        self.pub_odom.publish(odom_msg)
        
        imu_msg = Imu()
        imu_msg.header.stamp = time_cb
        imu_msg.header.frame_id = rospy.get_namespace() + 'base_link'
        
        imu_msg.orientation.w = data.imu.q0
        imu_msg.orientation.x = data.imu.q1
        imu_msg.orientation.y = data.imu.q2
        imu_msg.orientation.z = data.imu.q3        
        imu_msg.angular_velocity.x = data.imu.gyro_x
        imu_msg.angular_velocity.y = data.imu.gyro_y
        imu_msg.angular_velocity.z = data.imu.gyro_z
        imu_msg.linear_acceleration.x = data.imu.acc_x
        imu_msg.linear_acceleration.y = data.imu.acc_y
        imu_msg.linear_acceleration.z = data.imu.acc_z
        
        self.pub_imu.publish(imu_msg)

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

def main():
    try:
        TelloDriver()
    except KeyboardInterrupt:
        pass
    

if __name__ == '__main__':
    main()
