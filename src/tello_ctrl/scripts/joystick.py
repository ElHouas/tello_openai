#!/usr/bin/env python

import numpy as np
import rospy
import roslib
import subprocess
import time
from geometry_msgs.msg  import Twist
from sensor_msgs.msg import Joy
import sys
import signal
import tellopy
import pygame
import pygame.locals
from subprocess import Popen, PIPE

#Helpers
from helpers.rc import JoystickPS4
ps4_js = JoystickPS4()

class Ctrl():
    def __init__(self):
        rospy.init_node('ctrl_node', anonymous=True)
        drone = tellopy.Tello()
        drone.connect()
        self.rate = rospy.Rate(30)
        rospy.loginfo('connected to drone')

        global buttons
        pygame.init()
        pygame.joystick.init()

        rospy.spin()
        
        try:
            js = pygame.joystick.Joystick(0)
            js.init()
            js_name = js.get_name()

            buttons = ps4_js

        except pygame.error:
              pass

        try:
            while 1:
                # loop with pygame.event.get() is too much tight w/o some sleep
                time.sleep(0.01)
                for e in pygame.event.get():
                    ps4_js.handle_input_event(drone, e)
        except KeyboardInterrupt as e:
            print(e)
        except Exception as e:
            print(e)

        drone.quit()
        exit(1)

def main():
    try:
        Ctrl()
    except KeyboardInterrupt:
        pass
    

if __name__ == '__main__':
    main()