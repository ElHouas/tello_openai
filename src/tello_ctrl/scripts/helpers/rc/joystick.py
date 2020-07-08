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


class Ctrl():
    def __init__(self):
        rospy.init_node('ctrl_node', anonymous=True)
        self.rate = rospy.Rate(30)

        self.pose_sub = rospy.Subscriber('/joy', Joy, self.ctrl_callback)
        rospy.spin()
        
    def ctrl_callback(self, data):
        self.joy_buttons = data.buttons
        self.joy_axes = data.axes
        print(self.joy_buttons)
        print(self.joy_axes)

def main():
    try:
        Ctrl()
    except KeyboardInterrupt:
        pass
    

if __name__ == '__main__':
    main()