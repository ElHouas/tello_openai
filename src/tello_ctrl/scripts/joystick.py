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


buttons = None
speed = 100
throttle = 0.0
yaw = 0.0
pitch = 0.0
roll = 0.0

class Ctrl():
    def __init__(self):
        drone = tellopy.Tello()
        drone.connect()
        
        #INIT
        rospy.init_node('ctrl_node', anonymous=True)
        self.rate = rospy.Rate(30)
        rospy.loginfo('connected to drone')
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

        try:
            while 1:
                # loop with pygame.event.get() is too much tight w/o some sleep
                time.sleep(0.01)
                for e in pygame.event.get():
                    self.handle_input_event(drone, e)
        except KeyboardInterrupt as e:
            print(e)
        except Exception as e:
            print(e)

        drone.quit()
        exit(1)

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
        Ctrl()
    except KeyboardInterrupt:
        pass
    

if __name__ == '__main__':
    main()