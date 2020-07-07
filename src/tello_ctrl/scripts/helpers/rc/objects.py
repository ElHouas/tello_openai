#!/usr/bin/env python

import time
import sys
import tellopy
import pygame
import pygame.locals
from subprocess import Popen, PIPE
import rospy


class JoystickPS4:
    def __init__(self):
        # d-pad
        self.UP = -1  # UP
        self.DOWN = -1  # DOWN
        self.ROTATE_LEFT = -1  # LEFT
        self.ROTATE_RIGHT = -1  # RIGHT

        # bumper triggers
        self.TAKEOFF = 5  # R1
        self.LAND = 4  # L1
        # UNUSED = 7 #R2
        # UNUSED = 6 #L2

        # buttons
        self.FORWARD = 3  # TRIANGLE
        self.BACKWARD = 1  # CROSS
        self.LEFT = 0  # SQUARE
        self.RIGHT = 2  # CIRCLE

        # axis
        self.LEFT_X = 0
        self.LEFT_Y = 1
        self.RIGHT_X = 2
        self.RIGHT_Y = 3
        self.LEFT_X_REVERSE = 1.0
        self.LEFT_Y_REVERSE = -1.0
        self.RIGHT_X_REVERSE = 1.0
        self.RIGHT_Y_REVERSE = -1.0
        self.DEADZONE = 0.08




