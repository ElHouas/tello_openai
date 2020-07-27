#!/usr/bin/env python
#https://github.com/ros-teleop/teleop_twist_keyboard/blob/master/teleop_twist_keyboard.py

import rospy
from std_msgs.msg import Int8, String
import sys, select, termios, tty
import threading
import time

class Keypress(threading.Thread):
    def __init__(self):
        super(Keypress, self).__init__()
        self.publisher = rospy.Publisher('keypress', String, queue_size = 1)
        self.condition = threading.Condition()
        self.done = False
        self.key = ""

        self.start()
        
    def update(self, key):
        self.condition.acquire()
        self.key = key
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update("")
        self.join()

    def run(self):
        while not self.done:
            self.condition.acquire()
            key = str(self.key)
            self.condition.release()
            self.publisher.publish(key)

        key = ""
        self.publisher.publish(key)
        

def getKey(key_timeout):
    tty.setcbreak(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], key_timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('keypress_node')
    keypress = Keypress()

    try:
        while(1):
            key = getKey(None)
            if (key == '\x03'):
                break
            keypress.update(key)

    except Exception as e:
        print(e)
    finally:
        keypress.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)