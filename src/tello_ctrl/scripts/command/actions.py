#!/usr/bin/env python

import rospy
import tellopy
from time import sleep

class TelloAction(object):
    def __init__(self):
        rospy.init_node('tello_action_node', anonymous=False)
        self.rate = rospy.Rate(1)

        # Connect to the drone
        self._drone = tellopy.Tello()
        self._drone.connect()
        self._drone.wait_for_connection(60.0)
        rospy.loginfo('connected to drone')
        
        self._drone.takeoff()
        rospy.on_shutdown(self.shutdown)
        self._drone.clockwise(360)
        sleep(5)
        self._drone.clockwise(0)


        while not rospy.is_shutdown():
            # self._drone.clockwise(360)
            sleep(5)

            # self._drone.counter_clockwise(20)
            # self._drone.up(0)
            # self._drone.down(0)
            # self._drone.forward(0)
            # self._drone.backward(0)
            # self._drone.left(0)
            # self._drone.right(0)

            self.rate.sleep()

    def shutdown(self):
        self._drone.land()
        self._drone.quit()
        self._drone = None

def main():
    try:
        TelloAction()
    except KeyboardInterrupt:
        pass
    

if __name__ == '__main__':
    main()