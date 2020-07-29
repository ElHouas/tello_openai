# Tello OpenAI
DJI Tello drone running OpenAI Stable Baselines

## Env
    sudo apt-get install ros-melodic-joy
    pip install tellopy
    rosrun joy joy_node 
    sudo chmod a+rw /dev/input/js0
    catkin_create_pkg tello_ctrl std_msgs rospy roscpp
    cd src
    git submodule init
    git submodule update
    cd ..
    catkin_make -DPYTHON_EXECUTABLE:FILEPATH=~/.virtualenvs/py3venv/bin/python
## Run
    source devel/setup.bash

## TRT Pose

- Resnet18 model weights [here](https://drive.google.com/file/d/1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd/view)

## Reference
1. https://github.com/dji-sdk/Tello-Python
2. https://github.com/clydemcqueen/flock
3. https://github.com/dbaldwin/DroneBlocks-Tello-Python
4. https://github.com/geaxgx/tello-openpose
5. https://github.com/Ubotica/telloCV/blob/master/telloCV.py
6. https://github.com/hanyazou/TelloPy
