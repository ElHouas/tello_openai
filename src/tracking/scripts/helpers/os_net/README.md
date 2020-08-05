# OSNet

### Installation

```
#torch & torchvison
#torch=1.4.0
#torchvision= 0.5

wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip3 install Cython
pip3 install numpy torch-1.4.0-cp36-cp36m-linux_aarch64.whl

sudo apt-get install libjpeg-dev zlib1g-dev
git clone --branch v0.5.0 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install

#Torch2trt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```


### Create an engine

```
git clone https://github.com/KaiyangZhou/deep-person-reid

#Install missing with pip install...
numpy
Cython
h5py
Pillow
six
scipy
opencv-python
matplotlib
future
yacs
gdown
flake8
yapf
isort

```

then place the osnet2trt.py in the deep-person-reid folder