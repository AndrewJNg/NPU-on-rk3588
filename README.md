# openpilot-on-rk3588
for rknn-toolkit2 V1.6.0

## System setup
[https://github.com/airockchip/rknn-toolkit2/tree/master]

#### Ubuntu PC (ubuntu 20.04)
1) git clone https://github.com/airockchip/rknn-toolkit2/tree/master
2) cd rknn-toolkit2/rknn-toolkit2/packages/
3) pip3 install rknn_toolkit2-1.6.0+81f21f4d-cp38-cp38-linux_x86_64.whl (python 3.8 for this example)
4) Run in python terminal (library test, successful when no errors are returned)
    - from rknnlite.api import RKNNLite

#### Orange pi 5 (ubuntu 22.04 by Joshua-Riek) [https://github.com/Joshua-Riek/ubuntu-rockchip/wiki/Orange-Pi-5-Plus]
1) git clone https://github.com/airockchip/rknn-toolkit2/tree/master
2) cd rknn-toolkit2/rknn_toolkit_lite2/packages/
3) pip3 install rknn_toolkit_lite2-1.6.0-cp310-cp310-linux_aarch64.whl (python 3.10 for this example)
4) Run in python terminal (library test, successful when no errors are returned)
    - from rknnlite.api import RKNNLite





## Running the conversion on Ubuntu PC (python)
#### Running dmonitoring as example
1) cd dmonitoring_python
2) sudo python convert_dmonitoring.py

## Running the NPU on Orange Pi (python)
#### Running dmonitoring as example
1) cd dmonitoring_python
2) sudo python run_dmonitoring.py


## troubleshoot
When the library test returns an error, it means that the library version is mismatch
- suggest to make python virtualenv using 
"python[version] -m venv [virtual-environment-name]"
    https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/