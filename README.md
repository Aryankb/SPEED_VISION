# DISCLAIMER
Hey, i have created a vehicle speed detection and no. plate capturing system by modifying the code base and fine tuning of YOLO V9 Model.



# FEW THINGS TO NOTE
**changed coco.yaml for training**

**changed yolov9-c.yaml for training**

**changed yolov9-e.yaml for training**

**changed utils/general.py line 903 ,, addad [1]**

**changed detect.py for reading text from liscence during detection**





# INSTALLATION_FOR_SPEED_VISION
can use docker if want follow the installation guide **below** if using docker

make virtual env / conda env with python 3.10.11

cloone the repo

go into project folder

install nvidia gpu driver if gpu available (chat gpt se puch lena kaise kar)

pip install -r requirements.txt (do this after removing torch and torchvision from requirements.txt if GPU AVAILABLE)

if gpu not available simple install requirements.txt

install torch Version: 2.1.0+cu118 and torchvision Version: 0.16.0+cu118 if GPU available

pip install "paddleocr>=2.0.1"

pip install prox

pip install common

data

tight

dual

flask

pip install paddle

pip install paddlepaddle

ultralytics

motpy

collections

math

subprocess

csv

change the command in app.py for sourcing and environment activation according to your pc directories

if not using gpu, in the command in app.py , write cpu in place of 0  in --device 0

Install some additional files if they are the dependencies of other installed files (error aa jayega to dikh jayega kya install krna hai)

go into env and **python app.py**


