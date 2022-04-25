# Detection using detectron2 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZNCchcQD1RhbVlKvwKijkYPZiEl9D9W8?usp=sharing) <br />
**Detectron2** is a framework written on Pytorch that helps training an object detection model in a snap with a custon dataset. It presents multiple models [model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) trained on coco dataset so we just need to fine tune our custom dataset on one of the pre-trained model. <br />
In this repository, we are going to deal with identifying empty and occupied parking lots using Faster RCNN model from detectron2 model zoo: **The faster_rcnn_R_50_FPN_3x** <br />
![model](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/image.png)
 <br />
To train detectron2 we need to follow these steps: <br />
1. Install detectron2 <br />
2. Prepare and register the dataset <br />
  For our case, we are using [PKLot dataset](https://public.roboflow.com/object-detection/pklot). The dataset is downloaded in a coco format from [Roboflow](https://public.roboflow.com/object-detection/pklot/1/download/coco) <br />
3. Train the model <br />
  The training curves are visualised using tensorboard: <br />
  ![plot1](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/1.png)
  ![plot2](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/2.png)
  ![plot3](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/3.png) <br />
  You can download the trained model from 
4. Inference using the trained model <br />
  Here are some results: <br />
  ![Res1](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/Res1.png)
  ![Res2](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/Res2.png) <br />
  ![Res3](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/Res3.png) <br />
  Usually, the model is evaluated following the COCO Standards of evaluation. Mean Average Precision (mAP) is used to evaluate the performance of the model. <br />
  We get an accuracy of around **94.69%** for an IoU of 0.5 and around **88.93%** for an IoU of 0.75 which is not bad! <br />
  ![Eval](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/4.png)
  
## Dependences 
python>=3.6 <br />
torch==1.3.0+cu100 <br />
torchvision==0.4.1+cu100 <br />
tensorboard <br />
cython <br />
jupyter <br />
scikit-image <br />
numpy <br />
opencv-python <br />
pycocotools <br />
pyyaml==5.1 <br />

## Install Detectron2
```
# Install detectron2 that matches the above pytorch version
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
pip install detectron2 
```
## Download Dataset
```
cd .\Detection-using-Detectron2
curl -L "https://public.roboflow.com/ds/gPbookuOTI?key=kfody3xy1u" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip --output /content/sample_data/
```
## Trained Model
Please dowload from [Google Drive](https://drive.google.com/file/d/1ltLQukzgkEOC6fNUAbrTHrWyag7R7bbM/view?usp=sharing) and put in  .\Detection-using-Detectron2 <br />

## Train
```
python Pklot.py --mode train
```

## Test
```
python Pklot.py --mode test
```


Please check [Colab](https://colab.research.google.com/drive/14y3ThHeopbAJLiQymn9Z9GLFP-l2sw04?usp=sharing)  for other application of detectron2 (training a balloon segmentation model). 

 



