# Detection-using-Detectron2 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZNCchcQD1RhbVlKvwKijkYPZiEl9D9W8?usp=sharing) <br />
**Detectron2** is a framework written on Pytorch that helps training an object detection model in a snap with a custon dataset. It presents multiple models [model zoo] (https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) trained on coco dataset so we just need to fine tune our custom dataset on the pre-trained model. <br />
In this repository, we are going to deal with identifying empty and occupied parking lots using Faster RCNN model from detectron2 model zoo: **The faster_rcnn_R_50_FPN_3x** <br />
![model](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/image.png)
 <br />
To train detectron2 we need to follow these steps: <br />
1. Installing detectron2 
2. Preparing and registering the dataset
  For our case, we are using [PKLot dataset](https://public.roboflow.com/object-detection/pklot) and we got it in a coco formatfrom [Roboflow] (https://public.roboflow.com/object-detection/pklot/1/download/coco) <br />
3. Training the model
  The training curves are visualised using tensorboard: <br />
  ![plot1](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/1.png)
  ![plot2](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/2.png)
  ![plot3](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/3.png)
4. Inference using the trained model
  Here are some results: <br />
  ![Res1](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/t%C3%A9l%C3%A9chargement%20(3).png)
  ![Res2](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/t%C3%A9l%C3%A9chargement%20(4).png)
  ![Res3](https://github.com/WidedCHERIF/Detection-using-Detectron2/blob/test/t%C3%A9l%C3%A9chargement%20(5).png)
  
 



