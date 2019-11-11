# ya-yolo
Yet Another Yolo Implementation in Pytorch - convertible via onnx to coreml

Many Yolo v3 implementations can be found on but most of them are either not 
actively supported or it remains a big hassle to re-train them on custom 
datasets and to convert them to coreml.
 
This goal of this implementation is to provide:
* an accurate implementation of the original Yolo v3 pytorch
* to make it easy to plug in custom datasets and re-train with adjustable loss functions
* to provide code to train and test custom models  
* to keep the model convertible to coreml



### How to develop ya-yolo

1. Have [pipenv](https://pipenv.readthedocs.io/en/latest/) installed with a Python 3.6 environment
2. ``` cd ~/ya-yolo ```
3. Create your virtual Python 3.6 environment with all needed dependencies using pipenv
``` pip install```  





### How to use ya-yolo as a pip dependency in your project


