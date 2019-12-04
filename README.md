# ya-yolo
Yet Another YOLOv3 Implementation in PyTorch

There might already be too many implementations of [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) 
on github but unfortunately none of them has it all. Either they are incomplete, buggy, 
not out of the box convertible to coreml or even onnx, or it remains just a big hassle to 
re-train them on custom datasets.
 
This goal of this implementation is:
* to provide an accurate implementation of the original YOLOv3 paper in PyTorch
* to make it easy to plug in custom datasets and re-train with adjustable loss functions
* to provide code to train and test these custom models  
* to keep the PyTorch model convertible to [coreml](https://developer.apple.com/documentation/coreml) via [onnx](https://onnx.ai/)


### How to develop ya-yolo

1. Have [pipenv](https://pipenv.readthedocs.io/en/latest/) installed with a Python 3.6 environment
2. Check out this repo 
    ```shell script
    git clone git@github.com:petersiemen/ya-yolo.git
    ```
3. Create your virtual Python 3.6 environment with all needed dependencies using pipenv
    ```shell script
    cd ~/ya-yolo
    pip install
    ```  
4. Download the original weight parameters used in YOLOv3
    ```shell script
    cd ~/ya-yolo/cfg
    ./get_weights.sh
    ```

### How to run jupyter notebooks on an ec2 instance
1. ssh into your ec2 instance
2. Follow the steps from 'How to develop ya-yolo'
3. Generate config for jupyter and run notebook
    ```shell script
    jupyter notebook --generate-config    
    jupyter notebook --ip=0.0.0.0 --no-browser    
    ```


### How to use ya-yolo as a pip dependency in your project


#### Notes from Yolo papers how to train from scratch

#####Pretraining (classification only)
Dataset: ImageNet 1000-class competition dataset  
Train first 20 convolutional layers followed by a average-pooling layer and a fully connected layer
Achieving a top-5 accuracy (target label is among the top 5 predictions) on the ImageNet 2012 validation set

#####Training

Add four convolutional layers and two fully connected layers with randomly initialized weights
Increase input resolution from 224x224 to 448x448
Normalize bounding-box (width and height by image width and height, x and y offsets to grid cell location) so that 
all output values are bound between 0 and 1

  



### Datasets Used
ImageNet 2012:
##### Classification Task
1000 categories 
train images: 1.2 mio
test images: 150k
validation images: 150k


