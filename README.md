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


