# Classification at PyTorch
Various implementations of classification models using PyTorch framework.


## Environments
* python==3.7
* torch==1.5.0
* torchvision==0.6.0
* others: pyyaml, torchsummary


## Run
### 1. Train
```
# CIFAR100
python main.py --name base --idx 0 -g 0 -j 8 --dataset cifar100 --datapath ../data --arch resnet --layers 20 --batch-size 128 --run-type train --epochs 300 --lr 0.1 --sched cosine --sched-batch --wd 1e-4 --print-freq 100

# ImageNet
python main.py --name base --idx 0 -g 0 1 2 3 -j 16 --dataset imagenet --datapath /dataset/ImageNet --arch resnet --layers 18 --batch-size 256 --run-type train --epochs 90 --lr 0.1 --sched cosine --sched-batch --wd 1e-4 --print-freq 1000
```

### 2. Resume
```
python main.py --gpu 0 --workers 8 --dataset cifar100 --datapath ../data --run-type train --resume logs/resnet20/cifar100/base/0
```

### 3. Evaluate
```
python main.py --gpu 0 --workers 8 --dataset cifar100 --datapath ../data --run-type evaluate --eval logs/resnet20/cifar100/base/0
```


## Experiments
| Model | CIFAR10 (%) | CIFAR100 (%) | ImageNet (%) |
| :------- | :-------: | :-------: | :-------: |
| ResNet-20 [[1]](#1) | - | - | - |
| ResNet-56 | - | 71.56 | - |
| PreActResNet-20 [[2]](#2) | - | - | - |
| PreActResNet-56 | - | - | - |
| WideResNet | - | - | - |
| ShuffleNetV2 | - | - | - |
| MobileNetV2 | - | - | - |
| ReXNet | - | - | - |
| - | - | - | - |


## Citations
<a id="1">[1]</a> He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

<a id="2">[2]</a> He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.
