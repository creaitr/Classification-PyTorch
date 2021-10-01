# Classification at PyTorch
Various implementations of classification models using PyTorch framework.


## Test Environments
* python==3.9
* torch==1.9.0
* torchvision==0.10.0


## Run
### 1. Train
```
# CIFAR100
python main.py --name base --idx 0 -g 0 -j 8 --dataset cifar100 --datapath ../data --arch resnet --layers 20 --batch-size 128 --run-type train --epochs 300 --wd 1e-4 --lr 0.1 --lr-scheduler cosine --step-location batch --print-freq 100

# ImageNet
python main.py --name base --idx 0 -g 0 1 2 3 -j 16 --dataset imagenet --datapath /dataset/ImageNet --arch resnet --layers 18 --batch-size 256 --run-type train --epochs 100 --wd 1e-4 --lr 0.1 --lr-scheduler cosine --step-location batch --print-freq 1000
```

### 2. Resume
```
python main.py --gpu 0 --workers 8 --dataset cifar100 --datapath ../data --run-type train --resume --load logs/resnet-20/cifar100/base/0
```

### 3. Evaluate
```
python main.py --gpu 0 --workers 8 --dataset cifar100 --datapath ../data --run-type validate --load logs/resnet-20/cifar100/base/0
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


## References
* [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)