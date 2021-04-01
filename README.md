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
python main.py --name base --idx 0 -g 0 -j 8 --dataset cifar100 --datapath ../data --arch resnet --layers 20 --batch-size 128 --run-type train --epochs 300 --lr 0.1 --sched cosine --sched-batch --wd 1e-4

# ImageNet
python main.py --name base --idx 0 -g 0 1 2 3 -j 16 --dataset imagenet --datapath /dataset/ImageNet --arch resnet --layers 18 --batch-size 256 --run-type train --epochs 90 --lr 0.1 --sched cosine --sched-batch --wd 1e-4
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
---

- DoReFa-net [[1]](#1)
- PACT [[2]](#2)
- QIL [[3]](#3) (in progress)
- LSQ [[4]](#4)
- SAT [[5]](#5) (in progress)

Training Command
---
For the first, to train a pre-trained and full-precision model,

    python3 main.py --name FP --gpu 0

Then you can load the pre-trained checkpoint to initialize at low-precision training,

    python3 main.py --name 4B --gpu 0 --quatizer lsq --bitw 4 --bita 4 --init pre-trained/full-precision/model/ckpt.pth

Evaluation
---
Evaludation of ResNet-18 on CIFAR-100 (full-precision: 77.05)

| Bit-W / Bit-A        | 2 / 2 | 3 / 3 | 4 / 4 | 5 / 5 | 8 / 8 |
| :---                 | :---: | :---: | :---: | :---: | :---: |
| LSQ [[4]](#4)        | 00.00 | 00.00 | 76.94 | 00.00 | 00.00 |
| SAT [[5]](#5)        | 00.00 | 00.00 | 00.00 | 00.00 | 00.00 |
| QIL [[3]](#3)        | 00.00 | 00.00 | 00.00 | 00.00 | 00.00 |
| PACT [[2]](#2)       | 00.00 | 00.00 | 00.00 | 00.00 | 00.00 |
| DoReFa-net [[1]](#1) | 00.00 | 00.00 | 00.00 | 00.00 | 00.00 |


| | CIFAR10 || CIFAR100 ||
| Acc (%) | Top-1 | Top-5 | Top-1 | Top-5 |
| :------- | :-------: | :-------: | :-------: | :-------: |
| ResNet-20 | - | - | - | - |
| ResNet-56 | - | - | 71.56 | - |

Citations
---
<a id="1">[1]</a> Zhou, Shuchang, et al. "Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients." arXiv preprint arXiv:1606.06160 (2016).

<a id="2">[2]</a> Choi, Jungwook, et al. "Pact: Parameterized clipping activation for quantized neural networks." arXiv preprint arXiv:1805.06085 (2018).

<a id="3">[3]</a> Jung, Sangil, et al. "Learning to quantize deep networks by optimizing quantization intervals with task loss." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

<a id="4">[4]</a> Esser, Steven K., et al. "Learned step size quantization." arXiv preprint arXiv:1902.08153 (2019).

<a id="5">[5]</a> Jin, Qing, Linjie Yang, and Zhenyu Liao. "Towards efficient training for neural network quantization." arXiv preprint arXiv:1912.10207 (2019).
