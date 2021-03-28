# Classification at PyTorch
Various implementations of classification models using PyTorch framework.


## Requirements
* python==3.7
* torch==1.5.0
* torchvision==0.6.0
* pyyaml, torchsummary


## Implementations
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

Citations
---
<a id="1">[1]</a> Zhou, Shuchang, et al. "Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients." arXiv preprint arXiv:1606.06160 (2016).

<a id="2">[2]</a> Choi, Jungwook, et al. "Pact: Parameterized clipping activation for quantized neural networks." arXiv preprint arXiv:1805.06085 (2018).

<a id="3">[3]</a> Jung, Sangil, et al. "Learning to quantize deep networks by optimizing quantization intervals with task loss." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

<a id="4">[4]</a> Esser, Steven K., et al. "Learned step size quantization." arXiv preprint arXiv:1902.08153 (2019).

<a id="5">[5]</a> Jin, Qing, Linjie Yang, and Zhenyu Liao. "Towards efficient training for neural network quantization." arXiv preprint arXiv:1912.10207 (2019).
