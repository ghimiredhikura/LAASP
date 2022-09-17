# LAASP

Implementation of LAASP with PyTorch. Pruning the selected set of filters & restructuring the network is based on [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning).

![alt text](images/LAASP_flyer.png)

## Table of Contents

- [Requirements](#requirements)
- [Models](#models)
- [VGGNet on CIFAR10](#vggnet-on-cifar10)
  - [Training-Pruning]()
  - [Evaluation]()
- [ResNet on CIFAR-10](#resnet-on-cifar10)
  - [Training-Pruning]()
  - [Evaluation]()
- [ResNet on ImageNet](#resnet-on-imagenet)
  - [Prepare ImageNet dataset]()
  - [Training-Pruning]()
  - [Evaluation]()

## Requirements
- Python 3.9.7
- PyTorch 1.10.2
- TorchVision 0.11.2
- matplotlib 3.5.1
- scipy 1.8.0

`Note: These are the verified version of the tools used in the experiment. You can test with other versions as well.` 

-- in--progress --